#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import subprocess
import sys
from datetime import timedelta

import torch
import transformers
from transformers import set_seed

import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from h4.data import get_datasets
from h4.training import DataArguments, DPOTrainingArguments, ModelArguments, init_wandb_training
from h4.utils import (
    H4ArgumentParser,
    apply_chat_template,
    convert_to_safetensors,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    hf_login,
    is_slurm_available,
    push_to_hub_revision,
    run_mt_bench_job,
)
from trl import DPOTrainer


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOTrainingArguments))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Setup WandB
    if training_args.wandb_enabled:
        init_wandb_training(training_args)

    # Login to HuggingFace Hub if needed
    hf_login()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=6 * 1800))])

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map(),
        quantization_config=get_quantization_config(model_args),
    )

    ref_model = model_args.model_name_or_path
    ref_model_kwargs = model_kwargs

    if model_args.use_peft:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    dpo_trainer = DPOTrainer(
        model_args.model_name_or_path,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_seq_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = dpo_trainer.evaluate(eval_dataset=raw_datasets["test"])
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["test"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["test"]))
        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    dpo_trainer.save_model(training_args.output_dir)

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        kwargs["dataset"] = list(data_args.dataset_mixer.keys())
        dpo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dpo_trainer.model.config.use_cache = True
        # Fix custom code paths
        if model_args.trust_remote_code is True:
            auto_map = dpo_trainer.model.config.auto_map
            dpo_trainer.model.config.auto_map = {k: v.split("--")[-1] for k, v in auto_map.items()}
        dpo_trainer.model.config.save_pretrained(training_args.output_dir)
        # FSDP/DeepSpeed save the model as a single `pytorch_model.bin` file, so we need to shard it.
        # We run this in a subprocess to avoid interference from the accelerators.
        subprocess.run(
            [
                "python",
                "scripts/training/shard_checkpoint.py",
                f"--output_dir={training_args.output_dir}",
                f"--trust_remote_code={model_args.trust_remote_code}",
            ],
            check=True,
        )
        # Convert torch weights to safetensors for deployment with TGI
        convert_to_safetensors(training_args.output_dir)
        if training_args.push_to_hub_revision:
            is_model_on_hub = push_to_hub_revision(training_args, model_args)
            # Run automatic evaluation once the model is pushed to the Hub
            if is_slurm_available() and is_model_on_hub is True and training_args.do_eval is True:
                logger.info("*** Launching MT Bench ***")
                run_mt_bench_job(training_args, model_args)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()
    wandb.finish()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
