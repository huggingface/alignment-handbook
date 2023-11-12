
# Scripts to Train and Evaluate Chat Models

## Fine-tuning

In the handbook, we provide three main ways to align LLMs for chat:

- Full fine-tuning on a multi-GPU machine with DeepSpeed ZeRO-3 (tested on an 8 x A100 (80GB) node).
- LoRA or QLoRA fine-tuning on a single consumer 24GB GPU (tested on an RTX 4090).
- LoRA fine-tuning on a multi-GPU machine with DeepSpeed ZeRO-3 (tested on a 2 x A100s (80GB)).

In practice, we find comparable performance for both full and LoRA fine-tuning, with the latter having the advantage of producing small adapter weights that are fast to upload and download from the Hugging Face Hub. Here are the general commands to fine-tune your models:

```shell
# Full training with ZeRO-3 on 8 GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml

# LoRA training on a single GPU
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml

# QLoRA 4-bit training on a single GPU
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml --load_in_4bit=true

# LoRA training with ZeRO-3 on two or more GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes={num_gpus} scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml
```

Here `{task}` refers to the type of training you wish to run (SFT, DPO, etc), while `{model_name}` refers to the choice of a recipe in the `recipes` directory. For example, to replicate Zephyr-7B-β you can run:

```shell
# Step 1 - train SFT policy
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml

# Step 2 - align with DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

**💡 Tip:** If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant (and thus replicate our results).

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows:

```shell
# Change batch size, number of epochs etc
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml --per_device_train_batch_size=42 --num_train_epochs=5
```

By default all training metrics are logged with TensorBoard. If you have a [Weights and Biases](https://wandb.ai/site) account and are logged in, you can view the training metrics by appending `--report_to=wandb`, e.g.

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml --report_to=wandb
```

## Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `recipes/launch.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=handbook_{task} --nodes=1 recipes/launch.slurm {model_name} {task} {precision} {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{precision}` refers to the type of training (`full` vs `lora`) and `{accelerator}` refers to the choice of 🤗 Accelerate config in `recipes/accelerate_configs`. If you wish to override the default config parameters, you can provide them by appending a space-separated string like `'--arg1=value1 --arg2=value2'. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
# Launch on Slurm and override default hyperparameters
sbatch --job-name=handbook_sft --nodes=1 recipes/launch.slurm zephyr-7b-beta sft full deepspeed_zero3 '--per_device_train_batch_size=42 --num_train_epochs=5'
```

You can scale the number of nodes by increasing the `--nodes` flag.

**⚠️ Note:** the configuration in `recipes/launch.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.

## Fine-tuning on your datasets

Under the hood, each training script uses the `get_datasets()` function which allows one to easily combine multiple datasets with varying proportions. For instance, this is how one can specify multiple datasets and which splits to combine in one of the YAML configs:

```yaml
datasets_mixer:
    dataset_1: 0.5  # Use 50% of the training examples
    dataset_2: 0.66 # Use 66% of the training examples
    dataset_3: 0.10 # Use 10% of the training examples
dataset_splits:
- train_xxx         # The training splits to mix
- test_xxx          # The test splits to mix
```

If you want to fine-tune on your datasets, the main thing to keep in mind is how the chat templates are applied to the dataset blend. Since each task (SFT, DPO, etc), requires a different format, we assume the datasets have the following columns:

**SFT**

* `messages`: A list of `dicts` in the form `{"role": "{role}", "content": {content}}`. 
* See [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) for an example.

**DPO**

* `chosen`: A list of `dicts` in the form `{"role": "{role}", "content": {content}}` corresponding to the preferred dialogue.
* `rejected`: A list of `dicts` in the form `{"role": "{role}", "content": {content}}` corresponding to the dispreferred dialogue.
* See [ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) for an example.

We also find it useful to include dedicated splits per task in our datasets, so e.g. we have:

* `{train,test}_sft`: Splits for SFT training.
* `{train,test}_gen`: Splits for generation ranking like rejection sampling or PPO.
* `{train,test}_prefs`: Splits for preference modelling, like reward modelling or DPO.

If you format your dataset in the same way, our training scripts should work out of the box!

## Evaluating chat models

We recommend benchmarking chat models on:

* [MT-Bench](https://huggingface.co/spaces/lmsys/mt-bench): a multi-turn benchmark spanning 80 dialogues and 10 domains.
* [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval): a single-turn benchmark which evaluates the helpfulness of chat and instruct models against `text-davinci-003`.

For both benchmarks, we have added support for the [Zephyr chat template](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full/blob/ac6e600eefcce74f5e8bae1035d4f66019e93190/tokenizer_config.json#L30) (which is the default produced by our scripts), so you can evaluate models produced by our scripts as follows:

**MT-Bench**

* Follow the installation instructions [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
* Make sure the word `zephyr` exists in the `--model-path` argument when generating the model responses [here](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#step-1-generate-model-answers-to-mt-bench-questions). This will ensure the correct chat template is loaded. For example, the following model name is valid: `--model-path {hub_username}/my-baby-zephyr`
* Generate the model responses and GPT-4 rankings.

**AlpacaEval**

* Follow the installation instructions [here](https://github.com/tatsu-lab/alpaca_eval#quick-start)
* Copy-paste the [config](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml) for `zephyr-7b-beta` and place it in the `model_configs` directory under `{your_zephyr_model}`.
  * Next, update the [config name](https://github.com/tatsu-lab/alpaca_eval/blob/2daa6e11b194653043ca74f735728dc068e04aae/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml#L1) and [Hub model ID](https://github.com/tatsu-lab/alpaca_eval/blob/2daa6e11b194653043ca74f735728dc068e04aae/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml#L5) to match your model name.
* Follow the steps to evaluate your model [here](https://github.com/tatsu-lab/alpaca_eval/tree/main#evaluating-a-model).

Note that MT-Bench and AlpacaEval rely on LLMs like GPT-4 to judge the quality of the model responses, and thus the ranking exhibit various biases including a preference for models distilled from GPTs. For that reason, we also recommend submitting your best models for human evaluation in:

* [Chatbot Arena](https://chat.lmsys.org): a live, human evaluation of chat models in head-to-head comparisons.