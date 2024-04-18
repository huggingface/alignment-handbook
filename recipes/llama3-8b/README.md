
# Instructions to train Zephyr-141B-A35B with ORPO

This model is fine-tuned via a novel alignment algorithm called [Odds Ratio Preference Optimization (ORPO)](https://huggingface.co/papers/2403.07691). ORPO does not require an SFT step to achieve high performance and is thus much more computationally efficient than methods like DPO and PPO. To train Zephyr-141B-A35B, we used the [`argilla/distilabel-capybara-dpo-7k-binarized`](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized) preference dataset, which consists of synthetic, high-quality, multi-turn preferences that have been scored via LLMs.

See below for commands to train these models using FSDP. **Note:** we found it was not possible to train this large model with DeepSpeed ZeRO-3 due to unresolved NCCL errors which cause GPUs to hang. 

## Full training examples

You will require 4 nodes of 8 GPUs (80GB of VRAM) to train the full model - alternatively, you may be able to train on fewer GPUs by adjusting `per_device_train_batch_size` and `gradient_accumulation_steps` and `num_train_epochs` to keep the global batch size constant. A recipe involving QLoRA will come later 🤗.

To run with Slurm, use:

```shell
sbatch --job-name=handbook_sft --nodes=4 recipes/launch.slurm zephyr-141b-A35b orpo full fsdp
```

Under the hood, this calls the following script which can be adapted to other models and datasets:


```shell
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file recipes/accelerate_configs/fsdp.yaml scripts/run_orpo.py recipes/zephyr-141b-A35b/orpo/config_full.yaml
```