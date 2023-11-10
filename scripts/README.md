
## Scripts to Train and Evaluate Chat Models

### Fine-tuning

In the handbook, we provide three main ways to align LLMs for chat:

- Full fine-tuning on a multi-GPU machine with DeepSpeed ZeRO-3 (tested on an 8 x A100 (80GB) node).
- LoRA fine-tuning on a single consumer 24GB GPU (tested on a RTX 4090).
- LoRA fine-tuning on a multi-GPU machine with DeepSpeed ZeRO-3 (tested on a 2 x A100s (80GB)).

In practice, we find comparable performance for both full and LoRA fine-tuning, with the latter having the advantage of producing small adapter weights that are fast to upload and download from the Hugging Face Hub. Here's the two general commands to fine-tune your models:

```shell
# Full training with ZeRO-3 on 8 GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml

# LoRA training on a single GPU
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml

# LoRA training with ZeRO-3 on two or more GPUs
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes={num_gpus} scripts/run_{task}.py recipes/{model_name}/{task}/config_lora.yaml
```

Here `{task}` refers to type of training you wish to run (SFT, DPO, etc), while `{model_name}` refers to the choice of recipe in the `recipes` directory. For example, to replicate Zephyr-7B-β you can run:

```shell
# Step 1 - train SFT policy
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml

# Step 2 - align with DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

** 💡 Tip:** If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant (and thus replicate our results).

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows:

```shell
# Change batch size, number of epochs etc
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml --per_device_train_batch_size=42 --num_train_epochs=5
```

By default all training metrics are logged with TensorBoard. If you have a [Weights and Biases](https://wandb.ai/site) account and are logged in, you can view the training metrics by appending `--report_to=wandb`, e.g.

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_{task}.py recipes/{model_name}/{task}/config_full.yaml --report_to=wandb
```

### Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `recipes/launch.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=handbook_{task} --nodes=1 recipes/launch.slurm {model_name} {task} {precision} {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{precision}` refers to the type of training (full vs LoRA) and `{accelerator}` refers to the choice of 🤗 Accelerate config in `recipes/accelerate_configs`. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
sbatch --job-name=handbook_sft --nodes=1 recipes/launch.slurm zephyr-7b-beta sft full deepspeed_zero3
```

You can scale the number of nodes by increasing the `--nodes` flag.

**⚠️ Note:** the configuration in `recipes/launch.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.