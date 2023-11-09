
# Instructions

In the handbook, for each training step we provide two sets of recipes:
- Full training on a multi-GPU machine (tested on a 8xA100 node), using slurm to queue jobs.
- LORA taining on a single consumer 24GB GPU (tested on a RTX 4090)

The full training jobs will scale to a multi-node setting, by adjusting `--nodes=1`, we advise adjusting the gradient accumulation steps and/or batch size if you want to replicate our results.


## Full training examples 

### SFT

```shell
sbatch --job-name=handbook_sft --nodes=1 recipes/launch.slurm zephyr-7b sft full deepspeed_zero3
```

## DPO
```shell
sbatch --job-name=handbook_sft --nodes=1 recipes/launch.slurm zephyr-7b sft full deepspeed_zero3
```

## LORA training examples

### SFT
```shell
# locally on 1 gpu
accelerate launch scripts/run_sft.py recipes/zephyr-7b/sft/config_lora.yaml
```

```shell
# on a cluster
sbatch --job-name=handbook_sft_lora --nodes=1 recipes/launch.slurm zephyr-7b sft lora multi_gpu "--gradient_accumulation_steps=16"
```

### SFT

```shell
# locally on 1 gpu
accelerate launch scripts/run_dpo.py recipes/zephyr-7b/dpo/config_lora.yaml
```

```shell
# on a cluster
sbatch --job-name=handbook_dpo_lora --nodes=1 recipes/launch.slurm zephyr-7b dpo lora multi_gpu "--gradient_accumulation_steps=8"
```