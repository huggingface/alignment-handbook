
# Gemma 7B Alignment

## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model.
```shell
# Step 1 - SFT
TRANSFORMERS_VERBOSITY=info ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/gemma-7b/sft/config_full_1k.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

## QLoRA training examples

```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_qlora.yaml --load_in_4bit=true

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_qlora.yaml
```