# Instructions to train SmolLM3-3B

We are open-sourcing all the artifacts to train [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). You can find the configuration files for the three post-training stages (mid-training, SFT, and DPO) in the [`sft`](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3/sft) and [`dpo`](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm3/dpo) directories.

## Setup

Make sure you followed the installation instructions in the [README.md](README.md) file. We tested the training setup with 8 GPUs (80GB of VRAM) to train the full model.

## Full training examples

```shell
# Step 1 - Mid-Training
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/smollm3/sft/mid.yaml --gradient_accumulation_steps 16

# Step 2 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/smollm3/sft/sft.yaml --gradient_accumulation_steps 16

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/dpo.py --config recipes/smollm3/dpo/apo.yaml --gradient_accumulation_steps 4
```