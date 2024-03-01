
# Instructions to Replicate Zephyr 7B Gemma

Similar to how we trained Zephyr 7B Beta in [technical report](https://huggingface.co/papers/2310.16944), training this model proceeds in two steps:

1. Apply SFT to fine-tune Gemma 7B on a filtered version of the UltraChat dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)). The result is an SFT model like [`zephyr-7b-sft-full`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full) or [`zephyr-7b-sft-qlora`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-qlora).
2. Align the SFT model to AI feedback via DPO on a preprocessed version of the UltraFeedback dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)). The result is an DPO model like [`zephyr-7b-dpo-full`](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-full) or [`zephyr-7b-dpo-qlora`](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-qlora).


See below for commands to train these models using either DeepSpeed ZeRO-3 or LoRA.

## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model - alternatively, you can train on 1 GPU by adjusting the micro batch size and gradient accumulation steps to keep the global batch size constant. A recipe involving QLoRA will come later 🤗.

```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-gemma/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-gemma/dpo/config_full.yaml
```