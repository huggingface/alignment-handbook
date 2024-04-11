
# Instructions to train StarChat2

Similar to how we trained Zephyr 7B Beta in our [technical report](https://huggingface.co/papers/2310.16944), training this model proceeds in two steps:

1. Apply SFT to fine-tune [StarCoder2 15B](https://huggingface.co/bigcode/starcoder2-15b) on a blend of chat, code, and math datastets. The result is an SFT model like [`starchat2-15b-sft-v0.1`](https://huggingface.co/HuggingFaceH4/starchat2-15b-sft-v0.1).
2. Align the SFT model to AI feedback via DPO on the UltraFeedback and Orca DPO Pairs datasets. The result is a DPO model like [`starchat2-15b-v0.1`](https://huggingface.co/HuggingFaceH4/starchat2-15b-v0.1).

See below for commands to train these models using DeepSpeed ZeRO-3.

## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model - alternatively, you can train on 1 GPU by adjusting `per_device_train_batch_size` and `gradient_accumulation_steps` to keep the global batch size constant. A recipe involving QLoRA will come later 🤗.

```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/starchat2-15b/sft/config_v0.1.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/starchat2-15b/dpo/config_v0.1.yaml
```
