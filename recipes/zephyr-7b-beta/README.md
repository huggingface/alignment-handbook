
# Instructions to Replicate Zephyr-7b-β

As described in the Zephyr [technical report](https://huggingface.co/papers/2310.16944), training this model proceeds in two steps:

1. Apply SFT to fine-tune Mistral 7B on a filtered version of the UltraChat dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)). The result is an SFT model like [`zephyr-7b-sft-full`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-full) or [`zephyr-7b-sft-qlora`](https://huggingface.co/alignment-handbook/zephyr-7b-sft-qlora).
2. Align the SFT model to AI feedback via DPO on a preprocessed version of the UltraFeedback dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)). The result is a DPO model like [`zephyr-7b-dpo-full`](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-full) or [`zephyr-7b-dpo-qlora`](https://huggingface.co/alignment-handbook/zephyr-7b-dpo-qlora).

**Note:** after the release of Zephyr, the team at [Argilla](https://argilla.io) found that the source UltraFeedback dataset had a few thousand incorrect preference labels from GPT-4. Additionally, TRL's `SFTTrainer` had a bug in the learning rate scheduler which terminated training early. Accounting for these changes led us to find a better set of hyperparameters from those described in the technical report. In particular, for DPO training we found that training for 1 epoch with `beta=0.01` was sufficient to achieve comparable performance to `zephyr-7b-beta` (vs. 3 epochs with `beta=0.1`).

See below for commands to train these models using either DeepSpeed ZeRO-3 or LoRA.

## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model.
```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full.yaml
```

## QLoRA training examples

Train faster with flash-attention 2 (GPU supporting FA2: A100, H100, etc)
```````shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_qlora.yaml --load_in_4bit=true

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_qlora.yaml
```````

P.S. Using Flash Attention also allows you to drastically increase the batch size (x2 in my case)

Train without flash-attention (i.e. via PyTorch's scaled dot product attention):
```````shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_qlora.yaml --load_in_4bit=true --attn_implementation=sdpa

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_qlora.yaml --attn_implementation=sdpa
```````