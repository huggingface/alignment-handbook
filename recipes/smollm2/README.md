
# Instructions to train SmolLM2-1.7B-Instruct

We build the [SmolLM2-Instruct](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) by doing SFT on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) and then DPO on [UltraFeedBack](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

## Setup

Follow the installation instructions in https://github.com/huggingface/alignment-handbook/tree/main?tab=readme-ov-file#installation-instructions 

## Training
We train the 1.7B on 8 GPUs using the following command:

```shell
# SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/smollm2/sft/config.yaml

# DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/smollm2/dpo/config.yaml
```

For the 135M and 360M we use [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) dataset for SFT and UltraFeedback for DPO:
```shell
# SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/smollm2/sft/config_smol.yaml

# DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/smollm2/dpo/config_smol.yaml
```