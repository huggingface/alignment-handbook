
# Instructions to train Zephyr TinyLlama with SFT

This model is fine-tuned via SFT, we used the [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset, which is a heavily filtered version of the UltraChat dataset and was used to train Zephyr-7B-β.

See below for commands to train these models.

## Full training examples

You can finetune a model on any 16GB GPU, using QLoRA. A recipe involving DPO will come soon 🤗.


```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zephyr-tinyllama/sft/config_qlora.yaml --load_in_4bit=false
```
