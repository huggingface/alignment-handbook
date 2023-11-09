
## Supervised Fine-Tuning (SFT)

We provide 3 main ways to train SFT models:

* Distributed fine-tuning of all model weights with ZeRO-3
* Fine-tuning with LoRA adapters and ZeRO-3
* Fine-tuning with QLoRA adapters and DDP

```shell
# Full training with ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/{model_name}/sft/config_full.yaml

# LoRA training with ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/{model_name}/sft/config_16bit.yaml

# QLoRA training with DDP
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_sft.py recipes/{model_name}/sft/config_8bit.yaml
```

You can override the parameters in each YAML config by appending them to the command as follows:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/{model_name}/sft/config_full.yaml --per_device_train_batch_size=2 --num_train_epochs=3
```

## Direct Preference Optimisation (DPO)

```shell
# Full training with ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/{model_name}/dpo/config_full.yaml

# LoRA training with ZeRO-3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/{model_name}/dpo/config_16bit.yaml

# QLoRA training with DDP
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml scripts/run_dpo.py recipes/{model_name}/dpo/config_8bit.yaml
```