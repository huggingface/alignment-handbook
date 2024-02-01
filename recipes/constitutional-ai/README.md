# Constitutional AI 

This repo includes the recipe for training the following models:

* https://huggingface.co/HuggingFaceH4/mistral-7b-anthropic
* https://huggingface.co/HuggingFaceH4/mistral-7b-grok


## Full training examples

You will require 8 GPUs (80GB of VRAM) to train the full model.
```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/constitutional-ai/sft/config_{grok,anthropic}.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/constitutional-ai/dpo/config_{grok,anthropic}.yaml
```


## Advanced: generating you own dataset

To generate the constitutional AI dataset, see https://github.com/huggingface/llm-swarm/tree/main/examples/hh for detailed instructions if you want build or customize the dataset. 
