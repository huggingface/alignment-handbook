# Language Adaptation through Continued Pretraining

This directory shows a base example of how to use continued pretraining and further tuning to adapt a language model to new data (e.g. a new language or domain).

Three steps are needed: continued pretraining (`cpt`), supervised finetuning (`sft`), and direct preference optimisation (`dpo`). In this dummy example, we'll continue pretraining gpt2 on Dutch raw data, then sft-tuning it, and finally aligning it with DPO. Note that no extensive hyperparameters were tested in this example and that the output models are bad - it is just to show you how you can use the scripts for LM adaptation. The scripts work on 4x 3090s (24GB VRAM). If you have less powerful hardware you may need to reduce the batch size.

## Continued pretraining

This step will further pretrain the original `gpt2` model on plain Dutch text. Note that the script will by default use the `text` column in the dataset but you can change that by specifying `text_column` in the yaml file or on the command-line.

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes 4 \
    scripts/run_cpt.py \
    recipes/gpt2-nl/cpt/config_full.yaml
```

## Supervised finetuning

As other recipes, such as the famous zephyr-7b-beta recipe, have shown, we can then teach our model how to hold a conversation by finetuning it on chat-formatted data. As a base model, we'll make use of the output of the previous step.

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes 4 \
    scripts/run_sft.py recipes/gpt2-nl/sft/config_full.yaml
```

## Direct preference optimisation

Finally, to align the model better with feedback, we can finetune the SFT output with the DPO algorithm. This should improve the quality of the chat capabilities of the model.

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes 4 \
    scripts/run_dpo.py recipes/gpt2-nl/dpo/config_full.yaml
```

## Conclusion

With the steps above you can adapt an LM to a new domain, more data, or even a different language. Then, with sft and dpo, you can end up building a powerful chatbot, too! All within just three simple commands. It should be obvious that all of these follow a very similar approach, which makes them suitable to apply in parameterized slurm jobs. The neat part is that you can easily overwrite arguments in the yaml files by specifying the overwriting argument as a command-line argument, so the adaptability is also great.
