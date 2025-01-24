<p align="center">
  <img src="https://raw.githubusercontent.com/huggingface/alignment-handbook/main/assets/handbook.png">
</p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">Models & Datasets</a> | 📃 <a href="https://arxiv.org/abs/2310.16944" target="_blank">Technical Report</a>
</p>

# The Alignment Handbook

Robust recipes to continue pretraining and to align language models with human and AI preferences.

Note that this repository is derived from [the original alignment-handbook](https://github.com/huggingface/alignment-handbook). We implemented our proposed method based on this useful repository. Thanks for their contributions.

## PoFT
In our paper **Preference-Oriented Supervised Fine-Tuning: Favoring Target Model Over Aligned Large Language Models**, we introduced a novel preference-oriented supervised fine-tuning approach, namely **PoFT**, to boost SFT by imposing a particular preference: favoring the target model over aligned LLMs on the same SFT data. This preference encourages the target model to predict a higher likelihood than that predicted by the aligned LLMs, incorporating assessment information on data quality  into the training process. PoFT achieves stable and consistent improvements over the SFT baselines across different training datasets and base models.

## Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n handbook python=3.10 && conda activate handbook
```

Next, install PyTorch `v2.1.2` - the precise version is important for reproducibility! Since this is hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

You can then install the remaining package dependencies as follows:

```shell
git clone https://github.com/Savannah120/alignment-handbook-PoFT.git
cd ./alignment-handbook-PoFT/
pip install -e .
```

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

> **Note**
> If your machine has less than 96GB of RAM and many CPU cores, reduce the `MAX_JOBS` arguments, e.g. `MAX_JOBS=4 pip install flash-attn --no-build-isolation`

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```


## Run PoFT
### Overall training script:
```shell
bash run_poft.sh
```
, which contains the following steps:

#### First step: generate preference scores

```shell
python generate_preference_scores.py \
      --data_path /path/to/raw/data \
      --model_name_or_path /path/to/the/aligned/LLM \
      --json_save_path /path/to/the/save/file
```
Note that you should convert your data into the format of Ultrachat_200k, which contains ```['prompt','prompt_id','messages'] ```

https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

#### Second step: merge preference scores
```shell
python merge_prefrence_scores.py \
        --data_root /path/to/preference/scores/root \
        --save_path /path/to/merged/results
```

#### Third step: training
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_poft.py 
                            recipes/zephyr-7b-beta/sft/config_full_poft.yaml \
                            --dataset_mixer=/path/to/merged/results \
                            --model_name_or_path=/path/to/the/initial/model \
                            --output_dir=/path/to/save/model 
```



## Citation

If you find the content of this repo useful in your work, please cite it as follows via `\usepackage{biblatex}`:

```bibtex
@misc{fan2024preferenceorientedsupervisedfinetuningfavoring,
      title={Preference-Oriented Supervised Fine-Tuning: Favoring Target Model Over Aligned Large Language Models}, 
      author={Yuchen Fan and Yuzhong Hong and Qiushi Wang and Junwei Bao and Hongfei Jiang and Yang Song},
      year={2024},
      eprint={2412.12865},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.12865}, 
}
@software{Tunstall_The_Alignment_Handbook,
  author = {Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and Rajani, Nazneen and Huang, Shengyi and Rasul, Kashif and Bartolome, Alvaro and M. Rush, Alexander and Wolf, Thomas},
  license = {Apache-2.0},
  title = {{The Alignment Handbook}},
  url = {https://github.com/huggingface/alignment-handbook},
  version = {0.3.0.dev0}
}
```
