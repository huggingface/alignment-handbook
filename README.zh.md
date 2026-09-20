<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/huggingface/alignment-handbook/main/assets/handbook.png">
</p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/alignment-handbook/handbook-v01-models-and-datasets-654e424d22e6880da5ebc015" target="_blank">模型与数据集 (Models & Datasets)</a> | 📃 <a href="https://arxiv.org/abs/2310.16944" target="_blank">技术报告 (Technical Report)</a>
</p>

# 对齐手册 (The Alignment Handbook)

用于持续预训练以及将语言模型与人类及 AI 偏好相对齐的稳健训练配方。

## 项目简介

就在几年前，聊天机器人还远未普及，大多数人甚至未曾听说过利用人类反馈强化学习（RLHF）将语言模型与人类偏好相对齐的技术。随后，OpenAI 凭借 ChatGPT 轰动全球，Meta 紧接着开源了 Llama 系列语言模型，使得整个机器学习社区能够亲手构建专属于自己的强大聊天机器人。这催生了一个丰富的数据集与模型生态系统，但大部分实践都集中在通过监督微调（SFT）训练语言模型遵循指令。

然而，从 [InstructGPT](https://huggingface.co/papers/2203.02155) 和 [Llama 2](https://huggingface.co/papers/2307.09288) 的研究论文中我们得知，通过引入人类（或 AI）偏好来增强 SFT，可以在有用性（Helpfulness）和安全性（Safety）方面取得巨大收益。与此同时，将语言模型与一组偏好相对齐依然是一个相对前沿的领域，关于如何训练此类模型、收集何种数据以及评测何种指标以获得最佳下游性能，业界公开的高质量资源仍然相当匮乏。

《对齐手册》（The Alignment Handbook）旨在填补这一空白，为开源社区提供一系列贯穿全流程的稳健训练配方（Recipes）。

## 最新动态 🗞️
* **2025年7月24日**：发布了 SmolLM3-3B 背后的完整[后训练配方 (Post-Training Recipe)](recipes/smollm3/README.md)：顶尖的混合推理模型 💭
* **2024年11月21日**：发布了微调 SmolLM2-Instruct 的[训练配方](recipes/smollm2/README.md)。
* **2024年8月18日**：发布 SmolLM-Instruct v0.2，以及微调轻量级端侧小语言模型的[训练配方](recipes/smollm/README.md) 💻
* **2024年4月12日**：与 Argilla 及 KAIST AI 联合发布 Zephyr 141B (A35B)，并提供使用 ORPO 微调 Mixtral 8x22B 的训练配方 🪁
* **2024年3月12日**：发布 StarChat2 15B，以及训练高性能代码助手的训练配方 🌟
* **2024年3月1日**：发布 Zephyr 7B Gemma，这是使用 RLAIF（基于 AI 反馈的强化学习）对齐 Gemma 7B 的全新配方 🔥
* **2024年2月1日**：发布使用宪政 AI（Constitutional AI）对齐开源 LLM 的训练配方 📜！详情参阅[训练配方](https://github.com/huggingface/alignment-handbook/tree/main/recipes/constitutional-ai)与[官方博客](https://huggingface.co/blog/constitutional_ai)。
* **2024年1月18日**：发布针对 DPO vs KTO vs IPO 的全套评估基准，详情参阅[训练配方](recipes/pref_align_scan/README.md)与[官方博客](https://huggingface.co/blog/pref-tuning)。
* **2023年11月10日**：开源复现 Zephyr-7b-β 的全部训练代码 🪁！同时开源 [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) 数据集，包含 10,000 条完全由熟练人类专家编写的高质量指令与演示数据。

## 相关链接 🔗

* [Zephyr 7B 模型、数据集与演示合集](https://huggingface.co/collections/HuggingFaceH4/zephyr-7b-6538c6d6d5ddd1cbb1744a66)

## 项目导航指南 🧭

本项目结构清晰紧凑，主要包含以下核心模块：

* [`scripts`](./scripts/)：用于训练与评估模型的脚本。涵盖四个关键阶段：持续预训练（Continued Pretraining）、面向对话的监督微调（SFT）、基于 DPO 的偏好对齐，以及基于 ORPO 的 SFT 与偏好对齐融合训练。每个脚本均原生支持使用 DeepSpeed ZeRO-3 进行全参数分布式训练，或使用 LoRA/QLoRA 进行参数高效微调（PEFT）。
* [`recipes`](./recipes/)：用于复现 Zephyr 7B 等模型的训练配方。每个配方均以 YAML 文件形式呈现，包含了单次完整训练任务的所有超参数配置。项目中还提供了一个 `gpt2-nl` 配方，用于展示如何利用本手册进行语言或领域适配（例如在另一种语言上持续预训练，并对输出结果进行 SFT 与 DPO 微调）。

我们还在持续撰写一系列实践指南，用于剖析直接偏好优化（DPO）等算法的工作机理，并总结在实际落地中收集人类偏好数据的宝贵经验。上手推荐步骤如下：

1. 遵循[安装指引](#安装指引)配置开发环境。
2. 按照[配方说明指引](./recipes/zephyr-7b-beta/README.md)复现 Zephyr-7b-β。

如果您希望在自定义数据集上微调对话模型，建议参考[此处的数据集格式规范说明](./scripts/README.md#fine-tuning-on-your-datasets)。

## 核心对齐技术

手册的初始发布聚焦于以下关键技术：

* **持续预训练 (Continued pretraining)**：将语言模型适配到新的语言或垂直领域，或通过在全新数据集上执行持续因果语言建模（Causal Language Modeling）进一步增强基座能力。
* **监督微调 (Supervised fine-tuning, SFT)**：让语言模型学会遵循指令，并提供关于如何构建与清洗训练数据集的实用技巧。
* **奖励建模 (Reward modeling)**：训练语言模型根据人类或 AI 的偏好评判并区分不同候选回复的质量。
* **拒绝采样 (Rejection sampling)**：一种简单却极其强大的技术，能够大幅提升 SFT 模型的输出表现。
* **直接偏好优化 (Direct preference optimisation, DPO)**：一种相较于 PPO 更加轻量稳定且极具前景的对齐算法。
* **比值比偏好优化 (Odds Ratio Preference Optimisation, ORPO)**：一种将 SFT 与 DPO 合并为单阶段执行的人类偏好微调前沿技术。

## 安装指引

运行本项目代码，首先推荐使用 `uv` 创建 Python 虚拟环境：

```shell
uv venv handbook --python 3.11 && source handbook/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> 如需安装 `uv`，请参阅 [UV 安装指南](https://docs.astral.sh/uv/getting-started/installation/)。

接下来，安装 PyTorch `v2.6.0`：

```shell
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

请注意，确切的版本对于复现实验结果至关重要！由于 PyTorch 版本依赖具体的硬件环境，建议同时参考 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)。

随后安装本项目的其余依赖项：

```shell
uv pip install .
```

您还需要安装 Flash Attention 2，可通过以下命令安装：

```shell
uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

接着，登录您的 Hugging Face 账户：

```shell
huggingface-cli login
```

最后，安装 Git LFS 以便将模型权重推送到 Hugging Face Hub：

```shell
sudo apt-get install git-lfs
```

现在，您可以探索 `scripts` 和 `recipes` 目录，开启精彩的模型训练之旅 🪁！

## 项目目录结构

```
├── LICENSE
├── Makefile                    <- 包含类似 `make style` 等常用命令的 Makefile
├── README.md                   <- 面向开发者的顶层英文项目说明
├── recipes                     <- 训练配方配置 (YAML)、Accelerate 配置及 Slurm 脚本
├── scripts                     <- 用于训练与评估对话模型的完整脚本
├── setup.cfg                   <- 安装配置（主要用于代码质量与测试配置）
├── setup.py                    <- 使项目支持 pip 安装 (pip install -e .)，从而支持直接导入 `alignment`
├── src                         <- 本项目使用的核心源代码库
└── tests                       <- 单元测试用例
```

## 引用

如果您在研究或工程落地中发现本仓库的内容有所帮助，请按如下格式（支持 `\usepackage{biblatex}`）进行引用：

```bibtex
@software{Tunstall_The_Alignment_Handbook,
  author = {Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and Rajani, Nazneen and Huang, Shengyi and Rasul, Kashif and Bartolome, Alvaro, and M. Patiño, Carlos and M. Rush, Alexander and Wolf, Thomas},
  license = {Apache-2.0},
  title = {{The Alignment Handbook}},
  url = {https://github.com/huggingface/alignment-handbook},
  version = {0.4.0.dev0}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月20日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
