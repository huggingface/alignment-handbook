# The RLHF Handbook

Robust recipes for Reinforcement Learning from Human Feedback (RLHF) and related techniques to steer language models.

## What is this?

Just one year ago, chatbots were out of fashion and most people hadn't heard about techniques like RLHF to align language models with human preferences. Then, OpenAI broke the internet with ChatGPT and Meta followed suit by releasing the Llama series of language models which enabled the ML community to build their very own capable chatbots. This has led to a rich ecosystem of datasets and models that have mostly focused on teaching language models to follow instructions through supervised fine-tuning (SFT).

However, we know from the [InstructGPT](https://huggingface.co/papers/2203.02155) and [Llama2](https://huggingface.co/papers/2307.09288) papers that significant gains in helpfulness and safety can be had by augmenting SFT with human (or AI) preferences. At the same time, combining reinforcement learning with NLP is a fairly novel idea and there are few pubic resources available on how to train these models, what data to collect, and what metrics to measure for best downstream performance.

The RLHF Handbook aims to fill that gap by providing the community with a series of robust training recipes that span the whole RLHF pipeline.

## Contents

The initial release of the handbook will focus on the following techniques:

* **Supervised fine-tuning:** teach languge language models to follow instructions and tips on how to collect and curate your own training dataset.
* **Reward modeling:** teach language models to distinguish model responses according to human or AI preferences.
* **Rejection sampling:** a simple, but powerful technique to boost the performance of your SFT model.
* **Proximal policy optimization (PPO):** the workhorse of RLHF, but notoriously sensitive to hyperparameters (like all good RL algorithms).
* **Direct preference optimisation (DPO):** a powerful and promising alternative to PPO.

## Citation

If you find the content of this repo useful in your work, please cite it as follows:

```bibtex
@misc{rlhf_handbook2023,
  author = {Lewis Tunstall and Edward Beeching and Nathan Lambert and Nazneen Rajani and Thomas Wolf},
  title = {The RLHF Handbook},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/rlhf-handbook}}
}
```