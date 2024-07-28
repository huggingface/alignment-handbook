# Comparing Preference Alignment Algorithms
This directory contains various comparisons for three algorithms: DPO, IPO, and KTO. Each algorithm has been run in different hyperparameter configurations to study their performance. Two different models and datasets have been used to compare the performance of each algorithm:

- zephyr-beta-sft and Ultrafeedback
- OpenHermes-2.5 and the OpenOrca datasets 

We release a collection containing the datasets and models used for these experiments, if you require the other trained models, we can release them on request.
You can find a longer description of these results in our [blogpost](https://huggingface.co/blog/pref-tuning)

## Comparisons
For each algorithm, we aim to tune the beta parameter for a fixed learning rate. We vary beta from 0.1-0.9 in steps of 0.1, we have also found that in certain configurations a tiny value of beta, 0.01, can be effective. So we have included this smaller value in all our comparisons.

## Usage
The experiments can be launched with the following bash script:
```bash
#!/bin/bash

# Define an array containing the base configs we wish to fine tune
configs=("zephyr" "openhermes")
# Define an array of loss types
loss_types=("sigmoid" "kto_pair" "ipo")

# Define an array of beta values
betas=("0.01" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

# Outer loop for loss types
for config in "${configs[@]}"; do
    for loss_type in "${loss_types[@]}"; do

        # Inner loop for beta values
        for beta in "${betas[@]}"; do

            # Determine the job name and model revision based on loss type
            job_name="$config_${loss_type}_beta_${beta}"
            model_revision="${loss_type}-${beta}"

            # Submit the job
            sbatch --job-name=${job_name} recipes/launch.slurm pref_align_scan dpo $config deepspeed_zero3 \
            "--beta=${beta} --loss_type=${loss_type} --output_dir=data/$config-7b-align-scan-${loss_type}-beta-${beta} --hub_model_revision=${model_revision}"
        done
    done
done
```






