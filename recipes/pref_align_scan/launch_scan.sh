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