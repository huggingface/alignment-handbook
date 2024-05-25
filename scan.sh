# Llama 3-8b - base
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_base deepspeed_zero3
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_base deepspeed_zero3 '--beta=0.05 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-8b-base-beta-0.05 --output_dir=data/zephyr-orpo-llama3-8b-base-beta-0.05'
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_base deepspeed_zero3 '--beta=0.1 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-8b-base-beta-0.1 --output_dir=data/zephyr-orpo-llama3-8b-base-beta-0.1'
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_base deepspeed_zero3 '--beta=0.5 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-8b-base-beta-0.5 --output_dir=data/zephyr-orpo-llama3-8b-base-beta-0.5'

# Llama 3-8b - instruct
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_instruct deepspeed_zero3
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_instruct deepspeed_zero3 '--beta=0.05 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-8b-instruct-beta-0.05 --output_dir=data/zephyr-orpo-llama3-8b-instruct-beta-0.05'
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_instruct deepspeed_zero3 '--beta=0.1 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-8b-instruct-beta-0.1 --output_dir=data/zephyr-orpo-llama3-8b-instruct-beta-0.1'
sbatch --qos=normal --job-name=h4_llama3 --nodes=1 recipes/launch.slurm zephyr-orpo orpo llama3_8b_instruct deepspeed_zero3 '--beta=0.5 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-8b-instruct-beta-0.5 --output_dir=data/zephyr-orpo-llama3-8b-base-beta-0.5'

# Llama 3-70b - base
sbatch --qos=normal --job-name=h4_llama3 --nodes=4 recipes/launch.slurm zephyr-orpo orpo llama3_70b_base deepspeed_zero3
sbatch --qos=normal --job-name=h4_llama3 --nodes=4 recipes/launch.slurm zephyr-orpo orpo llama3_70b_base deepspeed_zero3 '--beta=0.05 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-70b-base-beta-0.05 --output_dir=data/zephyr-orpo-llama3-70b-base-beta-0.05'
sbatch --qos=normal --job-name=h4_llama3 --nodes=4 recipes/launch.slurm zephyr-orpo orpo llama3_70b_base deepspeed_zero3 '--beta=0.1 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-70b-base-beta-0.1 --output_dir=data/zephyr-orpo-llama3-70b-base-beta-0.1'
sbatch --qos=normal --job-name=h4_llama3 --nodes=4 recipes/launch.slurm zephyr-orpo orpo llama3_70b_base deepspeed_zero3 '--beta=0.5 --hub_model_id=orpo-explorers/zephyr-orpo-llama3-70b-base-beta-0.5 --output_dir=data/zephyr-orpo-llama3-8b-base-beta-0.5'
