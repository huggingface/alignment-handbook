
input_data_path=/path/to/raw/data
save_data_root=/path/to/preference/scores/root

##First step: generate preference scores
save_path=$save_data_root/zephyr-7b-sft-full.json
CUDA_VISIBLE_DEVICES=0 python generate_preference_scores.py --data_path ${input_data_path} --model_name_or_path alignment-handbook/zephyr-7b-sft-full --json_save_path ${save_path} &

save_path=$save_data_root/Yi-6B-Chat.json
CUDA_VISIBLE_DEVICES=1 python generate_preference_scores.py --data_path ${input_data_path} --model_name_or_path 01-ai/Yi-6B-Chat --json_save_path ${save_path} & 

save_path=$save_data_root/Meta-Llama-3-8B-Instruct.json
CUDA_VISIBLE_DEVICES=2 python generate_preference_scores.py --data_path ${input_data_path} --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --batch_size 4 --json_save_path ${save_path}&

wait
sleep 30

##Second step: merge preference scores
merge_data_save_path=/path/to/merged/results
python merge_prefrence_scores.py --data_root ${save_data_root} --save_path ${merge_data_save_path} &

wait

poft_model_path=alignment-handbook/zephyr-7b-sft-full
output_path=/path/to/save/model

#### Third step: training
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_poft.py recipes/zephyr-7b-beta/sft/config_full_poft.yaml --dataset_mixer=${merge_data_save_path}  --model_name_or_path=${poft_model_path} --output_dir=${output_path} 

