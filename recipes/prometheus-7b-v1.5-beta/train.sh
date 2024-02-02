#!/bin/bash

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/prometheus-7b-v1.5-beta/sft/config_qlora.yaml --load_in_4bit=true
