#!/usr/bin/env bash
set -euo pipefail

python code/train_glue.py --task_name sst2 --mode full --seed 42 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 4 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 8 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 16 --output_dir results/glue

python code/train_glue.py --task_name mrpc --mode full --seed 42 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 4 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 8 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 16 --output_dir results/glue
