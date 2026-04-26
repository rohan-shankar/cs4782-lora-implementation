#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline:
# 1) run GLUE experiments
# 2) run WikiText-2 experiments
# 3) generate poster figures

mkdir -p results/.mplconfig

python code/train_glue.py --task_name sst2 --mode full --seed 42 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 1 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 4 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 8 --output_dir results/glue
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --lora_rank 16 --output_dir results/glue

python code/train_glue.py --task_name mrpc --mode full --seed 42 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 1 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 4 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 8 --output_dir results/glue
python code/train_glue.py --task_name mrpc --mode lora --seed 42 --lora_rank 16 --output_dir results/glue

python code/train_wikitext2.py --mode full --seed 42 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 1 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 4 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 8 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 16 --output_dir results/wikitext2

python code/plot_results.py --root results --outdir results/figures --task sst2
python code/plot_results.py --root results --outdir results/figures --task mrpc

echo "Done. See results/figures for poster diagrams."
