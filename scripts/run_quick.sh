#!/usr/bin/env bash
set -euo pipefail

# Quick mode:
# - small number of optimization steps
# - enough runs to generate core poster diagrams for SST-2
# - optional lightweight WikiText-2 sanity check

mkdir -p results/.mplconfig

# GLUE (SST-2): full FT + LoRA rank sweep (r=1,4,8)
python code/train_glue.py --task_name sst2 --mode full --seed 42 --epochs 1 --max_train_steps 40 --output_dir results/glue_quick
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --epochs 1 --max_train_steps 40 --lora_rank 1 --output_dir results/glue_quick
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --epochs 1 --max_train_steps 40 --lora_rank 4 --output_dir results/glue_quick
python code/train_glue.py --task_name sst2 --mode lora --seed 42 --epochs 1 --max_train_steps 40 --lora_rank 8 --output_dir results/glue_quick

# WikiText-2 quick sanity (small steps) to verify LM path and collect efficiency logs.
python code/train_wikitext2.py --mode full --seed 42 --epochs 1 --max_train_steps 30 --output_dir results/wikitext2_quick
python code/train_wikitext2.py --mode lora --seed 42 --epochs 1 --max_train_steps 30 --lora_rank 8 --output_dir results/wikitext2_quick

# Generate main poster figures from quick results.
python code/plot_results.py --root results --outdir results/figures_quick --task sst2

echo "Quick run done. See results/figures_quick."
