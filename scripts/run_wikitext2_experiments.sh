#!/usr/bin/env bash
set -euo pipefail

python code/train_wikitext2.py --mode full --seed 42 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 4 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 8 --output_dir results/wikitext2
python code/train_wikitext2.py --mode lora --seed 42 --lora_rank 16 --output_dir results/wikitext2
