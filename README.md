# CS4782 Final Project: LoRA Re-Implementation (From Scratch)

This repository reproduces key results from:
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022).

## Important Compliance Rule

This codebase does **not** call a library LoRA API (for example PEFT LoRA wrappers).  
LoRA is implemented directly in this repo with custom modules:

- `code/lora/layers.py`
- `code/lora/inject.py`

We use standard libraries only for base models, tokenizers, and datasets.

## Repository Structure

- `code/`: training code and LoRA implementation
- `data/`: dataset notes/instructions
- `results/`: experiment outputs (`summary.json`, model checkpoints)
- `report/`: final 2-page report PDF
- `poster/`: poster PDF

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GLUE (BERT): Baseline vs LoRA

Full fine-tuning:

```bash
python code/train_glue.py \
  --task_name sst2 \
  --mode full \
  --output_dir results/glue
```

LoRA (custom implementation):

```bash
python code/train_glue.py \
  --task_name sst2 \
  --mode lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_targets query,value \
  --output_dir results/glue
```

## WikiText-2 (GPT-2): Baseline vs LoRA

Full fine-tuning:

```bash
python code/train_wikitext2.py \
  --mode full \
  --output_dir results/wikitext2
```

LoRA (custom implementation):

```bash
python code/train_wikitext2.py \
  --mode lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_targets c_attn \
  --output_dir results/wikitext2
```

## Outputs

Each run writes:

- `best_model.pt`
- `summary.json`

`summary.json` includes:

- task/dataset + model
- baseline/LoRA mode
- LoRA configuration and replaced module names
- total/trainable parameter counts
- peak GPU memory usage
- best validation metrics (`accuracy`/`f1`/`mcc` or `perplexity`)
