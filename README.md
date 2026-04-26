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

## One-Command Run (Experiments + Diagrams)

```bash
chmod +x scripts/run_everything.sh
./scripts/run_everything.sh
```

This runs:

- GLUE: SST-2 and MRPC (full fine-tune + LoRA ranks 1/4/8/16)
- WikiText-2: full fine-tune + LoRA ranks 1/4/8/16
- Poster diagrams for SST-2 and MRPC via `code/plot_results.py`

## Poster Diagram Outputs

Generated under `results/figures/`:

- `01_performance_vs_method_<task>.png`
- `02_performance_vs_trainable_params_<task>.png`
- `03_performance_vs_lora_rank_<task>.png`
- `04_memory_savings_<task>.png` (or storage variant)

If you want to include non-LoRA baselines (linear probe, adapters, prefix tuning), fill:

- `results/manual_methods.csv`

## Colab / Notebook Run

Use:

- `notebooks/lora_project_colab.ipynb`
- `notebooks/lora_project_colab_quick.ipynb` (fast smoke mode)

It is set up to:

1. clone or open the repo in Colab
2. install dependencies
3. run the full pipeline
4. generate and list poster figures

Quick shell mode (non-notebook):

```bash
chmod +x scripts/run_quick.sh
./scripts/run_quick.sh
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
- checkpoint size
- per-epoch history for training curves
- best validation metrics (`accuracy`/`f1`/`mcc` or `perplexity`)
