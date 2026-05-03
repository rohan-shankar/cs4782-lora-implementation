# Results

Experiment outputs and figures from our LoRA re-implementation.

## Figures

Summary plots used in the poster and report:

- `paramsvsrank.png` — trainable parameters vs LoRA rank (BERT + GPT-2, log scale)
- `perfvslorarank.png` — SST-2 accuracy and MRPC F1 vs LoRA rank
- `perplexityvsrank.png` — GPT-2 perplexity on WikiText-2 vs rank

## Full Experiment Data

The `full_outputs.zip` (not tracked in git due to size) contains all raw outputs from our Colab runs, including:
- Per-task figures (bar charts, scatter plots, training curves)
- `summary.json` files for each experiment with metrics, param counts, GPU memory, and per-epoch history
- Model checkpoints (`best_model.pt`)
