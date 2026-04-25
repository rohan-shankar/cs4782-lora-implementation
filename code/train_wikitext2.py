from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from lora.inject import apply_lora_to_model, count_parameters, freeze_non_lora_parameters
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="From-scratch LoRA vs full finetuning on WikiText-2")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="results/wikitext2")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="c_attn")
    parser.add_argument("--max_train_steps", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    output_dir = Path(args.output_dir) / f"{args.mode}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"])

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples: dict) -> dict:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = [x.copy() for x in result["input_ids"]]
        return result

    lm_data = tokenized.map(group_texts, batched=True)
    lm_data.set_format(type="torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(
        lm_data["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        lm_data["validation"], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    lora_replaced = []
    if args.mode == "lora":
        targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
        inject_result = apply_lora_to_model(
            model,
            target_module_suffixes=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        lora_replaced = inject_result.replaced_modules
        freeze_non_lora_parameters(model)

    model.to(device)
    total_params, trainable_params = count_parameters(model)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    max_steps = args.max_train_steps if args.max_train_steps > 0 else args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    global_step = 0
    best_ppl = float("inf")
    best = {}
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            train_losses.append(loss.item())
            progress.set_postfix({"loss": f"{np.mean(train_losses):.4f}"})
            if global_step >= max_steps:
                break

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="eval"):
                batch = {k: v.to(device) for k, v in batch.items()}
                eval_losses.append(model(**batch).loss.item())

        eval_loss = float(np.mean(eval_losses))
        perplexity = float(math.exp(eval_loss))
        if perplexity < best_ppl:
            best_ppl = perplexity
            best = {"eval_loss": eval_loss, "perplexity": perplexity, "epoch": epoch + 1}
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if global_step >= max_steps:
            break

    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    summary = {
        "dataset": "wikitext-2-raw-v1",
        "mode": args.mode,
        "model_name": args.model_name,
        "seed": args.seed,
        "lora_rank": args.lora_rank if args.mode == "lora" else None,
        "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
        "lora_dropout": args.lora_dropout if args.mode == "lora" else None,
        "lora_targets": args.lora_targets if args.mode == "lora" else None,
        "replaced_modules": lora_replaced,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params,
        "peak_gpu_memory_gb": peak_mem_gb,
        "best_metrics": best,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
