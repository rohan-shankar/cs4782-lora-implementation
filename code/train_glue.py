from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from lora.inject import apply_lora_to_model, count_parameters, freeze_non_lora_parameters
from utils.seed import set_seed


TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="From-scratch LoRA vs full finetuning on GLUE")
    parser.add_argument("--task_name", type=str, default="sst2", choices=list(TASK_TO_KEYS.keys()))
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="results/glue")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="query,value")
    parser.add_argument("--max_train_steps", type=int, default=-1)
    return parser.parse_args()


def accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def f1_binary_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


def matthews_binary_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
    return float(num / den)


def compute_metrics(task_name: str, logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    out = {"accuracy": accuracy_np(labels, preds)}
    if task_name in ("mrpc", "qqp"):
        out["f1"] = f1_binary_np(labels, preds)
    if task_name == "cola":
        out["mcc"] = matthews_binary_np(labels, preds)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    output_dir = Path(args.output_dir) / f"{args.task_name}_{args.mode}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence1_key, sentence2_key = TASK_TO_KEYS[args.task_name]
    raw = load_dataset("glue", args.task_name)
    num_labels = raw["train"].features["label"].num_classes

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(examples: dict) -> dict:
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, max_length=args.max_length)
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=args.max_length,
        )

    encoded = raw.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(
        encoded["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    valid_key = "validation_matched" if args.task_name == "mnli" else "validation"
    val_loader = DataLoader(
        encoded[valid_key],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

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
        freeze_non_lora_parameters(model, extra_trainable_keywords=("classifier",))

    model.to(device)
    total_params, trainable_params = count_parameters(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    max_steps = args.max_train_steps if args.max_train_steps > 0 else args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    global_step = 0
    best_acc = -1.0
    best_metrics = {}
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            losses.append(loss.item())
            progress.set_postfix({"loss": f"{np.mean(losses):.4f}"})
            if global_step >= max_steps:
                break

        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="eval"):
                labels = batch["labels"].numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits.detach().cpu().numpy()
                all_logits.append(logits)
                all_labels.append(labels)

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics(args.task_name, logits_np, labels_np)
        metrics["train_loss"] = float(np.mean(losses)) if losses else float("nan")
        metrics["epoch"] = epoch + 1
        history.append(dict(metrics))

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = dict(metrics)
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if global_step >= max_steps:
            break

    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    ckpt_path = output_dir / "best_model.pt"
    checkpoint_size_mb = ckpt_path.stat().st_size / (1024**2) if ckpt_path.exists() else 0.0

    summary = {
        "task_name": args.task_name,
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
        "checkpoint_size_mb": checkpoint_size_mb,
        "best_metrics": best_metrics,
        "history": history,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
