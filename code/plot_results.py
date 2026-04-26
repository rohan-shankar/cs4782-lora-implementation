from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate poster-ready LoRA diagrams from run summaries.")
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--outdir", type=str, default="results/figures")
    parser.add_argument("--task", type=str, default="sst2", help="Primary GLUE task for main poster figures.")
    parser.add_argument(
        "--manual_methods_csv",
        type=str,
        default="results/manual_methods.csv",
        help="Optional CSV for extra baselines (linear probing, adapters, prefix tuning).",
    )
    return parser.parse_args()


def _load_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("summary.json"):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        obj["path"] = str(p)
        best = obj.get("best_metrics", {})
        for k, v in best.items():
            obj[f"best_{k}"] = v
        rows.append(obj)
    if not rows:
        raise RuntimeError("No summary.json files found. Run experiments first.")
    return pd.DataFrame(rows)


def _task_metric_column(task: str) -> str:
    if task in {"mrpc", "qqp"}:
        return "best_f1"
    if task == "cola":
        return "best_mcc"
    return "best_accuracy"


def _method_label(mode: str, rank: float | int | None) -> str:
    if mode == "full":
        return "Full FT"
    if rank is None or np.isnan(rank):
        return "LoRA"
    return f"LoRA r={int(rank)}"


def _add_manual_methods(df_methods: pd.DataFrame, csv_path: Path, task: str, metric_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        return df_methods
    manual = pd.read_csv(csv_path)
    required = {"task", "method", "metric", "trainable_params"}
    missing = required - set(manual.columns)
    if missing:
        raise ValueError(f"manual methods csv missing columns: {sorted(missing)}")
    manual = manual[manual["task"] == task].copy()
    if manual.empty:
        return df_methods
    manual["method_label"] = manual["method"]
    manual[metric_col] = manual["metric"]
    if "peak_gpu_memory_gb" not in manual.columns:
        manual["peak_gpu_memory_gb"] = np.nan
    if "checkpoint_size_mb" not in manual.columns:
        manual["checkpoint_size_mb"] = np.nan
    return pd.concat(
        [
            df_methods,
            manual[
                [
                    "method_label",
                    metric_col,
                    "trainable_params",
                    "peak_gpu_memory_gb",
                    "checkpoint_size_mb",
                ]
            ],
        ],
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _load_summaries(root)
    df.to_csv(outdir / "all_summaries.csv", index=False)

    metric_col = _task_metric_column(args.task)
    if metric_col not in df.columns:
        raise RuntimeError(f"Missing metric column '{metric_col}' in summaries for task '{args.task}'.")

    glue = df[df.get("task_name", pd.Series(dtype=str)) == args.task].copy()
    if glue.empty:
        raise RuntimeError(f"No GLUE summaries found for task '{args.task}'.")

    glue["method_label"] = glue.apply(
        lambda r: _method_label(r.get("mode"), r.get("lora_rank")),
        axis=1,
    )
    glue = glue.sort_values(["mode", "lora_rank"], na_position="first")

    fig_methods = glue[["method_label", metric_col, "trainable_params", "peak_gpu_memory_gb", "checkpoint_size_mb"]].copy()
    fig_methods = _add_manual_methods(fig_methods, Path(args.manual_methods_csv), args.task, metric_col)
    fig_methods = fig_methods.dropna(subset=[metric_col]).copy()

    # Figure 1: Performance vs Method
    order = []
    if "Full FT" in set(fig_methods["method_label"]):
        order.append("Full FT")
    lora_methods = sorted(
        [m for m in fig_methods["method_label"].unique() if m.startswith("LoRA r=")],
        key=lambda x: int(x.split("=")[-1]),
    )
    for m in ("Linear Probe", "Adapters", "Prefix Tuning"):
        if m in set(fig_methods["method_label"]):
            order.append(m)
    order.extend([m for m in lora_methods if m not in order])
    order.extend([m for m in fig_methods["method_label"].unique() if m not in order])
    fig_methods["method_label"] = pd.Categorical(fig_methods["method_label"], categories=order, ordered=True)
    fig_methods = fig_methods.sort_values("method_label")

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(fig_methods["method_label"].astype(str), fig_methods[metric_col], color="#2f6b9a")
    plt.ylabel(metric_col.replace("best_", "").upper())
    plt.title(f"Performance vs Method ({args.task.upper()})")
    plt.xticks(rotation=22, ha="right")
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"01_performance_vs_method_{args.task}.png", dpi=220)
    plt.close()

    # Figure 2: Performance vs Trainable Parameters
    scatter = fig_methods.dropna(subset=["trainable_params"]).copy()
    if not scatter.empty:
        plt.figure(figsize=(7.2, 4.8))
        for _, row in scatter.iterrows():
            plt.scatter(row["trainable_params"], row[metric_col], s=65)
            plt.annotate(row["method_label"], (row["trainable_params"], row[metric_col]), fontsize=8, xytext=(5, 5), textcoords="offset points")
        plt.xscale("log")
        plt.xlabel("Trainable Parameters (log scale)")
        plt.ylabel(metric_col.replace("best_", "").upper())
        plt.title(f"Performance vs Trainable Parameters ({args.task.upper()})")
        plt.tight_layout()
        plt.savefig(outdir / f"02_performance_vs_trainable_params_{args.task}.png", dpi=220)
        plt.close()

    # Figure 3: Performance vs LoRA Rank
    lora = glue[glue["mode"] == "lora"].copy()
    lora = lora.dropna(subset=["lora_rank", metric_col]).sort_values("lora_rank")
    if not lora.empty:
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(lora["lora_rank"], lora[metric_col], marker="o", color="#1f8a70", linewidth=2)
        plt.xlabel("LoRA Rank r")
        plt.ylabel(metric_col.replace("best_", "").upper())
        plt.title(f"Performance vs LoRA Rank ({args.task.upper()})")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(outdir / f"03_performance_vs_lora_rank_{args.task}.png", dpi=220)
        plt.close()

    # Figure 4: Memory or Storage Savings
    resource_col = None
    if fig_methods["peak_gpu_memory_gb"].fillna(0).max() > 0:
        resource_col = "peak_gpu_memory_gb"
        y_label = "Peak GPU Memory (GB)"
        title = f"Memory Savings Across Methods ({args.task.upper()})"
        fname = f"04_memory_savings_{args.task}.png"
    elif fig_methods["checkpoint_size_mb"].fillna(0).max() > 0:
        resource_col = "checkpoint_size_mb"
        y_label = "Checkpoint Size (MB)"
        title = f"Storage Savings Across Methods ({args.task.upper()})"
        fname = f"04_storage_savings_{args.task}.png"
    if resource_col is not None:
        mem = fig_methods.dropna(subset=[resource_col]).copy()
        if not mem.empty:
            plt.figure(figsize=(9, 4.8))
            plt.bar(mem["method_label"].astype(str), mem[resource_col], color="#8f3b76")
            plt.ylabel(y_label)
            plt.title(title)
            plt.xticks(rotation=22, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / fname, dpi=220)
            plt.close()

    # Lower-priority training/validation curves if available in history
    curve_rows = []
    for _, r in glue.iterrows():
        hist = r.get("history")
        if not isinstance(hist, list):
            continue
        for e in hist:
            curve_rows.append(
                {
                    "method_label": _method_label(r.get("mode"), r.get("lora_rank")),
                    "epoch": e.get("epoch"),
                    "train_loss": e.get("train_loss"),
                    "metric": e.get("accuracy", e.get("f1", e.get("mcc"))),
                }
            )
    if curve_rows:
        curves = pd.DataFrame(curve_rows).dropna(subset=["epoch", "train_loss"])
        if not curves.empty:
            plt.figure(figsize=(7.2, 4.8))
            for label, sub in curves.groupby("method_label"):
                plt.plot(sub["epoch"], sub["train_loss"], marker="o", label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Train Loss")
            plt.title(f"Training Curves ({args.task.upper()})")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(outdir / f"05_training_curves_{args.task}.png", dpi=220)
            plt.close()

    print(f"Saved figures and CSV to: {outdir}")


if __name__ == "__main__":
    main()
