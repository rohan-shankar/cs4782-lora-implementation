from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--outdir", type=str, default="results/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in root.rglob("summary.json"):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        obj["path"] = str(p)
        best = obj.get("best_metrics", {})
        for k, v in best.items():
            obj[f"best_{k}"] = v
        summaries.append(obj)

    if not summaries:
        raise RuntimeError("No summary.json files found.")

    df = pd.DataFrame(summaries)
    df.to_csv(outdir / "all_summaries.csv", index=False)

    glue = df[df["task_name"].notna()] if "task_name" in df.columns else pd.DataFrame()
    if not glue.empty:
        for task in glue["task_name"].dropna().unique():
            sub = glue[glue["task_name"] == task].copy()
            sub["rank"] = sub["lora_rank"].fillna(0)
            sub = sub.sort_values("rank")
            plt.figure(figsize=(6, 4))
            plt.plot(sub["rank"], sub["best_accuracy"], marker="o")
            plt.title(f"{task}: Accuracy vs LoRA rank (0 = full FT)")
            plt.xlabel("LoRA rank")
            plt.ylabel("Validation Accuracy")
            plt.tight_layout()
            plt.savefig(outdir / f"glue_{task}_acc_vs_rank.png", dpi=180)
            plt.close()

    wt = df[df["dataset"].notna()] if "dataset" in df.columns else pd.DataFrame()
    if not wt.empty:
        sub = wt.copy()
        sub["rank"] = sub["lora_rank"].fillna(0)
        sub = sub.sort_values("rank")
        if "best_perplexity" in sub.columns:
            plt.figure(figsize=(6, 4))
            plt.plot(sub["rank"], sub["best_perplexity"], marker="o")
            plt.title("WikiText-2: Perplexity vs LoRA rank (0 = full FT)")
            plt.xlabel("LoRA rank")
            plt.ylabel("Validation Perplexity")
            plt.tight_layout()
            plt.savefig(outdir / "wikitext2_ppl_vs_rank.png", dpi=180)
            plt.close()


if __name__ == "__main__":
    main()
