"""
Reads results/runs.csv and produces the core plots + efficiency metrics
for the report:
  1. task_accuracy (performance) vs. data_fraction, one line per method
     -- the central "efficiency frontier" figure the assignment asks for
  2. task_accuracy vs. gpu_minutes, one line per method (the compute frontier)
  3. derived efficiency ratios: performance-per-GPU-hour, performance-per-
     million-training-tokens
  4. extended-method ablation bar charts (threshold band, cluster count) --
     only produced if those runs exist in runs.csv

Usage:
    python src/analyze.py --config configs/config.yaml
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_config

NUMERIC_COLS = ["data_fraction", "data_reduction_pct", "eval_loss", "eval_ppl",
                 "task_accuracy", "rouge_l", "gpu_minutes", "peak_vram_gb", "tokens_seen"]


def plot_frontier(df, x, y, out_path, title, prefix_exclude="ablation"):
    plt.figure(figsize=(6, 4))
    core = df[~df["run_name"].str.startswith(prefix_exclude)]
    for method, sub in core.groupby("method"):
        sub = sub.sort_values(x)
        plt.plot(sub[x], sub[y], marker="o", label=method)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def plot_ablation(df, prefix, out_path, title, metric="eval_loss"):
    sub = df[df["run_name"].str.startswith(prefix)]
    if sub.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.bar(sub["run_name"], sub[metric])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def add_efficiency_ratios(df):
    """
    Derived metrics answering the assignment's central question directly:
    how much capability are we getting per unit of compute / data.
    Uses task_accuracy as the performance number (swap to rouge_l or
    1/eval_ppl if that fits your chosen eval better).
    """
    gpu_hours = df["gpu_minutes"] / 60.0
    df["performance_per_gpu_hour"] = df["task_accuracy"] / gpu_hours.replace(0, pd.NA)
    df["performance_per_million_tokens"] = df["task_accuracy"] / (df["tokens_seen"] / 1e6).replace(0, pd.NA)
    return df


def main(cfg):
    df = pd.read_csv(cfg["results"]["runs_csv"])
    df = df[df["status"] == "complete"].copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_efficiency_ratios(df)

    plots_dir = cfg["results"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # --- primary efficiency frontier: performance vs. data retained ---
    plot_frontier(df, "data_fraction", "task_accuracy",
                  os.path.join(plots_dir, "performance_vs_data_fraction.png"),
                  "Task performance vs. % of training data retained (H1, H2, H4)")

    # --- primary compute frontier: performance vs. GPU time spent ---
    plot_frontier(df, "gpu_minutes", "task_accuracy",
                  os.path.join(plots_dir, "performance_vs_gpu_minutes.png"),
                  "Task performance vs. GPU-minutes spent training (H3)")

    # --- secondary quality signal ---
    plot_frontier(df, "data_fraction", "eval_loss",
                  os.path.join(plots_dir, "eval_loss_vs_data_fraction.png"),
                  "Held-out eval loss vs. % of training data retained")

    plot_frontier(df, "data_fraction", "rouge_l",
                  os.path.join(plots_dir, "rouge_l_vs_data_fraction.png"),
                  "ROUGE-L (generation quality, secondary) vs. % of training data retained")

    # --- efficiency ratios: performance per unit compute / data ---
    plot_frontier(df, "data_fraction", "performance_per_gpu_hour",
                  os.path.join(plots_dir, "performance_per_gpu_hour.png"),
                  "Performance per GPU-hour, by method and data fraction")

    # --- extended-method ablations, only rendered if those runs exist ---
    plot_ablation(df, "ablation_threshold",
                  os.path.join(plots_dir, "ablation_threshold.png"),
                  "[Extended] Threshold-band ablation (perplexity, 30% data)")
    plot_ablation(df, "ablation_clusters",
                  os.path.join(plots_dir, "ablation_clusters.png"),
                  "[Extended] Cluster-count ablation (diversity method, 30% data)")

    print("\nSummary table:")
    cols = ["run_name", "method", "data_fraction", "data_reduction_pct", "gpu_minutes",
            "flops_estimate", "peak_vram_gb", "eval_loss", "eval_ppl", "task_accuracy",
            "rouge_l", "performance_per_gpu_hour", "performance_per_million_tokens"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
