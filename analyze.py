"""Generate reproducible analysis tables for Experiments 1, 2, 3, and 5."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_runs(results_dir: str) -> pd.DataFrame:
    rows = []
    runs_root = Path(results_dir) / "runs"
    if not runs_root.exists():
        return pd.DataFrame()
    compute_path = Path(results_dir) / "compute_log.json"
    compute = _load_json(compute_path) if compute_path.exists() else []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        parts = run_id.split("_")
        metrics_path = run_dir / "metrics.json"
        selection_path = run_dir / "selection.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        selection = _load_json(selection_path) if selection_path.exists() else {}
        phases = [c for c in compute if c.get("run_id") == run_id]
        experiment = parts[0] if parts else None
        strategy = "_".join(parts[1:-2]) if len(parts) > 3 else selection.get("strategy")
        seed = int(parts[-1].replace("seed", "")) if parts and parts[-1].startswith("seed") else selection.get("seed")
        if experiment == "exp3" and strategy == "initial":
            selection_run_id = f"selection_seed{seed}_epochs0"
        elif experiment == "exp3" and strategy == "dynamics_1epoch":
            selection_run_id = f"selection_seed{seed}_epochs1"
        else:
            selection_run_id = f"selection_seed{seed}_epochs2"
        selection_phases = [c for c in compute if c.get("run_id") == selection_run_id]
        subset_selection = sum(c["wall_seconds"] for c in phases if c["phase"] == "selection")
        final_training = sum(c["wall_seconds"] for c in phases if c["phase"] == "final_training")
        evaluation = sum(c["wall_seconds"] for c in phases if c["phase"] == "evaluation")
        own_tokens = sum(c["tokens_processed"] for c in phases)
        selection_cost = sum(c["wall_seconds"] for c in selection_phases)
        selection_tokens = sum(c["tokens_processed"] for c in selection_phases)
        row = {
            "run_id": run_id,
            "experiment": experiment,
            "strategy": strategy,
            "budget_percent": int(parts[-2]) if len(parts) > 2 and parts[-2].isdigit() else selection.get("budget", 0) * 100,
            "seed": seed,
            "data_retained_percent": selection.get("data_retained_percent"),
            "data_removed_percent": selection.get("data_removed_percent"),
            "selected_examples": selection.get("n_examples"),
            "training_tokens": selection.get("approx_training_tokens_whitespace"),
            "selection_cost_seconds": selection_cost + subset_selection,
            "final_training_seconds": final_training,
            "evaluation_seconds": evaluation,
            "total_cost_seconds": selection_cost + subset_selection + final_training,
            "total_tokens_processed": selection_tokens + own_tokens,
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = [
        "validation_loss",
        "validation_perplexity",
        "test_loss",
        "test_perplexity",
        "task_specific_accuracy",
        "data_retained_percent",
        "final_training_seconds",
        "selection_cost_seconds",
        "total_cost_seconds",
        "total_tokens_processed",
    ]
    existing = [c for c in numeric if c in df.columns]
    grouped = df.groupby(["experiment", "strategy", "budget_percent"], dropna=False)[existing]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def collect_exp5(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in (Path(results_dir) / "runs").glob("exp5_*/leave_k_out.json"):
        data = _load_json(path)
        corr = data.get("correlations_with_true_marginal_value", {})
        for signal, stats in corr.items():
            rows.append({"run_id": data.get("run_id", path.parent.name), "signal": signal, **stats})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.results_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.results_dir)
    summary = summarize(runs)
    exp5 = collect_exp5(args.results_dir)
    runs.to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    exp5.to_csv(out_dir / "experiment5_correlations.csv", index=False)
    print(f"Wrote analysis tables to {out_dir}")


if __name__ == "__main__":
    main()
