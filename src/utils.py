"""
Shared helpers used by every script: reproducible seeding, a simple
run registry (so a killed Colab session can resume without redoing
finished runs), GPU-time / token-count / FLOPs / VRAM logging into
results/runs.csv, and a one-time environment snapshot for reproducibility.
"""
import os
import time
import random
import csv
import subprocess
import yaml
import numpy as np


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RunTimer:
    """Context manager that measures wall-clock GPU time for one run."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed_min = (time.time() - self.start) / 60.0


def already_done(run_name: str, runs_csv: str) -> bool:
    """Check the run registry so a re-launched grid script skips finished runs."""
    if not os.path.exists(runs_csv):
        return False
    with open(runs_csv, "r") as f:
        reader = csv.DictReader(f)
        return any(row["run_name"] == run_name and row.get("status") == "complete" for row in reader)


def reset_peak_memory_stats():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def get_peak_vram_gb() -> float:
    """Peak allocated GPU memory since the last reset, in GB. 0.0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except ImportError:
        pass
    return 0.0


def estimate_flops(n_params: int, n_tokens: int) -> float:
    """
    Standard transformer training-FLOPs approximation: FLOPs ~= 6 * N * D
    (N = model parameters, D = tokens processed), per Kaplan et al. 2020 /
    the Chinchilla scaling-law convention. This counts full forward+backward
    compute through the base model; it does NOT discount for LoRA freezing
    most parameters, since the forward pass (and most of the backward pass,
    for gradient flow through frozen layers) still costs the same as full
    fine-tuning. Treat this as an upper-bound estimate, not a measured value
    -- report it as such in the write-up.
    """
    return 6.0 * n_params * n_tokens


def snapshot_environment(out_path: str = "./results/env_freeze.txt"):
    """Write `pip freeze` once, so reviewers can recreate the exact environment.
    Safe to call every run -- only writes if the file doesn't already exist."""
    if os.path.exists(out_path):
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        freeze = subprocess.check_output(["pip", "freeze"], text=True)
        with open(out_path, "w") as f:
            f.write(freeze)
    except Exception as e:
        with open(out_path, "w") as f:
            f.write(f"# pip freeze failed: {e}\n")


def log_run(runs_csv: str, row: dict):
    """Append one row (one run) to the shared results table. Creates the file with a header if needed."""
    os.makedirs(os.path.dirname(runs_csv), exist_ok=True)
    file_exists = os.path.exists(runs_csv)
    fieldnames = [
        "run_name", "method", "data_fraction", "data_reduction_pct", "n_examples", "seed",
        "gpu_minutes", "tokens_seen", "flops_estimate", "peak_vram_gb", "train_loss_final",
        "eval_loss", "eval_ppl", "task_accuracy", "rouge_l", "status", "notes",
    ]
    with open(runs_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
