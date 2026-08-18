"""
Compute accounting so we never claim a "compute saving" that ignores the cost
of the selection process itself.

Every phase (warmup tracking, scoring, selection, final training, leave-k-out)
should be wrapped with `ComputeTracker` and its `.report()` appended to a shared
`results/compute_log.json`. `combine_phases()` then sums the right phases into
C_total = C_selection + C_training for a given run, per the assignment's
compute-accounting requirement.
"""
import json
import os
import time

try:
    import torch
except ImportError:  # allows --help / dry runs without torch installed
    torch = None


class ComputeTracker:
    """Context manager measuring wall time, peak VRAM, and (optionally) tokens processed
    for one phase of the pipeline (e.g. 'warmup_tracking', 'train_selected_30pct')."""

    def __init__(self, phase_name: str, run_id: str = None):
        self.phase_name = phase_name
        self.run_id = run_id
        self.tokens_processed = 0
        self._start = None
        self._report = None

    def add_tokens(self, n: int) -> None:
        self.tokens_processed += int(n)

    def __enter__(self):
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        peak_vram_gb = None
        if torch is not None and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self._report = {
            "phase": self.phase_name,
            "run_id": self.run_id,
            "wall_seconds": elapsed,
            "peak_vram_gb": peak_vram_gb,
            "tokens_processed": self.tokens_processed,
            "timestamp": time.time(),
        }
        return False  # do not suppress exceptions

    def report(self) -> dict:
        assert self._report is not None, "report() called before the `with` block exited"
        return self._report


def append_to_log(report: dict, log_path: str = "results/compute_log.json") -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entries = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            entries = json.load(f)
    entries.append(report)
    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2)


def combine_phases(log_path: str, phase_names: list) -> dict:
    """Sum wall_seconds and tokens_processed across the given phase names in the log.
    Use this to compute C_total = C_selection + C_training for a specific run,
    e.g. combine_phases(log, ['warmup_tracking', 'scoring', 'select_30pct', 'train_selected_30pct'])."""
    with open(log_path, "r") as f:
        entries = json.load(f)
    matched = [e for e in entries if e["phase"] in phase_names]
    total_seconds = sum(e["wall_seconds"] for e in matched)
    total_tokens = sum(e["tokens_processed"] for e in matched)
    peak_vram = max((e["peak_vram_gb"] or 0) for e in matched) if matched else None
    return {
        "phases_included": phase_names,
        "n_phase_entries_found": len(matched),
        "total_wall_seconds": total_seconds,
        "total_wall_minutes": total_seconds / 60.0,
        "total_tokens_processed": total_tokens,
        "peak_vram_gb_across_phases": peak_vram,
    }
