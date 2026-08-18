#!/usr/bin/env bash
# Run ONCE before any experiment. Produces results/trajectories.json and
# results/scores.csv, which every experiment below reuses (this is also why
# the warm-up tracking cost is logged separately in compute_log.json — it's a
# shared, one-time cost across all strategies/budgets, not per-run).
set -e
python main.py track-warmup
python main.py score
