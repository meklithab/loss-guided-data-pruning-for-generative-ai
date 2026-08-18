#!/usr/bin/env bash
# Experiment 1 — Does pruning work?
# Compares: full dataset (100%) vs. random vs. loss-based (dynamics) pruning,
# across the budget ladder. Run scripts/00_setup_shared.sh first.
set -e

python main.py run-experiment1
python analyze.py

echo "Experiment 1 done. See results/runs/*/metrics.json and results/tables/summary_mean_std.csv."
