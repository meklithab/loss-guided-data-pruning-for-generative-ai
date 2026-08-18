#!/usr/bin/env bash
# Experiment 2 — Which loss signal works?
# Compares all selection strategies at a SINGLE fixed budget (default 30%,
# a reasonable middle point per the proposal's ideal-result sketch).
# Run scripts/00_setup_shared.sh first.
set -e

python main.py run-experiment2
python analyze.py

echo "Experiment 2 done. Compare results/tables/summary_mean_std.csv."
