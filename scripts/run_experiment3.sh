#!/usr/bin/env bash
# Experiment 3 - How early can useful examples be identified?
set -e
python main.py run-experiment3
python analyze.py
echo "Experiment 3 done. Compare initial, 1-epoch dynamics, and 2-epoch dynamics in results/tables/summary_mean_std.csv."
