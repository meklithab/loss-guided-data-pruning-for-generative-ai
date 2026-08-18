#!/usr/bin/env bash
# Experiment 5 — Does the method predict actual data value?
# Uses the `dynamics` selection at a fixed budget (default 30%) as the subset
# to probe. Requires results/selection_dynamics_<budget>pct.json to already
# exist (i.e. run Experiment 1 or 2 first, or run `select` manually).
#
# COST WARNING: this trains (n_examples_sampled / group_size) + 1 models
# end-to-end (see src/leave_k_out.py). With config defaults (200/10) that is
# 21 training runs at the chosen budget's size. Lower these in
# configs/default.yaml if you need it to run faster.
set -e

BUDGET="${1:-0.30}"
python main.py run-experiment5 --budget "$BUDGET"
python analyze.py

echo "Experiment 5 done. See results/leave_k_out_dynamics_*.json for per-group"
echo "true marginal value and its Spearman correlation with each cheap signal."
