#!/usr/bin/env bash
# Experiment 1 — Does pruning work?
# Compares: full dataset (100%) vs. random vs. loss-based (dynamics) pruning,
# across the budget ladder. Run scripts/00_setup_shared.sh first.
set -e

BUDGETS=(0.10 0.30 0.50 1.00)
STRATEGIES=(random dynamics)

for BUDGET in "${BUDGETS[@]}"; do
  # "full dataset" is budget=1.00 regardless of strategy, so only run one
  # strategy at 100% to avoid redundant identical training runs.
  if [ "$BUDGET" == "1.00" ]; then
    python main.py select --strategy random --budget "$BUDGET"
    python main.py train --strategy random --budget "$BUDGET"
    python main.py evaluate --strategy random --budget "$BUDGET"
    continue
  fi
  for STRATEGY in "${STRATEGIES[@]}"; do
    echo "=== Experiment 1: strategy=$STRATEGY budget=$BUDGET ==="
    python main.py select --strategy "$STRATEGY" --budget "$BUDGET"
    python main.py train --strategy "$STRATEGY" --budget "$BUDGET"
    python main.py evaluate --strategy "$STRATEGY" --budget "$BUDGET"
  done
done

echo "Experiment 1 done. See results/eval_*.json for held-out loss/perplexity per run,"
echo "and results/compute_log.json for the cost of each run."
