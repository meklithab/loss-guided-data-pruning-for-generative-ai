#!/usr/bin/env bash
# Experiment 2 — Which loss signal works?
# Compares all selection strategies at a SINGLE fixed budget (default 30%,
# a reasonable middle point per the proposal's ideal-result sketch).
# Run scripts/00_setup_shared.sh first.
set -e

BUDGET="${1:-0.30}"
STRATEGIES=(random low_loss high_loss loss_delta dynamics dynamics_diversity)

for STRATEGY in "${STRATEGIES[@]}"; do
  echo "=== Experiment 2: strategy=$STRATEGY budget=$BUDGET ==="
  python main.py select --strategy "$STRATEGY" --budget "$BUDGET"
  python main.py train --strategy "$STRATEGY" --budget "$BUDGET"
  python main.py evaluate --strategy "$STRATEGY" --budget "$BUDGET"
done

echo "Experiment 2 done. Compare results/eval_<strategy>_$(python -c "print(int(${BUDGET}*100))")pct.json across strategies."
