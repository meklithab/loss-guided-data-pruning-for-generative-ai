#!/bin/bash
# EXTENDED / optional methods: perplexity, diversity, hybrid composite.
# Only run this if the core grid + ablations are done with time to spare.
set -e
CONFIG=configs/config.yaml

for method in perplexity diversity hybrid; do
  for frac in 10 30 50 70; do
    run_name="${method}_${frac}pct"
    data_path="data/subsets/${run_name}.jsonl"
    echo "=== $run_name ==="
    python src/train.py --config $CONFIG --data $data_path --run_name $run_name
  done
done
