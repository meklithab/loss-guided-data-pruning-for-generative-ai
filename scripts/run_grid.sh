#!/bin/bash
# CORE grid: loss_high / loss_low / loss_delta at every matched data
# fraction. This is the actual research question -- does any loss-based
# signal beat matched-budget random, and which one?
set -e
CONFIG=configs/config.yaml

for method in loss_high loss_low loss_delta; do
  for frac in 10 30 50 70; do
    run_name="${method}_${frac}pct"
    data_path="data/subsets/${run_name}.jsonl"
    echo "=== $run_name ==="
    python src/train.py --config $CONFIG --data $data_path --run_name $run_name
  done
done
