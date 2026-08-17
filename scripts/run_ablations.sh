#!/bin/bash
# EXTENDED ablations (threshold band + cluster count) -- only relevant
# if you've run the extended methods. The CORE loss-signal comparison
# (loss_high vs loss_low vs loss_delta vs random) is already an ablation
# by construction via run_grid.sh -- no separate script needed for it.
set -e
CONFIG=configs/config.yaml

for name in lowest midband; do
  run_name="ablation_threshold_${name}_30pct"
  python src/train.py --config $CONFIG --data data/subsets/${run_name}.jsonl --run_name $run_name
done

for k in 10 50 100; do
  run_name="ablation_clusters_k${k}_30pct"
  python src/train.py --config $CONFIG --data data/subsets/${run_name}.jsonl --run_name $run_name
done
