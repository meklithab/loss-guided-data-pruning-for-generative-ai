#!/bin/bash
# Full-data baseline + matched-budget random baselines (the reference
# points every loss-guided run gets compared against).
set -e
CONFIG=configs/config.yaml

python src/train.py --config $CONFIG --data data/subsets/random_100pct.jsonl --run_name random_100pct

for frac in 10 30 50 70; do
  python src/train.py --config $CONFIG --data data/subsets/random_${frac}pct.jsonl --run_name random_${frac}pct
done
