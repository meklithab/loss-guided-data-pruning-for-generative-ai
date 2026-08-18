#!/usr/bin/env bash
# Kaggle workflow:
#   clone/upload repo -> bash scripts/setup_kaggle.sh -> python main.py smoke-test -> run an experiment script
set -e
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export TOKENIZERS_PARALLELISM=false
export RESULTS_DIR="${RESULTS_DIR:-/kaggle/working/results}"
mkdir -p "$RESULTS_DIR"
echo "Kaggle setup complete. Results will be written to $RESULTS_DIR"
