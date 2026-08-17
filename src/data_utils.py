"""
Loads the raw instruction dataset ONCE, carves out a held-out eval split
that nothing else in the pipeline is allowed to touch, and writes the
remaining "train pool" to disk so every selection method scores/samples
from the exact same pool (fair comparison).

Run once at the start of the project:
    python src/data_utils.py --config configs/config.yaml
"""
import argparse
import os
import json
from datasets import load_dataset
from utils import load_config, set_seed


def format_example(ex, cfg) -> dict:
    """Turn a raw record into a standard {prompt, response} pair."""
    instr = ex[cfg["dataset"]["text_field_instruction"]]
    inp = ex.get(cfg["dataset"]["text_field_input"], "") or ""
    out = ex[cfg["dataset"]["text_field_output"]]
    if inp.strip():
        prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
    return {"prompt": prompt, "response": out}


def build_splits(cfg: dict):
    set_seed(cfg["seed"])
    raw = load_dataset(cfg["dataset"]["name"], split="train")
    raw = raw.shuffle(seed=cfg["seed"])

    n_eval = int(len(raw) * cfg["dataset"]["eval_holdout_frac"])
    eval_raw = raw.select(range(n_eval))
    pool_raw = raw.select(range(n_eval, len(raw)))

    eval_data = [format_example(ex, cfg) for ex in eval_raw]
    pool_data = [format_example(ex, cfg) for ex in pool_raw]

    splits_dir = cfg["dataset"]["splits_dir"]
    os.makedirs(splits_dir, exist_ok=True)

    with open(os.path.join(splits_dir, "eval.jsonl"), "w") as f:
        for row in eval_data:
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(splits_dir, "pool.jsonl"), "w") as f:
        for row in pool_data:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(pool_data)} pool examples and {len(eval_data)} held-out eval examples "
          f"to {splits_dir}/ (seed={cfg['seed']}).")


def load_jsonl(path: str) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_splits(cfg)
