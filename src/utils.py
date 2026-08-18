"""Small shared utilities: seeding, JSON/YAML IO, config merging."""
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_overrides(config: dict, overrides: dict) -> dict:
    """Shallow-merge CLI overrides (non-None values only) into a loaded config dict."""
    merged = dict(config)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _json_default(o):
    # numpy scalars / arrays -> plain python for json.dump
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


@contextmanager
def timer():
    """Usage: with timer() as t: ...   then t['seconds'] after the block exits."""
    state = {}
    start = time.time()
    try:
        yield state
    finally:
        state["seconds"] = time.time() - start
