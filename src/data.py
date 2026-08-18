"""
Dataset loading and formatting for instruction tuning.

Every example is given a stable integer `example_id` (its index in the ORIGINAL
full training split) so that per-example loss trajectories, scores, and
selections can all be joined back together reliably across the pipeline.
"""
import json
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(example: dict) -> str:
    if example.get("input"):
        return ALPACA_PROMPT_WITH_INPUT.format(
            instruction=example["instruction"], input=example["input"]
        )
    return ALPACA_PROMPT_NO_INPUT.format(instruction=example["instruction"])


def _dataset_loader_kwargs(revision: str = None) -> dict:
    return {"revision": revision} if revision else {}


def load_and_split(
    dataset_name: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    dataset_revision: str = None,
):
    """Load the raw dataset and produce fixed train/val/test splits.

    Returns three `datasets.Dataset` objects. `example_id` is attached as a
    column on the TRAIN split only (val/test don't need pruning-related bookkeeping).
    """
    if dataset_name == "tatsu-lab/alpaca":
        raw = load_dataset("tatsu-lab/alpaca", **_dataset_loader_kwargs(dataset_revision))["train"]
    elif dataset_name in ("databricks/databricks-dolly-15k", "dolly", "dolly-15k"):
        raw = load_dataset("databricks/databricks-dolly-15k", **_dataset_loader_kwargs(dataset_revision))["train"]
        raw = raw.rename_column("context", "input") if "context" in raw.column_names else raw
        raw = raw.rename_column("response", "output") if "response" in raw.column_names else raw
    else:
        raw = load_dataset(dataset_name, **_dataset_loader_kwargs(dataset_revision))["train"]

    raw = raw.shuffle(seed=seed)
    n = len(raw)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train = raw.select(range(0, n_train))
    val = raw.select(range(n_train, n_train + n_val))
    test = raw.select(range(n_train + n_val, n))

    train = train.add_column("example_id", list(range(len(train))))
    return train, val, test


def save_split_manifest(
    train,
    val,
    test,
    dataset_name: str,
    dataset_revision: str,
    seed: int,
    path: str,
) -> dict:
    manifest = {
        "dataset_identifier": dataset_name,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "training_ids": [int(ex["example_id"]) for ex in train],
        "validation_ids": list(range(len(val))),
        "test_ids": list(range(len(test))),
        "dataset_counts": {
            "training": len(train),
            "validation": len(val),
            "test": len(test),
            "total": len(train) + len(val) + len(test),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def tokenization_stats(hf_dataset, tokenizer, max_length: int, has_ids: bool = True) -> dict:
    total_tokens = []
    supervised_tokens = []
    invalid_ids = []
    for i, ex in enumerate(hf_dataset):
        prompt = format_prompt(ex)
        response = ex["output"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        n_supervised = sum(1 for y in labels if y != -100)
        eid = int(ex["example_id"]) if has_ids and "example_id" in ex else i
        if n_supervised <= 0:
            invalid_ids.append(eid)
        total_tokens.append(len(input_ids))
        supervised_tokens.append(n_supervised)

    return {
        "total_examples": len(hf_dataset),
        "invalid_examples": len(invalid_ids),
        "invalid_example_ids": invalid_ids,
        "average_tokens": float(sum(total_tokens) / max(len(total_tokens), 1)),
        "maximum_tokens": int(max(total_tokens) if total_tokens else 0),
        "average_supervised_tokens": float(sum(supervised_tokens) / max(len(supervised_tokens), 1)),
        "minimum_supervised_tokens": int(min(supervised_tokens) if supervised_tokens else 0),
        "total_supervised_tokens": int(sum(supervised_tokens)),
    }


class InstructionDataset(Dataset):
    """Tokenizes prompt+response, masking the prompt tokens from the loss so we
    only train (and score) on the response tokens, and keeps `example_id`
    alongside each item so per-example losses can be attributed correctly."""

    def __init__(self, hf_dataset, tokenizer, max_length: int, has_ids: bool = True):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_ids = has_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt = format_prompt(ex)
        response = ex["output"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        supervised_tokens = sum(1 for y in labels if y != -100)
        if supervised_tokens <= 0:
            eid = ex.get("example_id", idx)
            raise ValueError(
                f"Example {eid} has supervised_tokens=0 after tokenization. "
                "Increase max_length or filter/fix the example before training."
            )

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if self.has_ids:
            item["example_id"] = int(ex["example_id"])
        return item


@dataclass
class PaddingCollator:
    pad_token_id: int

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn, labels, ids = [], [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(
                torch.cat([x["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attn.append(torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            if "example_id" in x:
                ids.append(x["example_id"])
        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        if ids:
            out["example_id"] = torch.tensor(ids, dtype=torch.long)
        return out
