"""
Phase 5 — evaluation.

    evaluate_held_out()       -> mean held-out loss + perplexity on the val/test split
    generate_and_score()      -> greedy-decodes responses for a small eval subset and
                                   scores them against reference outputs with ROUGE-L
                                   (a cheap, dependency-light proxy for instruction-following
                                   quality; swap in an LLM-judge call here if you have API
                                   budget for one — see the docstring below).
"""
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import InstructionDataset, PaddingCollator, format_prompt


@torch.no_grad()
def evaluate_held_out(model, tokenizer, hf_dataset, max_length: int, batch_size: int = 8) -> dict:
    dataset = InstructionDataset(hf_dataset, tokenizer, max_length, has_ids=False)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = next(model.parameters()).device
    model.eval()

    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="evaluate_held_out"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_label_tokens = int((labels != -100).sum().item())
        total_loss += outputs.loss.item() * n_label_tokens
        total_tokens += n_label_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    return {"held_out_loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20))}


@torch.no_grad()
def generate_and_score(model, tokenizer, hf_dataset, max_new_tokens: int = 128, n_samples: int = None):
    """Greedy-decodes a completion for each example's prompt and scores it
    against the reference `output` with ROUGE-L as a cheap instruction-following
    proxy. For a more faithful "instruction-following performance" metric,
    replace the ROUGE scoring block with a call to a stronger LLM-as-judge
    (e.g. score 1-10 on instruction adherence) — the generation loop itself
    doesn't need to change, only how `predictions` are scored.
    """
    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")
    device = next(model.parameters()).device
    model.eval()

    data = hf_dataset if n_samples is None else hf_dataset.select(range(min(n_samples, len(hf_dataset))))

    predictions, references = [], []
    for ex in tqdm(data, desc="generate_and_score"):
        prompt = format_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(gen_text)
        references.append(ex["output"])

    scores = rouge.compute(predictions=predictions, references=references)
    return {
        "rougeL": scores["rougeL"],
        "rouge1": scores["rouge1"],
        "n_examples_scored": len(predictions),
    }
