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
from .compute_tracker import ComputeTracker, append_to_log


@torch.no_grad()
def evaluate_held_out(
    model,
    tokenizer,
    hf_dataset,
    max_length: int,
    batch_size: int = 8,
    run_id: str = None,
    split_name: str = "held_out",
    log_path: str = None,
) -> dict:
    dataset = InstructionDataset(hf_dataset, tokenizer, max_length, has_ids=False)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = next(model.parameters()).device
    model.eval()

    total_loss, total_tokens, total_correct = 0.0, 0, 0
    with ComputeTracker("evaluation", run_id=run_id) as tracker:
        for batch in tqdm(loader, desc=f"evaluate_{split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            n_label_tokens = int((labels != -100).sum().item())
            total_loss += outputs.loss.item() * n_label_tokens
            total_tokens += n_label_tokens
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != -100
            preds = shift_logits.argmax(dim=-1)
            total_correct += int(((preds == shift_labels) & mask).sum().item())
            tracker.add_tokens(int(attention_mask.sum().item()))
    if log_path:
        report = tracker.report()
        report["split"] = split_name
        append_to_log(report, log_path)

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        f"{split_name}_loss": mean_loss,
        f"{split_name}_perplexity": math.exp(min(mean_loss, 20)),
        f"{split_name}_token_accuracy": total_correct / max(total_tokens, 1),
    }


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
    exact = [
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    ]
    return {
        "rougeL": scores["rougeL"],
        "rouge1": scores["rouge1"],
        "task_specific_accuracy": sum(exact) / max(len(exact), 1),
        "n_examples_scored": len(predictions),
    }
