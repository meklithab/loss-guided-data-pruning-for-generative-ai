"""
Phase 4 — train a FRESH LoRA model from scratch on a selected subset of the
training data. "Fresh" matters: we must not reuse the warm-up-tracking model's
weights, or the comparison across strategies/budgets would be confounded by
how much extra training each condition happened to get during scoring.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compute_tracker import ComputeTracker, append_to_log
from .data import InstructionDataset, PaddingCollator
from .model import load_model_and_tokenizer


def train_on_subset(
    config: dict,
    selected_ids: list,
    train_hf_dataset,
    run_name: str,
    log_path: str = "results/compute_log.json",
):
    """Filters train_hf_dataset down to `selected_ids`, trains a fresh LoRA
    model for config['train_epochs'] epochs, and returns (model, tokenizer).
    Logs a ComputeTracker phase named f'train_{run_name}'."""
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model_name"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        use_4bit=config.get("use_4bit", False),
    )

    selected_set = set(selected_ids)
    subset = train_hf_dataset.filter(lambda ex: ex["example_id"] in selected_set)

    dataset = InstructionDataset(subset, tokenizer, config["max_length"], has_ids=True)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset, batch_size=config["train_batch_size"], shuffle=True, collate_fn=collator
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["train_lr"]
    )
    device = next(model.parameters()).device
    model.train()

    with ComputeTracker(f"train_{run_name}") as tracker:
        for epoch in range(config["train_epochs"]):
            for batch in tqdm(loader, desc=f"train_{run_name} epoch {epoch + 1}/{config['train_epochs']}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tracker.add_tokens(int(attention_mask.sum().item()))

    report = tracker.report()
    report["n_examples_trained_on"] = len(subset)
    append_to_log(report, log_path)

    return model, tokenizer
