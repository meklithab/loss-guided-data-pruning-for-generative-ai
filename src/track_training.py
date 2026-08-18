"""Exposure-balanced loss trajectory measurement.

Valid trajectory:
    base model -> score every training example (L0)
    train one complete epoch -> score every training example (L1)
    train one complete epoch -> score every training example (L2)

Scoring uses model.eval() and torch.no_grad(); it never updates parameters.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compute_tracker import ComputeTracker, append_to_log
from .data import InstructionDataset, PaddingCollator


def _per_example_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


@torch.no_grad()
def _score_full_dataset(model, loader, device, tracker: ComputeTracker) -> dict:
    model.eval()
    scores = {}
    for batch in tqdm(loader, desc="score all train examples"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        example_ids = batch["example_id"].tolist()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        per_ex_loss = _per_example_loss(outputs.logits, labels)
        tracker.add_tokens(int(attention_mask.sum().item()))

        for eid, loss_val in zip(example_ids, per_ex_loss.float().cpu().tolist()):
            scores[int(eid)] = float(loss_val)
    model.train()
    return scores


def _train_one_epoch(model, loader, optimizer, device, tracker: ComputeTracker, epoch_idx: int, run_id: str):
    model.train()
    for batch in tqdm(loader, desc=f"{run_id} warmup epoch {epoch_idx}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        tracker.add_tokens(int(attention_mask.sum().item()))


def run_exposure_balanced_tracking(
    model,
    tokenizer,
    train_hf_dataset,
    max_length: int,
    batch_size: int,
    lr: float,
    warmup_epochs: int = 2,
    run_id: str = "shared_seed42",
    log_path: str = "results/compute_log.json",
) -> dict:
    """Return {example_id: [loss_initial, loss_epoch_1, loss_epoch_2]}."""
    dataset = InstructionDataset(train_hf_dataset, tokenizer, max_length, has_ids=True)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    score_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_ids = [int(ex["example_id"]) for ex in train_hf_dataset]
    trajectories = {eid: [] for eid in all_ids}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device

    with ComputeTracker(f"{run_id}:selection_scoring") as tracker:
        sweep_scores = _score_full_dataset(model, score_loader, device, tracker)
        for eid in all_ids:
            trajectories[eid].append(sweep_scores[eid])

        for epoch_idx in range(1, warmup_epochs + 1):
            _train_one_epoch(model, train_loader, optimizer, device, tracker, epoch_idx, run_id)
            sweep_scores = _score_full_dataset(model, score_loader, device, tracker)
            for eid in all_ids:
                trajectories[eid].append(sweep_scores[eid])

    append_to_log(tracker.report(), log_path)
    return trajectories


def run_warmup_tracking(*args, **kwargs):
    kwargs.pop("checkpoint_fracs", None)
    return run_exposure_balanced_tracking(*args, **kwargs)
