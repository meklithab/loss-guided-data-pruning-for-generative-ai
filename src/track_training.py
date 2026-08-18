"""
Phase 1 — warm-up tracking run.

IMPORTANT DESIGN NOTE (read this before trusting the trajectories):
To get a real per-example TRAJECTORY (loss at 10%, 25%, 50%, 100% of warm-up)
we cannot just record each example's loss the one time it happens to pass
through a training batch — in a single epoch every example is seen exactly
once, so that would give one point, not a trajectory.

Instead we do it the way dataset-cartography-style methods do: train
normally, but PAUSE at each checkpoint fraction of the warm-up epoch and run
a full forward-only (no_grad) sweep over the ENTIRE training set to record
every example's current loss under the model-as-of-that-point. Training then
resumes. This is more expensive than a single pass (you pay for len(checkpoint_fracs)
extra forward sweeps) but it is the only way the "loss dynamics" signals
(delta, relative improvement, slope) are actually measuring the same
example's loss over time rather than different examples at different times.

This extra cost is exactly why Experiment 2 (which signal works) matters:
if static loss (needing only ONE sweep, no checkpoints) does nearly as well
as full dynamics (needing several), that's a real efficiency finding.

Output: {example_id: [loss_at_ckpt1, loss_at_ckpt2, ...]} written to
results/trajectories.json by main.py.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compute_tracker import ComputeTracker, append_to_log
from .data import InstructionDataset, PaddingCollator


def _per_example_loss(logits, labels):
    """Mean cross-entropy over the (label != -100) tokens of each example,
    returning one scalar loss per example. Keeps grad if logits requires_grad."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    per_example = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_example


@torch.no_grad()
def _score_full_dataset(model, loader, device, tracker: ComputeTracker) -> dict:
    """Forward-only sweep recording current loss for every example_id."""
    model.eval()
    scores = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        example_ids = batch["example_id"].tolist()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        per_ex_loss = _per_example_loss(outputs.logits, labels)
        tracker.add_tokens(int(attention_mask.sum().item()))

        for eid, loss_val in zip(example_ids, per_ex_loss.float().cpu().tolist()):
            scores[eid] = loss_val
    model.train()
    return scores


def run_warmup_tracking(
    model,
    tokenizer,
    train_hf_dataset,
    max_length: int,
    batch_size: int,
    lr: float,
    checkpoint_fracs: list,
    log_path: str = "results/compute_log.json",
) -> dict:
    """Trains one warm-up epoch, pausing at each checkpoint fraction to sweep
    the full dataset for current per-example loss. Returns
    {example_id: [loss_ckpt1, loss_ckpt2, ...]} covering every example."""
    dataset = InstructionDataset(train_hf_dataset, tokenizer, max_length, has_ids=True)
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    # Fixed order for scoring sweeps so every checkpoint scores in the same order (not required, just tidy).
    score_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    n_batches = len(train_loader)
    checkpoint_batches = sorted(set(max(int(n_batches * f), 1) for f in checkpoint_fracs))

    all_ids = [int(ex["example_id"]) for ex in train_hf_dataset]
    trajectories = {eid: [] for eid in all_ids}

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
    model.train()

    with ComputeTracker("warmup_tracking") as tracker:
        ckpt_set = set(checkpoint_batches)
        for step, batch in enumerate(tqdm(train_loader, desc="warmup_tracking (training)")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            per_ex_loss = _per_example_loss(outputs.logits, labels)
            batch_loss = per_ex_loss.mean()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            tracker.add_tokens(int(attention_mask.sum().item()))

            if (step + 1) in ckpt_set:
                sweep_scores = _score_full_dataset(model, score_loader, device, tracker)
                for eid in all_ids:
                    trajectories[eid].append(sweep_scores.get(eid))

        # Guard: if the epoch had fewer batches than checkpoints requested
        # (tiny smoke-test datasets), make sure we still have one sweep at the end.
        if len(next(iter(trajectories.values()))) < len(checkpoint_fracs):
            sweep_scores = _score_full_dataset(model, score_loader, device, tracker)
            for eid in all_ids:
                trajectories[eid].append(sweep_scores.get(eid))

    append_to_log(tracker.report(), log_path)
    return trajectories
