"""Research CLI for exposure-balanced loss-guided data pruning."""
import argparse
import os
import platform
import sys

import pandas as pd

from src.data import (
    format_prompt,
    load_and_split,
    save_split_manifest,
    tokenization_stats,
)
from src.scoring import compute_scores
from src.select import select_examples
from src.utils import load_config, load_json, merge_overrides, save_json, set_seed


def _config(args) -> dict:
    cfg = load_config(args.config)
    overrides = {
        "model_name": getattr(args, "model_name", None),
        "dataset_name": getattr(args, "dataset_name", None),
        "seed": getattr(args, "seed", None),
        "device": getattr(args, "device", None),
    }
    cfg = merge_overrides(cfg, overrides)
    cfg["results_dir"] = os.environ.get("RESULTS_DIR", cfg.get("results_dir", "results"))
    return cfg


def _paths(cfg: dict) -> dict:
    root = cfg.get("results_dir", "results")
    return {
        "root": root,
        "compute": os.path.join(root, "compute_log.json"),
        "splits": os.path.join(root, "split_manifest.json"),
        "stats": os.path.join(root, "dataset_stats.json"),
        "runs": os.path.join(root, "runs"),
        "checkpoints": os.path.join(root, "checkpoints"),
    }


def _splits(cfg):
    return load_and_split(
        cfg["dataset_name"],
        cfg["val_frac"],
        cfg["test_frac"],
        cfg["seed"],
        cfg.get("dataset_revision"),
    )


def _run_id(exp: str, strategy: str, budget: float, seed: int) -> str:
    return f"{exp}_{strategy}_{int(round(budget * 100))}_seed{seed}"


def _save_run_config(cfg, run_id):
    run_dir = os.path.join(_paths(cfg)["runs"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    save_json(cfg, os.path.join(run_dir, "config.json"))


def _env_metadata() -> dict:
    import torch

    def version(name):
        try:
            module = __import__(name)
            return getattr(module, "__version__", "installed")
        except Exception as exc:
            return f"unavailable: {exc.__class__.__name__}"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "datasets": version("datasets"),
        "trl": version("trl"),
    }


def cmd_prepare(args):
    from transformers import AutoTokenizer

    cfg = _config(args)
    paths = _paths(cfg)
    os.makedirs(paths["root"], exist_ok=True)
    set_seed(cfg["seed"])
    train, val, test = _splits(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg.get("model_revision"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifest = save_split_manifest(
        train,
        val,
        test,
        cfg["dataset_name"],
        cfg.get("dataset_revision"),
        cfg["seed"],
        paths["splits"],
    )
    stats = {
        "train": tokenization_stats(train, tokenizer, cfg["max_length"], has_ids=True),
        "validation": tokenization_stats(val, tokenizer, cfg["max_length"], has_ids=False),
        "test": tokenization_stats(test, tokenizer, cfg["max_length"], has_ids=False),
        "environment": _env_metadata(),
    }
    save_json(stats, paths["stats"])
    if any(s["invalid_examples"] for s in [stats["train"], stats["validation"], stats["test"]]):
        raise SystemExit(f"Invalid supervised-token examples found. See {paths['stats']}")
    print(f"Saved split manifest and tokenization stats for {manifest['dataset_counts']}")


def cmd_track(args):
    from src.model import load_model_and_tokenizer
    from src.track_training import run_exposure_balanced_tracking

    cfg = _config(args)
    paths = _paths(cfg)
    set_seed(cfg["seed"])
    train, _, _ = _splits(cfg)
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"],
        cfg.get("model_revision"),
        cfg["lora_r"],
        cfg["lora_alpha"],
        cfg["lora_dropout"],
        cfg["lora_target_modules"],
        cfg.get("use_4bit", False),
        cfg.get("device", "auto"),
    )
    run_id = f"selection_seed{cfg['seed']}_epochs{args.warmup_epochs}"
    trajectories = run_exposure_balanced_tracking(
        model,
        tokenizer,
        train,
        cfg["max_length"],
        cfg["warmup_batch_size"],
        cfg["warmup_lr"],
        warmup_epochs=args.warmup_epochs,
        run_id=run_id,
        log_path=paths["compute"],
    )
    save_json(trajectories, os.path.join(paths["root"], f"trajectories_{run_id}.json"))
    if args.warmup_epochs == cfg.get("warmup_epochs", 2):
        save_json(trajectories, os.path.join(paths["root"], "trajectories.json"))
    print(f"Saved exposure-balanced trajectories for {len(trajectories)} examples")


def cmd_score(args):
    cfg = _config(args)
    path = args.trajectories or os.path.join(_paths(cfg)["root"], "trajectories.json")
    trajectories = {int(k): v for k, v in load_json(path).items()}
    df = compute_scores(trajectories)
    out = args.output or os.path.join(_paths(cfg)["root"], "scores.csv")
    df.to_csv(out, index=False)
    print(f"Saved scores -> {out}")


def _select_and_save(cfg, scores_df, strategy, budget, seed, run_id):
    from src.compute_tracker import ComputeTracker, append_to_log

    paths = _paths(cfg)
    train, _, _ = _splits(cfg)
    example_texts = {int(ex["example_id"]): format_prompt(ex) + ex["output"] for ex in train}
    with ComputeTracker("selection", run_id=run_id) as tracker:
        selected = select_examples(scores_df, strategy, budget, seed=seed, example_texts=example_texts)
    append_to_log(tracker.report(), paths["compute"])
    selected_set = set(selected)
    token_count = 0
    for ex in train:
        if int(ex["example_id"]) in selected_set:
            token_count += len((format_prompt(ex) + ex["output"]).split())
    payload = {
        "run_id": run_id,
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "selected_ids": selected,
        "n_examples": len(selected),
        "data_retained_percent": 100 * len(selected) / max(len(train), 1),
        "data_removed_percent": 100 * (1 - len(selected) / max(len(train), 1)),
        "approx_training_tokens_whitespace": token_count,
    }
    out = os.path.join(paths["runs"], run_id, "selection.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_json(payload, out)
    return selected


def _train_eval_run(cfg, run_id, selected_ids, generate_eval=False):
    from src.evaluate import evaluate_held_out, generate_and_score
    from src.train_final import train_on_subset

    paths = _paths(cfg)
    _save_run_config(cfg, run_id)
    train, val, test = _splits(cfg)
    model, tokenizer = train_on_subset(cfg, selected_ids, train, run_id, log_path=paths["compute"])
    ckpt_dir = os.path.join(paths["checkpoints"], run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    metrics = {"run_id": run_id}
    metrics.update(evaluate_held_out(model, tokenizer, val, cfg["max_length"], cfg["train_batch_size"], run_id, "validation", paths["compute"]))
    metrics.update(evaluate_held_out(model, tokenizer, test, cfg["max_length"], cfg["train_batch_size"], run_id, "test", paths["compute"]))
    if generate_eval:
        metrics.update(generate_and_score(model, tokenizer, test, n_samples=cfg.get("n_generate_samples", 100)))
    else:
        metrics["task_specific_accuracy"] = metrics.get("test_token_accuracy")
    save_json(metrics, os.path.join(paths["runs"], run_id, "metrics.json"))
    return metrics


def cmd_run_single(args):
    cfg = _config(args)
    cfg["seed"] = args.seed or cfg["seed"]
    paths = _paths(cfg)
    set_seed(cfg["seed"])
    scores_df = pd.read_csv(os.path.join(paths["root"], "scores.csv"))
    run_id = args.run_id or _run_id(args.exp, args.strategy, args.budget, cfg["seed"])
    selected = _select_and_save(cfg, scores_df, args.strategy, args.budget, cfg["seed"], run_id)
    metrics = _train_eval_run(cfg, run_id, selected, args.generate_eval)
    print(metrics)


def cmd_experiment1(args):
    cfg = _config(args)
    scores_df = pd.read_csv(os.path.join(_paths(cfg)["root"], "scores.csv"))
    for seed in cfg.get("seeds", [cfg["seed"]]):
        cfg["seed"] = seed
        set_seed(seed)
        for budget in cfg["retention_percentages"]:
            for strategy in cfg["experiment1_strategies"]:
                if strategy == "full" and budget != 1.0:
                    continue
                if budget == 1.0 and strategy != "full":
                    continue
                run_id = _run_id("exp1", strategy, budget, seed)
                selected = _select_and_save(cfg, scores_df, strategy, budget, seed, run_id)
                _train_eval_run(cfg, run_id, selected, args.generate_eval)


def cmd_experiment2(args):
    cfg = _config(args)
    scores_df = pd.read_csv(os.path.join(_paths(cfg)["root"], "scores.csv"))
    for seed in cfg.get("seeds", [cfg["seed"]]):
        cfg["seed"] = seed
        for signal in cfg["experiment2_signals"]:
            run_id = _run_id("exp2", signal, cfg["main_budget"], seed)
            selected = _select_and_save(cfg, scores_df, signal, cfg["main_budget"], seed, run_id)
            _train_eval_run(cfg, run_id, selected, args.generate_eval)


def cmd_experiment3(args):
    cfg = _config(args)
    paths = _paths(cfg)
    for seed in cfg.get("seeds", [cfg["seed"]]):
        cfg["seed"] = seed
        for method, epochs in [("initial", 0), ("dynamics_1epoch", 1), ("dynamics_2epoch", 2)]:
            traj_path = os.path.join(paths["root"], f"trajectories_selection_seed{seed}_epochs{epochs}.json")
            if not os.path.exists(traj_path):
                args.seed = seed
                args.warmup_epochs = epochs
                cmd_track(args)
            scores_df = compute_scores({int(k): v for k, v in load_json(traj_path).items()})
            strategy = "initial_loss" if method == "initial" else "dynamics"
            run_id = _run_id("exp3", method, cfg["main_budget"], seed)
            selected = _select_and_save(cfg, scores_df, strategy, cfg["main_budget"], seed, run_id)
            _train_eval_run(cfg, run_id, selected, args.generate_eval)


def cmd_experiment5(args):
    from src.leave_k_out import run_leave_k_out

    cfg = _config(args)
    paths = _paths(cfg)
    cfg["seed"] = args.seed or cfg["seed"]
    set_seed(cfg["seed"])
    train, val, _ = _splits(cfg)
    scores_df = pd.read_csv(os.path.join(paths["root"], "scores.csv"))
    run_id = _run_id("exp5", args.strategy, args.budget, cfg["seed"])
    selected = _select_and_save(cfg, scores_df, args.strategy, args.budget, cfg["seed"], run_id)
    result = run_leave_k_out(cfg, selected, scores_df, train, val, run_id=run_id, log_path=paths["compute"])
    result["run_id"] = run_id
    save_json(result, os.path.join(paths["runs"], run_id, "leave_k_out.json"))
    print(result["correlations_with_true_marginal_value"])


def cmd_smoke(args):
    import torch
    from torch.utils.data import DataLoader
    from src.data import InstructionDataset, PaddingCollator
    from src.evaluate import evaluate_held_out
    from src.model import load_model_and_tokenizer

    cfg = _config(args)
    cfg["train_epochs"] = 1
    set_seed(cfg["seed"])
    print(_env_metadata())
    train, val, _ = _splits(cfg)
    train = train.select(range(min(4, len(train))))
    val = val.select(range(min(4, len(val))))
    model, tokenizer = load_model_and_tokenizer(
        cfg["model_name"], cfg.get("model_revision"), cfg["lora_r"], cfg["lora_alpha"],
        cfg["lora_dropout"], cfg["lora_target_modules"], cfg.get("use_4bit", False), cfg.get("device", "auto")
    )
    stats = tokenization_stats(train, tokenizer, cfg["max_length"])
    if stats["invalid_examples"]:
        raise SystemExit(stats)
    ds = InstructionDataset(train, tokenizer, cfg["max_length"])
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=PaddingCollator(tokenizer.pad_token_id))))
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["train_lr"])
    model.train()
    loss = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    ).loss
    loss.backward()
    opt.step()
    metrics = evaluate_held_out(model, tokenizer, val, cfg["max_length"], split_name="validation")
    print({"model_loading": "ok", "tokenizer": "ok", "lora": "ok", "one_training_step_loss": float(loss.detach().cpu()), "evaluation": metrics})


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model_name", default=None)
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    smoke = sub.add_parser("smoke-test")
    smoke.set_defaults(func=cmd_smoke)
    trk = sub.add_parser("track-warmup")
    trk.add_argument("--warmup_epochs", type=int, default=2)
    trk.set_defaults(func=cmd_track)
    score = sub.add_parser("score")
    score.add_argument("--trajectories", default=None)
    score.add_argument("--output", default=None)
    score.set_defaults(func=cmd_score)

    single = sub.add_parser("run-single")
    single.add_argument("--exp", default="manual")
    single.add_argument("--strategy", required=True)
    single.add_argument("--budget", type=float, required=True)
    single.add_argument("--run_id", default=None)
    single.add_argument("--generate_eval", action="store_true")
    single.set_defaults(func=cmd_run_single)

    for name, func in [("run-experiment1", cmd_experiment1), ("run-experiment2", cmd_experiment2), ("run-experiment3", cmd_experiment3)]:
        sp = sub.add_parser(name)
        sp.add_argument("--generate_eval", action="store_true")
        sp.set_defaults(func=func)

    exp5 = sub.add_parser("run-experiment5")
    exp5.add_argument("--strategy", default="dynamics")
    exp5.add_argument("--budget", type=float, default=0.30)
    exp5.set_defaults(func=cmd_experiment5)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
