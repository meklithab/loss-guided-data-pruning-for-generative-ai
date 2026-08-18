"""
CLI entry point. Every phase of the pipeline is a subcommand so you can run
them independently, inspect intermediate outputs, and re-run only what changed.

    python main.py track-warmup
    python main.py score
    python main.py select --strategy dynamics --budget 0.3
    python main.py train --strategy dynamics --budget 0.3
    python main.py evaluate --strategy dynamics --budget 0.3
    python main.py leave-k-out --strategy dynamics --budget 0.3
    python main.py run-experiment1
    python main.py run-experiment2

See RUNBOOK.md for the exact order and expected outputs.
"""
import argparse
import os

from datasets import Dataset

from src.data import load_and_split, format_prompt
from src.evaluate import evaluate_held_out, generate_and_score
from src.leave_k_out import run_leave_k_out
from src.model import load_model_and_tokenizer
from src.scoring import compute_scores
from src.select import select_examples
from src.track_training import run_warmup_tracking
from src.train_final import train_on_subset
from src.utils import load_config, merge_overrides, save_json, load_json, set_seed

RESULTS_DIR = "results"
COMPUTE_LOG = os.path.join(RESULTS_DIR, "compute_log.json")


def _load_config_with_overrides(args) -> dict:
    config = load_config(args.config)
    overrides = {
        "model_name": getattr(args, "model_name", None),
        "dataset_name": getattr(args, "dataset_name", None),
        "seed": getattr(args, "seed", None),
    }
    return merge_overrides(config, overrides)


def _splits(config):
    return load_and_split(config["dataset_name"], config["val_frac"], config["test_frac"], config["seed"])


def cmd_track_warmup(args):
    config = _load_config_with_overrides(args)
    set_seed(config["seed"])
    train, _, _ = _splits(config)

    model, tokenizer = load_model_and_tokenizer(
        config["model_name"], config["lora_r"], config["lora_alpha"],
        config["lora_dropout"], config["lora_target_modules"], config.get("use_4bit", False),
    )
    trajectories = run_warmup_tracking(
        model, tokenizer, train,
        max_length=config["max_length"],
        batch_size=config["warmup_batch_size"],
        lr=config["warmup_lr"],
        checkpoint_fracs=config["warmup_checkpoint_fracs"],
        log_path=COMPUTE_LOG,
    )
    save_json(trajectories, os.path.join(RESULTS_DIR, "trajectories.json"))
    print(f"Saved trajectories for {len(trajectories)} examples -> results/trajectories.json")


def cmd_score(args):
    trajectories = load_json(os.path.join(RESULTS_DIR, "trajectories.json"))
    trajectories = {int(k): v for k, v in trajectories.items()}
    df = compute_scores(trajectories)
    df.to_csv(os.path.join(RESULTS_DIR, "scores.csv"), index=False)
    print(f"Saved scores for {len(df)} examples -> results/scores.csv")
    print(df.describe())


def _run_name(strategy: str, budget: float) -> str:
    return f"{strategy}_{int(budget * 100)}pct"


def cmd_select(args):
    import pandas as pd

    config = _load_config_with_overrides(args)
    df = pd.read_csv(os.path.join(RESULTS_DIR, "scores.csv"))

    example_texts = None
    if args.strategy == "dynamics_diversity":
        train, _, _ = _splits(config)
        example_texts = {int(ex["example_id"]): format_prompt(ex) + ex["output"] for ex in train}

    selected = select_examples(df, args.strategy, args.budget, seed=config["seed"], example_texts=example_texts)
    out_path = os.path.join(RESULTS_DIR, f"selection_{_run_name(args.strategy, args.budget)}.json")
    save_json({"strategy": args.strategy, "budget": args.budget, "selected_ids": selected}, out_path)
    print(f"Selected {len(selected)} examples ({args.strategy} @ {args.budget:.0%}) -> {out_path}")


def cmd_train(args):
    config = _load_config_with_overrides(args)
    set_seed(config["seed"])
    train, val, test = _splits(config)

    sel_path = os.path.join(RESULTS_DIR, f"selection_{_run_name(args.strategy, args.budget)}.json")
    selection = load_json(sel_path)
    selected_ids = selection["selected_ids"]

    run_name = _run_name(args.strategy, args.budget)
    model, tokenizer = train_on_subset(config, selected_ids, train, run_name, log_path=COMPUTE_LOG)

    save_dir = os.path.join(RESULTS_DIR, "checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Trained on {len(selected_ids)} examples -> {save_dir}")


def cmd_evaluate(args):
    config = _load_config_with_overrides(args)
    _, val, test = _splits(config)

    run_name = _run_name(args.strategy, args.budget)
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints", run_name)

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    metrics = evaluate_held_out(model, tokenizer, test, config["max_length"])
    if args.generate_eval:
        gen_metrics = generate_and_score(model, tokenizer, test, n_samples=args.n_generate_samples)
        metrics.update(gen_metrics)

    out_path = os.path.join(RESULTS_DIR, f"eval_{run_name}.json")
    save_json(metrics, out_path)
    print(f"{run_name}: {metrics}")


def cmd_leave_k_out(args):
    import pandas as pd

    config = _load_config_with_overrides(args)
    set_seed(config["seed"])
    train, val, test = _splits(config)

    sel_path = os.path.join(RESULTS_DIR, f"selection_{_run_name(args.strategy, args.budget)}.json")
    selection = load_json(sel_path)
    scores_df = pd.read_csv(os.path.join(RESULTS_DIR, "scores.csv"))

    result = run_leave_k_out(config, selection["selected_ids"], scores_df, train, val, log_path=COMPUTE_LOG)
    out_path = os.path.join(RESULTS_DIR, f"leave_k_out_{_run_name(args.strategy, args.budget)}.json")
    save_json(result, out_path)
    print("Spearman correlations with TRUE marginal value:")
    for k, v in result["correlations_with_true_marginal_value"].items():
        print(f"  {k}: rho={v['spearman_rho']:.3f} (p={v['p_value']:.3f})")


def build_parser():
    p = argparse.ArgumentParser(description="Dynamic Loss-Guided Data Pruning pipeline")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model_name", default=None)
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("track-warmup").set_defaults(func=cmd_track_warmup)
    sub.add_parser("score").set_defaults(func=cmd_score)

    sel = sub.add_parser("select")
    sel.add_argument("--strategy", required=True,
                      choices=["random", "low_loss", "high_loss", "loss_delta", "dynamics", "dynamics_diversity"])
    sel.add_argument("--budget", type=float, required=True)
    sel.set_defaults(func=cmd_select)

    tr = sub.add_parser("train")
    tr.add_argument("--strategy", required=True)
    tr.add_argument("--budget", type=float, required=True)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--strategy", required=True)
    ev.add_argument("--budget", type=float, required=True)
    ev.add_argument("--generate_eval", action="store_true", help="also run ROUGE-L generation eval (slower)")
    ev.add_argument("--n_generate_samples", type=int, default=100)
    ev.set_defaults(func=cmd_evaluate)

    lko = sub.add_parser("leave-k-out")
    lko.add_argument("--strategy", required=True)
    lko.add_argument("--budget", type=float, required=True)
    lko.set_defaults(func=cmd_leave_k_out)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    args.func(args)
