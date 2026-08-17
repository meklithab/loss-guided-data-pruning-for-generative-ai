"""
Evaluates every trained checkpoint on the SAME held-out eval split
(data/splits/eval.jsonl, carved out once in data_utils.py and never
touched by training or selection) and writes eval_loss / eval_ppl /
task_accuracy back into results/runs.csv.

Usage:
    python src/evaluate.py --config configs/config.yaml --all_runs
    python src/evaluate.py --config configs/config.yaml --run_name loss_20pct
"""
import argparse
import os
import csv
import math
from utils import load_config
from data_utils import load_jsonl


def eval_held_out_loss(checkpoint_dir, eval_data, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], device_map="auto")
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model.eval()

    losses = []
    with torch.no_grad():
        for ex in eval_data:
            text = ex["prompt"] + ex["response"]
            ids = tok(text, truncation=True, max_length=cfg["model"]["max_seq_len"],
                       return_tensors="pt").to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            losses.append(out.loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def eval_generation_quality(checkpoint_dir, eval_data, cfg):
    """
    Generates a completion for a sample of held-out prompts and scores it
    against the reference response with:
      - ROUGE-L F1 (standard metric for instruction-following / summarization-
        style generation quality -- this is the primary quality number)
      - a crude exact/near-match rate as a secondary sanity-check signal
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from rouge_score import rouge_scorer

    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], device_map="auto")
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model.eval()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    n = min(cfg["evaluation"]["task_eval_n_prompts"], len(eval_data))
    sample = eval_data[:n]

    rouge_l_scores = []
    near_match = 0
    with torch.no_grad():
        for ex in sample:
            ids = tok(ex["prompt"], return_tensors="pt").to(model.device)
            gen = model.generate(**ids, max_new_tokens=64, do_sample=False)
            pred = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

            rl = scorer.score(ex["response"], pred)["rougeL"].fmeasure
            rouge_l_scores.append(rl)

            if ex["response"].strip()[:20].lower() in pred.lower():
                near_match += 1

    return (sum(rouge_l_scores) / len(rouge_l_scores)), (near_match / n)


def update_runs_csv(runs_csv, run_name, updates: dict):
    rows = []
    with open(runs_csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["run_name"] == run_name:
                row.update({k: str(v) for k, v in updates.items()})
            rows.append(row)
    with open(runs_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(run_name, cfg):
    checkpoint_dir = os.path.join(cfg["training"]["output_dir"], run_name)
    eval_data = load_jsonl(os.path.join(cfg["dataset"]["splits_dir"], "eval.jsonl"))

    avg_loss, ppl = eval_held_out_loss(checkpoint_dir, eval_data, cfg)
    rouge_l, near_match_acc = eval_generation_quality(checkpoint_dir, eval_data, cfg)

    update_runs_csv(cfg["results"]["runs_csv"], run_name, {
        "eval_loss": round(avg_loss, 4),
        "eval_ppl": round(ppl, 4),
        "task_accuracy": round(near_match_acc, 4),
        "rouge_l": round(rouge_l, 4),
    })
    print(f"[eval] {run_name}: loss={avg_loss:.4f} ppl={ppl:.2f} "
          f"rougeL={rouge_l:.3f} near_match={near_match_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--all_runs", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.all_runs:
        with open(cfg["results"]["runs_csv"], "r") as f:
            names = [row["run_name"] for row in csv.DictReader(f) if row["status"] == "complete"]
        for name in names:
            evaluate_run(name, cfg)
    elif args.run_name:
        evaluate_run(args.run_name, cfg)
    else:
        raise SystemExit("pass --run_name or --all_runs")
