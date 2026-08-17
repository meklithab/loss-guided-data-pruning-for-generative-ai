"""
One call to this script = one experiment cell trained end to end.
Designed to be looped over by scripts/run_grid.sh so a killed Colab
session can just re-run the loop -- finished run_names are skipped
(see utils.already_done).

Usage:
    python src/train.py --config configs/config.yaml \
        --data data/subsets/loss_20pct.jsonl \
        --run_name loss_20pct
"""
import argparse
import os
from utils import (
    load_config, set_seed, RunTimer, already_done, log_run,
    reset_peak_memory_stats, get_peak_vram_gb, estimate_flops, snapshot_environment,
)
from data_utils import load_jsonl


def count_tokens(dataset, tokenizer, max_len) -> int:
    total = 0
    for ex in dataset:
        text = ex["prompt"] + ex["response"]
        total += min(len(tokenizer.encode(text)), max_len)
    return total


def main(cfg: dict, data_path: str, run_name: str):
    runs_csv = cfg["results"]["runs_csv"]
    if already_done(run_name, runs_csv):
        print(f"[skip] {run_name} already complete in {runs_csv}")
        return

    set_seed(cfg["seed"])
    snapshot_environment()  # no-op after the first call this session

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from datasets import Dataset

    reset_peak_memory_stats()

    subset = load_jsonl(data_path)
    hf_dataset = Dataset.from_list(
        [{"text": ex["prompt"] + ex["response"]} for ex in subset]
    )

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_kwargs = {}
    if cfg["model"]["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **quant_kwargs)

    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
    ))

    out_dir = os.path.join(cfg["training"]["output_dir"], run_name)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_len"],
    )

    n_tokens = count_tokens(subset, tok, cfg["model"]["max_seq_len"])
    total_tokens_seen = n_tokens * cfg["training"]["epochs"]
    # count base-model params for the FLOPs estimate (see utils.estimate_flops
    # for why this uses total params rather than just LoRA's trainable subset)
    n_params = sum(p.numel() for p in model.parameters())

    status, notes, final_loss = "complete", "", None
    with RunTimer() as timer:
        try:
            result = trainer.train()
            final_loss = result.training_loss
        except Exception as e:
            status, notes = "failed", str(e)[:200]
            raise
        finally:
            trainer.save_model(out_dir)

    peak_vram_gb = get_peak_vram_gb()
    flops_estimate = estimate_flops(n_params, total_tokens_seen)

    # run_name format is "<method>_<NN>pct", but method itself may contain
    # underscores (loss_high, loss_low, loss_delta) -- so strip only the
    # trailing "_NNpct" token rather than splitting on the first underscore.
    parts = run_name.split("_")
    if parts[-1].endswith("pct"):
        method = "_".join(parts[:-1])
        data_fraction = float(parts[-1].replace("pct", "")) / 100.0
    else:
        method, data_fraction = run_name, None
    data_reduction_pct = round((1 - data_fraction) * 100, 1) if data_fraction is not None else None

    log_run(runs_csv, {
        "run_name": run_name,
        "method": method,
        "data_fraction": data_fraction,
        "data_reduction_pct": data_reduction_pct,
        "n_examples": len(subset),
        "seed": cfg["seed"],
        "gpu_minutes": round(timer.elapsed_min, 2),
        "tokens_seen": total_tokens_seen,
        "flops_estimate": f"{flops_estimate:.3e}",
        "peak_vram_gb": peak_vram_gb,
        "train_loss_final": final_loss,
        "status": status,
        "notes": notes,
    })
    print(f"[done] {run_name}: {timer.elapsed_min:.1f} GPU-min, {total_tokens_seen} tokens, "
          f"{peak_vram_gb:.2f} GB peak VRAM, {flops_estimate:.2e} FLOPs (est.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.data, args.run_name)
