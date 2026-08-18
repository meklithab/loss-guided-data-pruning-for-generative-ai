"""
Phase 6 — Experiment 5: does the cheap signal predict TRUE marginal value?

This is the strongest scientific experiment in the project (per the proposal)
and the one most likely to be skipped under time pressure, so keep its scope
small and fixed:

  1. Take the subset SELECTED by your best strategy at your main budget
     (e.g. `dynamics` at 30%) — this is your baseline training set.
  2. Randomly sample `n_examples_sampled` examples from that subset and split
     them into groups of size `group_size` (e.g. 200 examples / 10 per group
     = 20 groups).
  3. Train ONE baseline model on the full selected subset. Record its held-out
     loss on a fixed eval subset (`eval_subset_size` examples).
  4. For EACH group, retrain a fresh model on (selected subset - that group)
     and record held-out loss on the SAME eval subset. The loss increase
     (retrained-without-group minus baseline) is that group's *true* marginal
     contribution — the ground truth we don't have a cheap way to get in general.
  5. Correlate (Spearman) each group's true marginal contribution against the
     group's mean cheap signal (e.g. mean `learning_value` of its members).

COST WARNING: step 4 trains `n_examples_sampled / group_size` extra models
end-to-end. With the defaults (200/10 = 20 groups) that's 20 extra training
runs at your main budget's size — budget real time for this. If that's too
expensive, lower `n_examples_sampled` or raise `group_size` in
configs/default.yaml (fewer, larger groups = cheaper but coarser-grained
ground truth) rather than skipping the experiment.
"""
import random

import numpy as np
from scipy.stats import spearmanr

from .evaluate import evaluate_held_out
from .train_final import train_on_subset


def run_leave_k_out(
    config: dict,
    selected_ids: list,
    scores_df,
    train_hf_dataset,
    eval_hf_dataset,
    run_id: str = "exp5",
    log_path: str = "results/compute_log.json",
) -> dict:
    lko_cfg = config["leave_k_out"]
    seed = config["seed"]
    random.seed(seed)

    eval_subset = eval_hf_dataset.select(
        range(min(lko_cfg["eval_subset_size"], len(eval_hf_dataset)))
    )

    sample_ids = random.sample(selected_ids, min(lko_cfg["n_examples_sampled"], len(selected_ids)))
    groups = [
        sample_ids[i : i + lko_cfg["group_size"]]
        for i in range(0, len(sample_ids), lko_cfg["group_size"])
    ]

    # Step 3: baseline model on the full selected subset.
    baseline_model, baseline_tok = train_on_subset(
        config, selected_ids, train_hf_dataset, run_name=f"{run_id}_baseline", log_path=log_path
    )
    baseline_metrics = evaluate_held_out(
        baseline_model, baseline_tok, eval_subset, config["max_length"], run_id=f"{run_id}_baseline", split_name="validation", log_path=log_path
    )
    baseline_loss = baseline_metrics["validation_loss"]

    # Step 4: leave-group-out retraining.
    group_results = []
    for gi, group in enumerate(groups):
        remaining_ids = [i for i in selected_ids if i not in set(group)]
        model, tok = train_on_subset(
            config, remaining_ids, train_hf_dataset, run_name=f"{run_id}_group{gi}", log_path=log_path
        )
        metrics = evaluate_held_out(
            model,
            tok,
            eval_subset,
            config["max_length"],
            run_id=f"{run_id}_group{gi}",
            split_name="validation",
            log_path=log_path,
        )
        true_marginal_value = metrics["validation_loss"] - baseline_loss

        group_scores = scores_df[scores_df["example_id"].isin(group)]
        mean_cheap_signal = {
            "mean_static_loss": float(group_scores["static_loss"].mean()),
            "mean_initial_loss": float(group_scores["loss_initial"].mean()),
            "mean_loss_delta": float(group_scores["loss_delta"].mean()),
            "mean_relative_improvement": float(group_scores["relative_improvement"].mean()),
            "mean_dynamics_score": float(group_scores["dynamics_score"].mean()),
        }
        if "learning_value" in group_scores.columns:
            mean_cheap_signal["mean_learning_value"] = float(group_scores["learning_value"].mean())

        group_results.append(
            {
                "group_index": gi,
                "example_ids": group,
                "true_marginal_value": true_marginal_value,
                **mean_cheap_signal,
            }
        )

    # Step 5: correlate true marginal value against each cheap signal.
    correlations = {}
    signal_keys = [k for k in group_results[0].keys() if k.startswith("mean_")]
    true_values = [g["true_marginal_value"] for g in group_results]
    for key in signal_keys:
        signal_values = [g[key] for g in group_results]
        rho, pval = spearmanr(true_values, signal_values)
        correlations[key] = {"spearman_rho": rho, "p_value": pval}

    return {
        "baseline_held_out_loss": baseline_loss,
        "n_groups": len(groups),
        "group_size": lko_cfg["group_size"],
        "group_results": group_results,
        "correlations_with_true_marginal_value": correlations,
    }
