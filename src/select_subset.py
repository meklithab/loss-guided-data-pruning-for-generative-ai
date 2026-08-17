"""
Reads the raw score files produced by scoring.py and materializes the
actual training-data subsets for every (method, data_fraction) cell.

CORE grid (the research question): random / loss_high / loss_low / loss_delta
  all at the SAME data fractions, so every comparison is a matched-budget
  comparison ("at 30% of the data, does loss-guided beat random?").

EXTENDED methods (perplexity / diversity / hybrid) are built separately by
build_extended_grid(), only if there's time left after the core results.

Usage:
    python src/select_subset.py --config configs/config.yaml --build_core
    python src/select_subset.py --config configs/config.yaml --build_extended
    python src/select_subset.py --config configs/config.yaml --build_ablations
"""
import argparse
import json
import os
import random
import numpy as np
from utils import load_config, set_seed
from data_utils import load_jsonl


def _load_scores(scores_dir: str, method: str) -> dict:
    path = os.path.join(scores_dir, f"{method}.jsonl")
    rows = load_jsonl(path)
    return {r["idx"]: r for r in rows}


def _min_max_norm(values: dict) -> dict:
    vals = np.array(list(values.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def select_random(pool, fraction, seed) -> list:
    rng = random.Random(seed)
    n = int(len(pool) * fraction)
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    return sorted(idxs[:n])


def select_by_score(pool, scores_dir, score_key, method_file, fraction,
                     ascending=False, keep_low=0.0, keep_high=None) -> list:
    """
    Generic top-K / banded selector.
      ascending=False (default): keep the HIGHEST-scoring examples (e.g. loss_high, loss_delta)
      ascending=True: keep the LOWEST-scoring examples (loss_low -- the inverse test)
    keep_low/keep_high let the threshold ablation pick a percentile BAND
    instead of always the single highest/lowest tail.
    """
    scores = _load_scores(scores_dir, method_file)
    ranked = sorted(scores.items(), key=lambda kv: kv[1][score_key], reverse=not ascending)
    ranked_idxs = [idx for idx, _ in ranked]

    n_total = len(ranked_idxs)
    if keep_high is None:
        n = int(n_total * fraction)
        return sorted(ranked_idxs[:n])
    else:
        lo = int(n_total * keep_low)
        hi = int(n_total * keep_high)
        band = ranked_idxs[lo:hi]
        n = int(n_total * fraction)
        return sorted(band[:n])


def select_diversity(pool, scores_dir, fraction, n_clusters, seed) -> list:
    """[EXTENDED] Proportional sampling across clusters, preferring examples close to their centroid."""
    scores = _load_scores(scores_dir, "diversity")
    by_cluster = {}
    for idx, row in scores.items():
        by_cluster.setdefault(row["cluster_id"], []).append((idx, row["centroid_dist"]))

    rng = random.Random(seed)
    n_total = len(scores)
    n_target = int(n_total * fraction)
    per_cluster = max(1, n_target // max(1, len(by_cluster)))

    selected = []
    for cid, members in by_cluster.items():
        members.sort(key=lambda t: t[1])
        selected.extend([idx for idx, _ in members[:per_cluster]])

    rng.shuffle(selected)
    return sorted(selected[:n_target])


def select_composite(pool, scores_dir, fraction, weights: dict, n_clusters: int) -> list:
    """[EXTENDED] Weighted composite of loss_final + uncertainty + diversity, cluster-balanced."""
    loss = _load_scores(scores_dir, "loss")
    unc = _load_scores(scores_dir, "uncertainty")
    div = _load_scores(scores_dir, "diversity")

    loss_norm = _min_max_norm({i: r["loss_final"] for i, r in loss.items()})
    unc_norm = _min_max_norm({i: r["uncertainty_score"] for i, r in unc.items()})
    div_norm = _min_max_norm({i: r["centroid_dist"] for i, r in div.items()})

    composite = {i: (weights.get("loss", 0.0) * loss_norm[i]
                      + weights.get("uncertainty", 0.0) * unc_norm[i]
                      + weights.get("diversity", 0.0) * div_norm[i])
                 for i in loss_norm}

    by_cluster = {}
    for idx, row in div.items():
        by_cluster.setdefault(row["cluster_id"], []).append(idx)

    n_total = len(composite)
    n_target = int(n_total * fraction)
    per_cluster = max(1, n_target // max(1, len(by_cluster)))

    selected = []
    for cid, members in by_cluster.items():
        ranked = sorted(members, key=lambda i: composite[i], reverse=True)
        selected.extend(ranked[:per_cluster])

    if len(selected) < n_target:
        remaining = sorted((i for i in composite if i not in set(selected)),
                            key=lambda i: composite[i], reverse=True)
        selected.extend(remaining[: n_target - len(selected)])

    return sorted(selected[:n_target])


def write_subset(pool, idxs, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for i in idxs:
            f.write(json.dumps(pool[i]) + "\n")
    print(f"  -> {out_path}  (n={len(idxs)})")


def build_core(cfg):
    """random / loss_high / loss_low / loss_delta, all at matched data fractions."""
    set_seed(cfg["seed"])
    pool = load_jsonl(os.path.join(cfg["dataset"]["splits_dir"], "pool.jsonl"))
    scores_dir = os.path.join(cfg["dataset"]["splits_dir"], "..", "scores")
    subsets_dir = cfg["selection"]["subsets_dir"]

    for method in cfg["selection"]["methods"]:
        for frac in cfg["selection"]["data_fractions"]:
            if frac == 1.0:
                idxs = list(range(len(pool)))
            elif method == "random":
                idxs = select_random(pool, frac, cfg["seed"])
            elif method == "loss_high":
                idxs = select_by_score(pool, scores_dir, "loss_final", "loss", frac, ascending=False)
            elif method == "loss_low":
                idxs = select_by_score(pool, scores_dir, "loss_final", "loss", frac, ascending=True)
            elif method == "loss_delta":
                idxs = select_by_score(pool, scores_dir, "loss_delta", "loss", frac, ascending=False)
            else:
                raise ValueError(f"unknown core method {method}")

            tag = f"{method}_{int(frac * 100)}pct"
            write_subset(pool, idxs, os.path.join(subsets_dir, f"{tag}.jsonl"))


def build_extended(cfg):
    """[OPTIONAL] perplexity / diversity / hybrid, only if there's time left."""
    set_seed(cfg["seed"])
    pool = load_jsonl(os.path.join(cfg["dataset"]["splits_dir"], "pool.jsonl"))
    scores_dir = os.path.join(cfg["dataset"]["splits_dir"], "..", "scores")
    subsets_dir = cfg["selection"]["subsets_dir"]
    hybrid_weights = cfg["selection"]["hybrid_weights"]

    for method in cfg["selection"]["extended_methods"]:
        for frac in cfg["selection"]["data_fractions"]:
            if frac == 1.0:
                continue  # full-data baseline already built by build_core
            elif method == "perplexity":
                idxs = select_by_score(pool, scores_dir, "perplexity", "perplexity", frac)
            elif method == "diversity":
                idxs = select_diversity(pool, scores_dir, frac, cfg["diversity_scoring"]["n_clusters"], cfg["seed"])
            elif method == "hybrid":
                idxs = select_composite(pool, scores_dir, frac, hybrid_weights,
                                         cfg["diversity_scoring"]["n_clusters"])
            else:
                raise ValueError(f"unknown extended method {method}")

            tag = f"{method}_{int(frac * 100)}pct"
            write_subset(pool, idxs, os.path.join(subsets_dir, f"{tag}.jsonl"))


def build_ablations(cfg):
    """[EXTENDED] threshold-band and cluster-count ablations (only relevant if you run extended methods)."""
    set_seed(cfg["seed"])
    pool = load_jsonl(os.path.join(cfg["dataset"]["splits_dir"], "pool.jsonl"))
    scores_dir = os.path.join(cfg["dataset"]["splits_dir"], "..", "scores")
    subsets_dir = cfg["selection"]["subsets_dir"]
    fixed_frac = 0.30

    for band in cfg["ablations"]["threshold_bands"]:
        idxs = select_by_score(pool, scores_dir, "perplexity", "perplexity", fixed_frac,
                                keep_low=band["keep_low"], keep_high=band["keep_high"])
        tag = f"ablation_threshold_{band['name']}_30pct"
        write_subset(pool, idxs, os.path.join(subsets_dir, f"{tag}.jsonl"))

    for k in cfg["ablations"]["cluster_counts"]:
        idxs = select_diversity(pool, scores_dir, fixed_frac, k, cfg["seed"])
        tag = f"ablation_clusters_k{k}_30pct"
        write_subset(pool, idxs, os.path.join(subsets_dir, f"{tag}.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--build_core", action="store_true")
    parser.add_argument("--build_extended", action="store_true")
    parser.add_argument("--build_ablations", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.build_core:
        build_core(cfg)
    if args.build_extended:
        build_extended(cfg)
    if args.build_ablations:
        build_ablations(cfg)
