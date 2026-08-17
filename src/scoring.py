"""
Computes per-example data-value signals for the train pool.

CORE signal -- loss dynamics (the actual research question):
  A short (2-epoch) LoRA warm-up pass logs each example's loss at BOTH
  warm-up epochs. From that we derive:
    - loss_final: loss at the end of warm-up (EL2N-style snapshot)
    - loss_delta: loss_epoch1 - loss_epoch2 (how much the model improved
      on this example between the two passes -- a proxy for "the model is
      actively learning from this example" rather than "this example
      just happens to be hard/noisy")
  select_subset.py builds THREE methods from this single scoring pass:
    loss_high (top loss_final), loss_low (bottom loss_final, the inverse
    -- tests whether "high loss" is even the right signal), and
    loss_delta (top loss_delta -- the learning-dynamics signal).

EXTENDED / optional signals (only needed if you run the extended methods):
  - perplexity: inference-only pass with the frozen base model
  - uncertainty: predictive entropy, computed for free during the same
    warm-up pass as the loss signals
  - diversity: sentence-embedding + k-means cluster id
"""
import argparse
import json
import os
from utils import load_config, set_seed
from data_utils import load_jsonl


def score_perplexity(pool, cfg) -> list:
    """[EXTENDED] Inference-only perplexity of each example under the frozen base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    scores = []
    with torch.no_grad():
        for i, ex in enumerate(pool):
            text = ex["prompt"] + ex["response"]
            ids = tok(text, truncation=True, max_length=cfg["model"]["max_seq_len"],
                       return_tensors="pt").to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            ppl = float(torch.exp(out.loss).item())
            scores.append({"idx": i, "perplexity": ppl})
    return scores


def score_loss_dynamics_and_uncertainty(pool, cfg) -> tuple:
    """
    CORE scoring pass. Runs `loss_scoring.warmup_epochs` (default 2) short
    LoRA training epochs over the full pool, logging per-example loss at
    EACH epoch (not just a single snapshot), plus predictive entropy
    (uncertainty, [EXTENDED]) computed for free in the same pass.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
    ))
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    n_epochs = cfg["loss_scoring"]["warmup_epochs"]
    per_example_losses = [[] for _ in pool]     # loss at each warmup epoch, per example
    uncertainty_scores = [None] * len(pool)      # from the LAST warmup epoch only

    for epoch in range(n_epochs):
        for i, ex in enumerate(pool):
            text = ex["prompt"] + ex["response"]
            ids = tok(text, truncation=True, max_length=cfg["model"]["max_seq_len"],
                       return_tensors="pt").to(model.device)
            out = model(**ids, labels=ids["input_ids"])
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            per_example_losses[i].append(float(loss.item()))

            if epoch == n_epochs - 1:
                with torch.no_grad():
                    probs = F.softmax(out.logits, dim=-1)
                    token_entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)
                    uncertainty_scores[i] = {"idx": i, "uncertainty_score": float(token_entropy.mean().item())}

    loss_rows = []
    for i, losses in enumerate(per_example_losses):
        loss_final = losses[-1]
        loss_delta = losses[0] - losses[-1] if len(losses) > 1 else 0.0
        loss_rows.append({
            "idx": i,
            "loss_epochs": losses,
            "loss_final": loss_final,
            "loss_delta": loss_delta,
        })

    return loss_rows, uncertainty_scores


def score_diversity(pool, cfg) -> list:
    """[EXTENDED] Embed every example and cluster with k-means; each example gets a cluster id."""
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    embedder = SentenceTransformer(cfg["diversity_scoring"]["embedding_model"])
    texts = [ex["prompt"] + ex["response"] for ex in pool]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    n_clusters = cfg["diversity_scoring"]["n_clusters"]
    km = KMeans(n_clusters=n_clusters, random_state=cfg["seed"], n_init="auto")
    cluster_ids = km.fit_predict(embeddings)
    dists = km.transform(embeddings)
    centroid_dist = [float(dists[i, c]) for i, c in enumerate(cluster_ids)]

    return [{"idx": i, "cluster_id": int(c), "centroid_dist": d}
            for i, (c, d) in enumerate(zip(cluster_ids, centroid_dist))]


def run(cfg: dict, method: str):
    set_seed(cfg["seed"])
    pool = load_jsonl(os.path.join(cfg["dataset"]["splits_dir"], "pool.jsonl"))
    out_dir = os.path.join(cfg["dataset"]["splits_dir"], "..", "scores")
    os.makedirs(out_dir, exist_ok=True)

    def write(name, rows):
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  -> wrote {path}")

    if method in ("all", "core", "loss"):
        print(f"[CORE] Scoring pool with loss-dynamics pass "
              f"({cfg['loss_scoring']['warmup_epochs']} warmup epochs, n={len(pool)}) ...")
        loss_rows, uncertainty_rows = score_loss_dynamics_and_uncertainty(pool, cfg)
        write("loss", loss_rows)
        write("uncertainty", uncertainty_rows)   # extended signal, computed for free

    if method in ("all", "extended", "perplexity"):
        print(f"[EXTENDED] Scoring pool with method=perplexity (n={len(pool)}) ...")
        write("perplexity", score_perplexity(pool, cfg))

    if method in ("all", "extended", "diversity"):
        print(f"[EXTENDED] Scoring pool with method=diversity (n={len(pool)}) ...")
        write("diversity", score_diversity(pool, cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="core",
                         choices=["all", "core", "extended", "loss", "perplexity", "diversity"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.method)
