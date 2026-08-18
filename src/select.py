"""
Phase 3 — turn scores into a concrete subset of example_ids for a given
(strategy, budget) pair.

Strategies (matching the experimental-design ladder in the proposal):
    random               -- uniform random subset (the baseline everything must beat)
    low_loss              -- keep the examples the model already finds EASIEST
                              (tests whether "easy" examples are actually low-value)
    high_loss              -- keep the examples the model finds HARDEST
                              (tests whether "hard" examples are noisy/harmful vs. valuable)
    loss_delta              -- keep examples with the LARGEST raw improvement (learned the most)
    dynamics                -- keep examples with the best combination of relative_improvement,
                                 slope, and low variance (a z-scored composite "learning value")
    dynamics_diversity       -- like `dynamics`, but greedily downweights examples that are
                                 near-duplicates (by TF-IDF cosine similarity) of already-selected
                                 examples, implementing "Learning Value - Redundancy"

`low_loss` and `high_loss` only need ONE checkpoint (static_loss) and are the
cheapest strategies to compute; `dynamics` and `dynamics_diversity` need the
full trajectory and (for diversity) the example text — that cost difference
is exactly what Experiment 2 is meant to surface.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std > 1e-8 else s * 0.0


def _n_keep(n_total: int, budget_frac: float) -> int:
    return max(1, int(round(n_total * budget_frac)))


def select_examples(
    scores_df: pd.DataFrame,
    strategy: str,
    budget_frac: float,
    seed: int = 42,
    example_texts: dict = None,  # {example_id: raw text} — required only for dynamics_diversity
) -> list:
    n_keep = _n_keep(len(scores_df), budget_frac)
    df = scores_df.copy()

    if strategy == "random":
        return df.sample(n=n_keep, random_state=seed)["example_id"].tolist()

    if strategy == "low_loss":
        return df.sort_values("static_loss", ascending=True).head(n_keep)["example_id"].tolist()

    if strategy == "high_loss":
        return df.sort_values("static_loss", ascending=False).head(n_keep)["example_id"].tolist()

    if strategy == "loss_delta":
        return df.sort_values("loss_delta", ascending=False).head(n_keep)["example_id"].tolist()

    if strategy in ("dynamics", "dynamics_diversity"):
        # Composite "learning value": high relative improvement, steep negative
        # slope (fast learning), and low variance (stable, not noisy) are all
        # considered valuable. Equal weighting is the sane default; sweeping
        # these weights is a natural extension if time allows.
        value = (
            _zscore(df["relative_improvement"])
            + _zscore(-df["slope"])
            - _zscore(df["variance"])
        )
        df["learning_value"] = value

        if strategy == "dynamics":
            return df.sort_values("learning_value", ascending=False).head(n_keep)["example_id"].tolist()

        # dynamics_diversity: greedy selection that penalizes redundancy.
        assert example_texts is not None, "dynamics_diversity requires example_texts={id: text}"
        return _greedy_diverse_selection(df, example_texts, n_keep)

    raise ValueError(f"Unknown strategy: {strategy}")


def _greedy_diverse_selection(df: pd.DataFrame, example_texts: dict, n_keep: int) -> list:
    """Greedy facility-location-style selection: repeatedly pick the candidate
    with the highest (learning_value - redundancy_penalty), where the penalty
    is that candidate's max TF-IDF cosine similarity to anything already picked.
    This is O(n_keep * n_candidates) — fine at the dataset sizes here (tens of
    thousands of examples); switch to approximate nearest-neighbor pruning of
    the candidate pool first if scaling to much larger datasets."""
    ids = df["example_id"].tolist()
    texts = [example_texts[i] for i in ids]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)  # (n, vocab)

    values = df.set_index("example_id")["learning_value"]
    remaining = set(ids)
    id_to_row = {eid: i for i, eid in enumerate(ids)}

    # Start with the single highest-value example.
    selected = [values.idxmax()]
    remaining.discard(selected[0])

    # Track max similarity of each remaining candidate to the selected set so far.
    selected_matrix = tfidf[[id_to_row[selected[0]]]]
    max_sim = np.asarray(cosine_similarity(tfidf, selected_matrix)).max(axis=1)
    max_sim_series = pd.Series(max_sim, index=ids)

    while len(selected) < n_keep and remaining:
        candidates = list(remaining)
        redundancy_penalty = _zscore(max_sim_series.loc[candidates])
        score = values.loc[candidates] - redundancy_penalty
        pick = score.idxmax()
        selected.append(pick)
        remaining.discard(pick)

        pick_vec = tfidf[[id_to_row[pick]]]
        new_sim = np.asarray(cosine_similarity(tfidf, pick_vec)).flatten()
        max_sim_series = pd.Series(np.maximum(max_sim_series.values, new_sim), index=ids)

    return selected
