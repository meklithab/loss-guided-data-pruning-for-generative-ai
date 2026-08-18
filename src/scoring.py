"""
Phase 2 — turn per-example loss trajectories into scalar "value" signals.

Input: {example_id: [loss_ckpt1, loss_ckpt2, ..., loss_ckptN]} (see track_training.py)
Output: a pandas DataFrame, one row per example_id, with columns:

    static_loss           -- loss at the FIRST checkpoint (how hard is it, early on)
    loss_delta            -- static_loss - loss at LAST checkpoint (raw improvement)
    relative_improvement  -- loss_delta / static_loss (improvement relative to initial difficulty)
    slope                 -- linear-regression slope of loss vs. checkpoint index (learning speed)
    variance              -- variance of the trajectory (stability / noisiness)
    auc                   -- trapezoidal area under the loss curve (low = learned fast AND stayed low)

These map directly onto the five signal types in the proposal:
    1. Static loss          -> static_loss
    2. Loss improvement     -> loss_delta
    3. Relative improvement -> relative_improvement
    4. Learning dynamics    -> slope, variance, auc (full-trajectory features)
    5. Redundancy           -> handled separately in select.py (needs example TEXT, not just loss)
"""
import numpy as np
import pandas as pd

# numpy >=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _safe(values):
    """Replace any leftover Nones (shouldn't happen after backfill, but be defensive)
    with the first non-None value in the trajectory."""
    clean = [v for v in values if v is not None]
    fallback = clean[0] if clean else 0.0
    return [v if v is not None else fallback for v in values]


def compute_scores(trajectories: dict) -> pd.DataFrame:
    rows = []
    for eid, traj in trajectories.items():
        traj = _safe(traj)
        traj_arr = np.array(traj, dtype=float)
        x = np.arange(len(traj_arr))

        static_loss = float(traj_arr[0])
        final_loss = float(traj_arr[-1])
        loss_delta = static_loss - final_loss
        relative_improvement = loss_delta / static_loss if static_loss > 1e-8 else 0.0

        if len(traj_arr) >= 2 and np.std(x) > 0:
            slope = float(np.polyfit(x, traj_arr, 1)[0])
        else:
            slope = 0.0
        variance = float(np.var(traj_arr))
        auc = float(_trapz(traj_arr, x)) if len(traj_arr) >= 2 else static_loss

        rows.append(
            {
                "example_id": eid,
                "static_loss": static_loss,
                "final_loss": final_loss,
                "loss_delta": loss_delta,
                "relative_improvement": relative_improvement,
                "slope": slope,
                "variance": variance,
                "auc": auc,
            }
        )
    df = pd.DataFrame(rows).sort_values("example_id").reset_index(drop=True)
    return df
