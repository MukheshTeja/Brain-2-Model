"""Ladder decomposition: attribute zero-shot gain to each rung-to-rung step.

R3->R4 is explicitly NOT a single clean ingredient (it mixes "real vs
synthetic" with "correct vs wrong pairing" in one step) and is labeled as
such rather than reported alongside the other, single-ingredient steps.
"""
from __future__ import annotations

import pandas as pd

RUNG_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5"]

STEP_LABELS = {
    ("R0", "R1"): "+ per-channel power spectrum",
    ("R1", "R2"): "+ zero-lag cross-channel covariance",
    ("R2", "R3"): "+ full cross-spectral density (lagged cross-covariance)",
    ("R3", "R4"): "real data, wrong pairing (NOT a single clean ingredient: "
                  "conflates 'real vs synthetic' and 'correct vs wrong pairing')",
    ("R4", "R5"): "+ correct trial pairing",
}


def ladder_decomposition(results: pd.DataFrame, brain: str, alpha: float, metric_col: str = "value") -> pd.DataFrame:
    """results: tidy dataframe from training/sweep.py. Returns one row per
    rung-to-rung step with the mean (across seeds, variants) gain and its
    label, for a single (brain, alpha) slice."""
    subset = results[(results["brain"] == brain) & (results["alpha"] == alpha)]
    per_rung = subset.groupby("rung")[metric_col].mean()

    rows = []
    for a, b in zip(RUNG_ORDER[:-1], RUNG_ORDER[1:]):
        if a not in per_rung.index or b not in per_rung.index:
            continue
        rows.append({
            "step": f"{a}->{b}",
            "label": STEP_LABELS[(a, b)],
            "mean_metric_a": per_rung[a],
            "mean_metric_b": per_rung[b],
            "step_gain": per_rung[b] - per_rung[a],
        })
    return pd.DataFrame(rows)
