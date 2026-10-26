"""Paired Cohen's d for the verdict rule's first clause: R5 must beat every
rung, with R5 vs R4 (the hardest comparison — both are real data) checked
against a pre-registered minimum effect size across seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EffectSizeResult:
    mean_diff: float
    cohens_d: float
    n: int
    passed: bool


def paired_cohens_d(x: np.ndarray, y: np.ndarray, min_effect_size: float) -> EffectSizeResult:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return EffectSizeResult(mean_diff=float(diff.mean()), cohens_d=d, n=len(diff), passed=d > min_effect_size)


def rung_vs_rung_effect_size(
    results: pd.DataFrame,
    brain: str,
    alpha: float,
    rung_a: str,
    rung_b: str,
    min_effect_size: float,
    variant: str | None = None,
    metric_col: str = "value",
) -> EffectSizeResult:
    """Effect size of rung_a - rung_b (e.g. R5 - R4), paired across seeds."""
    subset = results[(results["brain"] == brain) & (results["alpha"] == alpha)]
    if variant is not None:
        subset = subset[subset["variant"] == variant]
    per_seed = subset.groupby(["rung", "seed"])[metric_col].mean().reset_index()
    a = per_seed[per_seed["rung"] == rung_a].sort_values("seed")
    b = per_seed[per_seed["rung"] == rung_b].sort_values("seed")
    common_seeds = sorted(set(a["seed"]) & set(b["seed"]))
    a_vals = a.set_index("seed").loc[common_seeds, metric_col].to_numpy()
    b_vals = b.set_index("seed").loc[common_seeds, metric_col].to_numpy()
    return paired_cohens_d(a_vals, b_vals, min_effect_size)
