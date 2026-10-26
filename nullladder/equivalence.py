"""TOST (two one-sided tests) equivalence testing.

Used for: knockout performance equivalent to R3 within a pre-registered
margin, and hollow-brain R5 equivalent to hollow-brain R4 within a margin —
both across seeds. Equivalence (not just "no significant difference") is
the right test here: failing to reject a null of "different" is not
evidence of "same."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TOSTResult:
    mean_diff: float
    margin: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool


def tost_equivalence(
    x: np.ndarray,
    y: np.ndarray,
    margin: float,
    alpha: float = 0.05,
    paired: bool = True,
) -> TOSTResult:
    """Test whether mean(x) - mean(y) lies within (-margin, +margin).

    x, y: per-seed metric values for the two arms being compared (e.g.
    knockout's zero-shot gain per seed vs R3's zero-shot gain per seed).
    paired=True assumes x[i], y[i] share seed i (recommended: it's strictly
    more powerful and seeds are shared by construction in this pipeline).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if paired:
        diff = x - y
        mean_diff = float(diff.mean())
        n = len(diff)
        se = diff.std(ddof=1) / np.sqrt(n)
        df = n - 1
        t_lower = (mean_diff - (-margin)) / se
        t_upper = (mean_diff - margin) / se
        p_lower = float(stats.t.sf(t_lower, df))  # H0: diff <= -margin
        p_upper = float(stats.t.cdf(t_upper, df))  # H0: diff >= margin
    else:
        mean_diff = float(x.mean() - y.mean())
        nx, ny = len(x), len(y)
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        se = float(np.sqrt(vx / nx + vy / ny))
        df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
        t_lower = (mean_diff - (-margin)) / se
        t_upper = (mean_diff - margin) / se
        p_lower = float(stats.t.sf(t_lower, df))
        p_upper = float(stats.t.cdf(t_upper, df))

    p_tost = max(p_lower, p_upper)
    return TOSTResult(
        mean_diff=mean_diff,
        margin=margin,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=p_tost < alpha,
    )
