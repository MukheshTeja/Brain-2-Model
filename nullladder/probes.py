"""Decodability probes used to verify surrogates destroy task structure and
knockout removes it. Every decodability check in this toolkit reports BOTH a
linear and an MLP probe score, since a linear probe passing is not sufficient
evidence against nonlinearly-coded task structure surviving in a surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor


@dataclass
class ProbeResult:
    linear_acc: float
    mlp_acc: float
    chance_acc: float

    @property
    def at_chance(self) -> bool:
        """True if both probes are within a small margin of chance."""
        margin = 0.07
        return (self.linear_acc - self.chance_acc) < margin and (self.mlp_acc - self.chance_acc) < margin


def _cv_accuracy(clf, x: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return float(np.max(np.bincount(y)) / len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv)
    return float(scores.mean())


def decodability(x: np.ndarray, labels: np.ndarray, seed: int = 0) -> ProbeResult:
    """Cross-validated linear + MLP decodability of `labels` from signal `x`.

    x: [N, C] feature matrix (e.g. per-trial-averaged or flattened channel signal).
    labels: [N] integer class labels.
    """
    chance = float(np.max(np.bincount(labels)) / len(labels))
    # C=0.1 (vs sklearn's default 1.0): an unregularized linear probe measurably
    # overfits null/chance-level data at the sample sizes this toolkit's tests
    # use (verified empirically — see tests/nullladder/conftest.py), inflating
    # false-positive "decodability" on surrogates that carry no real structure.
    linear = LogisticRegression(max_iter=2000, C=0.1)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, early_stopping=True, random_state=seed)
    linear_acc = _cv_accuracy(linear, x, labels, seed=seed)
    mlp_acc = _cv_accuracy(mlp, x, labels, seed=seed)
    return ProbeResult(linear_acc=linear_acc, mlp_acc=mlp_acc, chance_acc=chance)


@dataclass
class ContinuousProbeResult:
    linear_r2: float
    mlp_r2: float

    @property
    def at_chance(self) -> bool:
        """True if neither probe explains meaningfully more variance than a
        constant predictor (R^2 = 0 is chance for regression; CV R^2 can go
        slightly negative even at chance, so the margin is one-sided above 0)."""
        margin = 0.05
        return self.linear_r2 < margin and self.mlp_r2 < margin


def _cv_r2(reg, x: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    n_splits = min(n_splits, len(y))
    if n_splits < 2:
        return 0.0
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(reg, x, y, cv=cv, scoring="r2")
    return float(scores.mean())


def continuous_decodability(x: np.ndarray, y: np.ndarray, seed: int = 0) -> ContinuousProbeResult:
    """Cross-validated linear (Ridge) + MLP decodability of a CONTINUOUS
    target `y` from signal `x` — for continuous teacher latents, where
    `decodability()`'s classification probes don't apply. Chance R^2 is 0
    (a constant predictor). Regularized for the same reason `decodability`
    is: unregularized probes measurably overfit null data at small sample
    sizes.

    x: [N, C] feature matrix. y: [N] continuous target (one latent at a time).
    """
    linear = Ridge(alpha=10.0)
    mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, early_stopping=True,
                        alpha=1.0, random_state=seed)
    linear_r2 = _cv_r2(linear, x, y, seed=seed)
    mlp_r2 = _cv_r2(mlp, x, y, seed=seed)
    return ContinuousProbeResult(linear_r2=linear_r2, mlp_r2=mlp_r2)
