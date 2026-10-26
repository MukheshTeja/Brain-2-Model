"""Structure knockout: remove task-decodable linear directions from real
data (R5), then rescale so total variance matches the original — so a drop
in downstream transfer can't be explained by the knockout simply having
made the signal weaker rather than less structured.

The rescale is a SINGLE GLOBAL SCALAR (same factor for every channel and
every timestep), not a per-channel/per-frequency filter. This is a
deliberate, load-bearing choice: multiplying the erased signal by one
scalar c preserves erasure exactly, since Cov(c*X, Z) = c*Cov(X, Z) = c*0 = 0
for any c. An earlier version of this module used an independent
per-channel Fourier-magnitude filter to match the *detailed* per-channel
spectrum shape — empirically, that filter reintroduced a substantial amount
of the just-erased label-predictive structure (verified via the decodability
unit test: linear-probe accuracy jumped from below chance right after
erasure to well above chance after that filter), because a frequency-domain
filter mixes information ACROSS TIME and can reintroduce correlation with
block-structured labels even when applied identically to every channel. A
pointwise scalar has no such mixing, so it cannot undo the erasure. The
trade-off is that only TOTAL variance is matched exactly, not the detailed
per-channel/per-frequency spectral shape — that is the right trade-off here,
since reintroducing structure would invalidate the whole comparator.

Default eraser is LEACE (Belrose et al. 2023): the minimum-distortion linear
map that drives linear correlation between signal and concept labels to
exactly zero. `eraser="inlp"` (iterative nullspace projection) is accepted
as an interface point for a future swap but not implemented in milestone 1.

Unlike the R0-R4 surrogates, knockout is explicitly allowed to use labels —
removing label-predictive structure is the entire point. This does not
conflict with the "unconditional stats only" rule in surrogates/base.py,
which governs how NULLS are synthesized, not how a concept is erased from
real data.
"""
from __future__ import annotations

import numpy as np


def _sym_sqrt_and_inv_sqrt(cov: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, eps, None)
    sqrt_vals = np.sqrt(eigvals)
    cov_sqrt = eigvecs @ np.diag(sqrt_vals) @ eigvecs.T
    cov_inv_sqrt = eigvecs @ np.diag(1.0 / sqrt_vals) @ eigvecs.T
    return cov_sqrt, cov_inv_sqrt


def _one_hot(labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    return (labels[:, None] == classes[None, :]).astype(float)


def _leace_transform(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, A) such that x_erased = mean + (x - mean) @ A.T removes
    all linear predictability of z from x with minimum expected distortion."""
    mean_x = x.mean(axis=0)
    xc = x - mean_x
    zc = z - z.mean(axis=0)
    cov_xx = (xc.T @ xc) / x.shape[0]
    cov_xz = (xc.T @ zc) / x.shape[0]
    cov_sqrt, cov_inv_sqrt = _sym_sqrt_and_inv_sqrt(cov_xx)

    m = cov_inv_sqrt @ cov_xz  # whitened-space cross term, [C, k]
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    rank = int((s > 1e-8 * (s.max() if s.size else 1.0)).sum())
    u_r = u[:, :rank]
    proj = np.eye(x.shape[1]) - u_r @ u_r.T  # kills the predictive subspace, in whitened space
    a = cov_sqrt @ proj @ cov_inv_sqrt
    return mean_x, a


def rescale_total_variance(erased: np.ndarray, original: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Single global scalar so erased's TOTAL variance matches original's.
    Pointwise (no cross-time mixing) -> provably cannot reintroduce the
    just-erased label-predictive structure. See module docstring."""
    original_total_var = float(((original - mean) ** 2).sum())
    erased_total_var = float(((erased - mean) ** 2).sum())
    scale = np.sqrt(original_total_var / (erased_total_var + 1e-12))
    return mean + (erased - mean) * scale


class StructureKnockout:
    def __init__(self, eraser: str = "leace") -> None:
        if eraser != "leace":
            raise NotImplementedError(f"eraser={eraser!r} not implemented in milestone 1; use 'leace'")
        self.eraser = eraser
        self._fitted = False

    def fit(self, signal: np.ndarray, task_labels: np.ndarray) -> "StructureKnockout":
        z = task_labels if task_labels.ndim == 2 else _one_hot(task_labels)
        self.mean_, self.transform_ = _leace_transform(signal, z)
        self._fitted = True
        return self

    def apply(self, signal: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit() before apply()")
        erased = self.mean_ + (signal - self.mean_) @ self.transform_.T
        return rescale_total_variance(erased, signal, self.mean_)
