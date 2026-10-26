"""Oracle knockout: a sandbox-only diagnostic rung ("KO_oracle") that uses
the KNOWN ground-truth mixing matrix to remove ALL information about the
teacher's task variables exactly, by geometric projection — no label
estimation, no statistical fit, nothing that can be incomplete. Only
possible here because the sandbox knows its own generative process; real
fMRI data has no such ground truth, so this rung never ships to Phase B.

Purpose: isolate "is the LEACE-based knockout's erasure incomplete" from
"is the knockout_vs_r3_margin threshold too tight." If KO_oracle ~ R3 but
LEACE-knockout does not, the LEACE fit (its label choice, its phase
pooling, etc.) is the problem, not the margin.

Only defined for LINEAR mixing: structured.signal = Z @ W + noise, with
Z: [T, n_latent], W: [n_latent, n_channels] (brain.py's `Mixing.w1`). The
information about Z lives entirely in the ROW SPACE of W (equivalently,
the column space of W.T) — projecting every timestep onto the orthogonal
complement of that subspace removes it exactly, regardless of any
particular discretization of Z into labels.
"""
from __future__ import annotations

import numpy as np

from ..surrogates.knockout import rescale_total_variance
from .brain import Mixing


def oracle_knockout(signal: np.ndarray, mixing: Mixing) -> np.ndarray:
    if mixing.mixing_type != "linear":
        raise NotImplementedError(
            "oracle_knockout is only defined for linear mixing: nonlinear mixing has no "
            "exact linear subspace to project out."
        )
    w = mixing.w1  # [n_latent, n_channels]
    # Orthonormal basis of the row space of w (= column space of w.T).
    u, _, _ = np.linalg.svd(w.T, full_matrices=False)  # u: [n_channels, rank(w)]
    mean = signal.mean(axis=0)
    centered = signal - mean
    projected = centered - centered @ u @ u.T  # remove the col(w) subspace exactly
    erased = mean + projected
    return rescale_total_variance(erased, signal, mean)
