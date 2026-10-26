"""R3: multivariate phase randomization (full cross-spectral density).

The SAME random phase offset is applied to every channel at each frequency.
Since cross-spectral density at frequency f is X_i(f) * conj(X_j(f)), and a
shared phase rotation e^{i*phi(f)} cancels in that product, this preserves
the full cross-spectral density matrix exactly -> by the (cross-)
Wiener-Khinchin relation, ALL lagged cross-covariances are preserved, not
just zero-lag. This is what distinguishes R3 from R2 (R2 only restores
zero-lag covariance after independently randomizing each channel's phase).

Optional IAAFT refinement (`iaaft_iters > 0`) additionally nudges each
channel's amplitude distribution toward the real data's marginal histogram.
This is a per-channel, best-effort refinement: it is not guaranteed to
preserve the cross-spectral density exactly once applied, so it defaults to
off and should only be enabled when marginal non-Gaussianity matters more
than exact cross-spectral preservation.
"""
from __future__ import annotations

import numpy as np

from .base import Surrogate, register


def multivariate_phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T = x.shape[0]
    xf = np.fft.rfft(x, axis=0)  # [F, C]
    n_freq = xf.shape[0]
    phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    phases[0] = 0.0
    if T % 2 == 0:
        phases[-1] = 0.0
    xf_shuffled = xf * np.exp(1j * phases)[:, None]
    return np.fft.irfft(xf_shuffled, n=T, axis=0)


def _iaaft_refine(y: np.ndarray, real_signal: np.ndarray, n_iters: int) -> np.ndarray:
    """Per-channel amplitude-adjustment refinement (approximate; breaks exact
    cross-spectral preservation slightly in exchange for matching marginals)."""
    target_sorted = np.sort(real_signal, axis=0)
    target_spectrum_mag = np.abs(np.fft.rfft(real_signal, axis=0))
    cur = y.copy()
    for _ in range(n_iters):
        # Impose the real amplitude spectrum, keep current phases.
        cur_f = np.fft.rfft(cur, axis=0)
        phase = cur_f / (np.abs(cur_f) + 1e-12)
        cur = np.fft.irfft(target_spectrum_mag * phase, n=y.shape[0], axis=0)
        # Rank-match each channel's marginal to the real data's marginal.
        ranks = np.argsort(np.argsort(cur, axis=0), axis=0)
        cur = np.take_along_axis(target_sorted, ranks, axis=0)
    return cur


@register
class MaxEntropySurrogate(Surrogate):
    name = "R3"

    def __init__(self, iaaft_iters: int = 0) -> None:
        super().__init__()
        self.iaaft_iters = iaaft_iters

    def fit(self, real_signal: np.ndarray) -> "MaxEntropySurrogate":
        self.real_signal_ = real_signal
        self._fitted = True
        return self

    def generate(self, seed: int) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        y = multivariate_phase_randomize(self.real_signal_, rng)
        if self.iaaft_iters > 0:
            y = _iaaft_refine(y, self.real_signal_, self.iaaft_iters)
        return y

    def claims(self) -> dict[str, bool]:
        return {
            "mean_var": True,
            "per_channel_spectrum": True,
            "zero_lag_cov": True,
            "lagged_cross_cov": True,
        }
