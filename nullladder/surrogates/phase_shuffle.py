"""R1: independent per-channel phase randomization.

Each channel's Fourier phases are randomized independently of every other
channel, with a fresh random draw per call. This exactly preserves each
channel's own (unconditional) power spectrum / autocorrelation, since phase
randomization leaves |X(f)| untouched. Because each channel gets an
INDEPENDENT random phase, it destroys all cross-channel structure — both
zero-lag covariance and lagged cross-covariance are driven to ~0 in
expectation. R2 is literally "R1 + recolor to restore zero-lag covariance",
so this class is reused directly by covariance_matched.py.
"""
from __future__ import annotations

import numpy as np

from .base import Surrogate, register


def independent_phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomize Fourier phase of each channel of x: [T, C] independently."""
    T = x.shape[0]
    xf = np.fft.rfft(x, axis=0)
    n_freq = xf.shape[0]
    # Keep DC (and Nyquist, if T even) real so the output stays real-valued;
    # randomize every other frequency bin's phase independently per channel.
    phases = rng.uniform(0, 2 * np.pi, size=xf.shape)
    phases[0, :] = 0.0
    if T % 2 == 0:
        phases[-1, :] = 0.0
    xf_shuffled = xf * np.exp(1j * phases)
    return np.fft.irfft(xf_shuffled, n=T, axis=0)


@register
class PhaseShuffleSurrogate(Surrogate):
    name = "R1"

    def fit(self, real_signal: np.ndarray) -> "PhaseShuffleSurrogate":
        self.real_signal_ = real_signal
        self._fitted = True
        return self

    def generate(self, seed: int) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        return independent_phase_randomize(self.real_signal_, rng)

    def claims(self) -> dict[str, bool]:
        return {
            "mean_var": True,
            "per_channel_spectrum": True,
            "zero_lag_cov": False,
            "lagged_cross_cov": False,
        }
