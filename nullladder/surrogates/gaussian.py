"""R0: Gaussian noise matched to each channel's unconditional mean and variance.

No spectral or cross-channel structure is preserved — this is the weakest
rung and the control B2M's original paper actually tested against.
"""
from __future__ import annotations

import numpy as np

from .base import Surrogate, register


@register
class GaussianSurrogate(Surrogate):
    name = "R0"

    def fit(self, real_signal: np.ndarray) -> "GaussianSurrogate":
        self.mean_ = real_signal.mean(axis=0)
        self.std_ = real_signal.std(axis=0)
        self.shape_ = real_signal.shape
        self._fitted = True
        return self

    def generate(self, seed: int) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        return self.mean_ + self.std_ * rng.standard_normal(self.shape_)

    def claims(self) -> dict[str, bool]:
        return {
            "mean_var": True,
            "per_channel_spectrum": False,
            "zero_lag_cov": False,
            "lagged_cross_cov": False,
        }
