"""R2: covariance-matched surrogate.

Construction (per spec): independent per-channel phase randomization (R1),
then re-impose the real signal's zero-lag cross-channel covariance via a
whiten/recolor linear transform:

  1. z = x_real @ C_real^{-1/2,T}          # zero-lag-decorrelate the real signal
  2. z' = independent_phase_randomize(z)    # destroy each decorrelated
                                             # component's cross-component
                                             # temporal relationships, keep
                                             # its own spectrum
  3. y  = z' @ C_real^{1/2,T}               # recolor to the real zero-lag
                                             # covariance

Caveat (documented, not hidden — and empirically larger than you'd hope):
step 3's contemporaneous mixing of autocorrelated components leaks lagged
cross-covariance back into y, in proportion to how non-diagonal C_real is
and how strongly autocorrelated its principal components are. For strongly
autocorrelated sources (e.g. AR-driven latents, as in this toolkit's own
sandbox brain), the leakage is NOT small — empirically R2 can retain ~70-75%
of R3's lagged cross-covariance norm. This surrogate's claim is therefore
comparative and modest, not exact-zero or even "much smaller": R2 preserves
MEASURABLY LESS lagged cross-covariance than R3 (which preserves it
exactly), typically a 15-30% reduction for this toolkit's sandbox data, not
a destruction of it. Anyone using R2 as a rung should read its lagged
cross-covariance norm off the actual data rather than assume it behaves
like a clean "zero-lag-only" control.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from .base import Surrogate, register
from .phase_shuffle import independent_phase_randomize


@register
class CovarianceMatchedSurrogate(Surrogate):
    name = "R2"

    def fit(self, real_signal: np.ndarray) -> "CovarianceMatchedSurrogate":
        self.mean_ = real_signal.mean(axis=0)
        xc = real_signal - self.mean_
        cov = (xc.T @ xc) / xc.shape[0]
        cov = cov + 1e-10 * np.eye(cov.shape[0])  # numerical floor
        self.cov_sqrt_ = np.real(sqrtm(cov))
        self.cov_inv_sqrt_ = np.linalg.inv(self.cov_sqrt_)
        self.z_ = xc @ self.cov_inv_sqrt_.T
        self._fitted = True
        return self

    def generate(self, seed: int) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        z_prime = independent_phase_randomize(self.z_, rng)
        return z_prime @ self.cov_sqrt_.T + self.mean_

    def claims(self) -> dict[str, bool]:
        return {
            "mean_var": True,
            "per_channel_spectrum": True,
            "zero_lag_cov": True,
            "lagged_cross_cov": False,
        }
