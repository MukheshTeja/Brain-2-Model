"""R4: real recordings, wrong trial pairing.

Unlike R0-R3, R4 is not synthesized — it IS the real signal, trivially
matching every spectral/covariance statistic exactly. The only thing R4
destroys is the correspondence between a trial's brain data and that
trial's task condition: trial segments are permuted (derangement: no trial
keeps its own original slot) before being paired against task labels
downstream. `claims()` therefore adds `correct_trial_pairing: False` on top
of the four stats keys (all True, trivially).

fit() expects already-segmented trials, real_trials: [n_trials, trial_len, C].
generate() returns the same shape with trial order permuted; the caller is
responsible for pairing permuted trial i against task trial i's labels.
"""
from __future__ import annotations

import numpy as np

from .base import Surrogate, register


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


@register
class StimulusShuffleSurrogate(Surrogate):
    name = "R4"

    def fit(self, real_trials: np.ndarray) -> "StimulusShuffleSurrogate":
        if real_trials.ndim != 3:
            raise ValueError("R4 requires trial-segmented input [n_trials, trial_len, C]")
        self.real_trials_ = real_trials
        self._fitted = True
        return self

    def generate(self, seed: int) -> np.ndarray:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        perm = _derangement(self.real_trials_.shape[0], rng)
        return self.real_trials_[perm]

    def claims(self) -> dict[str, bool]:
        return {
            "mean_var": True,
            "per_channel_spectrum": True,
            "zero_lag_cov": True,
            "lagged_cross_cov": True,
            "correct_trial_pairing": False,
        }
