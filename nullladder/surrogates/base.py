"""Surrogate generator base class and registry.

Every rung in the Null Ladder fits on a real [T, C] signal and generates
new [T, C] samples. `claims()` declares which unconditional statistics the
rung preserves — the unit test for each rung asserts exactly those claims
and nothing more (e.g. R2 claims zero-lag covariance but NOT lagged
cross-covariance; R3 claims both).

All `fit()` implementations must use only unconditional (time-averaged)
statistics of the real signal. None of them may use trial/condition labels —
that would bake trial-locked (and therefore task-decodable) structure into a
surrogate that's supposed to be a null. Labels are only ever used downstream,
by probes, to check surrogates DON'T carry task structure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

_REGISTRY: dict[str, type["Surrogate"]] = {}


def register(cls: type["Surrogate"]) -> type["Surrogate"]:
    _REGISTRY[cls.name] = cls
    return cls


def get_surrogate(name: str) -> type["Surrogate"]:
    return _REGISTRY[name]


class Surrogate(ABC):
    name: ClassVar[str]

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, real_signal: np.ndarray) -> "Surrogate":
        """Fit to the real [T, C] signal using unconditional stats only."""

    @abstractmethod
    def generate(self, seed: int) -> np.ndarray:
        """Generate a [T, C] surrogate sample. Requires fit() first."""

    @abstractmethod
    def claims(self) -> dict[str, bool]:
        """Which statistics this rung preserves relative to the real signal.

        Keys: "mean_var", "per_channel_spectrum", "zero_lag_cov", "lagged_cross_cov".
        """

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before generate()")
