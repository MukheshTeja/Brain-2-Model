"""Structured and hollow simulated brains.

Structured brain: the teacher's task variables, mixed into `n_neurons`
simulated neurons via a (by default linear) random mixing, plus
temporally-correlated, heterogeneous-amplitude noise per neuron.

Hollow brain: must match the structured brain's FULL signal statistics
(per-neuron spectra, zero-lag covariance, dimensionality) without carrying
any task information. It is built by (1) fitting an AR(p) model to each
task-variable channel to capture its autocorrelation, (2) drawing an
INDEPENDENT realization of that AR(p) process (same spectral shape, no
shared randomness or trial-locking with the real task trajectory), then
(3) pushing that unrelated latent through the SAME mixing object and adding
a fresh draw from the SAME noise model as the structured brain. Reusing the
mixing/noise objects (not just their type) is what guarantees the two
brains' full signal statistics match, not just their noise.

Milestone 1 uses LINEAR mixing (`mixing_type="linear"`); nonlinear mixing is
exposed as a config option for a harder Phase A variant later.

CAVEAT — AR(p) autocorrelation-matching is a LINEAR, 2nd-order-only
approximation. If the teacher's task variables carry genuinely nonlinear /
non-Gaussian dynamics (plausible: the teacher is a GRU), a Gaussian-
innovation AR(p) surrogate will NOT exactly reproduce their spectral shape,
no matter how high `ar_order` is set (verified empirically on a tanh-bounded
synthetic latent: ar_order 2, 4, and 6 all left a ~0.5 relative spectral
mismatch — see tests/nullladder/test_brain.py). Treat "hollow matches
structured" as approximate, and check it empirically for your actual
teacher/task before trusting downstream Null Ladder conclusions — this is
exactly what scripts/verify_phaseA.py's headroom check and the hollow
R5-vs-R4 equivalence test are for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.regression.linear_model import yule_walker


class Mixing:
    def __init__(self, n_latent: int, n_channels: int, mixing_type: str, seed: int):
        if mixing_type not in ("linear", "nonlinear"):
            raise ValueError(f"unknown mixing_type={mixing_type!r}")
        self.mixing_type = mixing_type
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1.0 / np.sqrt(n_latent), size=(n_latent, n_channels))
        if mixing_type == "nonlinear":
            self.w2 = rng.normal(0, 1.0 / np.sqrt(n_channels), size=(n_channels, n_channels))

    def apply(self, latent: np.ndarray) -> np.ndarray:
        mixed = latent @ self.w1
        if self.mixing_type == "nonlinear":
            mixed = np.tanh(mixed) @ self.w2
        return mixed


class NoiseModel:
    """Per-channel AR(1) noise with heterogeneous amplitude and timescale."""

    def __init__(self, n_channels: int, seed: int, phi_range=(0.1, 0.6), std_range=(0.3, 1.5)):
        rng = np.random.default_rng(seed)
        self.n_channels = n_channels
        self.phi = rng.uniform(*phi_range, size=n_channels)
        self.std = rng.uniform(*std_range, size=n_channels)

    def generate(self, T: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise = np.zeros((T, self.n_channels))
        innovations = rng.normal(0, 1, size=(T, self.n_channels)) * self.std
        for t in range(1, T):
            noise[t] = self.phi * noise[t - 1] + innovations[t]
        return noise


@dataclass
class BrainResult:
    signal: np.ndarray  # [T, n_neurons]
    mixing: Mixing
    noise_model: NoiseModel


def make_structured_brain(
    task_variables: np.ndarray,
    n_neurons: int = 200,
    mixing_type: str = "linear",
    seed: int = 0,
) -> BrainResult:
    T, n_latent = task_variables.shape
    mixing = Mixing(n_latent, n_neurons, mixing_type, seed=seed)
    noise_model = NoiseModel(n_neurons, seed=seed + 1)
    mixed = mixing.apply(task_variables)
    noise = noise_model.generate(T, seed=seed + 2)
    return BrainResult(signal=mixed + noise, mixing=mixing, noise_model=noise_model)


def _independent_ar_surrogate(task_variables: np.ndarray, ar_order: int, seed: int) -> np.ndarray:
    """Fit an AR(ar_order) model per latent channel, then draw a fresh,
    independently-seeded realization with matched autocorrelation."""
    T, n_latent = task_variables.shape
    rng = np.random.default_rng(seed)
    out = np.zeros((T, n_latent))
    for c in range(n_latent):
        rho, sigma = yule_walker(task_variables[:, c], order=ar_order, result_object=False)
        innovations = rng.normal(0, sigma, size=T)
        series = np.zeros(T)
        for t in range(T):
            series[t] = innovations[t] + sum(
                rho[k] * series[t - k - 1] for k in range(min(ar_order, t))
            )
        out[:, c] = series
    return out


def make_hollow_brain(
    task_variables: np.ndarray,
    structured: BrainResult,
    ar_order: int = 2,
    seed: int = 0,
) -> BrainResult:
    T = task_variables.shape[0]
    unrelated_latent = _independent_ar_surrogate(task_variables, ar_order=ar_order, seed=seed)
    mixed = structured.mixing.apply(unrelated_latent)
    noise = structured.noise_model.generate(T, seed=seed + 1)
    return BrainResult(signal=mixed + noise, mixing=structured.mixing, noise_model=structured.noise_model)
