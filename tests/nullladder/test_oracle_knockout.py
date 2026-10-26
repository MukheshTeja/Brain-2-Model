import numpy as np
import pytest

from nullladder.probes import continuous_decodability
from nullladder.sandbox.brain import Mixing
from nullladder.sandbox.oracle_knockout import oracle_knockout


def _make_mixed_signal(seed=0, T=2000, n_latent=3, n_channels=20):
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, size=(T, n_latent))
    mixing = Mixing(n_latent, n_channels, mixing_type="linear", seed=seed)
    noise = rng.normal(0, 0.3, size=(T, n_channels))
    signal = mixing.apply(latent) + noise
    return signal, latent, mixing


def test_oracle_knockout_removes_all_linear_latent_info():
    signal, latent, mixing = _make_mixed_signal()
    erased = oracle_knockout(signal, mixing)

    for c in range(latent.shape[1]):
        result = continuous_decodability(erased, latent[:, c], seed=0)
        assert result.at_chance, (c, result)


def test_oracle_knockout_preserves_total_variance():
    signal, latent, mixing = _make_mixed_signal()
    erased = oracle_knockout(signal, mixing)
    mean = signal.mean(axis=0)
    original_var = ((signal - mean) ** 2).sum()
    erased_var = ((erased - mean) ** 2).sum()
    assert abs(original_var - erased_var) / original_var < 1e-6


def test_oracle_knockout_rejects_nonlinear_mixing():
    signal, latent, _ = _make_mixed_signal()
    nonlinear_mixing = Mixing(latent.shape[1], signal.shape[1], mixing_type="nonlinear", seed=0)
    with pytest.raises(NotImplementedError):
        oracle_knockout(signal, nonlinear_mixing)


def test_oracle_knockout_beats_latent_decodable_before_erasure():
    """Sanity: the real signal IS decodable before erasure (otherwise the
    other two assertions would be vacuous)."""
    signal, latent, mixing = _make_mixed_signal()
    result = continuous_decodability(signal, latent[:, 0], seed=0)
    assert result.linear_r2 > 0.5
