import numpy as np

from nullladder.probes import continuous_decodability


def test_continuous_decodability_detects_linear_signal():
    rng = np.random.default_rng(0)
    n = 300
    latent = rng.normal(0, 1, n)
    x = np.stack([latent + rng.normal(0, 0.05, n), rng.normal(0, 1, n)], axis=1)
    result = continuous_decodability(x, latent, seed=0)
    assert result.linear_r2 > 0.8
    assert not result.at_chance


def test_continuous_decodability_at_chance_for_unrelated_signal():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, size=(n, 4))
    y = rng.normal(0, 1, n)  # independent of x
    result = continuous_decodability(x, y, seed=0)
    assert result.at_chance, result
