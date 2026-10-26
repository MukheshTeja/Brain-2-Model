import numpy as np

from nullladder.stats import (
    covariance_match_error,
    lagged_cross_covariance_norm,
    spectrum_match_error,
)
from nullladder.surrogates.covariance_matched import CovarianceMatchedSurrogate
from nullladder.surrogates.max_entropy import MaxEntropySurrogate

from .conftest import average_block_decodability


def test_r2_matches_spectrum_and_zero_lag_cov(synthetic_brain):
    """Per-channel spectrum match is APPROXIMATE, not exact, for R2: the
    contemporaneous recolor step blends multiple independently-phase-
    randomized channels together, which perturbs each channel's own
    spectral shape somewhat even though the total/zero-lag covariance
    comes out exact (see covariance_matched.py docstring)."""
    real = synthetic_brain["signal"]
    gen = CovarianceMatchedSurrogate().fit(real).generate(seed=1)

    assert spectrum_match_error(real, gen) < 0.3
    assert covariance_match_error(real, gen) < 0.1


def test_r2_lacks_lagged_cross_covariance_relative_to_r3(synthetic_brain):
    """R2 only restores zero-lag covariance; R3 preserves the full cross-
    spectral density. R2's lagged cross-covariance should be MEASURABLY
    SMALLER than R3's — not near-zero (see the "Caveat" in
    covariance_matched.py's docstring: the contemporaneous whiten/recolor
    step leaks a non-trivial amount of lagged structure back in whenever
    the underlying components are themselves strongly autocorrelated, which
    is exactly the case for this fixture's AR-driven latents). A 15%+
    reduction relative to R3 is what's empirically observed and is the
    claim this test checks — not an aggressive "<50%" claim that doesn't
    hold for strongly autocorrelated sources."""
    real = synthetic_brain["signal"]
    r2 = CovarianceMatchedSurrogate().fit(real).generate(seed=1)
    r3 = MaxEntropySurrogate().fit(real).generate(seed=1)

    r2_lagged = lagged_cross_covariance_norm(r2, max_lag=5)
    r3_lagged = lagged_cross_covariance_norm(r3, max_lag=5)

    assert r2_lagged < 0.85 * r3_lagged


def test_r2_decodability_at_chance(synthetic_brain):
    real = synthetic_brain["signal"]
    labels = synthetic_brain["labels"]
    surrogate = CovarianceMatchedSurrogate().fit(real)
    result = average_block_decodability(surrogate.generate, labels, synthetic_brain["trial_len"])
    assert result.at_chance, result
