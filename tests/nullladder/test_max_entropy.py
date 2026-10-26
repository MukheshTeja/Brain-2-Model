import numpy as np

from nullladder.probes import decodability
from nullladder.stats import (
    covariance_match_error,
    lagged_cross_covariance_norm,
    spectrum_match_error,
)
from nullladder.surrogates.max_entropy import MaxEntropySurrogate

from .conftest import block_features


def test_r3_matches_spectrum_cov_and_lagged_cross_covariance(synthetic_brain):
    real = synthetic_brain["signal"]
    gen = MaxEntropySurrogate().fit(real).generate(seed=1)

    assert spectrum_match_error(real, gen) < 1e-6
    assert covariance_match_error(real, gen) < 1e-6

    real_lagged = lagged_cross_covariance_norm(real, max_lag=5)
    gen_lagged = lagged_cross_covariance_norm(gen, max_lag=5)
    assert abs(real_lagged - gen_lagged) / real_lagged < 0.02


def test_r3_decodability_at_chance(synthetic_brain):
    real = synthetic_brain["signal"]
    labels = synthetic_brain["labels"]
    gen = MaxEntropySurrogate().fit(real).generate(seed=1)
    features, block_labels = block_features(gen, labels, synthetic_brain["trial_len"])
    result = decodability(features, block_labels, seed=0)
    assert result.at_chance, result


def test_r3_iaaft_refinement_runs(synthetic_brain):
    real = synthetic_brain["signal"]
    gen = MaxEntropySurrogate(iaaft_iters=3).fit(real).generate(seed=1)
    assert gen.shape == real.shape
    assert np.isfinite(gen).all()
