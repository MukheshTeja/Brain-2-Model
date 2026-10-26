import numpy as np

from nullladder.probes import decodability
from nullladder.stats import covariance_match_error, spectrum_match_error
from nullladder.surrogates.gaussian import GaussianSurrogate

from .conftest import block_features


def test_r0_matches_mean_var_not_spectrum(synthetic_brain):
    real = synthetic_brain["signal"]
    surrogate = GaussianSurrogate().fit(real)
    gen = surrogate.generate(seed=1)

    assert np.allclose(gen.mean(axis=0), real.mean(axis=0), atol=0.3)
    assert np.allclose(gen.std(axis=0), real.std(axis=0), atol=0.3)
    assert spectrum_match_error(real, gen) > 0.3
    assert covariance_match_error(real, gen) > 0.3


def test_r0_decodability_at_chance(synthetic_brain):
    real = synthetic_brain["signal"]
    labels = synthetic_brain["labels"]
    gen = GaussianSurrogate().fit(real).generate(seed=1)
    features, block_labels = block_features(gen, labels, synthetic_brain["trial_len"])
    result = decodability(features, block_labels, seed=0)
    assert result.at_chance, result
