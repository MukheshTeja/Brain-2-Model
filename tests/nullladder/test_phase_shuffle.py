import numpy as np

from nullladder.stats import covariance_match_error, spectrum_match_error
from nullladder.surrogates.phase_shuffle import PhaseShuffleSurrogate

from .conftest import average_block_decodability


def test_r1_preserves_per_channel_spectrum_destroys_covariance(synthetic_brain):
    real = synthetic_brain["signal"]
    gen = PhaseShuffleSurrogate().fit(real).generate(seed=1)

    assert spectrum_match_error(real, gen) < 0.05
    assert covariance_match_error(real, gen) > 0.3


def test_r1_decodability_at_chance(synthetic_brain):
    real = synthetic_brain["signal"]
    labels = synthetic_brain["labels"]
    surrogate = PhaseShuffleSurrogate().fit(real)
    result = average_block_decodability(surrogate.generate, labels, synthetic_brain["trial_len"])
    assert result.at_chance, result
