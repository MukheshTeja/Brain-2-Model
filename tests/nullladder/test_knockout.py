import numpy as np

from nullladder.probes import decodability
from nullladder.surrogates.knockout import StructureKnockout

from .conftest import block_features


def test_knockout_destroys_decodability(synthetic_brain):
    real = synthetic_brain["signal"]
    labels = synthetic_brain["labels"]

    knockout = StructureKnockout(eraser="leace").fit(real, labels)
    erased = knockout.apply(real)

    trial_len = synthetic_brain["trial_len"]
    before_features, before_labels = block_features(real, labels, trial_len)
    after_features, after_labels = block_features(erased, labels, trial_len)
    before = decodability(before_features, before_labels, seed=0)
    after = decodability(after_features, after_labels, seed=0)

    assert before.linear_acc > before.chance_acc + 0.1  # real signal is decodable
    assert after.at_chance, after


def test_knockout_preserves_total_variance(synthetic_brain):
    """Only TOTAL variance is matched (a single global scalar rescale) — not
    the detailed per-channel spectrum shape. See knockout.py's module
    docstring for why a per-channel/per-frequency rescale was rejected: it
    empirically reintroduced the erased label-predictive structure."""
    real = synthetic_brain["signal"]
    labels = synthetic_brain["labels"]

    knockout = StructureKnockout(eraser="leace").fit(real, labels)
    erased = knockout.apply(real)

    original_total_var = ((real - real.mean(axis=0)) ** 2).sum()
    erased_total_var = ((erased - real.mean(axis=0)) ** 2).sum()
    assert abs(original_total_var - erased_total_var) / original_total_var < 1e-6


def test_knockout_rejects_unimplemented_eraser():
    try:
        StructureKnockout(eraser="inlp")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for eraser='inlp' in milestone 1")
