import numpy as np

from nullladder.equivalence import tost_equivalence


def test_tost_declares_equivalence_for_near_identical_arms():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=20)
    y = x + rng.normal(0, 0.02, size=20)  # paired, near-identical
    result = tost_equivalence(x, y, margin=0.1, paired=True)
    assert result.equivalent


def test_tost_rejects_equivalence_for_clearly_different_arms():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=20)
    y = rng.normal(2.0, 0.1, size=20)
    result = tost_equivalence(x, y, margin=0.1, paired=True)
    assert not result.equivalent


def test_tost_unpaired_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=20)
    y = rng.normal(0.52, 0.1, size=25)
    result = tost_equivalence(x, y, margin=0.3, paired=False)
    assert result.equivalent
