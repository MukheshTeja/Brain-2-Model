import numpy as np

from nullladder.probes import decodability
from nullladder.stats import effective_dimensionality, smoothed_spectrum_match_error
from nullladder.sandbox.brain import make_hollow_brain, make_structured_brain

from .conftest import block_features


def _fake_task_variables(n_trials=198, trial_len=20, n_latent=3, n_classes=3, seed=0):
    """Bounded (tanh-squashed), per-trial-independent latent — mimics a
    GRU-like hidden state rather than an unbounded/near-unit-root random
    walk. Each trial resets state to 0, matching the real pipeline where
    every task.generate() batch item is an independent sequence.

    Trial labels are randomly (not deterministically-periodically) ordered
    but class-balanced — see the comment in conftest.py's
    make_synthetic_brain_signal for why a strict 0,1,2,0,1,2,... cycle would
    make phase-randomized surrogates spuriously "decodable" regardless of
    surrogate correctness."""
    rng = np.random.default_rng(seed)
    assert n_trials % n_classes == 0
    trial_labels = rng.permutation(np.tile(np.arange(n_classes), n_trials // n_classes))
    labels = np.repeat(trial_labels, trial_len)
    bias = rng.normal(0, 1.0, size=(n_classes, n_latent))

    latent = np.zeros((n_trials, trial_len, n_latent))
    for b in range(n_trials):
        state, prev_state = np.zeros(n_latent), np.zeros(n_latent)
        for t in range(trial_len):
            raw = 0.6 * state + 0.2 * prev_state + 0.3 * bias[trial_labels[b]] + rng.normal(0, 0.3, n_latent)
            prev_state = state
            state = np.tanh(raw)
            latent[b, t] = state

    return latent.reshape(n_trials * trial_len, n_latent), labels, trial_len


def test_structured_brain_carries_task_structure():
    task_variables, labels, trial_len = _fake_task_variables()
    structured = make_structured_brain(task_variables, n_neurons=30, mixing_type="linear", seed=0)
    features, block_labels = block_features(structured.signal, labels, trial_len)
    result = decodability(features, block_labels, seed=0)
    assert result.linear_acc > result.chance_acc + 0.1


def test_hollow_brain_matches_structured_full_statistics_but_not_labels():
    task_variables, labels, trial_len = _fake_task_variables()
    structured = make_structured_brain(task_variables, n_neurons=30, mixing_type="linear", seed=0)
    hollow = make_hollow_brain(task_variables, structured, seed=0)

    # Structured and hollow are two INDEPENDENTLY drawn realizations (not one
    # derived from the other's exact Fourier bins), so compare SMOOTHED
    # spectra — raw periodogram bins are individually high-variance
    # (chi-squared(2)-ish) even for two draws of the identical process. See
    # stats.smoothed_spectrum_match_error.
    #
    # Threshold is looser than you might expect (0.6, not ~0.1-0.2): AR(p)
    # autocorrelation-matching is a LINEAR, 2nd-order-only approximation
    # (see make_hollow_brain's docstring). This fixture's latent is driven
    # through a tanh nonlinearity with regime-dependent bias — genuinely
    # nonlinear, non-Gaussian dynamics that no Gaussian-innovation AR(p)
    # model reproduces exactly, REGARDLESS of ar_order (verified empirically:
    # ar_order 2/4/6 all land in the same ~0.48-0.58 range). Since the real
    # pipeline's teacher is also a nonlinear (GRU) system, expect the same
    # approximate-not-exact match there — treat this as a standing caveat on
    # the hollow brain's fidelity, not a bug to chase to zero.
    rel_err = smoothed_spectrum_match_error(structured.signal, hollow.signal, bin_width=10)
    assert rel_err < 0.6

    dim_struct = effective_dimensionality(structured.signal)
    dim_hollow = effective_dimensionality(hollow.signal)
    assert abs(dim_struct - dim_hollow) < 0.5 * max(dim_struct, dim_hollow)

    features, block_labels = block_features(hollow.signal, labels, trial_len)
    result = decodability(features, block_labels, seed=0)
    assert result.at_chance, result


def test_linear_vs_nonlinear_mixing_config():
    task_variables, _, _ = _fake_task_variables()
    linear = make_structured_brain(task_variables, n_neurons=20, mixing_type="linear", seed=0)
    nonlinear = make_structured_brain(task_variables, n_neurons=20, mixing_type="nonlinear", seed=0)
    assert linear.signal.shape == nonlinear.signal.shape
    assert not np.allclose(linear.signal, nonlinear.signal)
