"""Shared synthetic fixture for Null Ladder surrogate/knockout unit tests.

Each "trial" is generated INDEPENDENTLY (latent state reset to 0 at the
start of every trial) — matching the real pipeline, where every
task.generate() batch item is an independent sequence with no state carried
over between trials. The per-step recursion is tanh-bounded, mimicking a
GRU-like hidden state (bounded, stationary) rather than an unbounded/
near-unit-root random walk — an earlier unbounded version produced huge
realization-specific DC drift that no surrogate could ever be expected to
match, which was a bug in the fixture, not in the surrogates.
"""
from __future__ import annotations

import numpy as np
import pytest

from nullladder.probes import ProbeResult, decodability


def make_synthetic_brain_signal(seed: int = 0, n_trials: int = 198, trial_len: int = 100, n_channels: int = 6):
    rng = np.random.default_rng(seed)
    n_classes = 3
    # Randomly (not deterministically-periodically) ordered but class-balanced
    # trial labels: a strict 0,1,2,0,1,2,... cycle makes the label sequence
    # itself exactly periodic, and ANY phase-randomization surrogate of a
    # label-correlated signal preserves exact periodicity (randomizing phase
    # of pure harmonics of a base frequency cannot destroy that base
    # frequency's period) — surviving "decodability" would then be an
    # artifact of this fixture's label design, not of the surrogate.
    assert n_trials % n_classes == 0
    trial_labels = rng.permutation(np.tile(np.arange(n_classes), n_trials // n_classes))
    labels = np.repeat(trial_labels, trial_len)

    n_latent = 3
    regime_bias = rng.normal(0, 1.0, size=(n_classes, n_latent))
    mixing = rng.normal(0, 1.0, size=(n_latent, n_channels))
    noise_std = rng.uniform(0.3, 1.2, size=n_channels)
    phi = rng.uniform(0.1, 0.5, size=n_channels)

    latent = np.zeros((n_trials, trial_len, n_latent))
    for b in range(n_trials):
        state = np.zeros(n_latent)
        prev_state = np.zeros(n_latent)
        bias = regime_bias[trial_labels[b]]
        for t in range(trial_len):
            raw = 0.6 * state + 0.2 * prev_state + 0.3 * bias + rng.normal(0, 0.3, n_latent)
            prev_state = state
            state = np.tanh(raw)
            latent[b, t] = state

    mixed = latent.reshape(n_trials * trial_len, n_latent) @ mixing

    noise = np.zeros((n_trials, trial_len, n_channels))
    for b in range(n_trials):
        prev = np.zeros(n_channels)
        innovations = rng.normal(0, 1, size=(trial_len, n_channels)) * noise_std
        for t in range(trial_len):
            prev = phi * prev + innovations[t]
            noise[b, t] = prev
    noise_flat = noise.reshape(n_trials * trial_len, n_channels)

    signal = mixed + noise_flat
    trials = signal.reshape(n_trials, trial_len, n_channels)
    return signal, trials, labels, trial_labels


def block_features(signal: np.ndarray, labels: np.ndarray, block_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Average each non-overlapping block of `block_len` timesteps into one
    feature row, with that block's (constant) label. Required for any
    decodability check on a temporally autocorrelated [T, C] signal: fitting
    a probe on raw per-timestep rows lets adjacent, nearly-identical
    timesteps leak across a random CV split, inflating accuracy regardless
    of whether real task structure survives."""
    n_blocks = signal.shape[0] // block_len
    features = signal[: n_blocks * block_len].reshape(n_blocks, block_len, -1).mean(axis=1)
    block_labels = labels[: n_blocks * block_len].reshape(n_blocks, block_len)[:, 0]
    return features, block_labels


def average_block_decodability(generate_fn, labels: np.ndarray, block_len: int, n_seeds: int = 5):
    """Average decodability over several independent surrogate draws. A
    single surrogate realization is one noisy sample of "does this rung
    carry label information" — at the block-feature sample sizes these
    tests use, a single draw's CV estimate has enough variance that a null
    rung can occasionally land just above a tight chance-margin by chance
    alone (verified empirically). Averaging several independent draws is
    the fix, not loosening the margin indefinitely."""
    linear_accs, mlp_accs, chance = [], [], None
    for seed in range(n_seeds):
        features, block_labels = block_features(generate_fn(seed), labels, block_len)
        result = decodability(features, block_labels, seed=seed)
        linear_accs.append(result.linear_acc)
        mlp_accs.append(result.mlp_acc)
        chance = result.chance_acc
    return ProbeResult(linear_acc=float(np.mean(linear_accs)), mlp_acc=float(np.mean(mlp_accs)), chance_acc=chance)


@pytest.fixture
def synthetic_brain():
    signal, trials, labels, trial_labels = make_synthetic_brain_signal(seed=0)
    return {
        "signal": signal,
        "trials": trials,
        "labels": labels,
        "trial_labels": trial_labels,
        "trial_len": 100,
    }
