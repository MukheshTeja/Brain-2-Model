import numpy as np

from nullladder.probes import decodability
from nullladder.stats import per_channel_power_spectrum
from nullladder.surrogates.stimulus_shuffle import StimulusShuffleSurrogate


def _mean_per_trial_spectrum(trials: np.ndarray) -> np.ndarray:
    """Average of each trial's OWN power spectrum — exactly invariant under
    reordering trials (unlike the spectrum of the globally-concatenated
    series, which is NOT reorder-invariant due to block-boundary
    discontinuities). The right comparison for "same trials, shuffled
    order"."""
    return np.mean([per_channel_power_spectrum(trial) for trial in trials], axis=0)


def test_r4_is_real_data_with_wrong_pairing(synthetic_brain):
    trials = synthetic_brain["trials"]
    trial_labels = synthetic_brain["trial_labels"]

    r4 = StimulusShuffleSurrogate().fit(trials)
    shuffled = r4.generate(seed=1)

    assert shuffled.shape == trials.shape
    # Every trial's content is untouched, just reassigned to a different slot.
    original_set = {tuple(t.ravel()) for t in trials}
    shuffled_set = {tuple(t.ravel()) for t in shuffled}
    assert original_set == shuffled_set
    # Derangement: no trial kept its original slot.
    assert all(not np.array_equal(shuffled[i], trials[i]) for i in range(len(trials)))

    # Compare the MEAN PER-TRIAL spectrum (exactly reorder-invariant), not
    # the global concatenated series' periodogram (which is NOT
    # reorder-invariant, since reordering changes which trials sit next to
    # each other at block boundaries).
    real_spec = _mean_per_trial_spectrum(trials)
    shuffled_spec = _mean_per_trial_spectrum(shuffled)
    rel_err = np.linalg.norm(real_spec - shuffled_spec) / np.linalg.norm(real_spec)
    assert rel_err < 1e-10


def test_r4_decodability_against_original_labels_at_chance(synthetic_brain):
    """Pairing trial i's shuffled (= some other trial's real) content against
    task trial i's original label should be at-chance: the content is real
    but belongs to the wrong trial."""
    trials = synthetic_brain["trials"]
    trial_labels = synthetic_brain["trial_labels"]

    r4 = StimulusShuffleSurrogate().fit(trials)
    shuffled = r4.generate(seed=1)

    features = shuffled.mean(axis=1)  # per-trial-averaged feature, [n_trials, C]
    result = decodability(features, trial_labels, seed=0)
    assert result.at_chance, result
