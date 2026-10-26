"""Null Ladder orchestration: build every rung from a real signal, then push
all of them through the SAME embedding pipeline.

The embedder is fit PER ARM (structured brain vs hollow brain each get their
own frozen PCA, fit once on that arm's real signal) and then applied
unchanged to every rung generated from that arm — R0 through R5 and the
knockout — so rung-to-rung comparisons within an arm are never confounded by
a differently-fit embedding, while the two arms are allowed to have
genuinely different brain-like statistics (that's the H1 vs H2 manipulation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from .surrogates.base import Surrogate
from .surrogates.gaussian import GaussianSurrogate
from .surrogates.phase_shuffle import PhaseShuffleSurrogate
from .surrogates.covariance_matched import CovarianceMatchedSurrogate
from .surrogates.max_entropy import MaxEntropySurrogate
from .surrogates.stimulus_shuffle import StimulusShuffleSurrogate
from .surrogates.knockout import StructureKnockout


class FrozenPCAEmbedder:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self._pca: PCA | None = None

    def fit(self, real_signal: np.ndarray) -> "FrozenPCAEmbedder":
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(real_signal)
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("call fit() before transform()")
        return self._pca.transform(signal)


@dataclass
class LadderArm:
    """One ladder (structured OR hollow), with its own frozen embedder."""

    real_signal: np.ndarray  # [T, C] continuous signal, unconditional stats only used by R0-R3
    real_trials: np.ndarray  # [n_trials, trial_len, C] segmented, used by R4
    task_labels: np.ndarray  # [T] integer labels, OR [T, k] one-hot/multi-concept matrix (e.g. several
    # task variables concatenated), aligned to real_signal's time axis. Used by knockout only — it
    # must cover every task variable the downstream evaluation depends on, or erasure is incomplete.
    n_pca_components: int = 10
    eraser: str = "leace"

    rungs: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    embedded: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    embedder: FrozenPCAEmbedder = field(init=False)

    def build(self, seed: int) -> "LadderArm":
        surrogate_classes: dict[str, Surrogate] = {
            "R0": GaussianSurrogate(),
            "R1": PhaseShuffleSurrogate(),
            "R2": CovarianceMatchedSurrogate(),
            "R3": MaxEntropySurrogate(),
        }
        for name, surrogate in surrogate_classes.items():
            surrogate.fit(self.real_signal)
            self.rungs[name] = surrogate.generate(seed=seed)

        r4 = StimulusShuffleSurrogate().fit(self.real_trials)
        shuffled_trials = r4.generate(seed=seed)
        self.rungs["R4"] = shuffled_trials.reshape(-1, shuffled_trials.shape[-1])[: self.real_signal.shape[0]]

        self.rungs["R5"] = self.real_signal

        knockout = StructureKnockout(eraser=self.eraser)
        knockout.fit(self.real_signal, self.task_labels)
        self.rungs["knockout"] = knockout.apply(self.real_signal)

        self.embedder = FrozenPCAEmbedder(self.n_pca_components).fit(self.real_signal)
        self.embedded = {name: self.embedder.transform(sig) for name, sig in self.rungs.items()}
        return self
