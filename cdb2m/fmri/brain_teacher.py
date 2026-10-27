"""Phase B teacher signals: Null Ladder rungs built from REAL fMRI betas (and
from planted / hollow synthetic betas for the Phase B positive control), all
pushed through ONE fixed embedding pipeline.

Per subject and task block (syllogism / transitive -- their voxel columns
differ, see datasets.py):
  1. z-score voxels, PCA to `n_comp` components -> S [n_problems, n_comp],
     rows in PRESENTATION ORDER (the "time" axis the spectral rungs act on).
  2. Build every rung from S using only unconditional statistics:
       R0 Gaussian, R1 phase-shuffled, R2 covariance-matched,
       R3 multivariate phase-randomised (max-entropy surrogate, all 1st/2nd
       order stats incl. lagged cross-covariance), R4 real data with a
       derangement of problem<->response pairing (event-related design ->
       trial-level shuffle, see design_check.py), R5 real data,
       knockout = LEACE erasure of task-design variables (validity,
       premises-required, conclusion polarity) + global variance rescale.
Across subjects (identical for every rung):
  3. correlation-distance RDM per subject, averaged over subjects (standard
     RSA way to pool subjects whose voxels differ), classical MDS to `k`
     dims, z-scored -> teacher embedding [n_problems, k] per block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from nullladder.sandbox.brain import Mixing, NoiseModel, _independent_ar_surrogate
from nullladder.sandbox.oracle_knockout import oracle_knockout
from nullladder.surrogates import (CovarianceMatchedSurrogate, GaussianSurrogate, MaxEntropySurrogate,
                                   PhaseShuffleSurrogate, StructureKnockout)
from nullladder.surrogates.stimulus_shuffle import _derangement

RUNGS = ["R0", "R1", "R2", "R3", "R4", "R5", "knockout"]


class SubjectEmbedder:
    """Frozen per-subject z-score + PCA, fit once on the subject's REAL betas
    and reused unchanged for every rung (as in nullladder.ladder)."""

    def __init__(self, x: np.ndarray, n_comp: int = 20):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-8
        n_comp = min(n_comp, x.shape[0] - 2, x.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=0).fit((x - self.mu) / self.sd)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform((x - self.mu) / self.sd)


def subject_rungs(x: np.ndarray, design: np.ndarray, seed: int, n_comp: int = 20, mixing=None) -> dict[str, np.ndarray]:
    """Rungs are generated where their claimed statistics are meaningful, then
    all passed through the SAME frozen subject embedder:
      R0, R1 in VOXEL space: per-voxel Gaussian / per-voxel independent phase
         shuffle genuinely destroy cross-voxel (zero-lag) covariance. (Done on
         PCA scores they would not: scores are already decorrelated, so R0/R1
         would silently keep the covariance and R2 would equal R1.)
      R2 in the embedder's decorrelated basis: independent phase shuffle of
         uncorrelated components keeps each component's spectrum and the full
         zero-lag covariance of the retained subspace, but no lagged
         cross-covariance.
      R3 multivariate phase randomisation (basis-invariant): all 1st/2nd-order
         statistics incl. lagged cross-covariance.
      R4 real responses, deranged problem pairing; R5 real; knockout = LEACE."""
    emb = SubjectEmbedder(x, n_comp)
    s = emb(x)
    out = {"R0": emb(GaussianSurrogate().fit(x).generate(seed=seed)),
           "R1": emb(PhaseShuffleSurrogate().fit(x).generate(seed=seed)),
           "R2": CovarianceMatchedSurrogate().fit(s).generate(seed=seed),
           "R3": MaxEntropySurrogate().fit(s).generate(seed=seed)}
    out["R4"] = s[_derangement(s.shape[0], np.random.default_rng(seed))]
    out["R5"] = s
    out["knockout"] = StructureKnockout().fit(s, design).apply(s)
    if mixing is not None:  # planted arm only: exact ground-truth erasure (diagnostic)
        out["KO_oracle"] = emb(oracle_knockout(x, mixing))
    return out


def rdm(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.corrcoef(x)


def classical_mds(d: np.ndarray, k: int) -> np.ndarray:
    n = d.shape[0]
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ (d ** 2) @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1][:k]
    emb = v[:, idx] * np.sqrt(np.clip(w[idx], 1e-12, None))
    return (emb - emb.mean(0)) / (emb.std(0) + 1e-8)


def group_embedding(subject_signals: list[np.ndarray], k: int = 8) -> np.ndarray:
    return classical_mds(np.mean([rdm(s) for s in subject_signals], axis=0), k)


def inter_subject_rdm_reliability(subject_signals: list[np.ndarray]) -> float:
    """Leave-one-subject-out correlation of each subject's RDM with the mean
    of the others -- the noise ceiling of the group teacher signal."""
    rd = [rdm(s)[np.triu_indices(s.shape[0], 1)] for s in subject_signals]
    vals = [np.corrcoef(rd[i], np.mean([r for j, r in enumerate(rd) if j != i], axis=0))[0, 1] for i in range(len(rd))]
    return float(np.mean(vals))


@dataclass
class BlockArm:
    """One arm (e.g. real DR network, planted, hollow) for one task block."""
    rung_embeddings: dict[str, np.ndarray]  # rung -> [n_block_problems, k]
    reliability: float


def build_block_arm(subject_betas: list[np.ndarray], design: np.ndarray, seed: int, k: int = 8,
                    n_comp: int = 20, mixings: list | None = None) -> BlockArm:
    per_rung: dict[str, list[np.ndarray]] = {}
    for si, x in enumerate(subject_betas):
        rungs = subject_rungs(x, design, seed=seed * 1000 + si, n_comp=n_comp,
                              mixing=None if mixings is None else mixings[si])
        for r, sig in rungs.items():
            per_rung.setdefault(r, []).append(sig)
    emb = {r: group_embedding(sigs, k) for r, sigs in per_rung.items()}
    return BlockArm(rung_embeddings=emb, reliability=inter_subject_rdm_reliability(per_rung["R5"]))


def planted_subject_betas(task_variables: np.ndarray, n_subjects: int, n_voxels: int, seed: int,
                          noise_scale: float = 1.0, hollow: bool = False, white_noise: bool = False):
    """Synthetic betas for the Phase B positive control. Each subject gets
    its own random linear voxel mixing of the (unit-variance) teacher task
    variables plus AR(1) heterogeneous noise (same generative family as the
    Phase A sandbox brain). hollow=True swaps the task variables for an
    independent AR(2) surrogate with matched autocorrelation (no task info).
    Returns (betas list, mixing list)."""
    tv = (task_variables - task_variables.mean(0)) / (task_variables.std(0) + 1e-8)
    n, d = tv.shape
    betas, mixes = [], []
    for si in range(n_subjects):
        rs = seed * 100 + si
        latent = _independent_ar_surrogate(tv, ar_order=2, seed=rs + 50) if hollow else tv
        latent = (latent - latent.mean(0)) / (latent.std(0) + 1e-8)
        mix = Mixing(d, n_voxels, "linear", seed=rs)
        if white_noise:  # no shared temporal autocorrelation across subjects (realistic-SNR arm)
            r = np.random.default_rng(rs + 2)
            noise = r.normal(size=(n, n_voxels)) * r.uniform(0.3, 1.5, size=n_voxels)
        else:
            noise = NoiseModel(n_voxels, seed=rs + 1).generate(n, seed=rs + 2)
        betas.append(mix.apply(latent) + noise_scale * noise)
        mixes.append(mix)
    return betas, mixes
