"""Tests for the Phase B / RQ3 components added on top of the scaffold."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cdb2m.fmri import reasoning_task as rt
from cdb2m.fmri import relational_task as rel
from cdb2m.fmri.brain_teacher import (build_block_arm, classical_mds, inter_subject_rdm_reliability,
                                      planted_subject_betas, subject_rungs, SubjectEmbedder)
from cdb2m.students.ppo import PPOStudent
from cdb2m.students.transformer import TransformerStudent
from cdb2m.training.online import BlockAlignment, train_online
from nullladder.probes import decodability

DATA = Path(__file__).resolve().parents[2] / "data"
needs_data = pytest.mark.skipif(not (DATA / "ds003076_betas").exists(), reason="run scripts/download_data.py first")


@pytest.mark.parametrize("n_premises", [3, 4, 5, 6])
def test_generator_labels_match_logic_engine(n_premises):
    probs = rt.generate_problems(1000, seed=n_premises, n_premises=n_premises)
    assert all(rt.evaluate_problem(p) == p.label for p in probs)
    labels = np.array([p.label for p in probs])
    assert 0.4 < labels.mean() < 0.6  # balanced


def test_propositional_labels_match_logic_engine():
    probs = rt.generate_problems(400, seed=0, n_premises=2, families=("propositional",))
    assert all(rt.evaluate_problem(p) == p.label for p in probs)


def test_batch_layout_answer_step():
    b = rt.problems_to_batch(rt.generate_problems(32, 0, 5))
    assert b["inputs"].shape[-1] == rt.INPUT_DIM
    assert b["probe_mask"].sum() == 32
    assert torch.all(b["inputs"][torch.arange(32), b["answer_idx"], -1] == 1)


@needs_data
def test_all_fmri_problems_parse_and_labels_verified():
    from cdb2m.fmri.datasets import load_ds003076

    ds = load_ds003076(DATA, networks=("deductive_reasoning",))
    assert len(ds.problems) == 70  # 72 minus the two contradictory transitive items
    assert (ds.table.task == "syllogism").sum() == 36
    assert all(x.shape == (70, 313) for x in ds.betas["deductive_reasoning"].values())


def test_relational_generator_rule_and_parity():
    for parity in (0, 1):
        for top, bot, lab in rel.generate_relational(500, 1, parity):
            dt = [i for i in range(2) if top[0][i] != top[1][i]]
            db = [i for i in range(2) if bot[0][i] != bot[1][i]]
            assert len(dt) == 1 and (dt[0] in db) == bool(lab)
            assert all((o[0] + o[1]) % 2 == parity for o in top + bot)


@needs_data
def test_hcp_stimuli_follow_relational_rule():
    from cdb2m.fmri.datasets import load_hcp_relational

    h = load_hcp_relational(DATA)
    for _, r in h.item_table.iterrows():
        top, bot, lab = rel.hcp_item_problem(r.positions, r.stim)
        dt = [i for i in range(2) if top[0][i] != top[1][i]]
        db = [i for i in range(2) if bot[0][i] != bot[1][i]]
        assert len(dt) == 1 and (dt[0] in db) == bool(lab)


def test_transformer_handles_longer_sequences_than_trained():
    m = TransformerStudent(10, 32, 3)
    for T in (5, 60, 180):
        logits, hidden = m(torch.randn(2, T, 10))
        assert logits.shape == (2, T, 3) and hidden.shape == (2, T, 32)
        assert torch.isfinite(logits).all()


def test_transformer_is_causal():
    m = TransformerStudent(6, 16, 2).eval()
    x = torch.randn(1, 12, 6)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 4, 6)  # change only the future
    assert torch.allclose(m(x)[0][:, :8], m(y)[0][:, :8], atol=1e-5)


def test_ppo_learns_reasoning_task_above_chance():
    rec = rt.problems_to_batch(rt.generate_problems(20, 99, 3))
    st = PPOStudent(rt.INPUT_DIM, 32, 2)
    train_online(st, lambda s: rt.problems_to_batch(rt.generate_problems(64, s, 3)), rec, 2, 600, algo="ppo", lr=1e-2, seed=0)
    b = rt.ReasoningEvalTask(3).generate(400, 0)
    with torch.no_grad():
        acc = (st(b["inputs"])[0][b["probe_mask"]].argmax(-1) == b["labels"][b["probe_mask"]]).float().mean()
    assert acc > 0.55


def _toy_betas(seed=0, n=36, n_subj=6, hollow=False, noise=1.0):
    tv = np.random.default_rng(1).normal(size=(n, 4))
    return planted_subject_betas(tv, n_subj, 120, seed=seed, hollow=hollow, noise_scale=noise)


def test_subject_rungs_shapes_and_r4_is_derangement():
    betas, mixes = _toy_betas()
    design = np.eye(2)[np.arange(36) % 2]
    r = subject_rungs(betas[0], design, seed=0, mixing=mixes[0])
    shapes = {k: v.shape for k, v in r.items()}
    assert set(shapes) == {"R0", "R1", "R2", "R3", "R4", "R5", "knockout", "KO_oracle"}
    assert len(set(shapes.values())) == 1
    rows_r5 = {tuple(np.round(x, 6)) for x in r["R5"]}
    assert {tuple(np.round(x, 6)) for x in r["R4"]} == rows_r5  # same rows...
    assert not np.any(np.all(np.isclose(r["R4"], r["R5"]), axis=1))  # ...none in place


def test_knockout_removes_design_decodability():
    rng = np.random.default_rng(0)
    lab = np.arange(60) % 2
    x = rng.normal(size=(60, 12))
    x[:, :3] += 3.0 * lab[:, None]  # strongly decodable
    design = np.eye(2)[lab]
    assert decodability(x, lab).linear_acc > 0.9
    ko = subject_rungs(x, design, seed=0, n_comp=10)["knockout"]
    assert decodability(ko, lab).linear_acc < 0.7


def test_planted_more_reliable_than_realistic_snr():
    hi = inter_subject_rdm_reliability([SubjectEmbedder(x)(x) for x in _toy_betas(noise=1.0)[0]])
    lo = inter_subject_rdm_reliability([SubjectEmbedder(x)(x) for x in
                                        planted_subject_betas(np.random.default_rng(1).normal(size=(36, 4)), 6, 120,
                                                              seed=0, noise_scale=12, white_noise=True)[0]])
    assert hi > 0.8 and lo < hi - 0.3


def test_group_embedding_dims_and_alignment_loss_runs():
    betas, _ = _toy_betas()
    arm = build_block_arm(betas, np.eye(2)[np.arange(36) % 2], seed=0, k=6, n_comp=10)
    assert arm.rung_embeddings["R5"].shape == (36, 6)
    assert np.allclose(arm.rung_embeddings["R5"].std(0), 1, atol=1e-3)
    al = BlockAlignment("linear_mse", 16, {"a": (torch.arange(36), torch.from_numpy(arm.rung_embeddings["R5"]))})
    assert torch.isfinite(al(torch.randn(36, 16)))


def test_classical_mds_recovers_distances():
    pts = np.random.default_rng(0).normal(size=(20, 3))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    emb = classical_mds(d, 3)
    d2 = np.linalg.norm(emb[:, None] - emb[None], axis=-1)
    assert np.corrcoef(d[np.triu_indices(20, 1)], d2[np.triu_indices(20, 1)])[0, 1] > 0.9


def test_real_data_rungs_are_graded_not_degenerate():
    """Regression test: R0/R1 built on decorrelated PCA scores silently kept the
    zero-lag covariance and made R2 identical to R1. They must be distinct, and
    R0/R1 must destroy covariance while R3/R4 keep it exactly."""
    from nullladder.stats import covariance_match_error

    betas, _ = _toy_betas()
    r = subject_rungs(betas[0], np.eye(2)[np.arange(36) % 2], seed=0)
    assert not np.allclose(r["R1"], r["R2"])
    err = {k: covariance_match_error(r["R5"], r[k]) for k in ("R0", "R1", "R2", "R3", "R4")}
    assert err["R0"] > 0.6 and err["R1"] > 0.6
    assert err["R2"] < err["R1"]
    assert err["R3"] < 1e-6 and err["R4"] < 1e-6
