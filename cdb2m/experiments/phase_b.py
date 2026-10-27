"""Phase B -- the validated test applied to REAL fMRI.

Arms (each is a full set of teacher signals per alignment block):
  real_dr          ds003076, deductive-reasoning network (primary)
  planted          Phase B positive control: a reasoning-teacher network's
                   task variables mixed into 10 synthetic subjects x 313
                   voxels (known answer: genuine structure)
  hollow           same, but task-free latents with matched statistics
                   (known answer: no structure)
  planted_realsnr  planted, with noise raised until its inter-subject RDM
                   reliability matches the real data's (power check: would
                   real structure even be detectable at fMRI's SNR?)
  rq2 coalitions   every non-empty coalition of {DR, MD, language} networks
  hcp              HCP relational task, MD network (replication)

Experiments:
  main    students x {real_dr, planted, hollow} x all rungs x alpha_main x seeds
  dose    GRU x {real_dr, planted} x {R0, R3, R5} x alpha grid x seeds      (RQ1)
  power   GRU x planted_realsnr x {R0, R3, R5} x alpha_main x seeds
  rq2     GRU x 7 network coalitions x {R5, R3} x alpha_main x seeds        (RQ2)
  hcp     GRU x hcp x all rungs x alpha_main x seeds                         (replication)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch
from sklearn.decomposition import PCA

from cdb2m.eval.zero_shot import eval_zero_shot_suite
from cdb2m.experiments.common import linear_cka, make_rows, make_student
from cdb2m.fmri import reasoning_task as rt
from cdb2m.fmri import relational_task as rel
from cdb2m.fmri.brain_teacher import (SubjectEmbedder, build_block_arm, inter_subject_rdm_reliability,
                                      planted_subject_betas)
from cdb2m.fmri.datasets import NETWORKS, design_labels, hcp_blocks, load_ds003076, load_hcp_relational
from cdb2m.training.online import BlockAlignment, train_online

NET_SHORT = {"deductive_reasoning": "DR", "md": "MD", "language": "LANG"}


@dataclass
class PhaseBSettings:
    data_dir: str = "data"
    seeds: tuple[int, ...] = tuple(range(10))
    seeds_other_students: tuple[int, ...] = tuple(range(5))
    students: tuple[str, ...] = ("gru", "transformer", "ppo")
    alpha_main: float = 0.2
    dose_alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    dose_rungs: tuple[str, ...] = ("R0", "R3", "R5")
    rq2_seeds: tuple[int, ...] = tuple(range(5))
    # follow-up seed extension (GRU only, real-data ladders + power arm): added after the
    # 10-seed run left real_dr R5-R3 neither significant nor equivalent within the SESOI.
    followup_seeds: tuple[int, ...] = tuple(range(10, 30))
    run_hcp: bool = True
    run_power: bool = True
    n_steps: int = 800
    batch_size: int = 64
    student_hidden_dim: int = 32
    lr: dict = field(default_factory=lambda: {"gru": 3e-3, "transformer": 1e-3, "ppo": 1e-2})
    n_eval: int = 1000
    k_embed: int = 8
    n_comp: int = 20
    teacher_steps: int = 3000
    n_teacher_vars: int = 4
    alignment_loss: str = "linear_mse"


# ------------------------------------------------------------------ contexts
def _teacher_task_variables(s: PhaseBSettings, recorded, blocks) -> dict[str, np.ndarray]:
    """A strong reasoning GRU (trained 3000 online steps) -- the 'brain' whose
    task variables are planted in the positive-control arm."""
    torch.manual_seed(12345)
    teacher = make_student("gru", rt.INPUT_DIM, 64, 2, seed=12345)
    train_online(teacher, lambda st: rt.problems_to_batch(rt.generate_problems(128, 5_000_000 + st, 3)),
                 recorded, 2, s.teacher_steps, lr=3e-3, seed=12345)
    with torch.no_grad():
        _, hid = teacher(recorded["inputs"])
    ans = hid[torch.arange(hid.shape[0]), recorded["answer_idx"]].numpy()
    return {b: PCA(s.n_teacher_vars, random_state=0).fit_transform(ans[rows]) for b, rows in blocks.items()}


def _calibrated_noise(tv: np.ndarray, n_vox: int, target: float, seed: int) -> float:
    best, best_err = 1.0, np.inf
    for ns in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 25):
        rel_ = np.mean([inter_subject_rdm_reliability(
            [SubjectEmbedder(x)(x) for x in planted_subject_betas(tv, 10, n_vox, seed + r, ns, white_noise=True)[0]])
            for r in range(2)])
        if abs(rel_ - target) < best_err:
            best, best_err = ns, abs(rel_ - target)
    return float(best)


def build_context(args: tuple[PhaseBSettings, int, dict]) -> dict:
    """All teacher signals for one seed (surrogate draws depend on the seed)."""
    s, seed, shared = args
    ds = load_ds003076(s.data_dir)
    blocks = {t: ds.task_rows(t) for t in ("syllogism", "transitive")}
    tv = shared["teacher_vars"]
    arms: dict[str, dict] = {}
    reliab: dict[str, float] = {}

    def add_arm(name, betas_by_block, mixings_by_block=None):
        arms[name] = {}
        for b in blocks:
            a = build_block_arm(betas_by_block[b], design_labels(ds, b), seed=seed, k=s.k_embed, n_comp=s.n_comp,
                                mixings=None if mixings_by_block is None else mixings_by_block[b])
            arms[name][b] = {r: e.astype(np.float32) for r, e in a.rung_embeddings.items()}
            reliab[f"{name}|{b}"] = a.reliability

    add_arm("real_dr", {b: [ds.betas["deductive_reasoning"][sub][r] for sub in ds.subjects] for b, r in blocks.items()})
    pl = {b: planted_subject_betas(tv[b], len(ds.subjects), 313, seed=7) for b in blocks}
    add_arm("planted", {b: v[0] for b, v in pl.items()}, {b: v[1] for b, v in pl.items()})
    ho = {b: planted_subject_betas(tv[b], len(ds.subjects), 313, seed=7, hollow=True) for b in blocks}
    add_arm("hollow", {b: v[0] for b, v in ho.items()}, {b: v[1] for b, v in ho.items()})
    if s.run_power:
        prs = {b: planted_subject_betas(tv[b], len(ds.subjects), 313, seed=7, noise_scale=shared["realsnr_noise"][b],
                                        white_noise=True) for b in blocks}
        add_arm("planted_realsnr", {b: v[0] for b, v in prs.items()})

    # RQ2: network coalitions. Each network is first reduced per subject (its own
    # frozen embedder), scaled to unit total variance so big networks (MD: 3082
    # voxels) don't drown small ones (DR: 313), then concatenated.
    coal = {}
    if seed in s.rq2_seeds:
        for r in range(1, len(NETWORKS) + 1):
            for combo in combinations(NETWORKS, r):
                name = "+".join(NET_SHORT[n] for n in combo)
                by_block = {}
                for b, rows in blocks.items():
                    subj = []
                    for sub in ds.subjects:
                        parts = []
                        for n in combo:
                            x = ds.betas[n][sub][rows]
                            z = SubjectEmbedder(x, s.n_comp)(x)
                            parts.append(z / np.sqrt((z ** 2).sum() / z.shape[0]))
                        subj.append(np.concatenate(parts, axis=1))
                    a = build_block_arm(subj, design_labels(ds, b), seed=seed, k=s.k_embed, n_comp=s.n_comp)
                    by_block[b] = {k: a.rung_embeddings[k].astype(np.float32) for k in ("R5", "R3")}
                    reliab[f"coal:{name}|{b}"] = a.reliability
                coal[name] = by_block

    hcp = None
    if s.run_hcp:
        hb = hcp_blocks(load_hcp_relational(s.data_dir))
        hcp = {}
        for b, d in hb.items():
            a = build_block_arm(d["betas"], d["design"], seed=seed, k=s.k_embed, n_comp=s.n_comp)
            hcp[b] = {r: e.astype(np.float32) for r, e in a.rung_embeddings.items()}
            reliab[f"hcp|{b}"] = a.reliability
    return {"seed": seed, "arms": arms, "coalitions": coal, "hcp": hcp, "reliability": reliab}


def shared_context(s: PhaseBSettings) -> dict:
    ds = load_ds003076(s.data_dir)
    rec = rt.problems_to_batch(ds.problems)
    blocks = {t: ds.task_rows(t) for t in ("syllogism", "transitive")}
    tv = _teacher_task_variables(s, rec, blocks)
    out = {"teacher_vars": tv}
    if s.run_power:
        out["realsnr_noise"] = {}
        for b, rows in blocks.items():
            target = inter_subject_rdm_reliability(
                [SubjectEmbedder(ds.betas["deductive_reasoning"][sub][rows])(ds.betas["deductive_reasoning"][sub][rows])
                 for sub in ds.subjects])
            out["realsnr_noise"][b] = _calibrated_noise(tv[b], 313, target, seed=3)
    return out


# ---------------------------------------------------------------- execution
@functools.lru_cache(maxsize=4)
def _recorded(dataset: str, data_dir: str):
    if dataset == "hcp":
        h = load_hcp_relational(data_dir)
        hb = hcp_blocks(h)
        probs, rows, start = [], {}, 0
        for b, d in hb.items():
            for i in d["items"]:
                r = h.item_table.iloc[i]
                probs.append(rel.hcp_item_problem(r.positions, r.stim))
            rows[b] = np.arange(start, start + len(d["items"]))
            start += len(d["items"])
        return rel.relational_to_batch(probs), rows
    ds = load_ds003076(data_dir, networks=("deductive_reasoning",))
    return rt.problems_to_batch(ds.problems), {t: ds.task_rows(t) for t in ("syllogism", "transitive")}


@functools.lru_cache(maxsize=2)
def _eval_suite(dataset: str):
    if dataset == "hcp":
        return rel.make_relational_zero_shot_suite()
    zs = rt.make_reasoning_zero_shot_suite()
    zs["iid"] = rt.ReasoningEvalTask(3)
    return zs


def execute_job(job: dict) -> list[dict]:
    s: PhaseBSettings = job["settings"]
    hcp = job["dataset"] == "hcp"
    recorded, rows = _recorded("hcp" if hcp else "ds003076", s.data_dir)
    in_dim = rel.INPUT_DIM if hcp else rt.INPUT_DIM
    seed = job["seed"]
    if hcp:
        batch_fn = lambda st: rel.relational_to_batch(rel.generate_relational(s.batch_size, 10_000 * seed + st, 0))  # noqa: E731
    else:
        batch_fn = lambda st: rt.problems_to_batch(rt.generate_problems(s.batch_size, 10_000 * seed + st, 3))  # noqa: E731
    student = make_student(job["student"], in_dim, s.student_hidden_dim, 2, seed)
    align = None
    if job["alpha"] > 0:
        align = BlockAlignment(s.alignment_loss, s.student_hidden_dim,
                               {b: (torch.as_tensor(rows[b]), torch.from_numpy(job["embedding"][b])) for b in rows})
    res = train_online(student, batch_fn, recorded, 2, s.n_steps, alpha=job["alpha"], alignment=align,
                       lr=s.lr[job["student"]], seed=seed, algo="ppo" if job["student"] == "ppo" else "supervised")
    accs = eval_zero_shot_suite(res.model, _eval_suite("hcp" if hcp else "ds003076"), 2, s.n_eval, seed=0)
    with torch.no_grad():
        res.model.eval()
        _, hid = res.model(recorded["inputs"])
    ans = hid[torch.arange(hid.shape[0]), recorded["answer_idx"]].numpy()

    def ckas(targets):  # mean over alignment blocks
        return {k: float(np.mean([linear_cka(ans[rows[b]], v[b]) for b in rows])) for k, v in targets.items()}

    if job["rung"] == "baseline":
        out = []
        for arm, targets in job["cka_targets"].items():
            out += make_rows({**job, "arm": arm}, accs, res.final_task_acc, ckas(targets))
        return out
    return make_rows(job, accs, res.final_task_acc, ckas(job["cka_targets"]))


def make_jobs(s: PhaseBSettings, contexts: dict[int, dict]) -> list[dict]:
    jobs = []

    def add(dataset, kind, student, arm, rung, alpha, seed, embedding, cka_targets):
        jobs.append({"job_id": f"B|{dataset}|{kind}|{student}|{arm}|{rung}|{alpha}|{seed}", "phase": "B", "dataset": dataset,
                     "student": student, "arm": arm, "rung": rung, "alpha": alpha, "seed": seed, "settings": s,
                     "embedding": embedding, "cka_targets": cka_targets})

    def pick(arm_blocks, rung):
        return {b: d[rung] for b, d in arm_blocks.items()}

    def tg(arm_blocks, rung):
        return {"rung": pick(arm_blocks, rung), "R5": pick(arm_blocks, "R5"), "R3": pick(arm_blocks, "R3")}

    main_arms = ("real_dr", "planted", "hollow")
    for student in s.students:
        for seed in (s.seeds if student == "gru" else s.seeds_other_students):
            arms = contexts[seed]["arms"]
            base_targets = {a: {"R5": pick(arms[a], "R5"), "R3": pick(arms[a], "R3")} for a in arms}
            add("ds003076", "main", student, "all", "baseline", 0.0, seed, None, base_targets)
            for arm in main_arms:
                for rung in arms[arm]["syllogism"]:
                    add("ds003076", "main", student, arm, rung, s.alpha_main, seed, pick(arms[arm], rung), tg(arms[arm], rung))
    for seed in s.seeds:
        arms = contexts[seed]["arms"]
        for arm in ("real_dr", "planted"):
            for rung in s.dose_rungs:
                for a in s.dose_alphas:
                    add("ds003076", "main", "gru", arm, rung, a, seed, pick(arms[arm], rung), tg(arms[arm], rung))
        if s.run_power:
            for rung in s.dose_rungs:
                add("ds003076", "main", "gru", "planted_realsnr", rung, s.alpha_main, seed,
                    pick(arms["planted_realsnr"], rung), tg(arms["planted_realsnr"], rung))
    for seed in s.rq2_seeds:
        for coal, d in contexts[seed]["coalitions"].items():
            for rung in ("R5", "R3"):
                add("ds003076", "rq2", "gru", coal, rung, s.alpha_main, seed, pick(d, rung), tg(d, rung))
    if s.run_hcp:
        for seed in s.seeds:
            h = contexts[seed]["hcp"]
            add("hcp", "main", "gru", "all", "baseline", 0.0, seed, None, {"hcp": {"R5": pick(h, "R5"), "R3": pick(h, "R3")}})
            for rung in h["list1"]:
                add("hcp", "main", "gru", "hcp", rung, s.alpha_main, seed, pick(h, rung), tg(h, rung))
    for seed in s.followup_seeds:  # GRU follow-up: real-data ladders only (+ power arm, + HCP)
        arms = contexts[seed]["arms"]
        add("ds003076", "main", "gru", "all", "baseline", 0.0, seed, None,
            {a: {"R5": pick(arms[a], "R5"), "R3": pick(arms[a], "R3")} for a in ("real_dr", "planted_realsnr") if a in arms})
        for rung in arms["real_dr"]["syllogism"]:
            add("ds003076", "main", "gru", "real_dr", rung, s.alpha_main, seed, pick(arms["real_dr"], rung), tg(arms["real_dr"], rung))
        if s.run_power:
            for rung in s.dose_rungs:
                add("ds003076", "main", "gru", "planted_realsnr", rung, s.alpha_main, seed,
                    pick(arms["planted_realsnr"], rung), tg(arms["planted_realsnr"], rung))
        if s.run_hcp:
            h = contexts[seed]["hcp"]
            add("hcp", "main", "gru", "all", "baseline", 0.0, seed, None, {"hcp": {"R5": pick(h, "R5"), "R3": pick(h, "R3")}})
            for rung in h["list1"]:
                add("hcp", "main", "gru", "hcp", rung, s.alpha_main, seed, pick(h, rung), tg(h, rung))
    seen, out = set(), []
    for j in jobs:
        if j["job_id"] not in seen:
            seen.add(j["job_id"])
            out.append(j)
    return out
