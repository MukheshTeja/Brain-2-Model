"""Phase A -- synthetic brain sandbox (known answer).

Per seed: train a teacher GRU on the goal-switching memory task, read out
its task variables, build a STRUCTURED brain (task variables mixed into 200
neurons + realistic noise) and a HOLLOW twin (same mixing/noise, task-free
latent), build each brain's Null Ladder R0-R5 + knockout (+ oracle
knockout diagnostic), then train students against every rung.

Experiments (all job-parallel, resumable):
  main  : students x {structured, hollow} x rungs x alpha_main x seeds
  dose  : GRU x {structured, hollow} x {R5, R3, R0} x alpha grid x seeds   (RQ1)
  rq2   : GRU x 15 region coalitions x {R5, R3} x seeds                     (RQ2 sandbox:
          4 regions with KNOWN content incl. one task-free region)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import torch

from nullladder.ladder import FrozenPCAEmbedder, LadderArm
from nullladder.sandbox.brain import SANDBOX_REGIONS, make_hollow_brain, make_regional_brain, make_structured_brain
from nullladder.sandbox.oracle_knockout import oracle_knockout
from nullladder.sandbox.task import GoalSwitchingTask, GoalSwitchingTaskConfig
from nullladder.sandbox.teacher import extract_task_variables_from_batch, train_teacher
from nullladder.sandbox.zero_shot import make_zero_shot_suite
from nullladder.surrogates import MaxEntropySurrogate

from cdb2m.eval.zero_shot import eval_zero_shot_suite
from cdb2m.experiments.common import linear_cka, make_rows, make_student
from cdb2m.losses import get_alignment_loss
from cdb2m.students.ppo import train_ppo_student
from cdb2m.training.loop import train_student


@dataclass
class PhaseASettings:
    seeds: tuple[int, ...] = tuple(range(10))
    seeds_other_students: tuple[int, ...] = tuple(range(5))
    students: tuple[str, ...] = ("gru", "transformer", "ppo")
    alpha_main: float = 0.2
    dose_alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)
    dose_rungs: tuple[str, ...] = ("R0", "R3", "R5")
    rq2_seeds: tuple[int, ...] = tuple(range(5))
    n_train_trials: int = 60
    data_seed: int = 999
    teacher_hidden_dim: int = 64
    teacher_train_steps: int = 600
    n_task_variables: int = 4
    n_neurons: int = 200
    n_pca_components: int = 8
    student_hidden_dim: int = 32
    n_train_epochs: int = 400
    lr: dict = field(default_factory=lambda: {"gru": 1e-3, "transformer": 1e-3, "ppo": 3e-3})
    n_eval_sequences: int = 150
    alignment_loss: str = "linear_mse"
    task: GoalSwitchingTaskConfig = field(default_factory=GoalSwitchingTaskConfig)


def _concept_labels(batch) -> np.ndarray:
    """Task-state variables the knockout must erase: active goal x remembered
    value at EVERY timestep (joint one-hot), i.e. the task's full latent state."""
    goal = batch["goal_trace"].numpy().reshape(-1)
    mem = batch["memory_trace"].numpy().reshape(-1)
    joint = goal * 100 + mem
    return (joint[:, None] == np.unique(joint)[None, :]).astype(float)


def build_context(args: tuple[PhaseASettings, int]) -> dict:
    """Everything that depends on the seed but not on the student: runs in a worker."""
    s, seed = args
    task = GoalSwitchingTask(s.task)
    train_batch = task.generate(batch_size=s.n_train_trials, seed=s.data_seed)
    teacher = train_teacher(task, hidden_dim=s.teacher_hidden_dim, n_steps=s.teacher_train_steps, seed=seed)
    tv = extract_task_variables_from_batch(teacher, train_batch, s.n_task_variables)
    n, T, _ = tv.shape
    flat = tv.reshape(-1, s.n_task_variables)
    labels = _concept_labels(train_batch)

    structured = make_structured_brain(flat, n_neurons=s.n_neurons, mixing_type="linear", seed=seed)
    hollow = make_hollow_brain(flat, structured, seed=seed)
    arms = {}
    for name, brain in (("structured", structured), ("hollow", hollow)):
        arm = LadderArm(real_signal=brain.signal, real_trials=brain.signal.reshape(n, T, -1), task_labels=labels,
                        n_pca_components=s.n_pca_components).build(seed=seed)
        emb = {r: e.reshape(n, T, -1).astype(np.float32) for r, e in arm.embedded.items()}
        emb["KO_oracle"] = arm.embedder.transform(oracle_knockout(brain.signal, brain.mixing)).reshape(n, T, -1).astype(np.float32)
        arms[name] = emb

    # RQ2 sandbox: regional brain, one frozen embedder per coalition (fit on that coalition's real signal)
    regional = make_regional_brain(flat, n_per_region=50, seed=seed)
    coalitions = {}
    regions = list(SANDBOX_REGIONS)
    for r in range(1, len(regions) + 1):
        for combo in combinations(regions, r):
            sig = regional.coalition(combo)
            embd = FrozenPCAEmbedder(s.n_pca_components).fit(sig)
            r3 = MaxEntropySurrogate().fit(sig).generate(seed=seed)
            coalitions["+".join(sorted(combo))] = {
                "R5": embd.transform(sig).reshape(n, T, -1).astype(np.float32),
                "R3": embd.transform(r3).reshape(n, T, -1).astype(np.float32)}
    return {"seed": seed, "arms": arms, "coalitions": coalitions,
            "train_batch": {k: v.numpy() for k, v in train_batch.items()}}


def execute_job(job: dict) -> list[dict]:
    s: PhaseASettings = job["settings"]
    task = GoalSwitchingTask(s.task)
    zs = make_zero_shot_suite(s.task, seed=s.data_seed)
    zs["iid"] = task  # same distribution as training, fresh sequences
    tb = {k: torch.from_numpy(v) for k, v in job["train_batch"].items()}
    student = make_student(job["student"], task.input_dim, s.student_hidden_dim, task.n_classes, job["seed"])
    emb = torch.from_numpy(job["embedding"]) if job["embedding"] is not None else None
    loss = get_alignment_loss(s.alignment_loss)(s.student_hidden_dim, s.n_pca_components) if job["alpha"] > 0 else None
    kw = dict(alpha=job["alpha"], alignment_loss=loss, teacher_embedding=emb, n_epochs=s.n_train_epochs,
              lr=s.lr[job["student"]], seed=job["seed"])
    if job["student"] == "ppo":
        res = train_ppo_student(student, tb, task.n_classes, **kw)
    else:
        res = train_student(student, tb, task.n_classes, **kw)
    accs = eval_zero_shot_suite(res.model, zs, task.n_classes, s.n_eval_sequences, seed=s.data_seed)
    with torch.no_grad():
        res.model.eval()
        _, hidden = res.model(tb["inputs"])
    h = hidden.reshape(-1, hidden.shape[-1]).numpy()

    def ckas(targets):
        return {k: linear_cka(h, v.reshape(-1, v.shape[-1])) for k, v in targets.items()}

    if job["rung"] == "baseline":  # one alpha=0 run serves every arm: emit one row-set per arm
        rows = []
        for arm, targets in job["cka_targets"].items():
            rows += make_rows({**job, "arm": arm}, accs, res.final_task_acc, ckas(targets))
        return rows
    return make_rows(job, accs, res.final_task_acc, ckas(job["cka_targets"]))


def make_jobs(s: PhaseASettings, contexts: dict[int, dict]) -> list[dict]:
    jobs = []

    def add(kind, student, arm, rung, alpha, seed, embedding, cka_targets):
        ctx = contexts[seed]
        jobs.append({"job_id": f"A|{kind}|{student}|{arm}|{rung}|{alpha}|{seed}", "phase": "A",
                     "dataset": f"sandbox_{kind}", "student": student, "arm": arm, "rung": rung, "alpha": alpha,
                     "seed": seed, "settings": s, "train_batch": ctx["train_batch"], "embedding": embedding,
                     "cka_targets": cka_targets})

    for student in s.students:
        seeds = s.seeds if student == "gru" else s.seeds_other_students
        for seed in seeds:
            arms = contexts[seed]["arms"]
            add("main", student, "all", "baseline", 0.0, seed, None,
                {arm: {"R5": e["R5"], "R3": e["R3"]} for arm, e in arms.items()})
            for arm, emb in arms.items():
                for rung, e in emb.items():
                    add("main", student, arm, rung, s.alpha_main, seed, e, {"rung": e, "R5": emb["R5"], "R3": emb["R3"]})
    for seed in s.seeds:  # RQ1 dose-response (GRU)
        for arm, emb in contexts[seed]["arms"].items():
            for rung in s.dose_rungs:
                for a in s.dose_alphas:
                    if a == s.alpha_main:
                        continue  # already in main
                    add("main", "gru", arm, rung, a, seed, emb[rung], {"rung": emb[rung], "R5": emb["R5"], "R3": emb["R3"]})
    for seed in s.rq2_seeds:  # RQ2 sandbox coalitions (GRU)
        for coal, d in contexts[seed]["coalitions"].items():
            for rung in ("R5", "R3"):
                add("rq2", "gru", coal, rung, s.alpha_main, seed, d[rung], {"rung": d[rung], "R5": d["R5"], "R3": d["R3"]})
    # de-duplicate (dose may overlap main)
    seen, out = set(), []
    for j in jobs:
        if j["job_id"] not in seen:
            seen.add(j["job_id"])
            out.append(j)
    return out
