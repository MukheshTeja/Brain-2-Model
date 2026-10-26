"""End-to-end Phase A orchestration: build one fixed training set, train a
teacher on it, build structured + hollow brains from the teacher's task
variables, build each brain's Null Ladder (per-arm frozen PCA embedder),
then sweep alpha x seed x rung, training a fresh student each time and
recording one tidy row per (student, rung, alpha, seed, variant, metric).

`seed` drives every source of randomness EXCEPT the training set itself
(teacher training, brain mixing/noise draws, surrogate generation, student
init/training) — the training set is fixed across seeds by design (see
training/loop.py) so the alpha x seed sweep isolates model/surrogate
stochasticity from "which trials happened to be trained on."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nullladder.sandbox.task import GoalSwitchingTask, GoalSwitchingTaskConfig
from nullladder.sandbox.zero_shot import make_zero_shot_suite
from nullladder.sandbox.teacher import train_teacher, extract_task_variables_from_batch
from nullladder.sandbox.brain import make_structured_brain, make_hollow_brain
from nullladder.ladder import LadderArm

from cdb2m.students.gru import GRUStudent
from cdb2m.losses import get_alignment_loss
from cdb2m.training.loop import train_student
from cdb2m.eval.zero_shot import eval_zero_shot_suite
from cdb2m.eval.cka_report import cka_report
from cdb2m.eval.headroom import headroom_check


@dataclass
class PhaseAConfig:
    task_cfg: GoalSwitchingTaskConfig
    n_train_trials: int = 60
    data_seed: int = 999
    teacher_hidden_dim: int = 64
    teacher_train_steps: int = 600
    n_task_variables: int = 4
    n_neurons: int = 200
    mixing_type: str = "linear"
    student_hidden_dim: int = 32
    n_pca_components: int = 8
    alignment_loss_name: str = "linear_mse"
    alphas: tuple[float, ...] = (0.0, 0.1, 0.2)
    seeds: tuple[int, ...] = (0, 1, 2)
    n_train_epochs: int = 400
    lr: float = 1e-3
    n_eval_sequences: int = 150
    eraser: str = "leace"
    headroom_alpha: float = 0.2
    headroom_epochs: int = 400  # must match n_train_epochs: the headroom gate has
    # to reflect the SAME training budget the main sweep uses, or it can report
    # "no headroom" purely from being undertrained relative to the real runs.


def _build_arm(real_signal: np.ndarray, real_trials: np.ndarray, task_labels: np.ndarray, cfg: PhaseAConfig, seed: int) -> LadderArm:
    arm = LadderArm(
        real_signal=real_signal,
        real_trials=real_trials,
        task_labels=task_labels,
        n_pca_components=cfg.n_pca_components,
        eraser=cfg.eraser,
    )
    arm.build(seed=seed)
    return arm


def run_phase_a(cfg: PhaseAConfig, results_csv_path: str | None = None) -> pd.DataFrame:
    task = GoalSwitchingTask(cfg.task_cfg)
    zero_shot_tasks = make_zero_shot_suite(cfg.task_cfg, seed=cfg.data_seed)
    train_batch = task.generate(batch_size=cfg.n_train_trials, seed=cfg.data_seed)

    def student_factory() -> GRUStudent:
        return GRUStudent(task.input_dim, cfg.student_hidden_dim, task.n_classes)

    def alignment_loss_factory() -> torch.nn.Module:
        return get_alignment_loss(cfg.alignment_loss_name)(cfg.student_hidden_dim, cfg.n_pca_components)

    rows: list[dict] = []
    headroom_rows: list[dict] = []

    for seed in cfg.seeds:
        teacher = train_teacher(task, hidden_dim=cfg.teacher_hidden_dim, n_steps=cfg.teacher_train_steps, seed=seed)
        task_variables = extract_task_variables_from_batch(teacher, train_batch, cfg.n_task_variables)
        n_trials, trial_len, _ = task_variables.shape
        task_vars_flat = task_variables.reshape(-1, cfg.n_task_variables)

        structured = make_structured_brain(task_vars_flat, n_neurons=cfg.n_neurons, mixing_type=cfg.mixing_type, seed=seed)
        hollow = make_hollow_brain(task_vars_flat, structured, seed=seed)

        # Knockout must erase predictability of EVERY task variable the
        # downstream zero-shot evaluation actually depends on — not just
        # active-goal identity. The task's real output is the remembered
        # value (what zero-shot accuracy measures), so both are erased
        # jointly via a combined one-hot label matrix. Erasing goal identity
        # alone left "remembered value" information intact, and knockout
        # tracked R5 almost exactly instead of R3 in an earlier smoke run.
        flat_goal = train_batch["goal_trace"].numpy().reshape(-1)
        flat_value = train_batch["labels"].numpy().reshape(-1)
        goal_onehot = (flat_goal[:, None] == np.unique(flat_goal)[None, :]).astype(float)
        value_onehot = (flat_value[:, None] == np.unique(flat_value)[None, :]).astype(float)
        flat_labels = np.concatenate([goal_onehot, value_onehot], axis=1)

        arms = {
            "structured": _build_arm(structured.signal, structured.signal.reshape(n_trials, trial_len, -1), flat_labels, cfg, seed),
            "hollow": _build_arm(hollow.signal, hollow.signal.reshape(n_trials, trial_len, -1), flat_labels, cfg, seed),
        }

        headroom = headroom_check(
            student_factory,
            train_batch,
            task.n_classes,
            r0_embedding=arms["structured"].embedded["R0"].reshape(n_trials, trial_len, -1),
            r5_embedding=arms["structured"].embedded["R5"].reshape(n_trials, trial_len, -1),
            alignment_loss_factory=alignment_loss_factory,
            zero_shot_tasks=zero_shot_tasks,
            alpha=cfg.headroom_alpha,
            n_epochs=cfg.headroom_epochs,
            seed=seed,
        )
        headroom_rows.append({"seed": seed, "r0_gain": headroom.r0_gain, "r5_gain": headroom.r5_gain, "passed": headroom.passed})

        for arm_name, arm in arms.items():
            rung_names = list(arm.rungs.keys())  # R0 R1 R2 R3 R4 R5 knockout

            baseline = train_student(student_factory(), train_batch, task.n_classes, alpha=0.0, n_epochs=cfg.n_train_epochs, seed=seed)
            baseline_accs = eval_zero_shot_suite(baseline.model, zero_shot_tasks, task.n_classes, cfg.n_eval_sequences, seed)

            for rung_name in rung_names:
                embedding = arm.embedded[rung_name].reshape(n_trials, trial_len, -1)
                teacher_embedding = torch.from_numpy(embedding).float()

                for alpha in cfg.alphas:
                    if alpha == 0.0:
                        result = baseline
                        variant_accs = baseline_accs
                    else:
                        result = train_student(
                            student_factory(),
                            train_batch,
                            task.n_classes,
                            alpha=alpha,
                            alignment_loss=alignment_loss_factory(),
                            teacher_embedding=teacher_embedding,
                            n_epochs=cfg.n_train_epochs,
                            seed=seed,
                        )
                        variant_accs = eval_zero_shot_suite(result.model, zero_shot_tasks, task.n_classes, cfg.n_eval_sequences, seed)

                    cka = cka_report(result.model, train_batch["inputs"], {rung_name: embedding})

                    for variant_name, acc in variant_accs.items():
                        rows.append({
                            "student": "gru",
                            "brain": arm_name,
                            "rung": rung_name,
                            "alpha": alpha,
                            "seed": seed,
                            "variant": variant_name,
                            "metric": "zero_shot_acc",
                            "value": acc,
                            "gain_vs_alpha0": acc - baseline_accs[variant_name],
                            "cka_to_rung": cka[rung_name],
                        })

    df = pd.DataFrame(rows)
    if results_csv_path:
        df.to_csv(results_csv_path, index=False)
        pd.DataFrame(headroom_rows).to_csv(results_csv_path.replace(".csv", "_headroom.csv"), index=False)
    return df
