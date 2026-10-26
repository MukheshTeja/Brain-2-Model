"""Headroom check (milestone-1 decision #5): before spending compute on the
full Null Ladder, confirm there is room for a transfer effect to show up at
all — i.e. training against the structured brain's real data (R5) actually
beats training against pure Gaussian noise (R0) on zero-shot generalization.
If R5 doesn't beat R0 here, no amount of ladder machinery downstream can be
trusted: either the task has no exploitable headroom (baseline too close to
ceiling) or the brain/mixing isn't carrying real task structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cdb2m.eval.zero_shot import eval_zero_shot_suite
from cdb2m.training.loop import train_student


@dataclass
class HeadroomResult:
    r0_gain: float
    r5_gain: float
    baseline_acc: float
    passed: bool


def headroom_check(
    student_factory,
    train_batch: dict[str, torch.Tensor],
    n_classes: int,
    r0_embedding: np.ndarray,
    r5_embedding: np.ndarray,
    alignment_loss_factory,
    zero_shot_tasks: dict,
    alpha: float,
    n_epochs: int,
    seed: int,
    min_r5_minus_r0_margin: float = 0.03,
) -> HeadroomResult:
    baseline = train_student(
        student_factory(), train_batch, n_classes, alpha=0.0, n_epochs=n_epochs, seed=seed
    )
    baseline_accs = eval_zero_shot_suite(baseline.model, zero_shot_tasks, n_classes, seed=seed)
    baseline_mean = float(np.mean(list(baseline_accs.values())))

    def run_rung(embedding: np.ndarray) -> float:
        teacher_embedding = torch.from_numpy(embedding).float()
        result = train_student(
            student_factory(),
            train_batch,
            n_classes,
            alpha=alpha,
            alignment_loss=alignment_loss_factory(),
            teacher_embedding=teacher_embedding,
            n_epochs=n_epochs,
            seed=seed,
        )
        accs = eval_zero_shot_suite(result.model, zero_shot_tasks, n_classes, seed=seed)
        return float(np.mean(list(accs.values()))) - baseline_mean

    r0_gain = run_rung(r0_embedding)
    r5_gain = run_rung(r5_embedding)
    return HeadroomResult(
        r0_gain=r0_gain,
        r5_gain=r5_gain,
        baseline_acc=baseline_mean,
        passed=(r5_gain - r0_gain) > min_r5_minus_r0_margin,
    )
