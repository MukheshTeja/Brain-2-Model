"""Generic training loop: loss = (1-alpha)*L_task + alpha*L_transfer.

A fixed training set (not fresh-every-step data) is used deliberately: the
same `train_batch` of trials is what the teacher RNN's hidden states were
extracted from and what the Null Ladder's rungs were built from, so the
alignment target lines up trial-for-trial with the task-loss batch. Picking
the number of training trials small enough to keep the task below ceiling
(so there's headroom for a transfer effect to show up) is a config
decision — see configs/sweep/smoke.yaml and the headroom check in
scripts/verify_phaseA.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cdb2m.losses.base import AlignmentLoss


@dataclass
class TrainResult:
    model: nn.Module
    final_task_loss: float
    final_task_acc: float


def _task_loss_and_acc(logits: torch.Tensor, labels: torch.Tensor, probe_mask: torch.Tensor, n_classes: int):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_per_t = loss_fn(logits.reshape(-1, n_classes), labels.reshape(-1))
    mask = probe_mask.reshape(-1).float()
    loss = (loss_per_t * mask).sum() / mask.sum().clamp_min(1.0)
    preds = logits.argmax(dim=-1)
    correct = ((preds == labels).float() * probe_mask.float()).sum()
    acc = correct / probe_mask.float().sum().clamp_min(1.0)
    return loss, acc.item()


def train_student(
    student: nn.Module,
    train_batch: dict[str, torch.Tensor],
    n_classes: int,
    alpha: float = 0.0,
    alignment_loss: AlignmentLoss | None = None,
    teacher_embedding: torch.Tensor | None = None,
    n_epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> TrainResult:
    torch.manual_seed(seed)
    params = list(student.parameters())
    if alignment_loss is not None:
        params += list(alignment_loss.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    if alpha > 0:
        assert alignment_loss is not None and teacher_embedding is not None, (
            "alpha > 0 requires both an alignment_loss module and a teacher_embedding"
        )

    final_loss, final_acc = 0.0, 0.0
    for epoch in range(n_epochs):
        logits, hidden = student(train_batch["inputs"])
        task_loss, task_acc = _task_loss_and_acc(logits, train_batch["labels"], train_batch["probe_mask"], n_classes)

        if alpha > 0:
            transfer_loss = alignment_loss(hidden, teacher_embedding)
            loss = (1 - alpha) * task_loss + alpha * transfer_loss
        else:
            loss = task_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss, final_acc = task_loss.item(), task_acc

    return TrainResult(model=student, final_task_loss=final_loss, final_task_acc=final_acc)
