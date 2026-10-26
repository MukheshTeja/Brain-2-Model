"""Zero-shot evaluation: run a trained student on each zero-shot variant
task, report accuracy and gain relative to an alpha=0 baseline accuracy
("generalization fingerprint").
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def eval_accuracy(student: nn.Module, task, n_classes: int, n_eval_sequences: int = 200, seed: int = 0) -> float:
    student.eval()
    batch = task.generate(batch_size=n_eval_sequences, seed=seed)
    logits, _ = student(batch["inputs"])
    preds = logits.argmax(dim=-1)
    mask = batch["probe_mask"]
    correct = ((preds == batch["labels"]).float() * mask.float()).sum()
    return (correct / mask.float().sum().clamp_min(1.0)).item()


def eval_zero_shot_suite(
    student: nn.Module,
    zero_shot_tasks: dict,
    n_classes: int,
    n_eval_sequences: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    return {
        name: eval_accuracy(student, task, n_classes, n_eval_sequences, seed)
        for name, task in zero_shot_tasks.items()
    }


def generalization_fingerprint(variant_accs: dict[str, float], baseline_variant_accs: dict[str, float]) -> dict[str, float]:
    """Per-variant zero-shot gain vs an alpha=0 baseline run with matching seed."""
    return {name: variant_accs[name] - baseline_variant_accs[name] for name in variant_accs}
