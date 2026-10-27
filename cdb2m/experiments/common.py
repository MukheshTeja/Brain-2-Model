"""Shared pieces for Phase A / Phase B experiments: student factory, CKA,
and the fixed result-row schema every job writes."""
from __future__ import annotations

import numpy as np
import torch

from cdb2m.students.gru import GRUStudent
from cdb2m.students.ppo import PPOStudent
from cdb2m.students.transformer import TransformerStudent

# per-architecture hyper-parameters (chosen in pilots so each learns the task
# below ceiling within the same gradient-step budget)
STUDENT_LR = {"gru": 3e-3, "transformer": 1e-3, "ppo": 3e-3}

ROW_KEYS = ["job_id", "phase", "dataset", "student", "arm", "rung", "alpha", "seed", "variant", "value",
            "train_acc", "cka_rung", "cka_R5", "cka_R3"]


def make_student(name: str, input_dim: int, hidden_dim: int, n_classes: int, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    if name == "gru":
        return GRUStudent(input_dim, hidden_dim, n_classes)
    if name == "transformer":
        return TransformerStudent(input_dim, hidden_dim, n_classes, n_heads=4, n_layers=2)
    if name == "ppo":
        return PPOStudent(input_dim, hidden_dim, n_classes)
    raise ValueError(name)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    hsic = np.linalg.norm(y.T @ x) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y) + 1e-12))


def make_rows(job: dict, accs: dict[str, float], train_acc: float, cka: dict[str, float]) -> list[dict]:
    rows = []
    for variant, v in accs.items():
        rows.append({
            "job_id": job["job_id"], "phase": job["phase"], "dataset": job["dataset"], "student": job["student"],
            "arm": job["arm"], "rung": job["rung"], "alpha": job["alpha"], "seed": job["seed"],
            "variant": variant, "value": v, "train_acc": train_acc,
            "cka_rung": cka.get("rung", np.nan), "cka_R5": cka.get("R5", np.nan), "cka_R3": cka.get("R3", np.nan),
        })
    return rows
