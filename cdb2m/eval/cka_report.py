"""CKA(student hidden states, teacher signal) per rung — used for the
ladder decomposition and RQ1's mediation analysis (CKA-to-brain vs
CKA-to-R3 as predictors of zero-shot gain).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _linear_cka_numpy(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(y.T @ x) ** 2
    norm_x = np.linalg.norm(x.T @ x)
    norm_y = np.linalg.norm(y.T @ y)
    return float(hsic / (norm_x * norm_y + 1e-12))


@torch.no_grad()
def cka_report(student: nn.Module, inputs: torch.Tensor, rung_embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    """inputs: [N, T, input_dim] batch whose trial order matches the rung
    embeddings' [N, T, n_components] layout. Returns CKA per rung name."""
    student.eval()
    _, hidden = student(inputs)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).numpy()
    report = {}
    for name, embedding in rung_embeddings.items():
        flat_embedding = embedding.reshape(-1, embedding.shape[-1])
        report[name] = _linear_cka_numpy(flat_hidden, flat_embedding)
    return report
