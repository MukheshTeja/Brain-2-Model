"""Linear-CKA alignment loss: minimizes 1 - CKA(hidden_states, teacher_embedding).
Pluggable alternative to linear_mse; swap via config, no training-loop changes.
"""
from __future__ import annotations

import torch

from .base import AlignmentLoss, register


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: [N, Dx], y: [N, Dy], both centered along N. Returns scalar CKA in [0, 1]."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (y.T @ x).norm() ** 2
    norm_x = (x.T @ x).norm()
    norm_y = (y.T @ y).norm()
    return hsic / (norm_x * norm_y + 1e-12)


@register
class CKAAlignmentLoss(AlignmentLoss):
    name = "cka"

    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        # hidden_dim/embedding_dim unused: CKA is dimension-agnostic. Accepted
        # so every AlignmentLoss subclass is constructed uniformly by the registry.

    def forward(self, hidden_states: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if teacher_embedding.dim() == 2:
            teacher_embedding = teacher_embedding.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        flat_teacher = teacher_embedding.reshape(-1, teacher_embedding.shape[-1])
        return 1.0 - _linear_cka(flat_hidden, flat_teacher)
