"""Default L_transfer: a learned linear map from student hidden states to
the teacher embedding's dimensionality, trained jointly via MSE.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import AlignmentLoss, register


@register
class LinearMSEAlignmentLoss(AlignmentLoss):
    name = "linear_mse"

    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, hidden_states: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.proj(hidden_states)
        if teacher_embedding.dim() == 2:
            teacher_embedding = teacher_embedding.unsqueeze(0).expand(projected.shape[0], -1, -1)
        return nn.functional.mse_loss(projected, teacher_embedding)
