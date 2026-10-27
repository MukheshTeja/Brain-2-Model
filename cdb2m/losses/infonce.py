"""B2M brain contrastive loss (Aquino et al. 2025, Eq. 2): InfoNCE between
model and brain embeddings of the SAME inputs. A learned linear head maps
student hidden states to the brain-embedding dimensionality; S = (B_brain
B_model^T) / tau on L2-normalised rows; positives are the diagonal (same
example), negatives every other example in the batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AlignmentLoss, register


@register
class InfoNCEAlignmentLoss(AlignmentLoss):
    name = "infonce"

    def __init__(self, hidden_dim: int, embedding_dim: int, temperature: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.temperature = temperature

    def forward(self, hidden_states: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        h = self.proj(hidden_states.reshape(-1, hidden_states.shape[-1]))
        if teacher_embedding.dim() == 2 and hidden_states.dim() == 3 and hidden_states.shape[0] > 1:
            teacher_embedding = teacher_embedding.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        b = teacher_embedding.reshape(-1, teacher_embedding.shape[-1])
        sim = F.normalize(b, dim=-1) @ F.normalize(h, dim=-1).T / self.temperature
        return F.cross_entropy(sim, torch.arange(sim.shape[0]))
