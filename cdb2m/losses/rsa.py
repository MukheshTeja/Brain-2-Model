"""RSA alignment loss: minimizes 1 - corr(RDM_hidden, RDM_teacher), where RDM
is the pairwise (1 - cosine similarity) representational dissimilarity
matrix over timesteps within each sequence. Pluggable alternative to
linear_mse / cka.
"""
from __future__ import annotations

import torch

from .base import AlignmentLoss, register


def _rdm(x: torch.Tensor) -> torch.Tensor:
    """x: [T, D] -> [T, T] pairwise (1 - cosine similarity) dissimilarity."""
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
    sim = x_norm @ x_norm.T
    return 1.0 - sim


def _upper_triangular(mat: torch.Tensor) -> torch.Tensor:
    n = mat.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    return mat[idx[0], idx[1]]


@register
class RSAAlignmentLoss(AlignmentLoss):
    name = "rsa"

    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        # hidden_dim/embedding_dim unused: RSA compares rank structure, not
        # raw dimensions. Accepted for uniform construction via the registry.

    def forward(self, hidden_states: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        if teacher_embedding.dim() == 2:
            teacher_embedding = teacher_embedding.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        losses = []
        for b in range(hidden_states.shape[0]):
            rdm_h = _upper_triangular(_rdm(hidden_states[b]))
            rdm_t = _upper_triangular(_rdm(teacher_embedding[b]))
            rdm_h = rdm_h - rdm_h.mean()
            rdm_t = rdm_t - rdm_t.mean()
            corr = (rdm_h * rdm_t).sum() / (rdm_h.norm() * rdm_t.norm() + 1e-12)
            losses.append(1.0 - corr)
        return torch.stack(losses).mean()
