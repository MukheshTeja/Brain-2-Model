"""Pluggable alignment loss: aligns a student's hidden states to a teacher
signal's embedding. Swappable so CKA/RSA-based losses can replace the
default learned-linear-map + MSE without touching the training loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

_REGISTRY: dict[str, type["AlignmentLoss"]] = {}


def register(cls: type["AlignmentLoss"]) -> type["AlignmentLoss"]:
    _REGISTRY[cls.name] = cls
    return cls


def get_alignment_loss(name: str) -> type["AlignmentLoss"]:
    return _REGISTRY[name]


class AlignmentLoss(nn.Module, ABC):
    name: str

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        """hidden_states: [B, T, H]. teacher_embedding: [B, T, E] or [T, E]
        broadcastable to the batch. Returns a scalar loss."""
