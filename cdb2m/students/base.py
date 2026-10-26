"""Student protocol: every student (GRU, Transformer, PPO) exposes the same
forward signature so training/eval code is architecture-agnostic.
"""
from __future__ import annotations

from typing import Protocol

import torch


class Student(Protocol):
    hidden_dim: int

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs: [B, T, input_dim] -> (task_output [B, T, n_classes], hidden_states [B, T, H])."""
        ...
