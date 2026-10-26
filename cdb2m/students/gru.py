from __future__ import annotations

import torch
import torch.nn as nn


class GRUStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_seq, _ = self.gru(inputs)
        logits = self.readout(hidden_seq)
        return logits, hidden_seq
