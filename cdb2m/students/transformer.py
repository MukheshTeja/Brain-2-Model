"""Small causal Transformer student (RQ3).

Same interface as GRUStudent: forward(inputs [B, T, D]) -> (logits [B, T, n_classes],
hidden [B, T, H]). Causal self-attention with ALiBi position biases (Press et al.
2022) instead of learned/sinusoidal position embeddings, so the model can be
evaluated zero-shot on sequences 3x longer than it was trained on (the
"longer_sequences" variant) without out-of-range position embeddings.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    start = 2 ** (-8.0 / n_heads)
    return torch.tensor([start ** (i + 1) for i in range(n_heads)])


class _Block(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.n_heads) + bias  # [B, H, T, T]
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.ff(self.ln2(x)))


class TransformerStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.inp = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([_Block(hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_classes)
        self.register_buffer("slopes", _alibi_slopes(n_heads), persistent=False)

    def _bias(self, T: int, device) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        dist = (pos[None, :] - pos[:, None]).float()  # j - i (<= 0 for allowed keys)
        bias = self.slopes[:, None, None] * dist[None]  # linear penalty on distance
        causal = torch.full((T, T), float("-inf"), device=device).triu(1)
        return (bias + causal)[None]  # [1, H, T, T]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inp(inputs)
        bias = self._bias(inputs.shape[1], inputs.device)
        for blk in self.blocks:
            x = blk(x, bias)
        hidden = self.ln_f(x)
        return self.readout(hidden), hidden
