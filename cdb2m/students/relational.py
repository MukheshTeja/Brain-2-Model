"""Small learned graph student for statement-encoded reasoning.

Entity slots are graph addresses, not learned identities. Typed directed
messages share weights across entities and reasoning iterations. There is no
logic solver or label access in forward. This is an answer-only classifier:
premise representations can condition on the question, unlike a causal GRU.
It is an architectural ablation, not evidence of brain transfer by itself.
"""
import torch
from torch import nn
from cdb2m.fmri.reasoning_task import N_SLOTS, STMT_TYPES, INPUT_DIM


class RelationalStudent(nn.Module):
    def __init__(self, hidden_dim=32, rounds=4):
        super().__init__()
        self.hidden_dim, self.rounds = hidden_dim, rounds
        nt = len(STMT_TYPES)
        self.initial = nn.Linear(2 + nt, hidden_dim)
        self.message = nn.Linear(hidden_dim, 2 * nt * hidden_dim, bias=False)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.answer = nn.Sequential(nn.Linear(3 * hidden_dim + nt, hidden_dim), nn.Tanh())
        self.readout = nn.Linear(hidden_dim, 2)

    def forward(self, inputs):
        b, t, d = inputs.shape
        if d != INPUT_DIM:
            raise ValueError("RelationalStudent requires statement-encoded reasoning inputs")
        nt, n, h = len(STMT_TYPES), N_SLOTS, self.hidden_dim
        flag = inputs[..., -1]
        query = (inputs * flag[..., None]).sum(1)
        src, dst = inputs[..., nt:nt+n], inputs[..., nt+n:nt+2*n]
        types = inputs[..., :nt] * (1 - flag[..., None])
        edges = torch.einsum("btr,bti,btj->brij", types, src, dst)
        adj = torch.cat([edges, edges.transpose(-1, -2)], dim=1)
        qsrc, qdst, qt = query[:, nt:nt+n], query[:, nt+n:nt+2*n], query[:, :nt]
        state = torch.tanh(self.initial(torch.cat([qsrc[...,None], qdst[...,None], qt[:,None].expand(-1,n,-1)], -1)))
        for _ in range(self.rounds):
            messages = self.message(state).reshape(b,n,2*nt,h)
            incoming = torch.einsum("brij,birh->bjh", adj, messages)
            state = self.update(incoming.reshape(-1,h), state.reshape(-1,h)).reshape(b,n,h)
        selected = torch.cat([(state*qsrc[...,None]).sum(1), (state*qdst[...,None]).sum(1), state.mean(1), qt], -1)
        answer = self.answer(selected)
        # Only conclusion positions carry predictions; compatible with probe masks.
        hidden = answer[:,None,:] * flag[...,None]
        return self.readout(hidden), hidden
