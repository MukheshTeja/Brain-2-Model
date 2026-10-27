"""Phase B training: fresh task problems every step + a FIXED set of
brain-recorded problems that carries the transfer loss.

    L = (1 - alpha) * L_task(fresh batch + recorded problems)
        + alpha * sum_blocks L_transfer(student answer-step state, brain embedding)

`L_transfer` is B2M's brain contrastive loss by default (same inputs shown
to humans and model, Aquino et al. 2025 Eq. 2), one projection head per
task block because the brain embeddings of the two blocks live in different
(voxel-set-specific) coordinate systems. `algo="ppo"` swaps the supervised
task loss for the clipped PPO objective (reward 1 for a correct verdict).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from cdb2m.losses import get_alignment_loss
from cdb2m.students.ppo import _gae
from cdb2m.training.loop import TrainResult


class BlockAlignment(nn.Module):
    def __init__(self, loss_name: str, hidden_dim: int, blocks: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        """blocks: name -> (row indices into the recorded batch, brain embedding [n_rows, k])."""
        super().__init__()
        self.rows = {b: r for b, (r, _) in blocks.items()}
        self.targets = {b: e.float() for b, (_, e) in blocks.items()}
        cls = get_alignment_loss(loss_name)
        self.heads = nn.ModuleDict({b: cls(hidden_dim, e.shape[-1]) for b, (_, e) in blocks.items()})

    def forward(self, answer_hidden: torch.Tensor) -> torch.Tensor:
        return sum(self.heads[b](answer_hidden[self.rows[b]][None], self.targets[b][None]) for b in self.heads)


def _answer_states(hidden: torch.Tensor, answer_idx: torch.Tensor) -> torch.Tensor:
    return hidden[torch.arange(hidden.shape[0]), answer_idx]


def train_online(
    student: nn.Module,
    batch_fn: Callable[[int], dict[str, torch.Tensor]],
    recorded: dict[str, torch.Tensor],
    n_classes: int,
    n_steps: int,
    alpha: float = 0.0,
    alignment: BlockAlignment | None = None,
    lr: float = 3e-3,
    seed: int = 0,
    algo: str = "supervised",
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> TrainResult:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 777)
    params = list(student.parameters()) + (list(alignment.parameters()) if alignment is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    n_rec = recorded["inputs"].shape[0]

    def merged(step: int) -> dict[str, torch.Tensor]:
        fresh = batch_fn(step)
        T = max(fresh["inputs"].shape[1], recorded["inputs"].shape[1])

        def pad(x):
            if x.shape[1] == T:
                return x
            shape = list(x.shape)
            shape[1] = T - x.shape[1]
            return torch.cat([x, torch.zeros(shape, dtype=x.dtype)], 1)

        return {k: torch.cat([pad(recorded[k]), pad(fresh[k])]) if k != "answer_idx"
                else torch.cat([recorded[k], fresh[k]]) for k in ("inputs", "labels", "probe_mask", "answer_idx")}

    def transfer(hidden, b):
        if alpha <= 0:
            return torch.zeros(())
        return alignment(_answer_states(hidden[:n_rec], b["answer_idx"][:n_rec]))

    final_acc, final_loss = 0.0, 0.0
    if algo == "supervised":
        ce = nn.CrossEntropyLoss()
        for step in range(n_steps):
            b = merged(step)
            logits, hidden = student(b["inputs"])
            m = b["probe_mask"]
            task_loss = ce(logits[m], b["labels"][m])
            loss = (1 - alpha) * task_loss + alpha * transfer(hidden, b) if alpha > 0 else task_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(task_loss.detach())
            final_acc = float((logits[m].argmax(-1) == b["labels"][m]).float().mean())
    elif algo == "ppo":
        for it in range(max(1, n_steps // ppo_epochs)):
            b = merged(it)
            m, mf = b["probe_mask"], b["probe_mask"].float()
            with torch.no_grad():
                logits, values, _ = student.forward_with_value(b["inputs"])
                actions = torch.multinomial(logits.softmax(-1).reshape(-1, n_classes), 1, generator=gen).reshape(m.shape)
                old_logp = torch.distributions.Categorical(logits=logits).log_prob(actions)
                rewards = (actions == b["labels"]).float() * mf
                adv, returns = _gae(rewards, values, m, gamma=0.0, lam=0.95)
                sel = adv[m]
                adv = (adv - sel.mean()) / (sel.std() + 1e-8) * mf
                final_acc = float((logits[m].argmax(-1) == b["labels"][m]).float().mean())
            for _ in range(ppo_epochs):
                logits, values, hidden = student.forward_with_value(b["inputs"])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(actions) - old_logp).exp()
                surr = torch.min(ratio * adv, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv)
                n = mf.sum().clamp_min(1.0)
                task_loss = (-(surr * mf).sum() + value_coef * (((values - returns) ** 2) * mf).sum()
                             - entropy_coef * (dist.entropy() * mf).sum()) / n
                loss = (1 - alpha) * task_loss + alpha * transfer(hidden, b) if alpha > 0 else task_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                final_loss = float(task_loss.detach())
    else:
        raise ValueError(algo)
    return TrainResult(model=student, final_task_loss=final_loss, final_task_acc=final_acc)
