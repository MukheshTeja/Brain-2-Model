"""Recurrent PPO agent (RQ3's reinforcement-learning student).

The goal-switching task is cast as a POMDP: at every PROBE step the agent
must ACT by reporting the remembered value (or "nothing remembered"), and
receives reward 1 if the report is correct, 0 otherwise. No label is ever
shown to the agent -- it only learns from scalar reward. Actions do not
change future observations (the stimulus stream is exogenous), so the
optimal discount is gamma=0 (each report's credit is its own reward); the
GAE code below stays general for other gamma/lambda.

Interface is identical to the supervised students (forward -> (policy
logits, hidden states)), so zero-shot evaluation (greedy action accuracy)
and the alignment losses are shared unchanged. Brain transfer enters the
same way as in B2M: loss = (1 - alpha) * L_PPO + alpha * L_transfer on the
recurrent core's hidden states.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cdb2m.losses.base import AlignmentLoss
from cdb2m.training.loop import TrainResult


class PPOStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.policy = nn.Linear(hidden_dim, n_classes)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.gru(inputs)
        return self.policy(hidden), hidden

    def forward_with_value(self, inputs: torch.Tensor):
        hidden, _ = self.gru(inputs)
        return self.policy(hidden), self.value(hidden).squeeze(-1), hidden


def _gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float):
    """GAE over the sequence of decision (probe) steps only. rewards/values/mask: [B, T]."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last_adv = torch.zeros(B)
    next_value = torch.zeros(B)
    for t in reversed(range(T)):
        m = mask[:, t]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        a = delta + gamma * lam * last_adv
        adv[:, t] = torch.where(m, a, torch.zeros_like(a))
        last_adv = torch.where(m, a, last_adv)
        next_value = torch.where(m, values[:, t], next_value)
    return adv, adv + values


def train_ppo_student(
    student: PPOStudent,
    train_batch: dict[str, torch.Tensor],
    n_classes: int,
    alpha: float = 0.0,
    alignment_loss: AlignmentLoss | None = None,
    teacher_embedding: torch.Tensor | None = None,
    n_epochs: int = 400,
    lr: float = 3e-3,
    seed: int = 0,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.2,
    gamma: float = 0.0,
    lam: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> TrainResult:
    """n_epochs counts GRADIENT STEPS (n_epochs // ppo_epochs rollouts x
    ppo_epochs updates each), so the PPO agent gets the same optimisation
    budget as the supervised students it is compared against."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 12345)
    params = list(student.parameters()) + (list(alignment_loss.parameters()) if alignment_loss is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    inputs, labels, mask = train_batch["inputs"], train_batch["labels"], train_batch["probe_mask"]
    maskf = mask.float()
    final_acc, final_loss = 0.0, 0.0

    for _ in range(max(1, n_epochs // ppo_epochs)):
        with torch.no_grad():  # rollout
            logits, values, _ = student.forward_with_value(inputs)
            dist = torch.distributions.Categorical(logits=logits)
            actions = torch.multinomial(
                logits.softmax(-1).reshape(-1, n_classes), 1, generator=gen).reshape(labels.shape)
            old_logp = dist.log_prob(actions)
            rewards = (actions == labels).float() * maskf
            adv, returns = _gae(rewards, values, mask, gamma, lam)
            sel = adv[mask]
            adv = (adv - sel.mean()) / (sel.std() + 1e-8) * maskf
            final_acc = float((logits.argmax(-1) == labels)[mask].float().mean())
        for _ in range(ppo_epochs):
            logits, values, hidden = student.forward_with_value(inputs)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ratio = (logp - old_logp).exp()
            surr = torch.min(ratio * adv, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv)
            n = maskf.sum().clamp_min(1.0)
            policy_loss = -(surr * maskf).sum() / n
            value_loss = (((values - returns) ** 2) * maskf).sum() / n
            entropy = (dist.entropy() * maskf).sum() / n
            task_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if alpha > 0:
                loss = (1 - alpha) * task_loss + alpha * alignment_loss(hidden, teacher_embedding)
            else:
                loss = task_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            final_loss = float(task_loss.detach())
    return TrainResult(model=student, final_task_loss=final_loss, final_task_acc=final_acc)
