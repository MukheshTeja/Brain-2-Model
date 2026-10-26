"""Teacher RNN: trained on the goal-switching task, then its hidden-state
trajectory is reduced to a small number of "task variables" — the
continuous signal that gets mixed into simulated neurons in brain.py.

Task variables are a PCA reduction of the trained teacher's hidden states,
not the raw ground-truth (goal, value) labels: B2M-style transfer is about
a teacher's *learned representation* of the task, not the task's ground
truth, so the structured brain should be built from what the teacher
actually represents internally.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class TeacherRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_seq, _ = self.gru(inputs)
        logits = self.readout(hidden_seq)
        return logits, hidden_seq


def train_teacher(
    task,
    hidden_dim: int = 64,
    n_steps: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> TeacherRNN:
    torch.manual_seed(seed)
    model = TeacherRNN(task.input_dim, hidden_dim, task.n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for step in range(n_steps):
        batch = task.generate(batch_size=batch_size, seed=seed * 100_000 + step)
        logits, _ = model(batch["inputs"])
        loss_per_t = loss_fn(logits.reshape(-1, task.n_classes), batch["labels"].reshape(-1))
        mask = batch["probe_mask"].reshape(-1).float()
        loss = (loss_per_t * mask).sum() / mask.sum().clamp_min(1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def extract_task_variables_from_batch(
    model: TeacherRNN,
    batch: dict[str, torch.Tensor],
    n_components: int,
) -> np.ndarray:
    """Run the trained teacher on a GIVEN batch (so the result's trial order
    lines up exactly with that batch, e.g. a fixed training set also used to
    build the Null Ladder), PCA-reduce hidden states to n_components.
    Returns [n_sequences, T, n_components]."""
    model.eval()
    with torch.no_grad():
        _, hidden_seq = model(batch["inputs"])  # [n_sequences, T, H]
    n_sequences, T, hidden_dim = hidden_seq.shape
    hidden = hidden_seq.reshape(-1, hidden_dim).numpy()
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(hidden)
    return reduced.reshape(n_sequences, T, n_components)


def extract_task_variables(
    model: TeacherRNN,
    task,
    n_components: int,
    n_sequences: int,
    seed: int,
) -> np.ndarray:
    """Convenience wrapper: generates a fresh batch internally, then calls
    extract_task_variables_from_batch. Use the _from_batch variant directly
    when the result must align with a training set used elsewhere."""
    batch = task.generate(batch_size=n_sequences, seed=seed)
    return extract_task_variables_from_batch(model, batch, n_components)
