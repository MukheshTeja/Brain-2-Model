"""Relational-processing task matched to the HCP relational fMRI paradigm
(Smith et al. 2007; HCP task battery) -- the Phase B replication dataset.

Each trial shows two pairs of objects; each object has a SHAPE and a
TEXTURE (6 values each, ids from the HCP stimulus mapping). The top pair
differs along exactly one dimension; the question is whether the bottom
pair ALSO differs along that dimension (YES/NO) -- the bottom pair may
differ along neither, the other, the same, or both dimensions (verified
against all 47 HCP stimuli used in the fMRI betas). Statement-level encoding, one
object per timestep: [shape one-hot | texture one-hot | top/bottom flag],
then a query step where the student answers.

Zero-shot variants:
  swapped_pairs  bottom pair shown first (relation is symmetric -> same answer)
  novel_combos   only object (shape, texture) combinations never seen in
                 training (training uses (shape + texture) even; test odd)
  both           the two shifts together
"""
from __future__ import annotations

import numpy as np
import torch

N_VAL = 6
INPUT_DIM = 2 * N_VAL + 2 + 1
N_CLASSES = 2


def _pair(rng, dims: tuple[int, ...], parity: int | None):
    while True:
        a = [int(rng.integers(N_VAL)), int(rng.integers(N_VAL))]
        b = list(a)
        for dim in dims:
            b[dim] = int((a[dim] + rng.integers(1, N_VAL)) % N_VAL)
        if parity is None or all((o[0] + o[1]) % 2 == parity for o in (a, b)):
            return [tuple(a), tuple(b)]


def generate_relational(n: int, seed: int, parity: int | None = 0, swap_pairs: bool = False):
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n):
        d_top = int(rng.integers(2))
        same = bool(j % 2)
        d_bot = ((d_top,), (0, 1))[int(rng.integers(2))] if same else ((), (1 - d_top,))[int(rng.integers(2))]
        top, bot = _pair(rng, (d_top,), parity), _pair(rng, d_bot, parity)
        out.append((bot, top, int(same)) if swap_pairs else (top, bot, int(same)))
    return out


def relational_to_batch(problems) -> dict[str, torch.Tensor]:
    B, T = len(problems), 5
    X = np.zeros((B, T, INPUT_DIM), dtype=np.float32)
    y = np.zeros((B, T), dtype=np.int64)
    m = np.zeros((B, T), dtype=bool)
    for b, (p1, p2, lab) in enumerate(problems):
        for t, (obj, pair) in enumerate([(p1[0], 0), (p1[1], 0), (p2[0], 1), (p2[1], 1)]):
            X[b, t, obj[0]] = 1.0
            X[b, t, N_VAL + obj[1]] = 1.0
            X[b, t, 2 * N_VAL + pair] = 1.0
        X[b, 4, -1] = 1.0
        y[b, 4], m[b, 4] = lab, True
    return {"inputs": torch.from_numpy(X), "labels": torch.from_numpy(y), "probe_mask": torch.from_numpy(m),
            "answer_idx": torch.full((B,), 4, dtype=torch.int64)}


class RelationalEvalTask:
    def __init__(self, parity: int | None, swap_pairs: bool):
        self.parity, self.swap = parity, swap_pairs
        self.input_dim, self.n_classes = INPUT_DIM, N_CLASSES

    def generate(self, batch_size: int, seed: int):
        return relational_to_batch(generate_relational(batch_size, 2_000_003 + seed, self.parity, self.swap))


def make_relational_zero_shot_suite() -> dict[str, RelationalEvalTask]:
    return {"swapped_pairs": RelationalEvalTask(0, True), "novel_combos": RelationalEvalTask(1, False),
            "both": RelationalEvalTask(1, True), "iid": RelationalEvalTask(0, False)}


def hcp_item_problem(positions: dict, stim_name: str):
    """HCP stimulus -> (top pair, bottom pair, label). ids in the mapping are 1-based."""
    obj = {k: (v["shape_id"] - 1, v["texture_id"] - 1) for k, v in positions.items()}
    top = [obj["TopLeft"], obj["TopRight"]]
    bot = [obj["BottomLeft"], obj["BottomRight"]]
    label = int("_YES_" in stim_name)
    return top, bot, label
