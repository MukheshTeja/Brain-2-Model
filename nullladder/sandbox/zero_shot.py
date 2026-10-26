"""Zero-shot evaluation suite: variants of the goal-switching task that a
model trained only on the standard training config never saw. Every
variant reuses GoalSwitchingTask unmodified — only its config changes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .task import GoalSwitchingTask, GoalSwitchingTaskConfig


def make_zero_shot_suite(base_cfg: GoalSwitchingTaskConfig, seed: int = 0) -> dict[str, GoalSwitchingTask]:
    rng = np.random.default_rng(seed)
    variants: dict[str, GoalSwitchingTaskConfig] = {}

    variants["longer_sequences"] = replace(base_cfg, seq_len=base_cfg.seq_len * 3)

    variants["fixed_cycle"] = replace(base_cfg, goal_sequence_mode="fixed_cycle")

    base_targets = list(range(base_cfg.n_targets))
    perm = rng.permutation(base_cfg.n_targets)
    while np.any(perm == np.arange(base_cfg.n_targets)):
        perm = rng.permutation(base_cfg.n_targets)
    swapped_ids = tuple(base_targets[i] for i in perm)
    variants["swapped_targets"] = replace(base_cfg, target_item_ids=swapped_ids)

    n_items = base_cfg.n_targets + base_cfg.n_distractors
    new_distractor_ids = tuple(range(n_items, n_items + base_cfg.n_distractors))
    variants["new_distractors"] = replace(base_cfg, distractor_item_ids=new_distractor_ids)

    return {name: GoalSwitchingTask(cfg) for name, cfg in variants.items()}
