"""Goal-switching memory task (Phase A sandbox).

At each timestep the sequence shows either a GOAL CUE (announcing which of
K goals is now active — a genuine switch, never cueing the same goal twice
in a row) or a STIMULUS item. Each of the K goals owns exactly one "target"
item id; D further item ids are pure distractors, relevant to no goal, plus
the OTHER goals' target items act as distractors for the currently active
goal. Whenever the active goal's own target item appears, it carries a
random value in {0..V-1} that must be remembered. At PROBE steps the task
asks: "what is the most recently remembered value for the active goal since
its last switch?" (class V = "nothing remembered yet").

This is deliberately a hybrid of task-switching (ignore off-goal items) and
working memory (hold the last value, not just detect occurrence), so a
network has to maintain goal identity AND goal-specific content — matching
B2M's "goal-switching memory task" framing.

Everything that zero-shot evaluation varies — sequence length, item-id
vocabulary (for "new distractors" / "swapped targets"), goal sequencing
mode (random vs fixed K-cycle) — is a constructor argument here, so the
zero-shot suite in zero_shot.py reuses this single generator unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GoalSwitchingTaskConfig:
    seq_len: int = 60
    n_targets: int = 3  # K
    n_distractors: int = 3  # D
    n_values: int = 4  # V
    goal_switch_period_range: tuple[int, int] = (8, 14)
    probe_prob: float = 0.3
    stimulus_prob: float = 0.7  # prob a non-cue step shows a stimulus (vs blank)
    target_hit_prob: float = 0.35  # prob a stimulus step shows the active goal's own target item
    goal_sequence_mode: str = "random"  # "random" | "fixed_cycle"
    target_item_ids: tuple[int, ...] | None = None  # override for "swapped targets" zero-shot
    distractor_item_ids: tuple[int, ...] | None = None  # override for "new distractors" zero-shot
    n_reserve_distractors: int | None = None  # extra unused-at-train-time item slots so
    # input_dim is identical across the "new distractors" zero-shot variant and training;
    # defaults to n_distractors (enough headroom to fully swap distractor identity).


class GoalSwitchingTask:
    """Vocabulary layout (one-hot, size = n_targets + n_distractors + 2):
    [0 .. K+D-1]           = item identity (K target items, D distractor items)
    [K+D .. K+D+V-1]       = value tag (only meaningful alongside a target item)
    last two channels      = {is_cue, is_probe} markers
    A goal cue step additionally one-hot encodes the new goal id in the
    item-identity slice (reusing the target item ids as goal ids, 1:1).
    """

    def __init__(self, cfg: GoalSwitchingTaskConfig):
        self.cfg = cfg
        K, D = cfg.n_targets, cfg.n_distractors
        reserve = cfg.n_reserve_distractors if cfg.n_reserve_distractors is not None else D
        self.n_items = K + D + reserve  # fixed regardless of id overrides, so input_dim
        # is identical across training config and every zero-shot variant.
        self.target_ids = list(cfg.target_item_ids) if cfg.target_item_ids else list(range(K))
        self.distractor_ids = (
            list(cfg.distractor_item_ids) if cfg.distractor_item_ids else list(range(K, K + D))
        )
        assert max(self.target_ids + self.distractor_ids) < self.n_items, (
            "target/distractor item ids must fit within n_targets + n_distractors + "
            "n_reserve_distractors slots"
        )
        self.input_dim = self.n_items + cfg.n_values + 2
        self.n_classes = cfg.n_values + 1  # + "nothing remembered"

    def _goal_sequence(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Returns list of (start_timestep, goal_idx) switch points covering seq_len."""
        cfg = self.cfg
        switches = []
        t = 0
        prev_goal = None
        goal_cycle = list(range(cfg.n_targets))
        cycle_pos = 0
        while t < cfg.seq_len:
            if cfg.goal_sequence_mode == "fixed_cycle":
                goal = goal_cycle[cycle_pos % cfg.n_targets]
                cycle_pos += 1
            else:
                choices = [g for g in range(cfg.n_targets) if g != prev_goal]
                goal = rng.choice(choices) if prev_goal is not None else rng.choice(cfg.n_targets)
            switches.append((t, goal))
            prev_goal = goal
            period = int(rng.integers(cfg.goal_switch_period_range[0], cfg.goal_switch_period_range[1] + 1))
            t += period
        return switches

    def generate(self, batch_size: int, seed: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        T = cfg.seq_len
        inputs = np.zeros((batch_size, T, self.input_dim), dtype=np.float32)
        labels = np.full((batch_size, T), cfg.n_values, dtype=np.int64)  # default "nothing remembered"
        probe_mask = np.zeros((batch_size, T), dtype=bool)
        goal_trace = np.full((batch_size, T), -1, dtype=np.int64)
        # remembered value for the ACTIVE goal at EVERY timestep (class V = nothing
        # remembered). `labels` only carries it at probe steps; the knockout needs
        # the full latent state, otherwise it leaves the memory variable decodable
        # at all non-probe timesteps (that was the Phase A knockout failure).
        memory_trace = np.full((batch_size, T), cfg.n_values, dtype=np.int64)

        for b in range(batch_size):
            switches = self._goal_sequence(rng)
            switch_idx = 0
            active_goal = switches[0][1]
            remembered = np.full(cfg.n_targets, cfg.n_values, dtype=np.int64)  # per-goal memory
            t = 0
            while t < T:
                is_switch = switch_idx < len(switches) and switches[switch_idx][0] == t
                if is_switch:
                    active_goal = switches[switch_idx][1]
                    remembered[active_goal] = cfg.n_values  # reset memory on switch
                    switch_idx += 1
                    inputs[b, t, self.target_ids[active_goal]] = 1.0
                    inputs[b, t, -2] = 1.0  # is_cue
                else:
                    roll = rng.random()
                    if roll < cfg.stimulus_prob:
                        if rng.random() < cfg.target_hit_prob:
                            item = self.target_ids[active_goal]
                            value = int(rng.integers(0, cfg.n_values))
                            remembered[active_goal] = value
                            inputs[b, t, item] = 1.0
                            inputs[b, t, self.n_items + value] = 1.0
                        else:
                            pool = [i for i in (self.target_ids + self.distractor_ids) if i != self.target_ids[active_goal]]
                            item = rng.choice(pool)
                            inputs[b, t, item] = 1.0
                            if item in self.target_ids:  # off-goal target also carries a (irrelevant) value
                                value = int(rng.integers(0, cfg.n_values))
                                inputs[b, t, self.n_items + value] = 1.0
                    if rng.random() < cfg.probe_prob:
                        inputs[b, t, -1] = 1.0  # is_probe
                        probe_mask[b, t] = True
                        labels[b, t] = remembered[active_goal]
                goal_trace[b, t] = active_goal
                memory_trace[b, t] = remembered[active_goal]
                t += 1

        return {
            "inputs": torch.from_numpy(inputs),
            "labels": torch.from_numpy(labels),
            "probe_mask": torch.from_numpy(probe_mask),
            "goal_trace": torch.from_numpy(goal_trace),
            "memory_trace": torch.from_numpy(memory_trace),
        }
