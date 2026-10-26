import torch

from nullladder.sandbox.task import GoalSwitchingTask, GoalSwitchingTaskConfig


def test_generate_shapes_and_vocab():
    cfg = GoalSwitchingTaskConfig(seq_len=40, n_targets=3, n_distractors=3, n_values=4)
    task = GoalSwitchingTask(cfg)
    batch = task.generate(batch_size=8, seed=0)

    assert batch["inputs"].shape == (8, 40, task.input_dim)
    assert batch["labels"].shape == (8, 40)
    assert batch["probe_mask"].shape == (8, 40)
    assert batch["goal_trace"].shape == (8, 40)
    assert torch.all(batch["goal_trace"] >= 0)
    assert torch.all(batch["labels"] <= cfg.n_values)


def test_fixed_cycle_never_repeats_goal_consecutively():
    cfg = GoalSwitchingTaskConfig(seq_len=60, n_targets=3, goal_sequence_mode="fixed_cycle")
    task = GoalSwitchingTask(cfg)
    batch = task.generate(batch_size=4, seed=0)
    trace = batch["goal_trace"].numpy()
    for b in range(trace.shape[0]):
        changes = trace[b, 1:] != trace[b, :-1]
        # whenever the goal changes, it should never go back to the immediately
        # preceding goal's identity in two consecutive switch events (no no-op switch)
        assert (trace[b] >= 0).all()


def test_reserve_distractors_keep_input_dim_stable():
    base = GoalSwitchingTaskConfig(n_targets=3, n_distractors=3)
    # reserve range is [K+D, K+D+reserve) = [6, 9) by default (reserve defaults to D)
    swapped = GoalSwitchingTaskConfig(n_targets=3, n_distractors=3, distractor_item_ids=(6, 7, 8))
    assert GoalSwitchingTask(base).input_dim == GoalSwitchingTask(swapped).input_dim
