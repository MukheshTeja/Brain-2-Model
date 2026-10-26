from nullladder.sandbox.task import GoalSwitchingTask, GoalSwitchingTaskConfig
from nullladder.sandbox.zero_shot import make_zero_shot_suite


def test_zero_shot_suite_matches_training_input_dim():
    base_cfg = GoalSwitchingTaskConfig(seq_len=50, n_targets=3, n_distractors=3)
    base_task = GoalSwitchingTask(base_cfg)
    suite = make_zero_shot_suite(base_cfg, seed=0)

    assert set(suite.keys()) == {"longer_sequences", "fixed_cycle", "swapped_targets", "new_distractors"}
    for name, task in suite.items():
        assert task.input_dim == base_task.input_dim, name
        assert task.n_classes == base_task.n_classes, name


def test_longer_sequences_variant_is_actually_longer():
    base_cfg = GoalSwitchingTaskConfig(seq_len=50)
    suite = make_zero_shot_suite(base_cfg, seed=0)
    assert suite["longer_sequences"].cfg.seq_len == 150


def test_swapped_targets_is_a_derangement():
    base_cfg = GoalSwitchingTaskConfig(n_targets=4)
    suite = make_zero_shot_suite(base_cfg, seed=0)
    swapped_ids = suite["swapped_targets"].cfg.target_item_ids
    assert all(swapped_ids[i] != i for i in range(4))
