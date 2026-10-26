from nullladder.sandbox.task import GoalSwitchingTask, GoalSwitchingTaskConfig
from nullladder.sandbox.teacher import extract_task_variables_from_batch, train_teacher


def test_teacher_trains_and_extracts_task_variables():
    cfg = GoalSwitchingTaskConfig(seq_len=30, n_targets=3, n_distractors=3)
    task = GoalSwitchingTask(cfg)
    model = train_teacher(task, hidden_dim=16, n_steps=20, batch_size=16, seed=0)

    batch = task.generate(batch_size=10, seed=1)
    task_vars = extract_task_variables_from_batch(model, batch, n_components=4)
    assert task_vars.shape == (10, cfg.seq_len, 4)
