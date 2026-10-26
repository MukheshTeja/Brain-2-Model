"""Single entrypoint: `python -m scripts.run_experiment [+sweep=smoke ...]`
reruns any experiment. Hydra composes task/student/loss/brain/sweep
configs; this script just translates the merged config into the plain
dataclasses training/sweep.py expects and writes the tidy CSV.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from nullladder.sandbox.task import GoalSwitchingTaskConfig
from cdb2m.training.sweep import PhaseAConfig, run_phase_a


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    task_cfg = GoalSwitchingTaskConfig(
        seq_len=cfg.task.seq_len,
        n_targets=cfg.task.n_targets,
        n_distractors=cfg.task.n_distractors,
        n_values=cfg.task.n_values,
        goal_switch_period_range=tuple(cfg.task.goal_switch_period_range),
        probe_prob=cfg.task.probe_prob,
        stimulus_prob=cfg.task.stimulus_prob,
        target_hit_prob=cfg.task.target_hit_prob,
        goal_sequence_mode=cfg.task.goal_sequence_mode,
    )
    phase_a_cfg = PhaseAConfig(
        task_cfg=task_cfg,
        n_train_trials=cfg.sweep.n_train_trials,
        data_seed=cfg.sweep.data_seed,
        teacher_hidden_dim=cfg.sweep.teacher_hidden_dim,
        teacher_train_steps=cfg.sweep.teacher_train_steps,
        n_task_variables=cfg.sweep.n_task_variables,
        n_neurons=cfg.brain.n_neurons,
        mixing_type=cfg.brain.mixing_type,
        student_hidden_dim=cfg.student.hidden_dim,
        n_pca_components=cfg.sweep.n_pca_components,
        alignment_loss_name=cfg.loss.name,
        alphas=tuple(cfg.sweep.alphas),
        seeds=tuple(cfg.sweep.seeds),
        n_train_epochs=cfg.sweep.n_train_epochs,
        lr=cfg.sweep.lr,
        n_eval_sequences=cfg.sweep.n_eval_sequences,
        eraser=cfg.sweep.eraser,
        headroom_alpha=cfg.sweep.headroom_alpha,
        headroom_epochs=cfg.sweep.headroom_epochs,
    )

    results_path = Path(hydra.utils.get_original_cwd()) / cfg.results_csv
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df = run_phase_a(phase_a_cfg, results_csv_path=str(results_path))
    print(f"Wrote {len(df)} rows to {results_path}")


if __name__ == "__main__":
    main()
