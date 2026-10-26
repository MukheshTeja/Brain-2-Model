from nullladder.ladder import LadderArm


def test_ladder_arm_builds_all_rungs(synthetic_brain):
    signal = synthetic_brain["signal"]
    trials = synthetic_brain["trials"]
    labels = synthetic_brain["labels"]

    arm = LadderArm(real_signal=signal, real_trials=trials, task_labels=labels, n_pca_components=4)
    arm.build(seed=0)

    expected_rungs = {"R0", "R1", "R2", "R3", "R4", "R5", "knockout"}
    assert set(arm.rungs.keys()) == expected_rungs
    assert set(arm.embedded.keys()) == expected_rungs
    for name, embedding in arm.embedded.items():
        assert embedding.shape == (signal.shape[0], 4), name
