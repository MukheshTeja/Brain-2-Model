# CD-B2M: Causally-Disentangled Brain-to-Model Transfer

Tests whether B2M-style brain-to-model transfer gains come from (H1)
genuine transfer of task structure or (H2) a regularization effect from
brain-like statistics, using a Null Ladder of surrogate teacher signals
(R0-R5 + structure knockout) passed through the same embedding pipeline.

## Layout

- `nullladder/` — standalone, releasable toolkit: surrogate generators
  (R0-R4), structure knockout, decodability probes, TOST equivalence
  testing, exact Shapley attribution, and the Phase A synthetic sandbox
  (goal-switching task, teacher RNN, structured/hollow brains, zero-shot
  suite).
- `cdb2m/` — experiment-specific code: students (GRU; Transformer/PPO
  scaffolded), pluggable alignment losses (linear+MSE default, CKA/RSA
  alternatives), training loop, alpha x seed sweep runner, zero-shot +
  CKA evaluation, ladder decomposition / equivalence / effect-size
  analysis. `cdb2m/fmri/` is a Phase B scaffold only.
- `configs/` — Hydra configs (task/student/loss/brain/sweep groups) plus
  `phaseA_thresholds.yaml`, the pre-registered pass/fail thresholds for
  `scripts/verify_phaseA.py`.
- `scripts/` — `run_experiment.py` (single entrypoint), `make_ladder_plot.py`,
  `verify_phaseA.py`.
- `tests/` — one test per surrogate/knockout verifying the statistics it
  claims to preserve and that task decodability (linear + MLP probe) drops
  to chance; sandbox tests for the task/teacher/brain/zero-shot generators.

## Setup

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Running

```
pytest                                   # unit tests
python -m scripts.run_experiment         # smoke sweep -> results/phaseA_smoke.csv
python -m scripts.make_ladder_plot results/phaseA_smoke.csv --alpha 0.2
python -m scripts.verify_phaseA results/phaseA_smoke.csv results/phaseA_smoke_headroom.csv --alpha 0.2
```

Override any config group from the command line, e.g.
`python -m scripts.run_experiment sweep.seeds='[0,1,2,3,4]' loss=cka`.
