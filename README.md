# CD-B2M: Causally-Disentangled Brain-to-Model Transfer

Do gains from aligning AI models to brain embeddings (B2M, Aquino et al. 2025)
reflect genuine transfer of computational structure (**H1**), or a regularizer
exploiting structured noise (**H2**)? We test this with a **Null Ladder** of
teacher signals that keep more and more of the brain's statistics, R0 (Gaussian,
B2M's only control) to R4 (real data with the wrong pairing), against the real
brain R5. We add a **structure knockout**, a **generalization fingerprint** and
causal analyses for RQ1–RQ3. Every experiment runs first in a sandbox with a known
answer (Phase A), then on real fMRI (Phase B).

## Quick start (CPU only, no GPU)

```bash
python -m venv .venv
.venv\Scripts\activate                   # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ~200 MB instead of ~2.5 GB CUDA build
pip install -e .

python -m scripts.download_data          # ~24 MB, pinned commits, writes data/MANIFEST.json (sha256)
python -m scripts.run_pipeline --preset quick   # ~15-20 min on an 8-core laptop
python -m scripts.make_report --preset quick    # -> results/report_quick/REPORT.md + figures

python -m scripts.run_pipeline --preset full    # deck protocol: 10 seeds (5 Transformer/PPO, 30 GRU on real data), ~4.7 CPU-hours
                                                #   = ~50 min with 6 workers
python -m scripts.make_report                   # -> results/report/REPORT.md
pytest                                          # 56 unit tests, ~30 s
```

Footprint: ~24 MB of data, <1 MB per results CSV, roughly 0.35 GB RAM per worker
(capped at 6 workers by default; set `--workers N`). Runs resume automatically:
finished jobs are appended to `results/*.csv` as they complete.

## Data (all public, downloaded by `scripts/download_data.py`)

| Dataset | What we use | Source |
|---|---|---|
| **OpenNeuro ds003076**, deductive reasoning fMRI (Lytle, Prado & Booth 2020; CC0) | trial-wise GLMsingle betas, 10 subjects × 70 problems (36 syllogisms + 34 transitive), for three ROI families: deductive-reasoning network (Prado 2011 meta-analysis spheres), multiple-demand (MD), language. Also the exact problem texts and behaviour. | betas: Xiao et al. 2026 release (github.com/pkuxmq/Brain-guided_LLM, MIT). Events/metadata: github.com/OpenNeuroDatasets/ds003076 |
| **HCP relational processing** fMRI | trial-wise betas (MD network), 19 subjects in two stimulus lists, plus the shape/texture layout of every stimulus | Xiao et al. 2026 release |
| ROI masks | `dr.nii`, MD and language parcels (MNI) | Xiao et al. 2026 release |

Raw BOLD is *not* needed. `--with-raw-bold` fetches the 10 subjects' raw NIfTIs
(~1.3 GB) from OpenNeuro S3, but only if you want to rebuild betas yourself.
The released betas are exactly the inputs to Xiao et al.'s published analyses.

## Pipeline (deck slide 6)

| Step | Phase A: sandbox (known answer) | Phase B: real fMRI |
|---|---|---|
| 1. Brain data | B2M's goal-switching memory task. A teacher GRU's task variables are mixed into 200 simulated neurons plus realistic noise (**structured brain**); the **hollow twin** has the same mixing and noise but task-free latents | ds003076 betas (DR network). Positive and negative controls: **planted** (a reasoning teacher's variables in 10 synthetic subjects), **hollow**, and **planted at real-fMRI SNR** (power check) |
| 2. Teacher signals | Null Ladder R0–R5, LEACE knockout of the task state (goal × remembered value), oracle knockout; one frozen PCA embedder per arm | the same rungs per subject (frozen per-subject PCA), pooled through **one** group-RDM to MDS pipeline for every rung |
| 3. Students | GRU · causal Transformer (ALiBi) · recurrent PPO agent, with loss = (1−α)·L_task + α·L_transfer (learned linear map + MSE) | same three students on a deductive-reasoning task built from the fMRI problems (syllogisms and transitive orderings); transfer loss on the 70 recorded problems |
| 4. Zero-shot tests | longer sequences, fixed goal cycles, swapped targets, new distractors | 4/5/6 premises, shuffled premise order, propositional logic (far transfer) |
| 5. Analysis | verdict rule, fingerprints, RQ1 mediation (α as randomised dose), RQ2 exact Shapley over 4 regions with known content, RQ3 per architecture | same, RQ2 over DR / MD / language networks, plus HCP replication |

**Verdict rule** (deck slide 5), tested inferentially across seeds: *genuine
transfer* iff R5 > every rung R0–R4 (Holm-corrected paired tests) and the knockout
falls to within one smallest-effect-of-interest (SESOI) of R3. *Regularizer* iff R5
is not above R3 and is TOST-equivalent to it. SESOI: Phase A 0.05 (pre-registered in
`configs/phaseA_thresholds.yaml`), Phase B 0.02 (set from the planted control). The
scaffold's original mean-based rule is reported alongside it; it fails the Phase B
hollow control, which is why the inferential version is used for headlines.

## Results (full preset): see `results/report/REPORT.md` and `results/report/COMPARISON.md` (vs prior work)

| Structure-specific gain R5 − R3 (GRU, 95% CI) | |
|---|---|
| Phase A structured brain (known structure) | +0.059 [+0.053, +0.066]: **genuine transfer** ✅ |
| Phase A hollow brain (no structure) | −0.001 [−0.008, +0.007]: **regularizer** ✅ |
| Phase B planted (known structure) | +0.035 [+0.023, +0.046]: **genuine transfer** ✅ |
| Phase B planted at real-fMRI SNR (power check) | +0.048 [+0.042, +0.055] |
| **ds003076 real fMRI (deductive-reasoning network)** | **+0.007 [−0.000, +0.014]**: inconclusive, at most ~29% of what genuine structure gives at this SNR, and below the SESOI |
| **HCP relational real fMRI (MD network)** | **−0.002 [−0.009, +0.005]**: **regularizer** |

The test is validated (it flags planted structure and rejects matched noise in both
phases, even at the real data's SNR). On real fMRI, B2M-style gains are explained by
the brain signal's statistics, not by transferred task structure: **H2 (regularizer)**
for HCP, and at most a negligible structure component for ds003076.

## Layout

- `nullladder/`: releasable toolkit with surrogates R0–R4, LEACE knockout, probes,
  TOST, exact Shapley, and the Phase A sandbox (task, teacher, structured, hollow
  and regional brains, zero-shot suite).
- `cdb2m/students/`: `gru.py`, `transformer.py` (causal, ALiBi), `ppo.py` (recurrent PPO).
- `cdb2m/losses/`: `linear_mse` (default), `cka`, `rsa`, `infonce` (B2M Eq. 2).
- `cdb2m/training/`: `loop.py` (fixed batch, Phase A), `online.py` (fresh problems
  plus recorded problems, supervised or PPO, Phase B).
- `cdb2m/fmri/`: `datasets.py` (loaders), `reasoning_task.py` (fMRI-matched
  syllogism/transitive task, parser of the real problems with a logic-engine check),
  `relational_task.py` (HCP), `brain_teacher.py` (real-data ladder plus group
  embedding, planted/hollow controls), `regions.py`.
- `cdb2m/experiments/`: `phase_a.py`, `phase_b.py`, `runner.py` (parallel, resumable).
- `cdb2m/analysis/core.py`: verdicts, fingerprints, mediation, Shapley.
- `scripts/`: `download_data.py`, `run_pipeline.py`, `make_report.py`; the original
  Hydra smoke entry points (`run_experiment.py`, `verify_phaseA.py`, `make_ladder_plot.py`) still work.
- `results/`: CSVs, `report*/REPORT.md`, figures, `verdicts.json`.

## Known limitations

- The released betas carry **no voxel coordinates**, so RQ2 on real data attributes
  the gain across the three ROI *networks* (DR, MD, language) rather than IFG, MFG,
  parietal cortex and basal ganglia separately. The 4-region version needs the raw
  BOLD (`--with-raw-bold`) plus SPM/fMRIPrep-style preprocessing.
- Xiao et al. selected voxels separately for syllogism and transitive problems, so
  the two tasks are separate alignment blocks throughout.
- Problems are encoded one statement per timestep (a small-model, CPU-budget
  choice); the logic is identical to the word-level problems, which are parsed and
  label-checked by an independent logic engine.
