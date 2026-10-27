# CD-B2M vs previous work

## 1. What each prior paper would have concluded on our data

Every prior paper validated brain transfer against one weak control. We replayed each
paper's control on our runs (GRU, α = 0.2, structure-specific gain = R5 minus control,
one-sided paired t-test across seeds, * = p < 0.05). The two **hollow** rows have no
task structure by construction, so any "transfer" verdict there is a false positive.

| teacher signal (ground truth) | B2M '25 / Xiao '26 control: Gaussian / random signal (R0) | Li '19 control: shuffled pairing (R4) | **Ours: max-entropy null (R3) + full ladder** |
|---|---|---|---|
| Phase A structured (structure) | +0.043 * | +0.073 * | +0.059 * → GENUINE TRANSFER ✅ |
| Phase A hollow (none) | −0.019 | +0.003 | −0.001 → REGULARIZER ✅ |
| Phase B planted (structure) | +0.047 * | +0.046 * | +0.035 * → GENUINE TRANSFER ✅ |
| **Phase B hollow (none)** | **+0.022 \* (false positive ❌)** | **+0.011 \* (false positive ❌)** | +0.007 (n.s.) → REGULARIZER ✅ |
| Real fMRI ds003076, DR network (n = 30) | +0.006 | +0.003 | +0.007 (p = 0.03 uncorrected, fails Holm): INCONCLUSIVE |
| Real fMRI HCP relational, MD (n = 30) | −0.003 | −0.001 | −0.002 → REGULARIZER |

**Key result.** On a brain signal that contains no task structure but does have realistic
statistics, the controls used by B2M (Gaussian noise) and Li et al. (shuffled pairing)
both report significant "brain transfer". Xiao et al.'s random-signal baseline is the
same kind of control as R0. Only the Null Ladder, with the max-entropy rung and the
knockout, rejects it. This is the confound the deck set out to expose (H1 vs H2), now
shown empirically: the controls in the literature cannot tell structure transfer from a
structured-noise regularizer.

## 2. Feature comparison (deck slide 8, now backed by an implementation)

| | B2M '25 | Li '19 | Xiao '26 | Sucholutsky '23 | Goldstein '22 | **Ours** |
|---|---|---|---|---|---|---|
| Brain-statistics-matched null | ✗ (Gaussian) | partial (shuffle) | ✗ (random signal) | ✗ | phase-shuffle, no training | **✓ R0–R4 ladder + knockout** |
| Null shown to avoid false positives | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ hollow controls, both phases** |
| Zero-shot / new-task tests | ✗ (iid test) | partial (noise robustness) | ✓ | ✓ (few-shot) | ✗ | **✓ + per-variant fingerprint** |
| Causal test of what transfers | ✗ | ✗ | partial | ✗ (correlational) | ✗ | **✓ dose-randomised mediation + knockout** |
| Known-answer validation (sandbox) | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ Phase A + planted Phase B + power check** |
| Region attribution | ✗ | ✗ | ✓ (ROI comparison) | ✗ | ✓ | **✓ exact Shapley (networks)** |
| RNN + Transformer + RL | partial (RNN, VAE) | ✗ (CNN) | ✗ (LLMs) | ✗ | ✗ | ✓ (GRU validated; Transformer/PPO need more seeds) |
| Compute | GPU | GPU | multi-GPU, 7–72B LLMs | GPU | GPU | **CPU laptop, ~24 MB data** |

## 3. Headline gains, side by side (not like-for-like)

| work | reported gain from brain data | control used |
|---|---|---|
| B2M '25 | memory-task RNN accuracy 0.967 → 0.987 (+2.0 pts); VAE loss 0.080 → 0.071 | Gaussian noise (did worse than no transfer) |
| Li '19 | CIFAR accuracy under strongest noise ≈ 20% → ≈ 50% | shuffled similarity matrix and VGG features (also helped "to some degree") |
| Xiao '26 | up to +13.2 pts (Mistral-7B, propositional, NARF + labels) | label-only fine-tuning (NARF); random signals / random directions (NARI) |
| **Ours, synthetic structure** | +4.5 pts zero-shot (Phase A), +4.7 pts (Phase B planted); +5.9 / +3.5 pts beyond R3 | full Null Ladder |
| **Ours, real fMRI** | +0.9 pts zero-shot (ds003076 and HCP) vs no transfer, but **≤ +1.4 pts (upper 95% CI) beyond R3** on ds003076 and **≈ 0** on HCP | full Null Ladder |

These rows differ in data, task, model scale and metric, so the numbers cannot be ranked
against each other. In particular, we did not re-run B2M's intracranial data (not public),
Li's mouse V1 data or Xiao's LLMs. What the comparison does show: gains of the size
reported before can appear even when there is no structure to transfer (the hollow row in §1).

## 4. Is our approach leading?

- **On what can be concluded: yes.** It is the only approach here that is validated
  against known answers. It detects planted structure even at the real data's SNR
  (+0.048, p ≈ 5 × 10⁻¹⁶, 30 seeds) and, unlike every prior control, does not produce false positives on
  structure-free brains. It also covers the most dimensions of the comparison (§2).
- **On raw accuracy gains: no, and that is not the goal.** Our students are small CPU
  models trained from scratch, so their absolute gains are smaller than LLM-scale results.
  The finding is that on real fMRI the structure-specific part of the gain is at most about
  1.4 pts (ds003076) or about 0 (HCP). Most of what a B2M-style analysis would credit to
  the brain is reproduced by statistics-matched noise, which supports **H2 (regularizer)**.
- **Where prior work is ahead:** model scale and real-world tasks (Xiao: LLMs on FOLIO;
  Li: CIFAR adversarial robustness); invasive recordings with higher SNR (B2M); and
  voxel-level regions (Xiao), where our RQ2 runs at the network level because the public
  betas lack coordinates.
- **Next step for a fair head-to-head:** apply the Null Ladder to Xiao et al.'s NARF
  pipeline (same ds003076 betas, an LLM student). The code here plugs in directly: their
  brain targets become R5, and R0–R4 and the knockout are built by `cdb2m/fmri/brain_teacher.py`.
