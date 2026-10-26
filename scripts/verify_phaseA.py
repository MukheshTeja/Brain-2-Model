"""Phase A pass criterion (explicit, pre-registered — see configs/phaseA_thresholds.yaml):

The full pipeline must flag the STRUCTURED brain as transferring and the
HOLLOW brain as NOT transferring. Checked in this order, each gating the
next:

  0. Headroom: structured R5 must beat R0 by >= headroom_min_margin
     (otherwise nothing below can be trusted).
  1. Structured R5 beats every other rung (R0-R4) on mean zero-shot
     accuracy, with R5 vs R4 additionally required to clear a minimum
     paired Cohen's d across seeds.
  2. Structured knockout is statistically equivalent to structured R3
     (paired TOST within a pre-registered margin).
  3. Hollow R5 is statistically equivalent to hollow R4 (paired TOST) —
     i.e. the hollow brain shows no genuine transfer to lose by shuffling.

Usage: python -m scripts.verify_phaseA results/phaseA_smoke.csv results/phaseA_smoke_headroom.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
from omegaconf import OmegaConf

from cdb2m.analysis.effect_size import rung_vs_rung_effect_size
from cdb2m.analysis.knockout_equivalence import hollow_r5_vs_r4_equivalence, knockout_vs_r3_equivalence

RUNGS_BELOW_R5 = ["R0", "R1", "R2", "R3", "R4"]

ALL_RUNG_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "knockout", "KO_oracle"]


def _print_rung_means(results: pd.DataFrame, brain: str, alpha: float) -> None:
    """Absolute mean zero-shot accuracy per rung — printed unconditionally so
    a TOST pass/fail can always be read against the raw numbers, not just
    the diff/p-value. TOST mean_diff below is signed x - y as passed to
    tost_equivalence (e.g. knockout - R3): POSITIVE means the first-named
    rung (knockout) scores HIGHER than the second (R3), i.e. knockout is
    retaining more transfer-relevant structure than R3, not less."""
    subset = results[(results.brain == brain) & (results.alpha == alpha)]
    means = subset.groupby("rung")["value"].mean()
    print(f"  Absolute mean zero-shot accuracy, brain={brain}, alpha={alpha}:")
    for rung in ALL_RUNG_ORDER:
        if rung in means.index:
            print(f"    {rung:10s} {means[rung]:.4f}")


def verify_phase_a(results_csv: str, headroom_csv: str, thresholds_path: str, alpha: float) -> bool:
    results = pd.read_csv(results_csv)
    headroom = pd.read_csv(headroom_csv)
    thresholds = OmegaConf.load(thresholds_path)

    print("=== Phase A verification ===")
    _print_rung_means(results, "structured", alpha)
    _print_rung_means(results, "hollow", alpha)
    all_passed = True

    headroom_mean_margin = (headroom["r5_gain"] - headroom["r0_gain"]).mean()
    headroom_passed = bool(headroom_mean_margin >= thresholds.headroom_min_margin)
    print(f"[0] Headroom (mean R5-R0 gain across seeds): {headroom_mean_margin:.4f} "
          f"(need >= {thresholds.headroom_min_margin}) -> {'PASS' if headroom_passed else 'FAIL'}")
    all_passed &= headroom_passed

    print("[1] Structured R5 vs every other rung (mean zero-shot accuracy):")
    structured_beats_all = True
    for rung in RUNGS_BELOW_R5:
        r5_mean = results[(results.brain == "structured") & (results.alpha == alpha) & (results.rung == "R5")]["value"].mean()
        rung_mean = results[(results.brain == "structured") & (results.alpha == alpha) & (results.rung == rung)]["value"].mean()
        beats = r5_mean > rung_mean
        structured_beats_all &= beats
        print(f"    R5 ({r5_mean:.4f}) vs {rung} ({rung_mean:.4f}): {'PASS' if beats else 'FAIL'}")

    effect = rung_vs_rung_effect_size(
        results, "structured", alpha, "R5", "R4", thresholds.min_effect_size_r5_vs_r4
    )
    print(f"    R5 vs R4 paired Cohen's d: {effect.cohens_d:.3f} "
          f"(need > {thresholds.min_effect_size_r5_vs_r4}) -> {'PASS' if effect.passed else 'FAIL'}")
    structured_beats_all &= effect.passed
    print(f"    -> clause 1: {'PASS' if structured_beats_all else 'FAIL'}")
    all_passed &= structured_beats_all

    knockout_eq = knockout_vs_r3_equivalence(results, "structured", alpha, thresholds.knockout_vs_r3_margin)
    print(f"[2] Structured knockout vs R3 TOST: mean_diff={knockout_eq.mean_diff:.4f}, "
          f"p_tost={knockout_eq.p_tost:.4f} (margin={thresholds.knockout_vs_r3_margin}) "
          f"-> {'PASS (equivalent)' if knockout_eq.equivalent else 'FAIL (not equivalent)'}")
    all_passed &= knockout_eq.equivalent

    # DIAGNOSTIC ONLY — does not gate all_passed. KO_oracle projects out the
    # sandbox's known ground-truth mixing exactly; comparing it to R3 gives
    # an upper bound on how close a PERFECT knockout could get, isolating
    # "is LEACE's erasure incomplete" from "is 0.05 too tight a margin."
    if "KO_oracle" in results["rung"].unique():
        oracle_eq = knockout_vs_r3_equivalence(
            results, "structured", alpha, thresholds.knockout_vs_r3_margin, knockout_rung="KO_oracle"
        )
        print(f"[2b, diagnostic] Structured KO_oracle vs R3 TOST: mean_diff={oracle_eq.mean_diff:.4f}, "
              f"p_tost={oracle_eq.p_tost:.4f} (same margin={thresholds.knockout_vs_r3_margin}) "
              f"-> {'equivalent' if oracle_eq.equivalent else 'not equivalent'} "
              f"(not part of the pass/fail gate)")

    hollow_eq = hollow_r5_vs_r4_equivalence(results, alpha, thresholds.hollow_r5_vs_r4_margin)
    print(f"[3] Hollow R5 vs R4 TOST: mean_diff={hollow_eq.mean_diff:.4f}, "
          f"p_tost={hollow_eq.p_tost:.4f} (margin={thresholds.hollow_r5_vs_r4_margin}) "
          f"-> {'PASS (equivalent)' if hollow_eq.equivalent else 'FAIL (not equivalent)'}")
    all_passed &= hollow_eq.equivalent

    print("=== Verdict ===")
    print(f"Structured brain flagged as TRANSFERRING: {structured_beats_all and knockout_eq.equivalent}")
    print(f"Hollow brain flagged as NOT transferring: {hollow_eq.equivalent}")
    print(f"PHASE A PASS CRITERION: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("headroom_csv")
    parser.add_argument("--thresholds", default="configs/phaseA_thresholds.yaml")
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()
    passed = verify_phase_a(args.results_csv, args.headroom_csv, args.thresholds, args.alpha)
    sys.exit(0 if passed else 1)
