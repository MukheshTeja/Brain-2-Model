"""Knockout vs R3 equivalence test (the verdict rule's second clause): if
removing task-decodable directions from R5 and rescaling to match R3's
total variance/spectrum leaves performance statistically equivalent to R3,
a drop from R5 to knockout is attributable to removed TASK STRUCTURE, not
to a weaker signal.
"""
from __future__ import annotations

import pandas as pd

from nullladder.equivalence import TOSTResult, tost_equivalence


def knockout_vs_r3_equivalence(
    results: pd.DataFrame,
    brain: str,
    alpha: float,
    margin: float,
    variant: str | None = None,
    metric_col: str = "value",
    knockout_rung: str = "knockout",
) -> TOSTResult:
    """results must have matching seeds for `knockout_rung` and 'R3' rows
    (paired TOST across seeds). If `variant` is given, restricts to that
    zero-shot variant; otherwise averages across variants per seed first.
    `knockout_rung` defaults to "knockout" (the LEACE-based rung) but can be
    set to "KO_oracle" to run the identical comparison against the
    ground-truth oracle knockout, as a diagnostic upper bound.

    Sign convention: TOSTResult.mean_diff = mean(knockout_rung) - mean(R3).
    Positive means the knockout rung scores HIGHER than R3, i.e. it is
    retaining MORE transfer-relevant structure than R3 — the opposite of
    what "knockout ~ R3" is supposed to show.
    """
    subset = results[(results["brain"] == brain) & (results["alpha"] == alpha)]
    if variant is not None:
        subset = subset[subset["variant"] == variant]

    per_seed = subset.groupby(["rung", "seed"])[metric_col].mean().reset_index()
    knockout = per_seed[per_seed["rung"] == knockout_rung].sort_values("seed")
    r3 = per_seed[per_seed["rung"] == "R3"].sort_values("seed")

    common_seeds = sorted(set(knockout["seed"]) & set(r3["seed"]))
    knockout_vals = knockout.set_index("seed").loc[common_seeds, metric_col].to_numpy()
    r3_vals = r3.set_index("seed").loc[common_seeds, metric_col].to_numpy()

    return tost_equivalence(knockout_vals, r3_vals, margin=margin, paired=True)


def hollow_r5_vs_r4_equivalence(
    results: pd.DataFrame,
    alpha: float,
    margin: float,
    variant: str | None = None,
    metric_col: str = "value",
) -> TOSTResult:
    """H2 check: in the hollow brain, real data (R5) should be equivalent to
    stimulus-shuffled real data (R4) — there's no genuine task structure to
    lose by shuffling, so R5 shouldn't beat R4 either."""
    subset = results[(results["brain"] == "hollow") & (results["alpha"] == alpha)]
    if variant is not None:
        subset = subset[subset["variant"] == variant]

    per_seed = subset.groupby(["rung", "seed"])[metric_col].mean().reset_index()
    r5 = per_seed[per_seed["rung"] == "R5"].sort_values("seed")
    r4 = per_seed[per_seed["rung"] == "R4"].sort_values("seed")

    common_seeds = sorted(set(r5["seed"]) & set(r4["seed"]))
    r5_vals = r5.set_index("seed").loc[common_seeds, metric_col].to_numpy()
    r4_vals = r4.set_index("seed").loc[common_seeds, metric_col].to_numpy()

    return tost_equivalence(r5_vals, r4_vals, margin=margin, paired=True)
