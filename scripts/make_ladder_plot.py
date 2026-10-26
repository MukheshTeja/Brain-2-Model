"""Render the first ladder plot: mean zero-shot accuracy per rung, one line
per brain (structured vs hollow), at a chosen alpha. Usage:

    python -m scripts.make_ladder_plot results/phaseA_smoke.csv --alpha 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nullladder.surrogates import RUNG_ORDER

RUNG_PLOT_ORDER = RUNG_ORDER + ["knockout"]


def make_ladder_plot(results_csv: str, alpha: float, out_path: str) -> None:
    df = pd.read_csv(results_csv)
    df = df[df["alpha"] == alpha]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"structured": "#2563eb", "hollow": "#9ca3af"}
    for brain in ["structured", "hollow"]:
        subset = df[df["brain"] == brain]
        means = subset.groupby("rung")["value"].mean().reindex(RUNG_PLOT_ORDER)
        sems = subset.groupby("rung")["value"].sem().reindex(RUNG_PLOT_ORDER)
        ax.errorbar(
            RUNG_PLOT_ORDER, means.values, yerr=sems.values,
            marker="o", label=brain, color=colors[brain], capsize=3,
        )

    ax.set_xlabel("Null Ladder rung")
    ax.set_ylabel("mean zero-shot accuracy (across variants, seeds)")
    ax.set_title(f"Null Ladder — alpha={alpha}")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", default="results/ladder_plot.png")
    args = parser.parse_args()
    make_ladder_plot(args.results_csv, args.alpha, args.out)
