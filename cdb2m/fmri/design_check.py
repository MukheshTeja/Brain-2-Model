"""Phase B scaffold. R4 (stimulus-shuffle) must respect the dataset's design:
shuffling at the wrong level (e.g. permuting individual trials within a
block-design study) would create artifacts unrelated to the Null Ladder's
intent. This inspects the events file and reports which level R4 should
shuffle at, BEFORE any surrogate is built.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DesignCheckResult:
    level: str  # "trial" | "block"
    n_units: int
    reason: str


def check_design_level(events: pd.DataFrame, min_trials_per_block: int = 2) -> DesignCheckResult:
    """events: dataframe with columns {onset, duration, trial_type, block_id}.

    Heuristic: if trials within the same block_id share very similar
    durations/spacing and there are multiple trials per block, this is a
    block design and R4 must shuffle at the block level (permuting blocks,
    never individual trials within a block). Otherwise it's event-related
    and R4 shuffles individual trial pairings directly.
    """
    required = {"onset", "duration", "trial_type", "block_id"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events file missing required columns: {missing}")

    trials_per_block = events.groupby("block_id").size()
    if (trials_per_block >= min_trials_per_block).mean() > 0.8:
        return DesignCheckResult(
            level="block",
            n_units=events["block_id"].nunique(),
            reason=f"median {trials_per_block.median():.0f} trials/block across "
                   f"{events['block_id'].nunique()} blocks -> shuffle at block level",
        )
    return DesignCheckResult(
        level="trial",
        n_units=len(events),
        reason="most blocks contain a single trial -> event-related design, "
               "shuffle individual trial pairings directly",
    )
