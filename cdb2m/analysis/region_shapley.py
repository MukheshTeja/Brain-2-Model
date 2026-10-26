"""RQ2 scaffold: exact Shapley attribution over the 4 fMRI regions (IFG,
MFG, parietal, basal ganglia — see cdb2m/fmri/regions.py), all 16
coalitions, per architecture (RQ3). Wraps nullladder.shapley.exact_shapley
with a value function specific to this experiment: zero-shot gain (or
CKA-to-brain) achieved when only the regions in a given coalition are kept
in the teacher embedding (other regions masked/zeroed before mixing).

Requires Phase B's region masks and reasoning task; not runnable in
milestone 1.
"""
from __future__ import annotations

from typing import Callable

from nullladder.shapley import exact_shapley

REGIONS = ("IFG", "MFG", "parietal", "basal_ganglia")


def region_shapley(value_fn: Callable[[frozenset], float]) -> dict[str, float]:
    """value_fn(coalition_of_region_names) -> a performance metric (e.g.
    zero-shot gain) for training with only those regions' signal retained.
    Caller is responsible for building value_fn from a trained-student
    evaluation per coalition (expensive: 16 coalitions x architecture)."""
    return exact_shapley(REGIONS, value_fn)
