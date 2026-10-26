"""Exact Shapley value attribution over a small set of players (e.g. the 4
fMRI regions in RQ2). Enumerates ALL 2^n coalitions rather than leave-
one-out, as required for n=4 (16 coalitions) in the brief.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Sequence


def exact_shapley(
    players: Sequence[str],
    value_fn: Callable[[frozenset], float],
) -> dict[str, float]:
    """Exact Shapley values for `players` under cooperative game `value_fn`.

    value_fn(coalition: frozenset[str]) -> float, defined for every subset
    of `players` including the empty set. Evaluates value_fn once per
    coalition (2^n calls total) and reuses cached values across players.
    """
    n = len(players)
    all_subsets = []
    for r in range(n + 1):
        all_subsets.extend(frozenset(c) for c in combinations(players, r))
    values = {s: value_fn(s) for s in all_subsets}

    phi: dict[str, float] = {}
    for p in players:
        others = [q for q in players if q != p]
        total = 0.0
        for r in range(len(others) + 1):
            for combo in combinations(others, r):
                s = frozenset(combo)
                weight = factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n)
                total += weight * (values[s | {p}] - values[s])
        phi[p] = total
    return phi
