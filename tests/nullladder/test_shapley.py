from itertools import combinations

from nullladder.shapley import exact_shapley


def test_shapley_efficiency():
    players = ["IFG", "MFG", "parietal", "basal_ganglia"]
    weights = {"IFG": 1.0, "MFG": 2.0, "parietal": 0.5, "basal_ganglia": 1.5}

    def value_fn(coalition):
        return sum(weights[p] for p in coalition)

    phi = exact_shapley(players, value_fn)
    assert len(phi) == 4
    assert abs(sum(phi.values()) - value_fn(frozenset(players))) < 1e-9
    for p in players:
        assert abs(phi[p] - weights[p]) < 1e-9  # additive game: Shapley = own weight


def test_shapley_enumerates_all_16_coalitions():
    players = ["a", "b", "c", "d"]
    seen = []

    def value_fn(coalition):
        seen.append(coalition)
        return len(coalition)

    exact_shapley(players, value_fn)
    assert len(set(seen)) == 16


def test_shapley_symmetry():
    players = ["a", "b", "c"]

    def value_fn(coalition):
        # symmetric game: value depends only on coalition size
        return {0: 0.0, 1: 1.0, 2: 2.5, 3: 3.0}[len(coalition)]

    phi = exact_shapley(players, value_fn)
    values = list(phi.values())
    assert max(values) - min(values) < 1e-9
