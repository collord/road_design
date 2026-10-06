"""Alternatives from plateaus: stretches that are the cheapest path from both ends."""

import numpy as np
import pytest

from roadsearch.params import load_params
from roadsearch.pipeline import (
    _alternatives_plateau, corridor_fields, locally_optimal, plateaus, route_through_state,
)
from roadsearch.preprocess import preprocess
from roadsearch.search import make_problem, path_cost

from .conftest import make_grid
from .test_search import random_terrain

S, G = (6, 6), (74, 70)


def setup(seed=3, relief=6.0, **over):
    p = load_params(overrides={"K_alt": 4, **over})
    z = random_terrain(seed, n=80, relief=relief)
    g = make_grid(z, 5.0)
    prob = make_problem(preprocess(g, g.xy(*S), g.xy(*G), p), p, hairpins=False)
    return prob, corridor_fields(prob, S, G)


def test_best_route_is_one_plateau():
    prob, fields = setup()
    pl = plateaus(prob, fields, fields.F_min * (1 + 1e-9), 0.0)
    assert len(pl["head"]) >= 1
    assert pl["route_cost"][0] == pytest.approx(fields.F_min, rel=1e-12)
    assert pl["cost"][0] == pytest.approx(fields.F_min, rel=1e-9)    # the whole best route


@pytest.mark.parametrize("seed", [3, 8])
def test_plateau_alternatives_are_locally_optimal(seed):
    prob, fields = setup(seed, alt_plateau=0.1)
    alts = _alternatives_plateau(prob, fields)
    assert len(alts) >= 2
    L = prob.lat
    for a in alts[1:]:
        assert path_cost(prob, a.edges) == pytest.approx(a.F, rel=1e-9)
        assert a.F <= (1 + prob.params.eps_alt) * fields.F_min * (1 + 1e-9)
        pc = a.local_check["plateau_cost"]
        assert pc >= prob.params.alt_plateau * fields.F_min * (1 - 1e-12)
        # The via node is the plateau's middle: any window inside the plateau is a cheapest path,
        # so the original method's check passes with a window of just under half the plateau.
        nodes = [S] + [(r + int(L.dr[e]), c + int(L.dc[e])) for r, c, e in a.edges]
        via_idx = nodes.index(a.via)
        ok, info = locally_optimal(prob, a.edges, via_idx, 0.45 * pc, 1e-9)
        assert ok, info
    # Distinct routes: none shares more than half its length with an earlier one.
    for i, a in enumerate(alts):
        for b in alts[:i]:
            assert len(set(a.edges) & set(b.edges)) <= 0.5 * len(a.edges) + 1


def test_route_through_a_plateau_state_costs_its_route_cost():
    prob, fields = setup()
    pl = plateaus(prob, fields, 1.25 * fields.F_min, 0.05 * fields.F_min)
    for i in range(min(5, len(pl["head"]))):
        edges = route_through_state(prob, fields, int(pl["tail"][i]))
        assert path_cost(prob, edges) == pytest.approx(pl["route_cost"][i], rel=1e-9)
