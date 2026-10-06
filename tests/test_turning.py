"""The turning cost: C_turn per radian of change in heading, charged on every edge."""

import math

import numpy as np
import pytest

from roadsearch.lattice import ACTION_HP_LEFT, ACTION_HP_RIGHT, ACTION_PRIM, heading_angles
from roadsearch.params import load_params
from roadsearch.preprocess import preprocess
from roadsearch.route import evaluate_route
from roadsearch.search import make_problem, path_cost, reconstruct, reverse_edges, run_search, single_edge_cost

from .conftest import make_grid
from .test_search import random_terrain


def problem(p, z, s=(6, 6), g=(70, 66), hairpins=True):
    grid = make_grid(z, 5.0)
    return make_problem(preprocess(grid, grid.xy(*s), grid.xy(*g), p), p, hairpins=hairpins)


def test_turn_angles():
    p = load_params()
    prob = problem(p, random_terrain(2, n=80, relief=3.0))
    L = prob.lat
    ang = heading_angles(p.N_theta)
    prim = np.flatnonzero(L.kind == ACTION_PRIM)
    d = (ang[L.h1[prim]] - ang[L.h0[prim]] + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(L.turn[prim], np.abs(d))
    assert (L.turn[L.h0 == L.h1] == 0).all()                       # straights pay nothing
    hp = np.isin(L.kind, [ACTION_HP_LEFT, ACTION_HP_RIGHT])
    assert hp.any() and np.allclose(L.turn[hp], np.pi)
    assert np.allclose(L.turn, L.turn[L.rev])                        # the same both ways


def test_turn_cost_is_symmetric_and_added_per_radian():
    z = random_terrain(2, n=80, relief=3.0)
    p0 = load_params()
    p1 = load_params(overrides={"C_turn": 50.0})
    a, b = problem(p0, z), problem(p1, z)
    L = b.lat
    n = 0
    for e in range(len(L.h0)):
        r, c = 40, 40
        c0, c1 = single_edge_cost(a, r, c, e), single_edge_cost(b, r, c, e)
        if not math.isfinite(c0):
            continue
        assert c1 == pytest.approx(c0 + 50.0 * L.turn[e], rel=1e-12)
        assert c1 == single_edge_cost(b, r + L.dr[e], c + L.dc[e], L.rev[e])
        n += 1
    assert n > 100


def test_astar_stays_exact_and_routes_straighten():
    z = random_terrain(5, n=80, relief=4.0)
    s, g = (6, 6), (70, 66)
    turning = []
    for c_turn in (0.0, 400.0):
        p = load_params(overrides={"C_turn": c_turn})
        prob = problem(p, z, s, g)
        dij = run_search(prob, s, g)
        ast = run_search(prob, s, g, astar=True)
        assert ast.goal_cost == pytest.approx(dij.goal_cost, rel=1e-9)
        edges = reconstruct(ast, prob.lat, ast.goal_state)
        assert path_cost(prob, edges) == pytest.approx(path_cost(prob, reverse_edges(edges, prob.lat)), rel=1e-12)
        segs = evaluate_route(prob, edges)
        t = sum(sg.turning for sg in segs)
        assert sum(sg.cost_operating for sg in segs) == pytest.approx(c_turn * t, rel=1e-9, abs=1e-9)
        turning.append(t)
    assert turning[1] < turning[0]
