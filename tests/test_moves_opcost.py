"""Binned straight moves (climbing near g_max on any slope) and lifetime
operating costs (C_travel, C_vertical)."""

import math

import numpy as np
import pytest

from roadsearch.lattice import binned_moves, heading_angles, lattice_arrays
from roadsearch.masshaul import true_cost
from roadsearch.params import load_params
from roadsearch.preprocess import preprocess
from roadsearch.route import evaluate_route
from roadsearch.search import make_problem, reconstruct, reverse_edges, run_search

from .conftest import make_grid
from .test_search import random_terrain

RES = 5.0


# --- binned moves -----------------------------------------------------------------

@pytest.mark.parametrize("n_theta", [8, 16, 32])
def test_binned_moves_respect_the_kink_limit(n_theta):
    ths = heading_angles(n_theta)
    moves = binned_moves(n_theta, 7, 10.0)
    assert moves
    for h0, X, Y, h1 in moves:
        assert math.gcd(X, Y) == 1 and max(abs(X), abs(Y)) <= 7
        a = math.atan2(Y, X)
        d = (a - ths + math.pi) % (2 * math.pi) - math.pi
        near = int(np.argmin(np.abs(d)))
        assert abs(d[h0]) <= math.radians(10.0) + 1e-12 and abs(d[h1]) <= math.radians(10.0) + 1e-12
        assert near in (h0, h1)
    L = lattice_arrays(n_theta, 4.0, move_max_cells=7)
    assert np.array_equal(L.rev[L.rev], np.arange(len(L.rev)))
    assert len(L.prims) > len(lattice_arrays(n_theta, 4.0).prims)


def test_every_pair_of_neighboring_headings_is_linked_by_a_move():
    moves = binned_moves(32, 7, 10.0)
    links = {(h0, h1) for h0, _, _, h1 in moves}
    for h in range(32):
        assert (h, (h + 1) % 32) in links and ((h + 1) % 32, h) in links


def climb_rate_between_hairpins(move_max_cells, S=0.25):
    p = load_params(overrides={"move_max_cells": move_max_cells})
    rows, cols = np.mgrid[0:200, 0:50]
    z = 1000 - S * rows * RES                       # rising north
    g = make_grid(z, RES)
    start, goal = (185, 25), (15, 25)
    t = preprocess(g, g.xy(*start), g.xy(*goal), p)
    prob = make_problem(t, p)
    res = run_search(prob, start, goal, astar=True)
    edges = reconstruct(res, prob.lat, res.goal_state)
    segs = evaluate_route(prob, edges)
    # Straight moves only: turn primitives curve through flatter headings by geometry.
    straight = [s for (_, _, a), s in zip(edges, segs)
                if prob.lat.kind[a] == 0 and math.isinf(prob.lat.prims[a].radius)]
    return sum(abs(s.grade) * s.length for s in straight) / sum(s.length for s in straight), res.goal_cost


def test_binned_moves_climb_near_g_max():
    """On a uniform 0.25 slope exact headings can only climb at 0.061 (or at
    0.112, over the limit); binned moves get close to g_max and the route is cheaper."""
    p = load_params()
    exact_rate, exact_cost = climb_rate_between_hairpins(0)
    moves_rate, moves_cost = climb_rate_between_hairpins(7)
    assert exact_rate < 0.07
    assert moves_rate > 0.09 and moves_rate <= p.g_max + 1e-9
    assert moves_cost < exact_cost


# --- operating costs ----------------------------------------------------------------

def test_flat_plane_operating_cost():
    p = load_params(overrides={"C_travel": 3.0, "C_vertical": 7.0})
    g = make_grid(np.full((60, 120), 1000.0), RES)
    t = preprocess(g, g.xy(30, 10), g.xy(30, 110), p)
    prob = make_problem(t, p)
    res = run_search(prob, (30, 10), (30, 110), astar=True)
    segs = evaluate_route(prob, reconstruct(res, prob.lat, res.goal_state))
    tc = true_cost(segs, p)
    assert res.goal_cost == pytest.approx((p.C_base + p.C_travel) * 500.0, rel=1e-12)
    assert tc.operating == pytest.approx(p.C_travel * 500.0) and tc.construction == pytest.approx(p.C_base * 500.0)


def test_operating_costs_keep_symmetry_and_astar_exactness():
    p = load_params(overrides={"C_travel": 2.0, "C_vertical": 30.0})
    g = make_grid(random_terrain(5, relief=5.0), RES)
    a, b = (12, 12), (78, 60)
    t = preprocess(g, g.xy(*a), g.xy(*b), p)
    prob = make_problem(t, p)
    dij = run_search(prob, a, b)
    ast = run_search(prob, a, b, astar=True)
    back = run_search(prob, b, a)
    opt = float(dij.min_over_headings()[b])
    assert ast.goal_cost == pytest.approx(opt, rel=1e-9)
    assert float(back.min_over_headings()[a]) == pytest.approx(opt, rel=1e-12)
    e = reconstruct(dij, prob.lat, dij.goal_state)
    segs = evaluate_route(prob, e)
    assert true_cost(segs, p).total == pytest.approx(opt, rel=2e-3)
    assert sum(s.cost_operating for s in segs) == pytest.approx(
        p.C_travel * sum(s.length for s in segs) + p.C_vertical * sum(s.vertical_travel for s in segs))


def test_vertical_cost_trades_length_for_less_rise_and_fall():
    rows, cols = np.mgrid[0:100, 0:120]
    z = 1000 + 15 * np.exp(-((cols - 60) / 8.0) ** 2) * (rows < 70)   # ridge with a flat way round
    g = make_grid(z, RES)
    s, gl = (30, 10), (30, 110)
    out = []
    for cv in (0.0, 5.0, 50.0):
        p = load_params(overrides={"C_vertical": cv, "C_travel": 2.0})
        t = preprocess(g, g.xy(*s), g.xy(*gl), p)
        prob = make_problem(t, p)
        res = run_search(prob, s, gl, astar=True)
        segs = evaluate_route(prob, reconstruct(res, prob.lat, res.goal_state))
        out.append((sum(x.length for x in segs), sum(x.vertical_travel for x in segs)))
    lengths, verts = zip(*out)
    assert list(verts) == sorted(verts, reverse=True) and verts[-1] < verts[0]
    assert list(lengths) == sorted(lengths) and lengths[-1] > lengths[0]
