"""The Rust kernel (roadsearch_core) against the numba one: same edge costs, same searches."""

import math

import numpy as np
import pytest

roadsearch_core = pytest.importorskip("roadsearch_core")

from roadsearch.lattice import ACTION_BEND_L, ACTION_FILL, ACTION_HP_LEFT, ACTION_PRIM  # noqa: E402
from roadsearch.params import load_params  # noqa: E402
from roadsearch.preprocess import preprocess  # noqa: E402
from roadsearch.search import edge_cost, make_problem, run_search, rust_args  # noqa: E402

from .conftest import make_grid  # noqa: E402
from .test_search import random_terrain  # noqa: E402


@pytest.fixture(scope="module")
def prob():
    p = load_params(overrides={"C_travel": 125.5, "C_vertical": 2576.0, "C_turn": 8000.0})
    z = random_terrain(7, n=90, relief=15.0)
    g = make_grid(z, 5.0)
    pr = make_problem(preprocess(g, g.xy(5, 5), g.xy(84, 80), p), p)
    pr.lam_fill = 3.7
    assert pr.channel.any()
    return pr


def test_edge_costs_match(prob):
    rng = np.random.default_rng(1)
    L = prob.lat
    n = 200_000
    rows = rng.integers(0, prob.nr, n)
    cols = rng.integers(0, prob.nc, n)
    acts = rng.integers(0, len(L.kind), n)
    # Keep the end node on the grid (the numba kernel indexes without checks).
    ok = ((rows + L.dr[acts] >= 0) & (rows + L.dr[acts] < prob.nr)
          & (cols + L.dc[acts] >= 0) & (cols + L.dc[acts] < prob.nc))
    rows, cols, acts = rows[ok], cols[ok], acts[ok]
    lat, terr = rust_args(prob)
    got = roadsearch_core.edge_costs(prob.n_theta, lat, terr, rows, cols, acts)
    want = np.array([edge_cost(int(r), int(c), int(a), prob.lat_tuple(), prob.terr_tuple())
                     for r, c, a in zip(rows, cols, acts)])
    finite = np.isfinite(want)
    kinds = set(L.kind[acts[finite]].tolist())
    # Primitives, fill crossings, and hairpins and bends in either direction.
    assert {ACTION_PRIM, ACTION_FILL} <= kinds, kinds
    assert kinds & {ACTION_HP_LEFT, ACTION_HP_LEFT + 1} and kinds & {ACTION_BEND_L, ACTION_BEND_L + 1}, kinds
    assert finite.sum() > 1000
    assert np.array_equal(np.isfinite(got), finite)
    assert np.array_equal(got[finite], want[finite])            # bit for bit


def test_dijkstra_matches(prob):
    a = run_search(prob, (5, 5), engine="numba")
    b = run_search(prob, (5, 5), engine="rust")
    assert np.array_equal(a.G, b.G)
    assert np.array_equal(a.parent, b.parent)
    assert np.array_equal(a.closed, b.closed)
    assert a.n_expanded == b.n_expanded


def test_astar_matches(prob):
    a = run_search(prob, (5, 5), (84, 80), astar=True, engine="numba")
    b = run_search(prob, (5, 5), (84, 80), astar=True, engine="rust")
    assert math.isfinite(a.goal_cost)
    assert b.goal_cost == pytest.approx(a.goal_cost, rel=1e-12)
    assert b.goal_state == a.goal_state
    assert b.n_expanded == pytest.approx(a.n_expanded, rel=1e-3)


def test_max_cost_matches(prob):
    a = run_search(prob, (40, 40), engine="numba", max_cost=2e5)
    b = run_search(prob, (40, 40), engine="rust", max_cost=2e5)
    assert np.array_equal(a.closed, b.closed) and np.array_equal(a.G, b.G)


def test_results_and_tables_hold_only_open_cells(prob):
    from roadsearch.pipeline import crop_problem

    res = run_search(prob, (5, 5), (84, 80), astar=True)
    open_cells = np.flatnonzero(prob.open_)
    extra = {5 * prob.nc + 5, 84 * prob.nc + 80} - set(open_cells.tolist())
    assert res.cells.size == open_cells.size + len(extra)
    assert res.G.shape == (res.cells.size, prob.n_theta)
    assert (res.lookup[res.cells] == np.arange(res.cells.size)).all()
    assert (res.lookup >= 0).sum() == res.cells.size
    # One table row per cell open when the tables were built, in row-major order.
    assert prob.hp.shape[0] == prob.bend.shape[0] == int((prob.macro_lookup >= 0).sum())
    assert (np.flatnonzero(prob.macro_lookup.reshape(-1) >= 0) == np.flatnonzero(prob.open_tables)).all()
    # A cropped problem shares the tables and crops the lookup.
    sub = crop_problem(prob, 10, 60, 20, 70)
    assert sub.hp is prob.hp and sub.macro_lookup.shape == (50, 50)
    assert (sub.macro_lookup == prob.macro_lookup[10:60, 20:70]).all()
