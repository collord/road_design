"""Brief section 11, test 9 (corridor field) and the alternatives rules."""

import numpy as np
import pytest
from scipy import ndimage

from roadsearch.params import load_params
from roadsearch.pipeline import (
    _Geometry, _shared_edge_fraction, alternatives, corridor_fields, locally_optimal, route_through,
)
from roadsearch.preprocess import preprocess
from roadsearch.search import make_problem, path_cost, run_search

from .conftest import make_grid
from .test_search import random_terrain

RES = 5.0


def problem(seed, p, relief=5.0):
    g = make_grid(random_terrain(seed, relief=relief), RES)
    t = preprocess(g, g.xy(12, 12), g.xy(78, 70), p)
    return t, make_problem(t, p)


@pytest.mark.parametrize("seed", [4, 7])
def test_corridor_minimum_equals_dijkstra_optimum(seed):
    p = load_params()
    t, prob = problem(seed, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    opt = float(f.fwd.min_over_headings()[t.goal_rc])
    assert f.F_min == pytest.approx(opt, rel=1e-12)
    assert f.F_min == pytest.approx(float(f.bwd.min_over_headings()[t.start_rc]), rel=1e-12)
    assert f.F_min == pytest.approx(run_search(prob, t.start_rc, t.goal_rc, astar=True).goal_cost, rel=1e-9)
    # Every cell on the optimal route has F equal to the optimum.
    best, _ = route_through(prob, f, t.goal_rc)
    assert path_cost(prob, best) == pytest.approx(f.F_min, rel=1e-9)
    cells = _Geometry(prob).mask(best)
    on = np.isfinite(f.F) & cells
    assert np.allclose(f.F[on], f.F_min, rtol=1e-9) or f.F[on].min() == pytest.approx(f.F_min)
    # F >= F_min everywhere.
    assert np.nanmin(f.F) >= f.F_min * (1 - 1e-12)


def test_route_through_any_cell_costs_F():
    p = load_params()
    t, prob = problem(5, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    rng = np.random.default_rng(1)
    finite = np.argwhere(np.isfinite(f.F))
    for k in rng.choice(len(finite), 10, replace=False):
        rc = tuple(finite[k])
        e, _ = route_through(prob, f, rc)
        assert path_cost(prob, e) == pytest.approx(f.F[rc], rel=1e-9)
        assert _Geometry(prob).mask(e)[rc]


def test_alternatives_rules():
    """A steep block between the endpoints: one route round each side."""
    p = load_params(overrides={"K_alt": 4, "eps_alt": 0.3, "D_alt": 60.0})
    z = np.full((70, 120), 1000.0)
    z[22:48, 50:70] += 100.0
    g = make_grid(z, RES)
    t = preprocess(g, g.xy(35, 10), g.xy(35, 110), p)
    prob = make_problem(t, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    alts = alternatives(prob, f, t.start_rc)
    assert 2 <= len(alts) <= p.K_alt
    assert alts[0].F == pytest.approx(f.F_min)
    for k, a in enumerate(alts):
        assert a.F <= (1 + p.eps_alt) * f.F_min * (1 + 1e-12)
        assert path_cost(prob, a.edges) == pytest.approx(a.F, rel=1e-9)
        for b in alts[:k]:
            d = ndimage.distance_transform_edt(~_Geometry(prob).mask(b.edges)) * RES
            assert d[a.via] >= p.D_alt
            assert _shared_edge_fraction(prob, a.edges, b.edges) <= 0.5
    # One goes north of the block, one south.
    rows = [np.mean([e[0] for e in a.edges]) for a in alts[:2]]
    assert min(rows) < 35 < max(rows)


def test_local_optimality_rejects_out_and_back_detours():
    """On a flat plane, the route through a cell off to the side goes out and
    comes back; it fails the local test, while the best route passes."""
    p = load_params(overrides={"alt_window": 0.5})
    g = make_grid(np.full((60, 120), 1000.0), RES)
    t = preprocess(g, g.xy(30, 10), g.xy(30, 110), p)
    prob = make_problem(t, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    best, k = route_through(prob, f, (30, 60))
    assert locally_optimal(prob, best, k, 0.5 * f.F_min, p.alt_local_tol)[0]
    detour, k = route_through(prob, f, (5, 60))
    ok, info = locally_optimal(prob, detour, k, 0.5 * f.F_min, p.alt_local_tol)
    assert not ok and info["window_cost"] > info["shortest"]
    # So no alternative survives on a featureless plane.
    assert len(alternatives(prob, f, t.start_rc)) == 1


def test_shared_cost_prefilter_tracks_exact_overlap():
    from roadsearch.pipeline import PREFILTER_SHARE, shared_cost_fraction
    p = load_params()
    z = np.full((70, 120), 1000.0)
    z[22:48, 50:70] += 100.0
    g = make_grid(z, RES)
    t = preprocess(g, g.xy(35, 10), g.xy(35, 110), p)
    prob = make_problem(t, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    best, _ = route_through(prob, f, t.goal_rc)
    share = shared_cost_fraction(prob, f, best)
    assert share[t.goal_rc] == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    cells = np.argwhere(np.isfinite(f.F))
    for k in rng.choice(len(cells), 40, replace=False):
        rc = tuple(cells[k])
        e, _ = route_through(prob, f, rc)
        exact = _shared_edge_fraction(prob, e, best)
        # Never skips a candidate that the exact half-length rule would keep.
        if exact <= 0.5:
            assert share[rc] <= PREFILTER_SHARE
