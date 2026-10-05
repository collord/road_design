"""Drainage crossings (brief section 7) and the true cost evaluator (section 8)."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from roadsearch.exclusion import MANUAL
from roadsearch.lattice import ACTION_FILL
from roadsearch.masshaul import solve_mass_haul, true_cost
from roadsearch.params import load_params
from roadsearch.pipeline import fill_price_iteration
from roadsearch.preprocess import preprocess
from roadsearch.route import Segment, evaluate_route
from roadsearch.search import make_problem, reconstruct, run_search, single_edge_cost

from .conftest import UTM, make_grid

RES = 5.0
N = 120


def gully(depth, half):
    """Plane falling south at 0.04 with a V gully along column 60."""
    rows, cols = np.mgrid[0:N, 0:N]
    return 1000 - 0.04 * rows * RES - depth * np.clip(1 - np.abs(cols - 60) / half, 0, None)


def setup(z, p, exclude=(), **kw):
    g = make_grid(z, RES)
    t = preprocess(g, g.xy(60, 20), g.xy(60, 100), p, list(exclude))
    return t, make_problem(t, p, **kw)


def fill_action(prob, h, m):
    L = prob.lat
    return next(i for i in range(len(L.h0))
                if L.kind[i] == ACTION_FILL and L.h0[i] == h and round(L.length[i] / L.length[L.by_heading[h, 0]]) == m
                and L.dr[i] == 0)


def solve(prob):
    res = run_search(prob, (60, 20), (60, 100), astar=True)
    assert res.goal_state is not None
    segs = evaluate_route(prob, reconstruct(res, prob.lat, res.goal_state))
    return res, segs


# --- at-grade crossings ----------------------------------------------------------

def test_at_grade_culvert_charged_once():
    p = load_params()
    t, prob = setup(gully(2.0, 10), p)
    res, segs = solve(prob)
    tc = true_cost(segs, p)
    crossed = [s for s in segs if s.cost_culvert > 0]
    assert tc.culverts == pytest.approx(p.culvert_cost(max(s.culvert_area_ha for s in crossed)))
    assert tc.total == pytest.approx(res.goal_cost, rel=2e-3)


def test_culvert_split_at_shared_node_is_symmetric():
    p = load_params()
    t, prob = setup(gully(2.0, 10), p)
    L = prob.lat
    straight = L.by_heading[0, 0]
    # Edges ending on, and starting from, the channel cell (60, 60).
    into = single_edge_cost(prob, 60, 59, straight)
    out = single_edge_cost(prob, 60, 60, straight)
    back_into = single_edge_cost(prob, 60, 61, L.rev[straight])
    assert into == back_into
    base_into = single_edge_cost(prob, 60, 30, straight)
    # The two halves around the node add up to one culvert (plus section costs).
    assert t.channel[60, 60] and not t.channel[60, 59] and not t.channel[60, 61]
    assert (into - base_into) + (out - base_into) == pytest.approx(
        p.culvert_cost(t.area_ha[60, 60]), abs=2.0)


# --- fill crossings -------------------------------------------------------------------

def test_fill_crossing_volume_and_cost():
    p = load_params(overrides={"A_min": 0.2})
    t, prob = setup(gully(6.0, 5), p)
    a = fill_action(prob, 0, 10)                     # 50 m east from the west shoulder
    cost = single_edge_cost(prob, 60, 55, a)
    seg = evaluate_route(prob, [(60, 55, a)])[0]
    assert seg.section_type == "fill_crossing" and seg.length == pytest.approx(50.0)
    # V of depth 6 over +-25 m, flat design: integral of w*h + h^2/f_rock.
    analytic = p.w * 6 * 25 + (2 * 36 * 25 / 3) / p.f_rock
    assert seg.fill_m3 == pytest.approx(analytic, rel=0.01)
    culvert = p.culvert_cost(np.nanmax(t.area_ha[60, 56:65]))
    assert cost == pytest.approx(p.C_base * 50 + p.C_borrow * seg.fill_m3 + culvert, rel=1e-9)
    assert seg.cost_culvert == pytest.approx(culvert)
    # Same cost in reverse.
    assert single_edge_cost(prob, 60, 65, prob.lat.rev[a]) == cost


def test_fill_crossing_limits():
    p = load_params(overrides={"A_min": 0.2})
    t, prob = setup(gully(6.0, 5), p)
    a = fill_action(prob, 0, 10)
    assert np.isfinite(single_edge_cost(prob, 60, 55, a))
    # Too high a fill.
    _, low = setup(gully(6.0, 5), load_params(overrides={"A_min": 0.2, "H_fill_max": 5.0}))
    assert np.isinf(single_edge_cost(low, 60, 55, a))
    # Too long: no such action.
    _, short = setup(gully(6.0, 5), load_params(overrides={"A_min": 0.2, "L_cross_max": 40.0}))
    assert all(short.lat.length[i] * RES <= 40.0 + 1e-9 for i in range(len(short.lat.h0))
               if short.lat.kind[i] == ACTION_FILL)
    # Not across a channel.
    assert np.isinf(single_edge_cost(prob, 60, 20, fill_action(prob, 0, 4)))
    # Ends in the channel.
    assert np.isinf(single_edge_cost(prob, 60, 56, fill_action(prob, 0, 4)))
    # The design may not run more than fill_tol below ground (a hump, not a gully).
    hump = 1000 + 2.0 * np.clip(1 - np.abs(np.mgrid[0:N, 0:N][1] - 60) / 5, 0, None)
    hump = hump - 3.0 * np.clip(1 - np.abs(np.mgrid[0:N, 0:N][1] - 63) / 1, 0, None)
    _, hp_prob = setup(hump, load_params(overrides={"A_min": 0.0}))
    assert np.isinf(single_edge_cost(hp_prob, 60, 55, fill_action(hp_prob, 0, 10)))


def test_fill_crossing_chosen_when_fill_is_cheap():
    p = load_params(overrides={"A_min": 0.2, "C_borrow": 0.1})
    t, prob = setup(gully(6.0, 5), p)
    res, segs = solve(prob)
    assert any(s.section_type == "fill_crossing" for s in segs)
    _, no_cross = setup(gully(6.0, 5), p, crossings=False)
    res2, _ = solve(no_cross)
    assert res.goal_cost < res2.goal_cost


def test_fill_crossing_cannot_span_manual_zone(tmp_path):
    p = load_params(overrides={"A_min": 0.2, "C_borrow": 0.1})
    g = make_grid(gully(6.0, 5), RES)
    t0, prob0 = setup(gully(6.0, 5), p)
    res0, segs0 = solve(prob0)
    cross = next(s for s in segs0 if s.section_type == "fill_crossing")
    # A small zone on the gully floor right under the chosen crossing.
    r = int(round(cross.pts_rc[:, 0].mean()))
    (x0, y0), (x1, y1) = g.xy(r - 1, 60), g.xy(r + 1, 60)
    f = tmp_path / "pond.gpkg"
    gpd.GeoDataFrame(geometry=[box(x0 - 1, y1, x1 + 1, y0)], crs=UTM).to_file(f)
    t, prob = setup(gully(6.0, 5), p, exclude=[f])
    res, segs = solve(prob)
    for s in segs:
        cells = np.floor(s.pts_rc + 0.5).astype(int)
        assert not (t.code[cells[:, 0], cells[:, 1]] & MANUAL).any()
    assert res.goal_cost >= res0.goal_cost


# --- mass haul ---------------------------------------------------------------------------

def mk(seq, length, z, kind="half_bench", spoil=0.0, fill=0.0):
    pts = np.array([[0.0, 0.0], [0.0, 1.0]])
    return Segment(seq, 0, 0, 0, kind, pts, np.array([z, z]), length, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, fill, spoil, 0, 10 * length, 0, 10 * length)


def test_mass_haul_prefers_downhill_and_respects_haul_max():
    p = load_params()
    segs = [mk(0, 100, 150, "full_bench", spoil=300),     # mid 50, uphill of the sink
            mk(1, 100, 50, "full_bench", spoil=300),      # mid 150, downhill of the sink
            mk(2, 100, 100, "fill_crossing", fill=200),   # mid 250
            mk(3, 1000, 100),
            mk(4, 100, 200, "full_bench", spoil=1000)]    # mid 1350, beyond haul_max
    m = solve_mass_haul(segs, p)
    # Source 0 hauls downhill over 200 m (2 per m3), source 1 uphill over 100 m (2 per m3): a tie;
    # any split costs 200 * 2.
    assert m.borrow_m3 == pytest.approx(0) and m.reused_m3 == pytest.approx(200)
    assert m.haul_cost == pytest.approx(200 * 2.0)
    assert all(a[0] != 4 for a in m.assignments)
    # Remove the near spoil: the far source is out of reach, so fill is borrowed.
    m2 = solve_mass_haul([segs[2], segs[3], segs[4]], p)
    assert m2.borrow_m3 == pytest.approx(200) and m2.total == pytest.approx(200 * p.C_borrow)
    assert m2.cost_per_m3 == pytest.approx(p.C_borrow)


def test_mass_haul_borrow_cheaper_than_long_haul():
    p = load_params(overrides={"C_borrow": 1.0})
    segs = [mk(0, 100, 50, "full_bench", spoil=500), mk(1, 400, 60), mk(2, 100, 100, "fill_crossing", fill=100)]
    m = solve_mass_haul(segs, p)   # uphill 500 m at 0.02 = 10 per m3 > borrow at 1
    assert m.borrow_m3 == pytest.approx(100) and m.haul_cost == pytest.approx(0)


def test_fill_price_iteration_sets_realized_price():
    p = load_params(overrides={"A_min": 0.2, "C_borrow": 0.1})
    t, prob = setup(gully(6.0, 5), p)
    it = fill_price_iteration(prob, (60, 20), (60, 100))
    assert it.history[0].lam_fill == p.C_borrow
    assert it.best is not None and it.history[0].n_crossings >= 1
    # No spoil nearby: crossing fill is all borrow, so the realized price is C_borrow
    # and the second search returns the same route.
    if len(it.history) > 1:
        assert it.history[1].lam_fill == pytest.approx(p.C_borrow)
        assert it.history[1].shared_with_previous >= 0.98
    assert it.best.cost.total == pytest.approx(min(h.true for h in it.history))
