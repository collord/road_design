"""Exact route evaluation and stacked-leg conflict detection."""

import numpy as np
import pytest

from roadsearch.params import load_params
from roadsearch.route import Segment, evaluate_route, stacked_leg_conflicts
from roadsearch.search import reconstruct, run_search

from .test_hairpin import plane_terrain
from .test_search import random_terrain, solve


def seg(seq, pts, z, catch=0.0, toe=0.0):
    pts = np.asarray(pts, dtype=float)
    length = float(np.hypot(*np.diff(pts, axis=0).T).sum()) * 5.0  # meters at res 5
    return Segment(seq, 0, 0, 0, "half_bench", pts, np.full(len(pts), z), length,
                   0, 0, 0, 0, 0, 0, catch, toe, 0, 0, 0, 0, 0, 0, 0, 0)


def test_conflict_between_stacked_legs():
    # Lower leg along row 10, then a U-turn, upper leg back along row 6 (4 cells = 20 m apart).
    s = [seg(0, [(10, 0), (10, 20)], 100, catch=8),
         seg(1, [(10, 20), (8, 24), (6, 20)], 101),
         seg(2, [(6, 20), (6, 0)], 102, toe=15)]
    cf = stacked_leg_conflicts(s, res=5.0)
    assert len(cf) == 1 and (cf[0].upper, cf[0].lower) == (2, 0)
    assert cf[0].separation == pytest.approx(20.0) and cf[0].required == pytest.approx(23.0)
    # Smaller slopes: no conflict.
    s[2] = seg(2, [(6, 20), (6, 0)], 102, toe=11)
    assert not stacked_leg_conflicts(s, res=5.0)


def test_no_conflict_along_a_straight_road():
    s = [seg(i, [(10, 2 * i), (10, 2 * i + 2)], 100 + i, catch=30, toe=30) for i in range(10)]
    assert not stacked_leg_conflicts(s, res=5.0)


def test_exact_evaluation_matches_surrogate_without_hairpins():
    p = load_params()
    t, prob, res, edges = solve(random_terrain(4, relief=5.0), (12, 12), (78, 60), p)
    segs = evaluate_route(prob, edges)
    exact = sum(sg.cost_total for sg in segs)
    assert sum(sg.cost_surrogate for sg in segs) == pytest.approx(res.goal_cost, rel=1e-12)
    assert exact == pytest.approx(res.goal_cost, rel=2e-3)
    assert sum(sg.length for sg in segs) * p.C_base == pytest.approx(sum(sg.cost_base for sg in segs))
    # Half-bench segments balance: no spoil.
    assert all(sg.spoil_m3 == 0 for sg in segs if sg.section_type in ("flat", "half_bench"))
    # Geometry is continuous.
    for a, b in zip(segs, segs[1:]):
        np.testing.assert_allclose(a.pts_rc[-1], b.pts_rc[0])


def test_exact_evaluation_with_hairpins():
    p = load_params()
    t, _ = plane_terrain(0.22, p, shape=(200, 50))
    from roadsearch.search import make_problem
    prob = make_problem(t, p)
    res = run_search(prob, (185, 25), (15, 25), astar=True)
    segs = evaluate_route(prob, reconstruct(res, prob.lat, res.goal_state))
    hp = [sg for sg in segs if sg.section_type == "hairpin"]
    assert hp and all(sg.extra["inner_edge_grade"] > sg.extra["arc_grade"] for sg in hp)
    assert sum(sg.cost_total for sg in segs) == pytest.approx(res.goal_cost, rel=2e-3)
    # Road elevation is continuous across hairpin ends.
    for a, b in zip(segs, segs[1:]):
        assert a.z[-1] == pytest.approx(b.z[0], abs=0.05)
