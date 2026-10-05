"""Routes that overlap themselves (balloon loops, out-and-back excursions)."""

import numpy as np
import pytest

from roadsearch.masshaul import true_cost
from roadsearch.params import load_params
from roadsearch.pipeline import (
    IterationResult, RouteResult, _Geometry, alternatives, avoid_self_overlap, corridor_fields, route_through,
)
from roadsearch.preprocess import preprocess
from roadsearch.route import evaluate_route, self_overlaps
from roadsearch.search import make_problem

from .conftest import make_grid

RES = 5.0


def line(*pts, n=50):
    pts = np.asarray(pts, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        out.extend(a + (b - a) * f for f in np.linspace(0, 1, n)[1:])
    return np.array(out)


def test_detects_out_and_back_but_not_curves_or_hairpin_legs():
    w = 5.0
    # Out along row 10, round a loop, back along row 10: overlapping.
    pieces = [line((10, 0), (10, 10)), line((10, 10), (10, 20)), line((10, 20), (6, 26), (2, 20)),
              line((2, 20), (6, 14), (10, 20)), line((10, 20), (10, 10)), line((10, 10), (10, 0))]
    ov = self_overlaps(pieces, RES, w)
    assert ov and min(s for _, _, s in ov) == pytest.approx(0.0, abs=1e-9)
    # A gentle curve: never.
    th = np.linspace(0, np.pi / 2, 7)
    curve = [line((40 * np.sin(a), 40 * np.cos(a)), (40 * np.sin(b), 40 * np.cos(b)))
             for a, b in zip(th[:-1], th[1:])]
    assert not self_overlaps(curve, RES, w)
    # Hairpin legs further apart than the road width: not an overlap.
    legs = [line((10, 0), (10, 20)), line((10, 20), (6, 24), (8, 20)), line((8, 20), (8, 0))]
    assert self_overlaps(legs, RES, 0.5 * 2 * RES) == []        # legs 10 m apart, road 5 m


def test_avoid_self_overlap_recovers_the_direct_route():
    p = load_params()
    g = make_grid(np.full((60, 140), 1000.0), RES)
    t = preprocess(g, g.xy(30, 10), g.xy(30, 90), p)
    prob = make_problem(t, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    # Forced through a cell beyond the goal, the route overshoots, turns round and comes back.
    loop, _ = route_through(prob, f, (30, 125))
    assert _Geometry(prob).overlaps(loop)
    segs = evaluate_route(prob, loop)
    fake = IterationResult(RouteResult(loop, segs, true_cost(segs, p), p.C_borrow))
    it, rounds, blocked = avoid_self_overlap(prob, t.start_rc, t.goal_rc, fake)
    assert rounds and rounds[0]["result"] == "re-searched"
    assert blocked.any() and not blocked[t.start_rc] and not blocked[t.goal_rc]
    assert not _Geometry(prob).overlaps(it.best.edges)
    assert it.best.cost.total < fake.best.cost.total
    assert it.best.cost.total == pytest.approx(p.C_base * 80 * RES, rel=1e-9)   # straight, flat


def test_alternatives_never_overlap_themselves():
    p = load_params(overrides={"K_alt": 4, "eps_alt": 0.5, "D_alt": 40.0})
    z = np.full((70, 120), 1000.0)
    z[22:48, 50:70] += 100.0
    g = make_grid(z, RES)
    t = preprocess(g, g.xy(35, 10), g.xy(35, 110), p)
    prob = make_problem(t, p)
    f = corridor_fields(prob, t.start_rc, t.goal_rc)
    geo = _Geometry(prob)
    for a in alternatives(prob, f, t.start_rc):
        assert not geo.overlaps(a.edges)
