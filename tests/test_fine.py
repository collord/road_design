"""The Rust fine stage (road_fine, through roadsearch_core) against the Python pipeline: the section,
the cost table, the problem's rasters and macro tables, and whole routes with their true costs."""

import json
import math

import numpy as np
import pytest

roadsearch_core = pytest.importorskip("roadsearch_core")

from roadsearch.params import load_params  # noqa: E402
from roadsearch.pipeline import avoid_self_overlap, fill_price_iteration  # noqa: E402
from roadsearch.preprocess import preprocess  # noqa: E402
from roadsearch.route import self_overlaps  # noqa: E402
from roadsearch.search import build_cost_table, make_problem  # noqa: E402
from roadsearch.section import section_arrays, section_params  # noqa: E402

from .conftest import make_grid  # noqa: E402
from .test_crossings import gully  # noqa: E402
from .test_overlap import line  # noqa: E402
from .test_search import random_terrain  # noqa: E402


def params_json(p) -> str:
    """Params as the browser and roadsearch_core read them: JSON, infinities as null."""
    fix = lambda v: None if isinstance(v, float) and math.isinf(v) else v  # noqa: E731
    return json.dumps({k: fix(v) for k, v in p.to_dict().items()})


def fine_inputs(t):
    """The fine grid's rasters as road_fine takes them."""
    return (np.ascontiguousarray(t.grid.z, dtype=np.float64), t.z_smooth, t.dzdx, t.dzdy,
            np.ascontiguousarray(t.area_ha, dtype=np.float64), ((t.code == 0) & t.clip).astype(np.uint8), t.grid.res)


def test_section_matches():
    p = load_params()
    rng = np.random.default_rng(3)
    s = np.concatenate([[0.0, 1e-13, p.s_mask, p.s_mask * 0.999], rng.uniform(0, 1.05 * p.s_mask, 20000)])
    d = np.concatenate([[0.0, 0.5, 1.0, 0.0], rng.uniform(-0.1, 3.0, 20000)])
    want = section_arrays(s, d, section_params(p))
    got = roadsearch_core.fine_section(s, d, params_json(p))
    assert np.array_equal(np.isinf(got), np.isinf(want))
    assert np.array_equal(got[np.isfinite(want)], want[np.isfinite(want)])      # bit for bit


def test_cost_table_matches():
    p = load_params(overrides={"g_max": 0.105})
    want = build_cost_table(p)
    table, ds, dg = roadsearch_core.fine_cost_table(params_json(p))
    assert (ds, dg) == (want.ds, want.dg)
    assert np.array_equal(table, want.table)


def test_self_overlaps_match():
    th = np.linspace(0, np.pi / 2, 7)
    cases = [
        [line((10, 0), (10, 10)), line((10, 10), (10, 20)), line((10, 20), (6, 26), (2, 20)),
         line((2, 20), (6, 14), (10, 20)), line((10, 20), (10, 10)), line((10, 10), (10, 0))],
        [line((40 * np.sin(a), 40 * np.cos(a)), (40 * np.sin(b), 40 * np.cos(b))) for a, b in zip(th[:-1], th[1:])],
        [line((10, 0), (10, 20)), line((10, 20), (6, 24), (8, 20)), line((8, 20), (8, 0))],
    ]
    for pieces in cases:
        for w in (5.0, 12.0):
            want = sorted(self_overlaps(pieces, 5.0, w))
            got = sorted(roadsearch_core.fine_self_overlaps([[tuple(x) for x in pc.tolist()] for pc in pieces], 5.0, w))
            assert [(i, j) for i, j, _ in got] == [(i, j) for i, j, _ in want]
            assert np.allclose([s for _, _, s in got], [s for _, _, s in want], atol=1e-9)


def random_case():
    p = load_params(overrides={"C_travel": 125.5, "C_vertical": 2576.0, "C_turn": 8000.0})
    g = make_grid(random_terrain(7, n=90, relief=15.0), 5.0)
    return p, preprocess(g, g.xy(5, 5), g.xy(84, 80), p)


def gully_case():
    p = load_params(overrides={"A_min": 0.2, "C_borrow": 0.1})
    g = make_grid(gully(6.0, 5), 5.0)
    return p, preprocess(g, g.xy(60, 20), g.xy(60, 100), p)


def steep_case():
    """Steep random ground, small channels: hairpins, bends, fill crossings and full benches (spoil
    for the crossings), over five fill-price searches."""
    p = load_params(overrides={"A_min": 0.3, "g_max": 0.12})
    g = make_grid(random_terrain(8, n=100, relief=28.0), 5.0)
    return p, preprocess(g, g.xy(10, 10), g.xy(90, 85), p)


CASES = {"random": random_case, "gully": gully_case, "steep": steep_case}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    p, t = CASES[request.param]()
    prob = make_problem(t, p)
    return request.param, p, t, prob


def test_problem_matches(case):
    _, p, t, prob = case
    got = roadsearch_core.fine_problem(*fine_inputs(t), params_json(p))
    assert np.array_equal(got["soil"], prob.soil)
    assert np.array_equal(got["channel"], prob.channel)
    assert np.array_equal(got["chdist"], prob.chdist)
    assert np.array_equal(got["macro_lookup"], prob.macro_lookup)
    assert np.array_equal(got["table"], prob.cost.table)
    assert np.array_equal(got["hp"], prob.hp)
    assert np.array_equal(got["bend"], prob.bend)


def close(a, b, rel=1e-9, abs_=1e-9):
    return a == pytest.approx(b, rel=rel, abs=abs_, nan_ok=True)


def test_route_matches(case):
    name, p, t, prob = case
    it = fill_price_iteration(prob, t.start_rc, t.goal_rc)
    it, rounds, blocked = avoid_self_overlap(prob, t.start_rc, t.goal_rc, it)
    got = roadsearch_core.fine_route(*fine_inputs(t), params_json(p), tuple(t.start_rc), tuple(t.goal_rc))
    assert it.best is not None and got["best"] is not None, name
    b = got["best"]
    assert [tuple(e) for e in b["edges"]] == [tuple(int(v) for v in e) for e in it.best.edges]
    assert close(b["lam_fill"], it.best.lam_fill)
    want_cost = it.best.cost
    for k, v in [("total", want_cost.total), ("surrogate", want_cost.surrogate), ("base", want_cost.base),
                 ("excavation", want_cost.excavation), ("culverts", want_cost.culverts),
                 ("crossing_fill", want_cost.crossing_fill), ("operating", want_cost.operating),
                 ("fill_m3", want_cost.mass.fill_m3), ("borrow_m3", want_cost.mass.borrow_m3)]:
        assert close(b["cost"][k], v), (name, k, b["cost"][k], v)
    assert len(b["segments"]) == len(it.best.segments)
    for gs, ws in zip(b["segments"], it.best.segments):
        assert gs["section_type"] == ws.section_type, (name, gs["seq"])
        for k in ("length", "grade", "cut_soil_m3", "cut_rock_m3", "fill_m3", "spoil_m3", "borrow_m3", "cost_base",
                  "cost_earthwork", "cost_surrogate", "cost_culvert", "vertical_travel", "turning", "cost_operating"):
            assert close(gs[k], getattr(ws, k)), (name, gs["seq"], ws.section_type, k, gs[k], getattr(ws, k))
        assert np.allclose(gs["pts"], ws.pts_rc, rtol=0, atol=1e-9)
        assert np.allclose(gs["z"], ws.z, rtol=0, atol=1e-9)
    assert len(got["history"]) == len(it.history)
    for gh, wh in zip(got["history"], it.history):
        assert close(gh["lam_fill"], wh.lam_fill) and close(gh["surrogate"], wh.surrogate) and close(gh["true"], wh.true)
        assert gh["n_crossings"] == wh.n_crossings
    assert [r["result"] for r in got["rounds"]] == [r["result"] for r in rounds]
    assert np.array_equal(got["blocked"], blocked)


def test_cases_exercise_the_model(case):
    """The cases between them cover hairpins or bends, fill crossings and spoil."""
    name, p, t, prob = case
    got = roadsearch_core.fine_route(*fine_inputs(t), params_json(p), tuple(t.start_rc), tuple(t.goal_rc))
    kinds = {s["section_type"] for s in got["best"]["segments"]}
    if name == "gully":
        assert "fill_crossing" in kinds
    if name == "steep":
        assert {"hairpin", "bend", "fill_crossing", "full_bench"} <= kinds
        assert len(got["history"]) > 1 and got["best"]["cost"]["haul_cost"] > 0


def test_lazy_macros_match_tables(case):
    """Hairpin and bend costs computed as the search reaches them (the browser's way) give the
    same route and costs as whole tables."""
    name, p, t, _ = case
    args = (*fine_inputs(t), params_json(p), tuple(t.start_rc), tuple(t.goal_rc))
    eager = roadsearch_core.fine_route(*args)
    lazy = roadsearch_core.fine_route(*args, lazy=True)
    assert lazy["best"]["edges"] == eager["best"]["edges"], name
    assert lazy["best"]["cost"] == eager["best"]["cost"]
    assert [h["true"] for h in lazy["history"]] == [h["true"] for h in eager["history"]]
