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


@pytest.fixture(scope="module")
def rough_prob():
    """The same ground with patches of roughness: soil-effective grades past g_max."""
    p = load_params(overrides={"C_travel": 125.5, "C_vertical": 2576.0, "C_turn": 8000.0, "rough_r0": 0.5})
    z = random_terrain(7, n=90, relief=15.0)
    g = make_grid(z, 5.0)
    g.rough = 0.8 * np.clip(random_terrain(11, n=90, relief=1.0) - 1000.0, 0.0, None)
    pr = make_problem(preprocess(g, g.xy(5, 5), g.xy(84, 80), p), p)
    pr.lam_fill = 3.7
    assert pr.k_rough > 0 and pr.cost.table.shape[1] > 21
    return pr


@pytest.mark.parametrize("which", ["prob", "rough_prob"])
def test_edge_costs_match(which, request):
    prob = request.getfixturevalue(which)
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


@pytest.mark.parametrize("kind", ["hairpin", "bend"])
def test_macro_tables_match(prob, kind):
    """Hairpin and bend tables from the Rust core against numba's: same feasibility everywhere,
    and float32 costs equal (sines and cosines may differ in the last bit before rounding)."""
    from roadsearch.bend import bend_table
    from roadsearch.hairpin import hairpin_table

    p = prob.params
    terr = prob.macro_terrain_arrays
    drain = (prob.channel, prob.area)
    cells = np.flatnonzero(prob.open_tables)
    if kind == "hairpin":
        build = lambda e: hairpin_table(terr, p, prob.res, drain, cells, engine=e)  # noqa: E731
    else:
        build = lambda e: bend_table(terr, p, prob.res, prob.bend_geo, drain, cells, engine=e)  # noqa: E731
    a, b = build("numba"), build("rust")
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    fa, fb = np.isfinite(a), np.isfinite(b)
    assert fa.sum() > 1000
    assert np.array_equal(fa, fb)
    same = a[fa] == b[fa]
    assert same.mean() > 0.999, same.mean()
    assert np.max(np.abs(a[fa] - b[fa]) / np.abs(a[fa])) < 1e-6


def test_preprocessing_matches_python():
    """Rust preprocessing against the Python path on terrain with nodata holes: smoothing,
    gradients, slope masks and connected regions identical; channels nearly so (flow routing on
    flats and through filled depressions differs from pysheds by design)."""
    from roadsearch.flow import contributing_area_ha
    from roadsearch.preprocess import gradient, reachable_from, slope_mask, smooth_dem

    p = load_params()
    z = random_terrain(11, n=120, relief=40.0)
    z[30:34, 50:60] = np.nan
    z[90, 10] = np.nan
    g = make_grid(z, 5.0)
    zs = {e: smooth_dem(g.z, p.sigma_m / g.res, e) for e in ("numba", "rust")}
    assert np.array_equal(zs["numba"], zs["rust"], equal_nan=True)
    gr = {e: gradient(zs["numba"], g, e) for e in ("numba", "rust")}
    for i in range(2):
        assert np.array_equal(gr["numba"][i], gr["rust"][i], equal_nan=True)
    S = np.hypot(*gr["numba"])
    m = {e: slope_mask(S, g, p, e) for e in ("numba", "rust")}
    assert m["numba"][1].any() and np.array_equal(m["numba"][0], m["rust"][0])
    assert np.array_equal(m["numba"][1], m["rust"][1]) and m["numba"][2] == m["rust"][2]
    open_ = ~m["numba"][1]
    seed = tuple(np.argwhere(open_)[0])
    assert np.array_equal(reachable_from(open_, seed, "numba"), reachable_from(open_, seed, "rust"))
    a, b = contributing_area_ha(g, "numba"), contributing_area_ha(g, "rust")
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ca, cb = np.nan_to_num(a) >= p.A_min, np.nan_to_num(b) >= p.A_min
    assert ca.sum() > 100 and np.mean(ca == cb) > 0.99


def test_run_id_records_the_terrain_engine(tmp_path, monkeypatch):
    import json

    from roadsearch import engine
    from roadsearch.cli import main

    from .test_run import args, dem_file

    path, s, g = dem_file(tmp_path)
    ids = {}
    for e in ("numba", "rust"):
        monkeypatch.setattr(engine, "ENGINE", e)
        out = tmp_path / f"out-{e}"
        assert main(args(path, s, g, out, "--K_alt", "1")) == 0
        run_dir = next((out / "runs").iterdir())
        ids[e] = run_dir.name
        rec = json.loads((run_dir / "params.json").read_text())
        assert rec["inputs"].get("terrain_engine") == (engine.TERRAIN_ENGINE_TAG if e == "rust" else None)
    assert ids["numba"] != ids["rust"]
