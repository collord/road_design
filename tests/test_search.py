"""Brief section 11, tests 4, 6, 7 and the search half of 8 (no hairpins or
crossings yet)."""

import math

import geopandas as gpd
import numpy as np
import pytest
from scipy import ndimage
from shapely.geometry import box

from roadsearch.exclusion import MANUAL
from roadsearch.lattice import lattice_arrays
from roadsearch.params import load_params
from roadsearch.preprocess import preprocess
from roadsearch.search import (
    make_problem, path_cost, path_xy, reconstruct, reverse_edges, run_search, single_edge_cost,
)

from .conftest import UTM, make_grid

RES = 5.0


def solve(z, start_rc, goal_rc, p=None, exclude=(), astar=False, hairpins=False):
    p = p or load_params()
    g = make_grid(z, RES)
    t = preprocess(g, g.xy(*start_rc), g.xy(*goal_rc), p, list(exclude))
    prob = make_problem(t, p, hairpins=hairpins)
    res = run_search(prob, start_rc, goal_rc, astar=astar)
    assert res.goal_state is not None, "no route found"
    edges = reconstruct(res, prob.lat, res.goal_state)
    return t, prob, res, edges


def random_terrain(seed, n=90, relief=5.0):
    """Smooth random hills. Without hairpins a road can only turn back where
    the ground is gentler than g_max, so relief is kept low enough (median
    slope ~0.08) for routes to exist."""
    rng = np.random.default_rng(seed)
    z = ndimage.gaussian_filter(rng.standard_normal((n, n)), 10.0)
    return 1000.0 + relief * z / z.std()


def route_cells(prob, edges, start_rc):
    pts = path_xy(prob, edges, start_rc)
    return np.unique(np.floor(pts + 0.5).astype(int), axis=0)


# --- test 4: flat plane ------------------------------------------------------

@pytest.mark.parametrize("start,goal", [((40, 10), (40, 110)), ((60, 10), (20, 90)), ((70, 20), (10, 80))])
def test_flat_plane_straight(start, goal):
    p = load_params()
    z = np.full((80, 120), 1000.0)
    t, prob, res, edges = solve(z, start, goal, p)
    # Straight: every edge is a straight move along the start-goal line. (Heading
    # labels may differ: binned moves along an exact heading can carry a
    # neighboring heading's label at identical geometry and cost.)
    d = np.array(goal) - np.array(start)
    for _, _, a in edges:
        v = np.array([prob.lat.dr[a], prob.lat.dc[a]])
        assert prob.lat.kind[a] == 0 and math.isinf(prob.lat.prims[a].radius), "route turns on a flat plane"
        assert v[0] * d[1] - v[1] * d[0] == 0 and v @ d > 0
    length = np.hypot(goal[0] - start[0], goal[1] - start[1]) * RES
    assert res.goal_cost == pytest.approx(p.C_base * length, rel=1e-12)
    assert sum(prob.lat.length[e[2]] for e in edges) * RES == pytest.approx(length, rel=1e-12)


def test_flat_plane_astar_expands_little():
    z = np.full((80, 120), 1000.0)
    _, prob, dij, _ = solve(z, (40, 10), (40, 110))
    _, _, ast, _ = solve(z, (40, 10), (40, 110), astar=True)
    assert ast.goal_cost == dij.goal_cost
    assert ast.n_expanded < dij.n_expanded / 100


# --- lattice -------------------------------------------------------------------

@pytest.mark.parametrize("n_theta", [8, 16, 32])
@pytest.mark.parametrize("res", [2.0, 5.0, 25.9])
def test_primitives_respect_curvature_and_reverse(res, n_theta):
    p = load_params(overrides={"N_theta": n_theta})
    lat = lattice_arrays(p.N_theta, p.R_min / res)
    assert all(pr.radius * res >= p.R_min - 1e-9 for pr in lat.prims)
    assert np.array_equal(lat.rev[lat.rev], np.arange(len(lat.rev)))
    half = p.N_theta // 2
    for i, pr in enumerate(lat.prims):
        q = lat.prims[lat.rev[i]]
        assert (q.X, q.Y) == (-pr.X, -pr.Y)
        assert q.h0 == (pr.h1 + half) % p.N_theta and q.h1 == (pr.h0 + half) % p.N_theta
        assert q.length == pytest.approx(pr.length, rel=1e-12)
        # Same curve: reversed samples coincide.
        np.testing.assert_allclose(q.pts[::-1] + [pr.X, pr.Y], pr.pts, atol=1e-9)


def test_edge_cost_bit_identical_in_reverse():
    p = load_params()
    g = make_grid(random_terrain(1, relief=10.0), RES)  # mixed feasible/infeasible edges
    prob = make_problem(preprocess(g, g.xy(10, 10), g.xy(80, 80), p), p, hairpins=True)
    L = prob.lat
    rng = np.random.default_rng(0)
    n_finite = 0
    for _ in range(3000):
        r, c, pidx = rng.integers(5, 85), rng.integers(5, 85), rng.integers(len(L.h0))
        a = single_edge_cost(prob, r, c, pidx)
        b = single_edge_cost(prob, r + L.dr[pidx], c + L.dc[pidx], L.rev[pidx])
        assert a == b
        n_finite += np.isfinite(a)
    assert n_finite > 500


# --- test 6: symmetry ----------------------------------------------------------

@pytest.mark.parametrize("seed,n_theta", [(1, 16), (2, 16), (3, 16), (1, 32), (2, 8)])
def test_symmetry(seed, n_theta):
    p = load_params(overrides={"N_theta": n_theta})
    z = random_terrain(seed)
    a, b = (8, 12), (78, 70)
    _, prob, fwd, e_fwd = solve(z, a, b, p)
    _, _, bwd, e_bwd = solve(z, b, a, p)
    assert fwd.goal_cost == pytest.approx(bwd.goal_cost, rel=1e-12)
    # Same route reversed: the same nodes in reverse order. (Binned moves along
    # an exact heading can carry a neighboring heading's label at identical
    # geometry and cost, so the action labels of a tie may differ.)
    nodes = lambda edges: [(r, c) for r, c, _ in edges]  # noqa: E731
    assert nodes(e_bwd) == nodes(reverse_edges(e_fwd, prob.lat))


# --- test 7: A* against Dijkstra -------------------------------------------------

@pytest.mark.parametrize("seed,n_theta", [(4, 16), (5, 16), (6, 16), (7, 16), (8, 32)])
def test_astar_matches_dijkstra(seed, n_theta):
    p = load_params(overrides={"N_theta": n_theta})
    z = random_terrain(seed, relief=5.0)
    _, prob, dij, e = solve(z, (12, 12), (78, 60), p)
    _, _, ast, e2 = solve(z, (12, 12), (78, 60), p, astar=True)
    assert ast.goal_cost == pytest.approx(dij.goal_cost, rel=1e-9)
    assert path_cost(prob, e2) == pytest.approx(ast.goal_cost, rel=1e-12)
    assert ast.n_expanded <= dij.n_expanded


def test_route_respects_grade_and_heuristic_bound():
    p = load_params()
    z = random_terrain(9, relief=5.0)
    t, prob, res, edges = solve(z, (5, 5), (84, 84), p)
    dz = abs(t.z_smooth[84, 84] - t.z_smooth[5, 5])
    dist = np.hypot(79, 79) * RES
    assert res.goal_cost >= p.C_base * max(dist, dz / p.g_max) - 1e-9


# --- test 8 (search part): masks --------------------------------------------------

def test_cliff_band_never_entered():
    rows, cols = np.mgrid[0:120, 0:120]
    z = 1000 + 0.02 * cols * RES
    z[58:62, 0:90] += 60.0  # wall, gap on the east side
    _, prob, res, edges = solve(z, (20, 30), (100, 30))
    cells = route_cells(prob, edges, (20, 30))
    assert prob.open_[cells[:, 0], cells[:, 1]].all()
    assert cells[:, 1].max() >= 90  # went round the end of the wall


def test_manual_zone_forces_detour(tmp_path):
    p = load_params()
    z = np.full((80, 120), 1000.0)
    start, goal = (40, 10), (40, 110)
    _, _, base, _ = solve(z, start, goal, p)
    g = make_grid(z, RES)
    (x0, y0), (x1, y1) = g.xy(25, 50), g.xy(55, 60)
    f = tmp_path / "block.gpkg"
    gpd.GeoDataFrame(geometry=[box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))], crs=UTM).to_file(f)
    t, prob, res, edges = solve(z, start, goal, p, [f])
    cells = route_cells(prob, edges, start)
    assert not (t.code[cells[:, 0], cells[:, 1]] & MANUAL).any()
    assert res.goal_cost > base.goal_cost
    # The same polygon in lon/lat gives the same route.
    f2 = tmp_path / "block_ll.gpkg"
    gpd.read_file(f).to_crs(4326).to_file(f2)
    _, _, res2, edges2 = solve(z, start, goal, p, [f2])
    assert edges2 == edges and res2.goal_cost == res.goal_cost


def test_no_corner_squeeze():
    """A diagonal line of excluded cells touching at corners is a barrier."""
    z = np.full((60, 60), 1000.0)
    g = make_grid(z, RES)
    p = load_params()
    t = preprocess(g, g.xy(10, 5), g.xy(50, 55), p)
    t.code[:] = 0
    for i in range(60):
        t.code[i, 59 - i] = 2  # anti-diagonal, corner-connected only
    prob = make_problem(t, p, hairpins=False)
    assert run_search(prob, (10, 5), (50, 55)).goal_state is None
    t.code[30, 29] = 0  # open one cell: now a 4-connected gap exists
    prob = make_problem(t, p, hairpins=False)
    assert run_search(prob, (10, 5), (50, 55)).goal_state is not None
