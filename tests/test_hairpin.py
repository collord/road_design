"""Hairpins (brief section 6): semicircles and keyholes, and test 5
(inclined plane with hairpins)."""

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon

from roadsearch.hairpin import (
    HP_FIELDS, HP_N, _seg_point, edge_cut, edge_fill, hairpin_centerline, hairpin_core, hairpin_detail,
    hairpin_params, hairpin_segments, hairpin_slots,
)
from roadsearch.lattice import ACTION_HP_LEFT, ACTION_HP_RIGHT, heading_vectors
from roadsearch.params import load_params
from roadsearch.preprocess import preprocess
from roadsearch.search import (
    canonical_hairpin, edge_points, make_problem, reconstruct, run_search, single_edge_cost,
)
from roadsearch.section import section, soil_depth

from .conftest import make_grid

RES = 5.0


def plane_terrain(S, p, shape=(100, 100), azimuth=90.0):
    """Plane rising at slope S toward `azimuth` (degrees from east, CCW)."""
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    x, y = cols * RES, -rows * RES
    a = math.radians(azimuth)
    z = 1000 + S * (x * math.cos(a) + y * math.sin(a))
    g = make_grid(z, RES)
    t = preprocess(g, g.xy(shape[0] // 2, 5), g.xy(shape[0] // 2, shape[1] - 6), p)
    arr = ((t.code == 0).astype(np.uint8), t.z_smooth, t.dzdx, t.dzdy, np.nan_to_num(t.soil), t.grid.z)
    return t, arr


def core(arr, p, r, c, h, k, R):
    """One hairpin of a given leg spacing (k lattice steps) and loop radius R."""
    open_, zs, gx, gy, soil, zr = arr
    v = heading_vectors(p.N_theta)[h].astype(float)
    vlen = float(np.hypot(*v))
    out = np.empty(HP_N)
    drain = (np.zeros(zs.shape, np.uint8), np.zeros(zs.shape), np.array(p.culvert_area_breaks),
             np.array(p.culvert_costs))
    hairpin_core(r, c, v[0] / vlen, v[1] / vlen, k, vlen, R, RES, zs.shape[0], zs.shape[1], open_, zs, gx, gy,
                 soil, zr, hairpin_params(p, RES), drain, out)
    return dict(zip(HP_FIELDS, out.tolist()))


def semicircle(arr, p, r=60, c=40, h=0):
    k = int(np.ceil(2 * p.R_hp / (np.hypot(*heading_vectors(p.N_theta)[h]) * RES) - 1e-9))
    return core(arr, p, r, c, h, k, 0.0)   # R below D/2 snaps to the semicircle


def brute_force_hairpin(S, azimuth, h, p, det, step=0.1):
    """Independent integration of a left hairpin (any shape) on a plane."""
    pts_rc, zc = hairpin_centerline(0, 0, h, det, p, RES, spacing=0.05)
    return brute_force_centerline(S, azimuth, pts_rc, zc, det["z0"], p.w_hp, p, step)


def brute_force_centerline(S, azimuth, pts_rc, zc, z0, width, p, step=0.1):
    """Independent integration of any road centerline on a plane, starting at
    (row, col) = (0, 0) with ground elevation z0 there.

    Footprint: shapely buffer of the centerline (flat caps), sampled on a fine
    grid; design elevation from the station found by projecting onto the
    centerline. Edges: offset curves of the centerline at +-width/2, with the
    wedge formulas applied at fine spacing.
    """
    a = math.radians(azimuth)
    grad = np.array([S * math.cos(a), S * math.sin(a)])
    d = float(soil_depth(S, p))
    xy = np.column_stack([pts_rc[:, 1] * RES, -pts_rc[:, 0] * RES])  # meters, origin at P0
    line = LineString(xy)
    station = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])

    def design(sig):
        return float(np.interp(sig, station, zc))

    def ground(x, y):
        return z0 + grad[0] * x + grad[1] * y

    foot = line.buffer(width / 2, cap_style="flat")
    minx, miny, maxx, maxy = foot.bounds
    X, Y = np.meshgrid(np.arange(minx + step / 2, maxx, step), np.arange(miny + step / 2, maxy, step))
    inside = shapely.contains_xy(foot, X.ravel(), Y.ravel())
    vs = vr = vf = 0.0
    for x, y in zip(X.ravel()[inside], Y.ravel()[inside]):
        e = design(line.project(shapely.Point(x, y))) - ground(x, y)
        if e > 0:
            vf += e * step * step
        else:
            vs += min(-e, d) * step * step
            vr += (-e - min(-e, d)) * step * step
    loose = p.k_soil * vs + p.k_rock * vr
    f = (p.k_soil * vs * p.f_soil + p.k_rock * vr * p.f_rock) / loose if loose > 0 else p.f_soil
    for side in (1, -1):
        edge = line.offset_curve(side * width / 2)
        n = int(edge.length / 0.05)
        for k in range(n):
            q = edge.interpolate((k + 0.5) * edge.length / n)
            q2 = edge.interpolate((k + 0.5) * edge.length / n + 1e-3)
            tx, ty = (q2.x - q.x) / 1e-3, (q2.y - q.y) / 1e-3
            nrm = np.array([ty, -tx])
            c0 = line.interpolate(line.project(q))
            if np.dot(nrm, [q.x - c0.x, q.y - c0.y]) < 0:
                nrm = -nrm
            s_o = float(grad @ nrm)
            hh = ground(q.x, q.y) - design(line.project(q))
            dl = edge.length / n
            if hh > 0:
                As, Ar, _ = edge_cut(hh, d, s_o, p.c_soil, p.c_rock, p.catch_max)
                vs += As * dl
                vr += Ar * dl
            else:
                Af, _ = edge_fill(-hh, s_o, f, p.catch_max)
                vf += Af * dl
    return vs, vr, vf


# --- geometry -------------------------------------------------------------------------

@pytest.mark.parametrize("D,R,rho,L", [(20, 10, 20, 0), (15, 10, 20, 0), (12, 20, 20, 5), (28, 25, 40, 30)])
def test_centerline_is_continuous_and_closes(D, R, rho, L):
    seg, alpha, Rr = hairpin_segments(D, R, rho, L)
    if R <= D / 2:
        assert alpha == 0 and Rr == D / 2
    else:
        assert 0 < alpha < math.pi / 2
        assert seg[2, 3] == pytest.approx(1 / R) and seg[1, 3] == pytest.approx(-1 / rho)
    prev = (0.0, 0.0, 0.0)
    for i in range(5):
        x, y, th = _seg_point(seg, i, 0.0)
        assert (x, y) == pytest.approx(prev[:2], abs=1e-9)
        assert math.cos(th - prev[2]) == pytest.approx(1.0, abs=1e-9)
        prev = _seg_point(seg, i, seg[i, 4])
    assert prev[:2] == pytest.approx((0.0, D), abs=1e-9)
    assert math.cos(prev[2] - math.pi) == pytest.approx(1.0, abs=1e-12)


def test_slots():
    p = load_params()
    s = hairpin_slots(p, RES)
    step = np.hypot(*heading_vectors(p.N_theta).T) * RES
    D = s * step[:, None]
    assert np.all(D[:, 0] - p.w_hp >= p.hp_gap_min - 1e-9)        # clear gap between the legs
    used = s[:, 1] > 0
    assert np.all(D[used, 1] >= 2 * p.R_hp - 1e-9)                  # semicircle spacing


# --- costs on planes -------------------------------------------------------------------

def test_flat_semicircle_costs_base_only():
    p = load_params()
    _, arr = plane_terrain(0.0, p)
    det = semicircle(arr, p)
    assert det["leg"] == 0 and det["alpha"] == 0 and det["radius"] == pytest.approx(p.R_hp)
    assert det["cost"] == pytest.approx(p.C_base * math.pi * det["radius"], rel=1e-12)
    assert det["cut_soil_m3"] == det["cut_rock_m3"] == det["fill_m3"] == 0


@pytest.mark.parametrize("S", [0.2, 0.4])
def test_semicircle_leg_length_is_minimal_for_the_climb(S):
    p = load_params()
    _, arr = plane_terrain(S, p)
    det = semicircle(arr, p)
    R = det["radius"]
    assert det["z1"] - det["z0"] == pytest.approx(S * 2 * R)
    assert det["leg"] == pytest.approx((S * 2 * R - p.g_hp * math.pi * R) / (2 * p.g_max))


def test_keyhole_needs_less_climb_and_is_cheaper_on_steep_ground():
    p = load_params()
    _, arr = plane_terrain(0.4, p)
    semi = semicircle(arr, p)
    key = hairpin_detail(60, 40, 0, 0, arr, p, RES)       # narrowest leg spacing
    assert key["alpha"] > 0 and key["leg_spacing"] < 2 * key["radius"]
    assert key["z1"] - key["z0"] == pytest.approx(0.4 * key["leg_spacing"])
    assert key["leg"] < semi["leg"]
    assert key["cost"] < semi["cost"]


def test_brief_hairpin_without_legs_needs_gentle_cross_slope():
    p = load_params(overrides={"hp_leg_max": 0})
    lim = p.g_hp * math.pi / 2
    _, arr = plane_terrain(0.95 * lim, p)
    assert math.isfinite(semicircle(arr, p)["cost"])
    _, arr = plane_terrain(1.05 * lim, p)
    assert math.isinf(semicircle(arr, p)["cost"])


def test_leg_cap():
    p = load_params(overrides={"hp_leg_max": 20})
    _, arr = plane_terrain(0.2, p)
    assert math.isfinite(semicircle(arr, p)["cost"])   # needs 13.7 m
    _, arr = plane_terrain(0.4, p)
    assert math.isinf(semicircle(arr, p)["cost"])      # needs 33.7 m


@pytest.mark.parametrize("h,S,az,slot", [(0, 0.2, 90.0, 1), (0, 0.4, 90.0, 0), (3, 0.3, 160.0, 0),
                                         (5, 0.25, 200.0, 0), (0, 0.3, 90.0, 0)])
def test_volumes_match_brute_force(h, S, az, slot):
    p = load_params()
    _, arr = plane_terrain(S, p, azimuth=az)
    det = hairpin_detail(60, 50, h, slot, arr, p, RES)
    assert math.isfinite(det["cost"]) and det["wall_m2"] == 0
    vs, vr, vf = brute_force_hairpin(S, az, h, p, det)
    assert det["cut_soil_m3"] + det["cut_rock_m3"] == pytest.approx(vs + vr, rel=0.03, abs=1.0)
    assert det["cut_rock_m3"] == pytest.approx(vr, rel=0.03, abs=1.0)
    assert det["fill_m3"] == pytest.approx(vf, rel=0.03, abs=1.0)


def test_walls_between_close_legs():
    """Steep ground: the slopes between keyhole legs overlap. Without walls the
    hairpin is flagged; with a wall cost they are replaced by priced walls."""
    S = 0.55
    p0 = load_params(overrides={"hp_gap_min": 2.0, "hp_leg_max": 200})
    _, arr = plane_terrain(S, p0)
    k = int(hairpin_slots(p0, RES)[0, 0])
    no_wall = core(arr, p0, 60, 40, 0, k, p0.R_hp)
    assert no_wall["alpha"] > 0 and no_wall["conflict"] == 1 and no_wall["wall_m2"] == 0
    p1 = load_params(overrides={"hp_gap_min": 2.0, "hp_leg_max": 200, "C_wall": 15.0})
    walled = core(arr, p1, 60, 40, 0, k, p1.R_hp)
    assert walled["conflict"] == 0 and walled["wall_m2"] > 0
    assert walled["fill_m3"] < no_wall["fill_m3"]
    assert walled["cost"] == pytest.approx(
        p1.C_base * walled["length"] + p1.C_soil * walled["cut_soil_m3"] + p1.C_rock * walled["cut_rock_m3"]
        + p1.C_borrow * walled["borrow_m3"] + p1.C_wall * walled["wall_m2"], rel=1e-9)


def test_edge_cut_matches_polygon():
    """Two-layer backslope wedge against an explicit polygon."""
    p = load_params()
    h, d, s_o = 3.0, 0.8, 0.3
    As, Ar, x2 = edge_cut(h, d, s_o, p.c_soil, p.c_rock, 1e9)
    hr = h - d
    x1 = hr / (p.c_rock - s_o)
    wedge = Polygon([(0, 0), (x1, p.c_rock * x1), (x2, h + s_o * x2), (0, h)])
    rock = Polygon([(0, 0), (x1, p.c_rock * x1), (0, hr)])
    assert wedge.is_valid and rock.is_valid
    assert Ar == pytest.approx(rock.area, rel=1e-9)
    assert As + Ar == pytest.approx(wedge.area, rel=1e-9)
    assert x2 == pytest.approx(x1 + d / (p.c_soil - s_o))
    As2, Ar2, x3 = edge_cut(h, d, s_o, p.c_soil, p.c_rock, 1.0)
    assert x3 == 1.0 and As2 + Ar2 < As + Ar


def test_edge_fill_formula():
    a, toe = edge_fill(2.0, -0.3, 0.8, 50.0)
    assert toe == pytest.approx(2.0 / 0.5) and a == pytest.approx(0.5 * 2.0 * 4.0)
    assert math.isinf(edge_fill(2.0, -0.9, 0.8, 50.0)[0])


def test_footprint_touching_mask_rejected():
    p = load_params()
    _, arr = plane_terrain(0.0, p)
    det = hairpin_detail(60, 40, 0, 0, arr, p, RES)
    assert math.isfinite(det["cost"])
    pts, _ = hairpin_centerline(60, 40, 0, det, p, RES)
    rr, cc = np.floor(pts[len(pts) // 2] + 0.5).astype(int)
    open_ = arr[0].copy()
    open_[rr, cc] = 0
    k = int(hairpin_slots(p, RES)[0, 0])
    assert math.isinf(core((open_,) + arr[1:], p, 60, 40, 0, k, det["radius"])["cost"])
    # The other loop radius may miss that cell: the slot then stays feasible but changes shape.
    other = hairpin_detail(60, 40, 0, 0, (open_,) + arr[1:], p, RES)
    assert math.isinf(other["cost"]) or other["radius"] != det["radius"]
    open_ = arr[0].copy()
    open_[60, 75] = 0
    assert math.isfinite(hairpin_detail(60, 40, 0, 0, (open_,) + arr[1:], p, RES)["cost"])


def test_cells_blocked_after_the_tables_keep_a_hairpins_price_and_shape():
    """Self-overlap avoidance blocks cells after the hairpin table is built. The
    search checks only a hairpin's end nodes against them, so its exact cost and
    shape must still come from the table's mask (a run once crashed here)."""
    from roadsearch.route import evaluate_edge
    from roadsearch.search import hairpin_geometry

    p = load_params()
    t, _ = plane_terrain(0.25, p, azimuth=110.0)
    prob = make_problem(t, p)
    L = prob.lat
    r, c = 50, 50
    a = next(a for a in np.flatnonzero(L.kind == ACTION_HP_LEFT) if math.isfinite(single_edge_cost(prob, r, c, a)))
    cost = single_edge_cost(prob, r, c, a)
    det, pts, _ = hairpin_geometry(prob, r, c, a)
    mid = tuple(np.floor(pts[len(pts) // 2] + 0.5).astype(int))
    prob.open_[mid] = 0                                    # inside the loop, not at either end node
    assert single_edge_cost(prob, r, c, a) == cost
    det2, pts2, _ = hairpin_geometry(prob, r, c, a)
    assert det2["cost"] == det["cost"] and np.array_equal(pts2, pts)
    assert math.isfinite(evaluate_edge(prob, 0, r, c, a).cost_total)


def test_hairpin_and_reverse_cost_the_same():
    p = load_params()
    t, _ = plane_terrain(0.25, p, azimuth=110.0)
    prob = make_problem(t, p)
    L = prob.lat
    n = 0
    for a in np.flatnonzero(np.isin(L.kind, [ACTION_HP_LEFT, ACTION_HP_RIGHT])):
        r, c = 50, 50
        fwd = single_edge_cost(prob, r, c, a)
        back = single_edge_cost(prob, r + L.dr[a], c + L.dc[a], L.rev[a])
        assert fwd == back
        assert L.h1[L.rev[a]] == (L.h0[a] + p.N_theta // 2) % p.N_theta
        n += math.isfinite(fwd)
    assert n > 10


# --- test 5: inclined plane steeper than g_max -------------------------------------

def edge_hand_cost(prob, r, c, a, S, azimuth, p):
    """Section cost of a primitive on a plane from the exact section function."""
    L = prob.lat
    grad = S * np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))])
    s0, n = L.s0[a], L.ns[a]
    w = L.length[a] * RES / n
    total = 0.0
    for k in range(s0, s0 + n):
        t = np.array([L.tx[k], L.ty[k]])
        g = abs(grad @ t)
        s = abs(grad @ np.array([-t[1], t[0]]))
        total += section(s, float(soil_depth(math.hypot(s, g), p)), p).cost * w
    return total


def best_lattice_climb(S, p):
    """Steepest climb at or under g_max available from a lattice heading on a plane."""
    from roadsearch.lattice import heading_angles
    rates = S * np.abs(np.sin(heading_angles(p.N_theta)))
    return rates[rates <= p.g_max].max()


def test_climb_rate_is_quantized_by_headings():
    """On a plane, straight lattice motion climbs only at S*sin(angle off the
    contour) for lattice angles; at S = 0.25 the best is 0.061, not g_max."""
    p = load_params()
    assert best_lattice_climb(0.25, p) == pytest.approx(0.25 * math.sin(math.atan(0.25)), rel=1e-9)
    assert best_lattice_climb(0.22, p) == pytest.approx(0.22 * math.sin(math.atan(0.5)), rel=1e-9)


def test_inclined_plane_uses_hairpins():
    # Exact lattice headings (the brief's lattice): the hand calculation below
    # relies on lattice climb rates. Binned moves are tested in test_moves_opcost.
    p = load_params(overrides={"move_max_cells": 0})
    # S chosen so a lattice heading (atan(1/2) off the contour) climbs at
    # 0.098, close to g_max; see test_climb_rate_is_quantized_by_headings.
    S, az = 0.22, 90.0
    shape = (200, 50)  # 1000 m upslope, 250 m of lateral room
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    z = 1000 + S * (-rows * RES)
    g = make_grid(z, RES)
    start, goal = (185, 25), (15, 25)
    t = preprocess(g, g.xy(*start), g.xy(*goal), p)
    prob = make_problem(t, p)
    res = run_search(prob, start, goal, astar=True)
    assert res.goal_state is not None
    edges = reconstruct(res, prob.lat, res.goal_state)
    n_hp = sum(prob.lat.kind[a] != 0 for _, _, a in edges)
    assert n_hp >= 3

    hand = 0.0
    prim_len = 0.0
    hp_climb = 0.0
    for r, c, a in edges:
        if prob.lat.kind[a] == 0:
            hand += edge_hand_cost(prob, r, c, a, S, az, p)
            prim_len += prob.lat.length[a] * RES
        else:
            hr, hc, h, slot = canonical_hairpin(prob, r, c, a)
            det = hairpin_detail(hr, hc, h, slot, prob.terrain_arrays, p, RES)
            vs, vr, vf = brute_force_hairpin(S, az, h, p, det, step=0.2)
            loose = p.k_soil * vs + p.k_rock * vr
            hand += (p.C_base * det["length"] + p.C_soil * vs + p.C_rock * vr
                     + p.C_borrow * max(vf - loose, 0.0))
            hp_climb += abs(det["z1"] - det["z0"])
    assert res.goal_cost == pytest.approx(hand, rel=0.01)

    # Between hairpins the road climbs at about g_max.
    dz = S * (start[0] - goal[0]) * RES
    assert (dz - hp_climb) / prim_len == pytest.approx(best_lattice_climb(S, p), rel=0.1)
    assert best_lattice_climb(S, p) == pytest.approx(p.g_max, rel=0.05)
    pts = np.vstack([edge_points(prob, *e) for e in edges])
    cells = np.floor(pts + 0.5).astype(int)
    assert prob.open_[cells[:, 0], cells[:, 1]].all()
