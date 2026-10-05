"""Bends: 90 and ~135 degree turns with their own design profile."""

import math

import numpy as np
import pytest

from roadsearch.bend import bend_centerline, bend_detail, bend_geometry
from roadsearch.lattice import ACTION_BEND_L, ACTION_BEND_R, bend_steps, heading_angles
from roadsearch.params import load_params
from roadsearch.route import evaluate_route
from roadsearch.search import make_problem, single_edge_cost

from .test_hairpin import RES, brute_force_centerline, plane_terrain


@pytest.mark.parametrize("h", [0, 1, 3, 4, 13])
def test_bend_geometry_ends_on_node_with_turned_heading(h):
    p = load_params()
    t, arr = plane_terrain(0.0, p)
    geo = bend_geometry(p, RES)
    angles = heading_angles(p.N_theta)
    for slot in range(len(bend_steps(p.N_theta))):
        det = bend_detail(50, 50, h, slot, arr, p, RES, geo)
        pts, z = bend_centerline(50, 50, h, slot, det, p, RES, geo, spacing=0.1)
        assert pts[0] == pytest.approx([50, 50])
        assert pts[-1] == pytest.approx([50 - geo.Y[h, slot], 50 + geo.X[h, slot]], abs=1e-9)
        step = np.hypot(*np.diff(pts, axis=0).T)
        assert step.max() < 0.1 / RES * 1.01                  # continuous
        d_end = pts[-1] - pts[-2]
        th_end = math.atan2(-d_end[0], d_end[1])
        h1 = int(geo.h1[h, slot])
        assert math.cos(th_end - angles[h1]) == pytest.approx(1.0, abs=1e-4)
        assert abs(math.degrees(geo.dth[h, slot])) == pytest.approx(det["angle_deg"])
        assert det["radius"] == pytest.approx(p.R_min)


def test_flat_bend_costs_base_only_and_is_symmetric():
    p = load_params()
    t, _ = plane_terrain(0.0, p)
    prob = make_problem(t, p, hairpins=False)
    L = prob.lat
    for a in np.flatnonzero(np.isin(L.kind, [ACTION_BEND_L, ACTION_BEND_R])):
        fwd = single_edge_cost(prob, 50, 50, a)
        back = single_edge_cost(prob, 50 + L.dr[a], 50 + L.dc[a], L.rev[a])
        assert fwd == back
        assert fwd == pytest.approx(p.C_base * L.length[a] * RES, rel=1e-6)   # float32 table


@pytest.mark.parametrize("S,az", [(0.12, 270.0), (0.1, 200.0), (0.2, 100.0)])
def test_bend_volumes_match_brute_force(S, az):
    p = load_params()
    _, arr = plane_terrain(S, p, azimuth=az)
    geo = bend_geometry(p, RES)
    # The first feasible bends (many climb too steeply for their length on these planes).
    found = [(h, slot) for h in range(p.N_theta) for slot in range(len(bend_steps(p.N_theta)))
             if math.isfinite(bend_detail(50, 50, h, slot, arr, p, RES, geo)["cost"])]
    assert found
    h, slot = found[len(found) // 2]
    det = bend_detail(50, 50, h, slot, arr, p, RES, geo)
    pts, z = bend_centerline(0, 0, h, slot, det, p, RES, geo, spacing=0.05)
    vs, vr, vf = brute_force_centerline(S, az, pts, z, det["z0"], p.w, p)
    assert det["cut_soil_m3"] + det["cut_rock_m3"] == pytest.approx(vs + vr, rel=0.03, abs=1.0)
    assert det["fill_m3"] == pytest.approx(vf, rel=0.03, abs=1.0)


def test_bend_turns_through_the_fall_line_on_moderate_slopes():
    """Just steeper than g_max, the fall line itself is too steep for an
    ordinary edge, but a bend sweeping across it averages chord/arc of the
    slope (0.90 for 90 degrees) and is feasible. A bend can only do this up
    to about g_max / 0.9 (90 degrees) or g_max / 0.78 (135 degrees): beyond
    that, turning through the fall line takes a hairpin."""
    p = load_params()
    geo = bend_geometry(p, RES)
    angles = heading_angles(p.N_theta)
    # Descending to the south-west and turning left to the south-east passes due south.
    h0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - math.radians(-135)))))))
    south = int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - math.radians(-90)))))))
    for S, ok in [(1.05 * p.g_max, True), (1.2 * p.g_max, False)]:
        t, arr = plane_terrain(S, p, azimuth=90.0)       # rising north
        det = bend_detail(50, 50, h0, 0, arr, p, RES, geo)
        assert math.isfinite(det["cost"]) == ok
        if ok:
            assert abs(det["grade"]) <= p.g_max
            assert det["max_cut"] > 0 or det["max_fill"] > 0       # off the ground somewhere
        # A straight edge down the fall line is infeasible on both slopes.
        prob = make_problem(t, p, hairpins=False, bends=False)
        L = prob.lat
        straight = next(int(i) for i in L.by_heading[south] if i >= 0 and L.kind[i] == 0 and L.h1[i] == south)
        assert math.isinf(single_edge_cost(prob, 50, 50, straight))


def test_bend_segment_in_route_output():
    p = load_params()
    t, _ = plane_terrain(0.1, p, azimuth=60.0)
    prob = make_problem(t, p, hairpins=False)
    a = int(np.flatnonzero(prob.lat.kind == ACTION_BEND_R)[3])
    seg = evaluate_route(prob, [(50, 50, a)])[0]
    assert seg.section_type == "bend"
    assert seg.cost_total == pytest.approx(seg.cost_surrogate, rel=1e-6)   # float32 table
    assert seg.pts_rc[0] == pytest.approx([50, 50]) and \
        seg.pts_rc[-1] == pytest.approx([50 + prob.lat.dr[a], 50 + prob.lat.dc[a]])


@pytest.mark.parametrize("n_theta,steps", [(8, (2, 3)), (16, (4, 6)), (32, (8, 12))])
def test_bend_angles_follow_heading_count(n_theta, steps):
    from roadsearch.lattice import build_bends
    assert bend_steps(n_theta) == steps
    geo = build_bends(n_theta, 4.0)
    assert np.allclose(np.degrees(np.abs(geo.dth[0])), [90.0, 135.0])
