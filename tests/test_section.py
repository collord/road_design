"""Brief section 11, tests 1-3: the cross-section in isolation."""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from roadsearch.params import load_params
from roadsearch.section import (
    cut_geometry, max_catch_at_mask, section, section_arrays, section_params, soil_depth,
)

P = load_params()
SP = section_params(P)


def tri_cut_area(a, s, c):
    return 0.5 * a * a * s * c / (c - s)


# --- Test 1: single material ------------------------------------------------

@pytest.mark.parametrize("s", [0.05, 0.2, 0.5, 0.8])
@pytest.mark.parametrize("a", [0.5, 2.0, 5.0])
def test_cut_all_soil_when_deep(a, s):
    As, Ar, catch = cut_geometry(a, s, 1e6, P.c_soil, P.c_rock)
    assert Ar == 0.0
    assert As == pytest.approx(tri_cut_area(a, s, P.c_soil), rel=1e-12)
    assert catch == pytest.approx(P.c_soil * a / (P.c_soil - s), rel=1e-12)


@pytest.mark.parametrize("s", [0.05, 0.5, 1.5, 3.0])
@pytest.mark.parametrize("a", [0.5, 2.0, 5.0])
def test_cut_all_rock_when_no_soil(a, s):
    As, Ar, catch = cut_geometry(a, s, 0.0, P.c_soil, P.c_rock)
    assert As == pytest.approx(0.0, abs=1e-12)
    assert Ar == pytest.approx(tri_cut_area(a, s, P.c_rock), rel=1e-12)
    assert catch == pytest.approx(P.c_rock * a / (P.c_rock - s), rel=1e-12)


@pytest.mark.parametrize("s", [0.1, 0.3, 0.6, 0.9])
def test_section_single_material_through_balance(s):
    deep = section(s, 1e6, P)
    assert deep.A_rock == 0.0
    assert deep.A_soil == pytest.approx(tri_cut_area(deep.cut_width, s, P.c_soil), rel=1e-9)
    bare = section(s, 0.0, P)
    assert bare.A_soil == pytest.approx(0.0, abs=1e-12)
    assert bare.A_rock == pytest.approx(tri_cut_area(bare.cut_width, s, P.c_rock), rel=1e-9)


@pytest.mark.parametrize("s,d,a", [
    (0.3, 0.5, 4.0), (0.6, 0.27, 3.2), (0.2, 0.3, 5.0), (0.9, 0.05, 2.0), (0.5, 1.0, 3.0),
])
def test_two_material_split_matches_polygon_clipping(s, d, a):
    """Independent check: build the cut region in shapely and clip by the rock surface."""
    As, Ar, x2 = cut_geometry(a, s, d, P.c_soil, P.c_rock)
    assert a * s > d  # the bench floor enters rock in these cases
    x1 = (P.c_rock * a - d) / (P.c_rock - s)
    cut = Polygon([(0, 0), (a, 0), (x1, P.c_rock * (x1 - a)), (x2, s * x2)])
    assert cut.is_valid
    big = 1e3
    below_rock = Polygon([(-big, -s * big - d), (big, s * big - d), (big, -big * 10), (-big, -big * 10)])
    rock = cut.intersection(below_rock).area
    assert Ar == pytest.approx(rock, rel=1e-9)
    assert As == pytest.approx(cut.area - rock, rel=1e-9)
    # The backslope reaches ground exactly at x2.
    assert x2 == pytest.approx(x1 + d / (P.c_soil - s), rel=1e-12)


def test_cut_area_continuous_at_rock_contact():
    """Crossing a = d/s switches formulas; the total area must not jump."""
    s, d = 0.4, 0.8
    a0 = d / s
    lo = sum(cut_geometry(a0 * (1 - 1e-9), s, d, P.c_soil, P.c_rock)[:2])
    hi = sum(cut_geometry(a0 * (1 + 1e-9), s, d, P.c_soil, P.c_rock)[:2])
    assert lo == pytest.approx(hi, rel=1e-6)


# --- Test 2: balance ---------------------------------------------------------

def test_half_bench_balances():
    n_checked = 0
    for s in np.linspace(0.005, P.s_mask * 0.999, 400):
        for d in (0.0, 0.05, 0.3, 1.0, 2.0, 50.0):
            r = section(s, d, P)
            if r.kind not in ("half_bench", "flat"):
                continue
            loose = P.k_soil * r.A_soil + P.k_rock * r.A_rock
            assert loose == pytest.approx(r.A_fill, rel=1e-8, abs=1e-12)
            # The fill really fits: fill area and toe match the stated slope.
            b, f = r.fill_width, r.fill_slope
            assert r.A_fill == pytest.approx(0.5 * b * b * s * f / (f - s), rel=1e-12)
            assert r.fill_toe == pytest.approx(f * b / (f - s), rel=1e-12)
            assert r.cut_width + r.fill_width == pytest.approx(P.w)
            assert r.fill_width >= P.b_min
            n_checked += 1
    assert n_checked > 400


def test_fill_slope_is_volume_weighted_blend():
    r = section(0.5, 0.378, P)
    vs, vr = P.k_soil * r.A_soil, P.k_rock * r.A_rock
    assert vs > 0 and vr > 0
    assert r.fill_slope == pytest.approx((vs * P.f_soil + vr * P.f_rock) / (vs + vr), rel=1e-12)


# --- Test 3: limits and monotonicity ------------------------------------------

def test_cost_tends_to_base_on_flat_ground():
    costs = [section(s, P.d0, P).cost for s in (1e-1, 1e-2, 1e-3, 1e-4, 0.0)]
    assert costs[-1] == P.C_base
    excess = [c - P.C_base for c in costs[:-1]]
    assert all(e > 0 for e in excess)
    assert all(e2 < e1 for e1, e2 in zip(excess, excess[1:]))
    assert excess[-1] < 1e-3


@pytest.mark.parametrize("d,f", [(0.0, P.f_rock), (1e6, P.f_soil)])
def test_full_bench_as_s_approaches_fill_slope(d, f):
    assert section(0.5 * f, d, P).kind == "half_bench"
    assert section(f * (1 - 1e-6), d, P).kind == "full_bench"
    assert section(f, d, P).kind == "full_bench"
    full = section(f * 1.01, d, P)
    assert full.kind == "full_bench"
    assert full.cut_width == P.w and full.fill_width == 0.0 and full.A_fill == 0.0
    assert full.spoil == pytest.approx(P.k_soil * full.A_soil + P.k_rock * full.A_rock)


def test_full_bench_when_fill_bench_too_narrow():
    # Find the half-bench to full-bench transition on bare rock and check b there.
    s_grid = np.linspace(0.3, P.f_rock, 2000)
    kinds = [section(s, 0.0, P).kind for s in s_grid]
    i = kinds.index("full_bench")
    assert section(s_grid[i - 1], 0.0, P).fill_width >= P.b_min


@pytest.mark.parametrize("d", [0.0, 0.1, 0.5, 2.0, 1e6])
def test_cost_monotonic_in_s_fixed_depth(d):
    """Monotonic on each side of c_soil.

    At s = c_soil the brief's model is discontinuous for d > 0: a soil
    backslope at c_soil never daylights on ground that steep (the cut is then
    clipped at catch_max), and at s >= c_soil the soil is removed (d = 0).
    The drop there is checked separately below.
    """
    for lo, hi in [(0.0, P.c_soil), (P.c_soil, P.s_mask)]:
        s = np.linspace(lo, hi, 3000, endpoint=False)
        out = section_arrays(s, np.full_like(s, d), SP)
        cost = out[:, 10]
        assert np.all(np.isfinite(cost))
        assert np.all(np.diff(cost) >= -1e-9)


def test_cost_monotonic_in_s_with_slope_dependent_depth():
    for lo, hi in [(0.0, P.c_soil), (P.c_soil, P.s_mask)]:
        s = np.linspace(lo, hi, 3000, endpoint=False)
        out = section_arrays(s, soil_depth(s, P), SP)
        assert np.all(np.diff(out[:, 10]) >= -1e-9)


def test_drop_at_soil_cut_slope_is_bounded_by_catch_cap():
    """Approaching c_soil from below, a soil backslope at c_soil never
    daylights; catch_max clips it, so the cost stays bounded and the drop at
    s = c_soil (where soil depth goes to 0) is under 1% with the defaults.
    """
    s = np.linspace(0.0, P.c_soil, 5000, endpoint=False)
    s = np.concatenate([s, P.c_soil * (1 - np.logspace(-3, -12, 50))])
    out = section_arrays(s, soil_depth(s, P), SP)
    at = section(P.c_soil, 0.0, P).cost
    assert out[:, 7].max() <= P.catch_max
    assert out[:, 10].max() <= 1.01 * at


@pytest.mark.parametrize("s,d,a", [(0.999, 0.072, 5.0), (0.9999, 0.072, 3.0), (0.9999, 2.0, 0.5), (3.55, 0.0, 5.0)])
def test_catch_cap_clips_cut_polygon(s, d, a):
    """Clipped areas match shapely's clip of the unclipped cut polygon."""
    cap = 20.0
    As, Ar, catch = cut_geometry(a, s, d, P.c_soil, P.c_rock, cap)
    As0, Ar0, x2 = cut_geometry(a, s, d, P.c_soil, P.c_rock)
    assert x2 > cap and catch == cap
    if a * s <= d:
        cut = Polygon([(0, 0), (a, 0), (x2, s * x2)])
        rock = Polygon()
    else:
        x1 = (P.c_rock * a - d) / (P.c_rock - s)
        cut = Polygon([(0, 0), (a, 0), (x1, P.c_rock * (x1 - a)), (x2, s * x2)])
        rock = Polygon([(d / s, 0), (a, 0), (x1, P.c_rock * (x1 - a))])
    keep = box(-1, -1e4, cap, 1e4)
    assert Ar == pytest.approx(rock.intersection(keep).area, rel=1e-9, abs=1e-12)
    assert As + Ar == pytest.approx(cut.intersection(keep).area, rel=1e-9)
    assert As + Ar < As0 + Ar0


def test_catch_cap_inactive_below_cap():
    assert cut_geometry(5.0, 0.5, 0.3, P.c_soil, P.c_rock, P.catch_max) ==         cut_geometry(5.0, 0.5, 0.3, P.c_soil, P.c_rock)


def test_infeasible_at_mask_slope():
    assert section(P.s_mask, 0.0, P).kind == "infeasible"
    assert math.isinf(section(P.s_mask * 1.5, 0.0, P).cost)
    assert section(P.s_mask * 0.999, 0.0, P).feasible


def test_sign_of_sidehill_does_not_matter():
    assert section(-0.4, 0.5, P) == section(0.4, 0.5, P)


def test_soil_cannot_stand_steeper_than_cut_slope():
    assert section(1.2, 2.0, P) == section(1.2, 0.0, P)
    assert float(soil_depth(P.c_soil, P)) == 0.0


def test_max_catch_at_mask():
    # At the threshold there is no soil, and a full bench in rock gives c_rock*w/(c_rock - s).
    assert max_catch_at_mask(P) == pytest.approx(P.w / (1 - P.mask_margin), rel=1e-12)


def test_width_parameter_override():
    narrow = section(0.4, 0.5, P, w=3.0)
    wide = section(0.4, 0.5, P, w=7.0)
    assert narrow.cut_width + narrow.fill_width == pytest.approx(3.0)
    assert wide.cost > narrow.cost
