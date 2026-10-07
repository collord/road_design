"""Hairpins as macro-actions (brief section 6), as semicircles or keyholes.

Local frame in meters: origin at the entry node P0, x along the entry
heading u, y along its left normal n. The exit node is P1 = (0, D), heading
-x, where D is a whole number of lattice steps along n. A left hairpin is
the centerline

    entry leg     (0, 0) -> (L, 0)                        straight, length L
    entry curve   right turn, radius rho, through alpha    reverse curve outward
    loop          left turn, radius R, through pi + 2 alpha
    exit curve    right turn, radius rho, through alpha
    exit leg      (L, D) -> (0, D)                        straight, length L

with the loop centered on the midline y = D/2. If R = D/2 there are no
reverse curves (alpha = 0) and this is the plain semicircle. If R > D/2 it
is a keyhole: the legs run D apart, closer than the loop's diameter, and
each swings outward on a reverse curve of radius rho = R_min before the
loop. Then

    alpha = pi/2 - atan2(D/2 + rho, sqrt((R + rho)^2 - (D/2 + rho)^2)).

A right hairpin from node Q with heading h is the left hairpin from
Q - D*n(h) traveled backwards, so the search prices both from one table
and a hairpin costs the same in both directions.

For each heading there are up to two leg spacings: the narrowest with a
clear gap of at least hp_gap_min between the road edges, and the
semicircle spacing (2 R_hp snapped up). Each is a separate search action.
For each, two loop radii are tried (max(R_hp, D/2) and 1.5 times that) and
the cheaper hairpin is kept.

Design profile (raw DEM elevations at P0 and P1): the loop climbs at up to
g_hp and the reverse curves and legs at up to g_max. If the loop and curves
cannot take up the difference z1 - z0, approach legs of length L (at most
hp_leg_max) are added. g_hp limits the centerline grade; the design is
level across the width, so the inside edge of the loop is steeper by
R / (R - w_hp/2), as on real turns.

Earthwork: design minus ground over the footprint (centerline buffered by
w_hp/2, sampled across the width with curvature-corrected area), cut split
into soil and rock by the soil depth raster, plus cut backslopes and fill
slopes along both edges with the section-4 catch geometry. Fill comes from
the hairpin's own cut first; a deficit is bought at C_borrow, a surplus is
spoil. Between the legs the facing slopes may overlap (a stacked-leg
conflict). If C_wall is finite, both slopes there are replaced by walls
priced per m2 of face; otherwise the hairpin is flagged as a conflict.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numba
import numpy as np
from numba import prange

from .culvert import _run_step, culvert_cost
from .lattice import heading_vectors
from .params import Params
from .section import _clipped_area

HairpinParams = namedtuple(
    "HairpinParams",
    "w_hp g_hp g_max leg_max c_soil c_rock f_soil f_rock k_soil k_rock "
    "C_base C_soil C_rock C_borrow catch_max delta R_hp rho C_wall C_travel C_vertical",
)

# Detail vector layout returned by hairpin_core.
HP_FIELDS = ("cost", "leg", "radius", "length", "cut_soil_m3", "cut_rock_m3", "fill_m3",
             "borrow_m3", "spoil_m3", "conflict", "max_cut", "max_fill", "z0", "z1",
             "arc_grade", "inner_edge_grade", "max_catch", "max_toe", "culvert", "culvert_area",
             "leg_spacing", "alpha", "rho", "wall_m2")
HP_N = len(HP_FIELDS)
_LEG, _RADIUS, _LEG_SPACING = (HP_FIELDS.index(f) for f in ("leg", "radius", "leg_spacing"))
N_SLOTS = 2
N_ACROSS = 6
R_FACTORS = (1.0, 1.5)


def hairpin_params(p: Params, res: float) -> HairpinParams:
    return HairpinParams(
        p.w_hp, p.g_hp, p.g_max, p.hp_leg_max, p.c_soil, p.c_rock, p.f_soil, p.f_rock,
        p.k_soil, p.k_rock, p.C_base, p.C_soil, p.C_rock, p.C_borrow, p.catch_max,
        min(0.5 * res, p.w_hp / N_ACROSS), p.R_hp, p.R_min, p.C_wall, p.C_travel, p.C_vertical,
    )


def hairpin_slots(p: Params, res: float) -> np.ndarray:
    """(N_theta, N_SLOTS) lattice steps k along the left normal for each leg
    spacing D = k |v| res; 0 marks an unused slot."""
    vecs = heading_vectors(p.N_theta)
    step = np.hypot(vecs[:, 0], vecs[:, 1]) * res
    k_gap = np.maximum(1, np.ceil((p.w_hp + p.hp_gap_min) / step - 1e-9)).astype(np.int64)
    k_semi = np.maximum(k_gap, np.ceil(2 * p.R_hp / step - 1e-9).astype(np.int64))
    slots = np.zeros((p.N_theta, N_SLOTS), dtype=np.int64)
    slots[:, 0] = k_gap
    slots[:, 1] = np.where(k_semi > k_gap, k_semi, 0)
    return slots


@numba.njit(cache=True, inline="always")
def _bil(a, fr, fc, nr, nc):
    if fr < 0.0 or fc < 0.0 or fr > nr - 1 or fc > nc - 1:
        return math.nan
    r0 = min(int(fr), nr - 2)
    c0 = min(int(fc), nc - 2)
    wr = fr - r0
    wc = fc - c0
    return ((1 - wr) * ((1 - wc) * a[r0, c0] + wc * a[r0, c0 + 1])
            + wr * ((1 - wc) * a[r0 + 1, c0] + wc * a[r0 + 1, c0 + 1]))


_BIG = 1e6


@numba.njit(cache=True)
def edge_cut(h, d, s_o, c_soil, c_rock, catch_max):
    """Cut wedge outside a footprint edge whose face is h high.

    x is outward from the edge, y up from the design level. Ground is
    y = h + s_o*x, rock surface y = h - d + s_o*x. The backslope rises at
    c_rock through rock, then c_soil through soil, as in section 4, and is
    clipped at x = catch_max. Returns (A_soil, A_rock, catch) per meter of edge.
    """
    if h <= 0.0:
        return 0.0, 0.0, 0.0
    if d >= h:
        den = c_soil - s_o
        x2 = h / den if den > 0.0 else _BIG
        xs = np.array([0.0, x2, 0.0])
        ys = np.array([0.0, c_soil * x2, h])
        area = _clipped_area(xs, ys, 3, catch_max)
        return area, 0.0, min(x2, catch_max)
    hr = h - d
    den_r = c_rock - s_o
    x1 = hr / den_r if den_r > 0.0 else _BIG
    y1 = c_rock * x1
    if d > 0.0:
        den_s = c_soil - s_o
        x2 = x1 + (d / den_s if den_s > 0.0 else _BIG)
    else:
        x2 = x1
    y2 = h + s_o * x2
    total = _clipped_area(np.array([0.0, x1, x2, 0.0]), np.array([0.0, y1, y2, h]), 4, catch_max)
    rock = _clipped_area(np.array([0.0, x1, 0.0]), np.array([0.0, y1, hr]), 3, catch_max)
    return max(total - rock, 0.0), rock, min(x2, catch_max)


@numba.njit(cache=True)
def edge_fill(h, s_o, f, catch_max):
    """Fill wedge outside an edge standing h above ground, ground rising
    outward at s_o. Returns (area, toe); area is inf if the slope cannot
    reach ground within catch_max."""
    if h <= 0.0:
        return 0.0, 0.0
    den = f + s_o
    if den <= 0.0:
        return math.inf, math.inf
    toe = h / den
    if toe > catch_max:
        return math.inf, math.inf
    return 0.5 * h * toe, toe


@numba.njit(cache=True)
def hairpin_segments(D, R, rho, L):
    """Five centerline segments of the left hairpin in the local frame.

    Rows: (x0, y0, heading0, curvature, length); curvature > 0 turns left.
    Returns (segments, alpha, R) with R = D/2 for a semicircle.
    """
    seg = np.zeros((5, 5))
    if R <= 0.5 * D + 1e-9:
        R = 0.5 * D
        alpha = 0.0
        phi = 0.5 * math.pi
        dx = 0.0
    else:
        dx = math.sqrt((R + rho) ** 2 - (0.5 * D + rho) ** 2)
        phi = math.atan2(0.5 * D + rho, dx)
        alpha = 0.5 * math.pi - phi
    xc = L + dx
    q1x = xc - R * math.cos(phi)
    q1y = 0.5 * D - R * math.sin(phi)
    q2y = 0.5 * D + R * math.sin(phi)
    seg[0, 0], seg[0, 1], seg[0, 2], seg[0, 3], seg[0, 4] = 0.0, 0.0, 0.0, 0.0, L
    seg[1, 0], seg[1, 1], seg[1, 2], seg[1, 3], seg[1, 4] = L, 0.0, 0.0, -1.0 / rho, rho * alpha
    seg[2, 0], seg[2, 1], seg[2, 2], seg[2, 3], seg[2, 4] = q1x, q1y, -alpha, 1.0 / R, R * (math.pi + 2 * alpha)
    seg[3, 0], seg[3, 1], seg[3, 2], seg[3, 3], seg[3, 4] = q1x, q2y, math.pi + alpha, -1.0 / rho, rho * alpha
    seg[4, 0], seg[4, 1], seg[4, 2], seg[4, 3], seg[4, 4] = L, D, math.pi, 0.0, L
    return seg, alpha, R


@numba.njit(cache=True, inline="always")
def _seg_point(seg, i, s):
    """(x, y, heading) at distance s along segment i."""
    th0 = seg[i, 2]
    k = seg[i, 3]
    if k == 0.0:
        return seg[i, 0] + s * math.cos(th0), seg[i, 1] + s * math.sin(th0), th0
    th = th0 + k * s
    return (seg[i, 0] + (math.sin(th) - math.sin(th0)) / k,
            seg[i, 1] - (math.cos(th) - math.cos(th0)) / k, th)


@numba.njit(cache=True)
def hairpin_grades(dz, A_loop, A_curves, g_hp, g_max):
    """Leg length and grades (leg, curves, loop) for a climb dz between the ends."""
    a = abs(dz)
    sgn = 1.0 if dz >= 0 else -1.0
    if a <= g_hp * (A_loop + A_curves):
        g = dz / (A_loop + A_curves)
        return 0.0, 0.0, g, g
    if A_curves > 0.0 and a <= g_hp * A_loop + g_max * A_curves:
        return 0.0, 0.0, (dz - sgn * g_hp * A_loop) / A_curves, sgn * g_hp
    L = (a - g_hp * A_loop - g_max * A_curves) / (2.0 * g_max)
    return L, sgn * g_max, sgn * g_max, sgn * g_hp


@numba.njit(cache=True)
def centerline_footprint(seg, nst, grades, zstart, r, c, ux, uy, res, width, nr, nc, open_, zr, soil):
    """Design minus ground over a centerline buffered by width/2.

    seg: (n, 5) segments in the local frame (x along (ux, uy), y to its
    left); nst: stations per segment; grades/zstart: design profile per
    segment. Returns (feasible, v_soil, v_rock, v_fill, max_cut, max_fill);
    infeasible if any sample lies on an excluded cell.
    """
    nx = -uy
    ny = ux
    w2 = 0.5 * width
    dt = width / N_ACROSS
    v_soil = 0.0
    v_rock = 0.0
    v_fill = 0.0
    max_cut = 0.0
    max_fill = 0.0
    for i in range(seg.shape[0]):
        n = nst[i]
        if n == 0:
            continue
        ds = seg[i, 4] / n
        kap = seg[i, 3]
        for q in range(n):
            s = (q + 0.5) * ds
            x, y, th = _seg_point(seg, i, s)
            zd = zstart[i] + grades[i] * s
            mx = -math.sin(th)
            my = math.cos(th)
            for j in range(N_ACROSS):
                t = -w2 + (j + 0.5) * dt
                lx = x + t * mx
                ly = y + t * my
                fr = r - (lx * uy + ly * ny) / res
                fc = c + (lx * ux + ly * nx) / res
                cr = int(math.floor(fr + 0.5))
                cc = int(math.floor(fc + 0.5))
                if not (0 <= cr < nr and 0 <= cc < nc and open_[cr, cc]):
                    return False, 0.0, 0.0, 0.0, 0.0, 0.0
                zg = _bil(zr, fr, fc, nr, nc)
                dA = ds * dt * (1.0 - kap * t)
                e = zd - zg
                if e > 0:
                    v_fill += e * dA
                    max_fill = max(max_fill, e)
                else:
                    dsoil = _bil(soil, fr, fc, nr, nc)
                    so = min(-e, dsoil)
                    v_soil += so * dA
                    v_rock += (-e - so) * dA
                    max_cut = max(max_cut, -e)
    ok = v_soil < math.inf and v_rock < math.inf and v_fill < math.inf
    return ok, v_soil, v_rock, v_fill, max_cut, max_fill


@numba.njit(cache=True, inline="always")
def edge_wedge(seg, i, s, sn, zd, r, c, ux, uy, res, w2, nr, nc, zr, gx, gy, soil, f, hp):
    """Cut or fill wedge outside one footprint edge (side sn = +1 left, -1 right).

    Returns (A_soil, A_rock, A_fill, reach, face height, edge length factor,
    local x, local y). A_fill is inf if the fill slope cannot reach ground.
    """
    nx = -uy
    ny = ux
    x, y, th = _seg_point(seg, i, s)
    mx = -math.sin(th)
    my = math.cos(th)
    lx = x + sn * w2 * mx
    ly = y + sn * w2 * my
    fr = r - (lx * uy + ly * ny) / res
    fc = c + (lx * ux + ly * nx) / res
    zg = _bil(zr, fr, fc, nr, nc)
    ox = sn * (mx * ux + my * nx)
    oy = sn * (mx * uy + my * ny)
    s_o = _bil(gx, fr, fc, nr, nc) * ox + _bil(gy, fr, fc, nr, nc) * oy
    h = zg - zd
    As = 0.0
    Ar = 0.0
    Af = 0.0
    if h > 0:
        As, Ar, xr = edge_cut(h, _bil(soil, fr, fc, nr, nc), s_o, hp.c_soil, hp.c_rock, hp.catch_max)
    else:
        Af, xr = edge_fill(-h, s_o, f, hp.catch_max)
    return As, Ar, Af, xr, abs(h), 1.0 - seg[i, 3] * sn * w2, lx, ly


@numba.njit(cache=True)
def centerline_culverts(seg, nst, r, c, r1, c1, ux, uy, res, drain):
    """Culverts where a centerline crosses channels, with the same run rule as
    edges (a run touching an end node is shared with the neighboring edge).
    Returns (cost, largest contributing area crossed)."""
    nx = -uy
    ny = ux
    channel, area, cbreaks, ccosts = drain
    culv = 0.0
    camax = 0.0
    in_run, run_max, run_ts, ch = _run_step(channel[r, c] != 0, area[r, c], False, 0.0, False, True,
                                            cbreaks, ccosts)
    for i in range(seg.shape[0]):
        n = nst[i]
        for q in range(n):
            s = (q + 0.5) * seg[i, 4] / n
            x, y, th = _seg_point(seg, i, s)
            cr = int(math.floor(r - (x * uy + y * ny) / res + 0.5))
            cc = int(math.floor(c + (x * ux + y * nx) / res + 0.5))
            in_run, run_max, run_ts, ch = _run_step(channel[cr, cc] != 0, area[cr, cc], in_run, run_max,
                                                    run_ts, False, cbreaks, ccosts)
            culv += ch
            if channel[cr, cc]:
                camax = max(camax, area[cr, cc])
    in_run, run_max, run_ts, ch = _run_step(channel[r1, c1] != 0, area[r1, c1], in_run, run_max,
                                            run_ts, False, cbreaks, ccosts)
    culv += ch
    if in_run:
        culv += culvert_cost(run_max, cbreaks, ccosts) * (1.0 - 0.5 * run_ts - 0.5)
    return culv, camax


@numba.njit(cache=True)
def hairpin_core(r, c, ux, uy, k, vlen, R, res, nr, nc, open_, zs, gx, gy, soil, zr, hp, drain, out):
    """Left hairpin from node (r, c), unit heading (ux, uy) (x east, y north),
    leg spacing k*vlen cells, loop radius R (m). Fills `out` (HP_N values)
    and returns the cost (inf if infeasible)."""
    for i in range(HP_N):
        out[i] = 0.0
    out[0] = math.inf
    nx = -uy
    ny = ux
    D = k * vlen * res
    w2 = 0.5 * hp.w_hp
    rho = hp.rho
    r1 = r - int(round(k * vlen * ny))
    c1 = c + int(round(k * vlen * nx))
    if not (0 <= r1 < nr and 0 <= c1 < nc and open_[r, c] and open_[r1, c1]):
        return math.inf
    seg, alpha, R = hairpin_segments(D, R, rho, 0.0)
    if R - w2 <= 0.0 or (alpha > 0.0 and rho - w2 <= 0.0) or D - hp.w_hp <= 0.0:
        return math.inf
    A_loop = seg[2, 4]
    A_curves = 2.0 * seg[1, 4]
    z0 = zr[r, c]
    z1 = zr[r1, c1]
    L, g_leg, g_curve, g_loop = hairpin_grades(z1 - z0, A_loop, A_curves, hp.g_hp, hp.g_max)
    if L > hp.leg_max:
        return math.inf
    total = 2.0 * L + A_curves + A_loop
    # The smoothed climb is bounded too, which keeps the A* heuristic consistent.
    if abs(zs[r1, c1] - zs[r, c]) > hp.g_max * total:
        return math.inf
    if L > 0.0:
        seg, alpha, R = hairpin_segments(D, R, rho, L)
    grades = np.array([g_leg, g_curve, g_loop, g_curve, g_leg])
    zstart = np.empty(5)
    acc = z0
    for i in range(5):
        zstart[i] = acc
        acc += grades[i] * seg[i, 4]

    delta = hp.delta
    nst = np.zeros(5, dtype=np.int64)
    for i in range(5):
        nst[i] = int(math.ceil(seg[i, 4] / delta)) if seg[i, 4] > 0.0 else 0

    ok, v_soil, v_rock, v_fill, max_cut, max_fill = centerline_footprint(
        seg, nst, grades, zstart, r, c, ux, uy, res, hp.w_hp, nr, nc, open_, zr, soil)
    if not ok:
        return math.inf
    loose_s = hp.k_soil * v_soil
    loose_r = hp.k_rock * v_rock
    f = (loose_s * hp.f_soil + loose_r * hp.f_rock) / (loose_s + loose_r) if loose_s + loose_r > 0 else hp.f_soil

    # --- edges. The left (+m) edge faces the loop's center and, on the two
    # arms (legs and reverse curves), the other arm across the gap. Arm
    # inner edges are collected and paired with their mirror images.
    n_arm = nst[0] + nst[1]
    arm = np.zeros((2, max(n_arm, 1), 8))   # [arm][station]: soil, rock, fill, reach, face, ds, x, y
    max_catch = 0.0
    max_toe = 0.0
    for i in range(5):
        n = nst[i]
        if n == 0:
            continue
        ds = seg[i, 4] / n
        for q in range(n):
            s = (q + 0.5) * ds
            zd = zstart[i] + grades[i] * s
            for side in range(2):
                sn = 1.0 if side == 0 else -1.0
                As, Ar, Af, xr, face, lf, lx, ly = edge_wedge(seg, i, s, sn, zd, r, c, ux, uy, res, w2, nr, nc,
                                                              zr, gx, gy, soil, f, hp)
                dl = ds * lf
                if As > 0 or Ar > 0:
                    max_catch = max(max_catch, xr)
                elif Af < math.inf:
                    max_toe = max(max_toe, xr)
                if side == 0 and i != 2:
                    a_ = 0 if i < 2 else 1
                    idx = (q if i == 0 else nst[0] + q) if i < 2 else (q if i == 3 else nst[3] + q)
                    arm[a_, idx, 0] = As * dl
                    arm[a_, idx, 1] = Ar * dl
                    arm[a_, idx, 2] = Af * dl if Af < math.inf else math.inf
                    arm[a_, idx, 3] = xr
                    arm[a_, idx, 4] = face
                    arm[a_, idx, 5] = dl
                    arm[a_, idx, 6] = lx
                    arm[a_, idx, 7] = ly
                    continue
                if Af == math.inf:
                    return math.inf
                v_soil += As * dl
                v_rock += Ar * dl
                v_fill += Af * dl
    conflict = 0.0
    wall = 0.0
    for e_ in range(n_arm):
        x_ = n_arm - 1 - e_                    # mirror station on the exit arm
        gap = math.hypot(arm[0, e_, 6] - arm[1, x_, 6], arm[0, e_, 7] - arm[1, x_, 7])
        reach = arm[0, e_, 3] + arm[1, x_, 3]
        overlap = reach > gap or arm[0, e_, 2] == math.inf or arm[1, x_, 2] == math.inf
        if overlap and hp.C_wall < math.inf:
            wall += arm[0, e_, 4] * arm[0, e_, 5] + arm[1, x_, 4] * arm[1, x_, 5]
            continue
        if arm[0, e_, 2] == math.inf or arm[1, x_, 2] == math.inf:
            return math.inf
        if overlap:
            conflict = 1.0
        for a_ in range(2):
            st = e_ if a_ == 0 else x_
            v_soil += arm[a_, st, 0]
            v_rock += arm[a_, st, 1]
            v_fill += arm[a_, st, 2]

    culv, camax = centerline_culverts(seg, nst, r, c, r1, c1, ux, uy, res, drain)

    loose = hp.k_soil * v_soil + hp.k_rock * v_rock
    borrow = max(v_fill - loose, 0.0)
    spoil = max(loose - v_fill, 0.0)
    cost = (hp.C_base * total + hp.C_soil * v_soil + hp.C_rock * v_rock + hp.C_borrow * borrow
            + culv + (hp.C_wall * wall if wall > 0.0 else 0.0)
            + hp.C_travel * total + hp.C_vertical * abs(zs[r1, c1] - zs[r, c]))
    out[0] = cost
    out[1] = L
    out[2] = R
    out[3] = total
    out[4] = v_soil
    out[5] = v_rock
    out[6] = v_fill
    out[7] = borrow
    out[8] = spoil
    out[9] = conflict
    out[10] = max_cut
    out[11] = max_fill
    out[12] = z0
    out[13] = z1
    out[14] = abs(g_loop)
    out[15] = abs(g_loop) * R / (R - w2)
    out[16] = max_catch
    out[17] = max_toe
    out[18] = culv
    out[19] = camax
    out[20] = D
    out[21] = alpha
    out[22] = rho if alpha > 0.0 else 0.0
    out[23] = wall
    return cost


@numba.njit(cache=True)
def best_hairpin(r, c, ux, uy, k, vlen, res, nr, nc, open_, zs, gx, gy, soil, zr, hp, drain, out, tmp):
    """Cheapest of the loop radii R_FACTORS * max(R_hp, D/2); result in out."""
    D = k * vlen * res
    R0 = max(hp.R_hp, 0.5 * D)
    best = math.inf
    for f_ in R_FACTORS:
        cost = hairpin_core(r, c, ux, uy, k, vlen, f_ * R0, res, nr, nc, open_, zs, gx, gy, soil, zr,
                            hp, drain, tmp)
        if cost < best:
            best = cost
            for i in range(HP_N):
                out[i] = tmp[i]
    if best == math.inf:
        out[0] = math.inf
    return best


@numba.njit(cache=True, parallel=True)
def _hairpin_table(cells, nr, nc, nh, vx, vy, slots, res, open_, zs, gx, gy, soil, zr, hp, drain):
    table = np.full((cells.size, nh, N_SLOTS), np.inf, dtype=np.float32)
    for i in prange(cells.size):
        out = np.empty(HP_N)
        tmp = np.empty(HP_N)
        r = cells[i] // nc
        c = cells[i] - r * nc
        if open_[r, c]:
            for h in range(nh):
                vlen = math.hypot(vx[h], vy[h])
                for j in range(N_SLOTS):
                    k = slots[h, j]
                    if k == 0:
                        continue
                    cost = best_hairpin(r, c, vx[h] / vlen, vy[h] / vlen, k, vlen, res, nr, nc, open_, zs,
                                        gx, gy, soil, zr, hp, drain, out, tmp)
                    if cost < math.inf:
                        table[i, h, j] = cost
    return table


@numba.njit(cache=True, parallel=True)
def _hairpin_plans(rows, cols, hs, ks, vx, vy, res, nr, nc, open_, zs, gx, gy, soil, zr, hp, drain):
    n = rows.size
    D = np.full(n, np.nan)
    R = np.full(n, np.nan)
    L = np.full(n, np.nan)
    for i in prange(n):
        out = np.zeros(HP_N)
        tmp = np.empty(HP_N)
        h = hs[i]
        vlen = math.hypot(vx[h], vy[h])
        cost = best_hairpin(rows[i], cols[i], vx[h] / vlen, vy[h] / vlen, ks[i], vlen, res, nr, nc, open_, zs,
                            gx, gy, soil, zr, hp, drain, out, tmp)
        if cost < math.inf:
            D[i] = out[_LEG_SPACING]
            R[i] = out[_RADIUS]
            L[i] = out[_LEG]
    return D, R, L


def hairpin_plans(rows, cols, hs, slots, terrain_arrays, p: Params, res: float, drainage=None):
    """Plan shape of many left hairpins at once: (leg spacing, loop radius, leg length), in meters,
    as hairpin_detail would give them (nan where infeasible). For drawing them in bulk."""
    open_, zs, gx, gy, soil, zr = terrain_arrays
    vecs = heading_vectors(p.N_theta).astype(np.float64)
    hs = np.asarray(hs, dtype=np.int64)
    ks = hairpin_slots(p, res)[hs, np.asarray(slots, dtype=np.int64)].astype(np.int64)
    return _hairpin_plans(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), hs, ks,
                          vecs[:, 0].copy(), vecs[:, 1].copy(), res, zs.shape[0], zs.shape[1], open_, zs, gx, gy,
                          np.nan_to_num(soil), zr, hairpin_params(p, res), _drain(p, drainage, zs.shape))


def _drain(p: Params, drainage, shape):
    if drainage is None:
        drainage = (np.zeros(shape, np.uint8), np.zeros(shape))
    channel, area = drainage
    return (np.ascontiguousarray(channel, dtype=np.uint8), np.ascontiguousarray(area, dtype=np.float64),
            np.array(p.culvert_area_breaks, dtype=np.float64), np.array(p.culvert_costs, dtype=np.float64))


def hairpin_table(terrain_arrays, p: Params, res: float, drainage=None, cells=None) -> np.ndarray:
    """Left-hairpin cost per (cell, heading, slot) for the given flat cells (default: the open
    ones, row-major); inf where infeasible.

    terrain_arrays: (open_, zs, gx, gy, soil, z_raw); drainage: (channel, area_ha).
    """
    open_, zs, gx, gy, soil, zr = terrain_arrays
    vecs = heading_vectors(p.N_theta).astype(np.float64)
    nr, nc = zs.shape
    cells = np.flatnonzero(open_) if cells is None else np.asarray(cells, dtype=np.int64)
    return _hairpin_table(cells.astype(np.int64), nr, nc, p.N_theta, vecs[:, 0].copy(), vecs[:, 1].copy(), hairpin_slots(p, res), res,
                          open_, zs, gx, gy, np.nan_to_num(soil), zr, hairpin_params(p, res),
                          _drain(p, drainage, zs.shape))


def hairpin_detail(r, c, h, slot, terrain_arrays, p: Params, res: float, drainage=None) -> dict:
    open_, zs, gx, gy, soil, zr = terrain_arrays
    v = heading_vectors(p.N_theta)[h].astype(float)
    vlen = float(np.hypot(*v))
    k = int(hairpin_slots(p, res)[h, slot])
    out = np.zeros(HP_N)          # stays zero, with cost inf, if infeasible
    best_hairpin(r, c, v[0] / vlen, v[1] / vlen, k, vlen, res, zs.shape[0], zs.shape[1], open_, zs, gx, gy,
                 np.nan_to_num(soil), zr, hairpin_params(p, res), _drain(p, drainage, zs.shape), out,
                 np.empty(HP_N))
    return dict(zip(HP_FIELDS, out.tolist()))


def hairpin_centerline(r, c, h, det: dict, p: Params, res: float, spacing: float = 1.0):
    """Centerline (row, col) points of a left hairpin and the design elevation at each."""
    v = heading_vectors(p.N_theta)[h].astype(float)
    u = v / np.hypot(*v)
    n = np.array([-u[1], u[0]])
    seg, alpha, R = hairpin_segments(det["leg_spacing"], det["radius"], p.R_min, det["leg"])
    A_loop, A_curves = seg[2, 4], 2.0 * seg[1, 4]
    _, g_leg, g_curve, g_loop = hairpin_grades(det["z1"] - det["z0"], A_loop, A_curves, p.g_hp, p.g_max)
    grades = [g_leg, g_curve, g_loop, g_curve, g_leg]
    pts, zs = [], []
    z = det["z0"]
    for i in range(5):
        Ls = seg[i, 4]
        if Ls <= 1e-9:
            continue
        m = max(2, int(np.ceil(Ls / spacing)) + 1)
        for s in np.linspace(0.0, Ls, m)[(0 if not pts else 1):]:
            x, y, _ = _seg_point(seg, i, s)
            pts.append(x * u + y * n)
            zs.append(z + grades[i] * s)
        z += grades[i] * Ls
    pts = np.array(pts)
    return np.column_stack([r - pts[:, 1] / res, c + pts[:, 0] / res]), np.array(zs)
