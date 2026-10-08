"""Bends: sharp turns as macro-actions with their own design profile.

A route that must turn through the fall line on ground steeper than g_max
cannot do it with ordinary curves, whose profile follows the ground: some
heading in between is too steep. Without a macro it may loop round instead.
A bend turns 90 or about 135 degrees (lattice.BEND_ANGLES, in heading bins): a straight
lead-in, an arc of radius R_min and a straight lead-out (lattice geometry in
lattice.build_bends). Its design runs at one grade from the raw-DEM
elevation of the start node to that of the end node, at most g_max, and
leaves the ground where it has to. The earthwork over the footprint (road
width w) is priced as for hairpins.

Right bends are the reverses of left bends, so one table of left bends
prices both directions identically.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numba
import numpy as np
from numba import prange

from .engine import use_rust
from .hairpin import _drain, _rust_macro_args, _seg_point, centerline_culverts, centerline_footprint, edge_wedge
from .lattice import BendGeometry, build_bends, heading_vectors
from .params import Params

BendParams = namedtuple(
    "BendParams",
    "w g_max c_soil c_rock f_soil f_rock k_soil k_rock C_base C_soil C_rock C_borrow catch_max delta "
    "C_travel C_vertical",
)

BEND_FIELDS = ("cost", "length", "cut_soil_m3", "cut_rock_m3", "fill_m3", "borrow_m3", "spoil_m3",
               "max_cut", "max_fill", "z0", "z1", "grade", "max_catch", "max_toe", "culvert",
               "culvert_area", "angle_deg", "radius")
BEND_N = len(BEND_FIELDS)


def bend_params(p: Params, res: float) -> BendParams:
    from .hairpin import N_ACROSS
    return BendParams(p.w, p.g_max, p.c_soil, p.c_rock, p.f_soil, p.f_rock, p.k_soil, p.k_rock,
                      p.C_base, p.C_soil, p.C_rock, p.C_borrow, p.catch_max, min(0.5 * res, p.w / N_ACROSS),
                      p.C_travel, p.C_vertical)


def bend_geometry(p: Params, res: float) -> BendGeometry:
    return build_bends(p.N_theta, p.R_min / res)


@numba.njit(cache=True)
def bend_segments(L1, R, dth, L2):
    """Three segments (straight, arc, straight) in the local frame, meters."""
    seg = np.zeros((3, 5))
    seg[0, 4] = L1
    seg[1, 0] = L1
    seg[1, 3] = (1.0 if dth > 0 else -1.0) / R
    seg[1, 4] = R * abs(dth)
    x, y, th = _seg_point(seg, 1, seg[1, 4])
    seg[2, 0] = x
    seg[2, 1] = y
    seg[2, 2] = th
    seg[2, 4] = L2
    return seg


@numba.njit(cache=True)
def bend_core(r, c, ux, uy, X, Y, L1, R, dth, L2, res, nr, nc, open_, zs, gx, gy, soil, zr, bp, drain, out):
    """Left bend from node (r, c), unit heading (ux, uy), ending at (r - Y, c + X).
    Lengths in cells. Fills out (BEND_N values); returns the cost or inf."""
    for i in range(BEND_N):
        out[i] = 0.0
    out[0] = math.inf
    r1 = r - Y
    c1 = c + X
    if not (0 <= r1 < nr and 0 <= c1 < nc and open_[r, c] and open_[r1, c1]):
        return math.inf
    seg = bend_segments(L1 * res, R * res, dth, L2 * res)
    total = seg[0, 4] + seg[1, 4] + seg[2, 4]
    z0 = zr[r, c]
    z1 = zr[r1, c1]
    g = (z1 - z0) / total
    if abs(g) > bp.g_max * (1.0 + 1e-12) or abs(zs[r1, c1] - zs[r, c]) > bp.g_max * total:
        return math.inf
    grades = np.array([g, g, g])
    zstart = np.array([z0, z0 + g * seg[0, 4], z0 + g * (seg[0, 4] + seg[1, 4])])
    nst = np.zeros(3, dtype=np.int64)
    for i in range(3):
        nst[i] = int(math.ceil(seg[i, 4] / bp.delta)) if seg[i, 4] > 0.0 else 0
    ok, v_soil, v_rock, v_fill, max_cut, max_fill = centerline_footprint(
        seg, nst, grades, zstart, r, c, ux, uy, res, bp.w, nr, nc, open_, zr, soil)
    if not ok:
        return math.inf
    loose_s = bp.k_soil * v_soil
    loose_r = bp.k_rock * v_rock
    f = (loose_s * bp.f_soil + loose_r * bp.f_rock) / (loose_s + loose_r) if loose_s + loose_r > 0 else bp.f_soil
    w2 = 0.5 * bp.w
    max_catch = 0.0
    max_toe = 0.0
    for i in range(3):
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
                                                              zr, gx, gy, soil, f, bp)
                if Af == math.inf:
                    return math.inf
                if As > 0 or Ar > 0:
                    max_catch = max(max_catch, xr)
                elif Af > 0:
                    max_toe = max(max_toe, xr)
                dl = ds * lf
                v_soil += As * dl
                v_rock += Ar * dl
                v_fill += Af * dl
    culv, camax = centerline_culverts(seg, nst, r, c, r1, c1, ux, uy, res, drain)
    loose = bp.k_soil * v_soil + bp.k_rock * v_rock
    borrow = max(v_fill - loose, 0.0)
    spoil = max(loose - v_fill, 0.0)
    cost = (bp.C_base * total + bp.C_soil * v_soil + bp.C_rock * v_rock + bp.C_borrow * borrow + culv
            + bp.C_travel * total + bp.C_vertical * abs(zs[r1, c1] - zs[r, c]))
    out[0] = cost
    out[1] = total
    out[2] = v_soil
    out[3] = v_rock
    out[4] = v_fill
    out[5] = borrow
    out[6] = spoil
    out[7] = max_cut
    out[8] = max_fill
    out[9] = z0
    out[10] = z1
    out[11] = g
    out[12] = max_catch
    out[13] = max_toe
    out[14] = culv
    out[15] = camax
    out[16] = math.degrees(abs(dth))
    out[17] = R * res
    return cost


@numba.njit(cache=True, parallel=True)
def _bend_table(cells, nr, nc, nh, vx, vy, BX, BY, BL1, BL2, BDT, R, res, open_, zs, gx, gy, soil, zr, bp,
                drain):
    ns = BX.shape[1]
    table = np.full((cells.size, nh, ns), np.inf, dtype=np.float32)
    for i in prange(cells.size):
        out = np.empty(BEND_N)
        r = cells[i] // nc
        c = cells[i] - r * nc
        if open_[r, c]:
            for h in range(nh):
                vlen = math.hypot(vx[h], vy[h])
                for j in range(ns):
                    cost = bend_core(r, c, vx[h] / vlen, vy[h] / vlen, BX[h, j], BY[h, j], BL1[h, j], R,
                                     BDT[h, j], BL2[h, j], res, nr, nc, open_, zs, gx, gy, soil, zr, bp, drain, out)
                    if cost < math.inf:
                        table[i, h, j] = cost
    return table


def bend_table(terrain_arrays, p: Params, res: float, geo: BendGeometry, drainage=None, cells=None,
               engine=None) -> np.ndarray:
    """Left-bend cost per (cell, heading, slot) for the given flat cells (default: the open
    ones, row-major); inf where infeasible. Built by the Rust core when it is the engine."""
    open_, zs, gx, gy, soil, zr = terrain_arrays
    vecs = heading_vectors(p.N_theta).astype(np.float64)
    nr, nc = zs.shape
    cells = np.flatnonzero(open_) if cells is None else np.asarray(cells, dtype=np.int64)
    if use_rust(engine):
        import roadsearch_core

        terrain, drain = _rust_macro_args(terrain_arrays, p, drainage)
        shapes = tuple(np.ascontiguousarray(a, dtype=dt) for a, dt in
                       ((geo.X, np.int64), (geo.Y, np.int64), (geo.L1, np.float64), (geo.L2, np.float64),
                        (geo.dth, np.float64)))
        return roadsearch_core.bend_table(cells.astype(np.int64), vecs[:, 0].copy(), vecs[:, 1].copy(), shapes,
                                          float(geo.R), float(res), terrain, drain,
                                          tuple(float(v) for v in bend_params(p, res)))
    return _bend_table(cells.astype(np.int64), nr, nc, p.N_theta, vecs[:, 0].copy(), vecs[:, 1].copy(), geo.X, geo.Y, geo.L1, geo.L2,
                       geo.dth, geo.R, res, open_, zs, gx, gy, np.nan_to_num(soil), zr, bend_params(p, res),
                       _drain(p, drainage, zs.shape))


def bend_detail(r, c, h, slot, terrain_arrays, p: Params, res: float, geo: BendGeometry, drainage=None) -> dict:
    open_, zs, gx, gy, soil, zr = terrain_arrays
    v = heading_vectors(p.N_theta)[h].astype(float)
    vlen = float(np.hypot(*v))
    out = np.empty(BEND_N)
    bend_core(r, c, v[0] / vlen, v[1] / vlen, int(geo.X[h, slot]), int(geo.Y[h, slot]), float(geo.L1[h, slot]),
              geo.R, float(geo.dth[h, slot]), float(geo.L2[h, slot]), res, zs.shape[0], zs.shape[1], open_, zs,
              gx, gy, np.nan_to_num(soil), zr, bend_params(p, res), _drain(p, drainage, zs.shape), out)
    return dict(zip(BEND_FIELDS, out.tolist()))


def bend_centerline(r, c, h, slot, det: dict, p: Params, res: float, geo: BendGeometry, spacing: float = 1.0):
    """Centerline (row, col) points of a left bend and the design elevation at each."""
    v = heading_vectors(p.N_theta)[h].astype(float)
    u = v / np.hypot(*v)
    n = np.array([-u[1], u[0]])
    seg = bend_segments(geo.L1[h, slot] * res, geo.R * res, geo.dth[h, slot], geo.L2[h, slot] * res)
    pts, s_all = [], []
    acc = 0.0
    for i in range(3):
        Ls = seg[i, 4]
        if Ls <= 1e-9:
            continue
        m = max(2, int(np.ceil(Ls / spacing)) + 1)
        for s in np.linspace(0.0, Ls, m)[(0 if not pts else 1):]:
            x, y, _ = _seg_point(seg, i, s)
            pts.append(x * u + y * n)
            s_all.append(acc + s)
        acc += Ls
    pts = np.array(pts)
    z = det["z0"] + det["grade"] * np.array(s_all)
    return np.column_stack([r - pts[:, 1] / res, c + pts[:, 0] / res]), z
