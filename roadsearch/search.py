"""Lattice search: Dijkstra and A* over (row, col, heading) (brief section 5).

Actions are motion primitives and hairpin macro-actions. Edges are evaluated
lazily inside the search loop. An edge's cost is always computed from its
canonical direction (the lower-numbered of an action and its reverse), so an
edge and its reverse have bit-identical cost.

Cost per meter comes from a table over (sidehill slope, |grade|) built with
the exact section function; soil depth at a sample is d0*exp(-S/S0) with
S = |grad z| = hypot(s, g), the same rule as the soil depth raster. Grade
is checked on the profile itself (smoothed-DEM elevation differences between
consecutive sample points), which telescopes to the node-to-node climb and
keeps the A* climb heuristic consistent.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numba
import numpy as np

from scipy import ndimage

from .culvert import _run_step, culvert_cost
from .hairpin import hairpin_slots, hairpin_table
from .lattice import (
    ACTION_BEND_L, ACTION_BEND_R, ACTION_FILL, ACTION_HP_LEFT, ACTION_PRIM, BendGeometry, LatticeArrays,
    lattice_arrays,
)
from .params import Params
from .preprocess import Terrain
from .section import section_arrays, section_params, soil_depth

TABLE_NS = 4097
TABLE_NG = 21


@dataclass
class CostTable:
    table: np.ndarray   # (TABLE_NS, TABLE_NG) cost per meter; inf where infeasible
    ds: float
    dg: float


def build_cost_table(p: Params) -> CostTable:
    s = np.linspace(0.0, p.s_mask, TABLE_NS)
    g = np.linspace(0.0, p.g_max, TABLE_NG)
    S, G = np.meshgrid(s, g, indexing="ij")
    d = soil_depth(np.hypot(S, G), p)
    out = section_arrays(S.ravel(), d.ravel(), section_params(p))
    table = out[:, 10].reshape(S.shape)
    table[-1, :] = np.inf  # s == s_mask is infeasible
    return CostTable(table, s[1] - s[0], g[1] - g[0])


@dataclass
class SearchProblem:
    """Everything the kernel needs, as plain arrays."""

    nr: int
    nc: int
    open_: np.ndarray    # uint8, 1 where the search may go
    gx: np.ndarray       # dz/dx of the smoothed DEM (NaN at nodata)
    gy: np.ndarray       # dz/dy, y north
    zs: np.ndarray       # smoothed elevation at cell centers
    soil: np.ndarray     # soil depth
    z_raw: np.ndarray    # working DEM (road elevations, fill heights)
    channel: np.ndarray  # uint8 channel cells
    area: np.ndarray     # contributing area, ha (0 at nodata)
    chdist: np.ndarray   # distance to the nearest channel cell, cells
    res: float
    params: Params
    lat: LatticeArrays
    cost: CostTable
    hp: np.ndarray       # (nr, nc, nh, slots) left-hairpin cost, float32, inf if infeasible
    lam_fill: float = math.nan   # current price of crossing fill per m3 (section 8)
    bend: np.ndarray | None = None          # (nr, nc, nh, slots) left-bend cost, float32
    bend_geo: BendGeometry | None = None
    open_tables: np.ndarray | None = None   # the mask the hairpin and bend tables were built on

    @property
    def n_theta(self) -> int:
        return self.lat.n_theta

    @property
    def terrain_arrays(self):
        return self.open_, self.zs, self.gx, self.gy, self.soil, self.z_raw

    @property
    def macro_terrain_arrays(self):
        """Terrain for re-evaluating a hairpin or bend: on the mask its table was built on.

        The tables are built once. Cells blocked later (self-overlap avoidance) are
        checked only at a macro-action's end nodes, so its exact cost and shape must
        come from the original mask, or a hairpin the search priced could come out
        infeasible."""
        open_ = self.open_tables if self.open_tables is not None else self.open_
        return open_, self.zs, self.gx, self.gy, self.soil, self.z_raw

    def lat_tuple(self):
        L = self.lat
        return (L.h0, L.h1, L.dr, L.dc, L.length, L.rev, L.s0, L.ns,
                L.sdr, L.sdc, L.tx, L.ty, L.by_heading, L.kind, L.slot, L.turn)

    def terr_tuple(self):
        c = self.cost
        p = self.params
        lam = p.C_borrow if math.isnan(self.lam_fill) else self.lam_fill
        return (self.open_, self.gx, self.gy, self.zs, self.hp, c.table, c.ds, c.dg,
                self.res, p.g_max, p.C_base,
                self.z_raw, self.channel, self.area, self.chdist,
                np.array(p.culvert_area_breaks, dtype=np.float64),
                np.array(p.culvert_costs, dtype=np.float64),
                float(lam), p.w, p.f_rock, p.H_fill_max, p.fill_tol,
                self.bend if self.bend is not None else np.full((1, 1, 1, 1), np.inf, dtype=np.float32),
                p.C_travel, p.C_vertical, p.C_turn)


def make_problem(t: Terrain, p: Params, hairpins: bool = True, crossings: bool = True,
                 bends: bool = True) -> SearchProblem:
    g = t.grid
    open_ = ((t.code == 0) & t.clip).astype(np.uint8)
    gx = np.ascontiguousarray(t.dzdx, dtype=np.float64)
    gy = np.ascontiguousarray(t.dzdy, dtype=np.float64)
    zs = np.ascontiguousarray(t.z_smooth, dtype=np.float64)
    soil = np.ascontiguousarray(np.nan_to_num(t.soil), dtype=np.float64)
    channel = t.channel.astype(np.uint8)
    area = np.ascontiguousarray(np.nan_to_num(t.area_ha), dtype=np.float64)
    if channel.any():
        chdist = ndimage.distance_transform_edt(channel == 0).astype(np.float64)
    else:
        chdist = np.full(channel.shape, np.inf)
    from .timing import stage

    slots = hairpin_slots(p, g.res) if hairpins else None
    z_raw = np.ascontiguousarray(g.z, dtype=np.float64)
    if hairpins:
        with stage("hairpin table", cells=int(open_.sum())):
            hp = hairpin_table((open_, zs, gx, gy, soil, z_raw), p, g.res, (channel, area))
    else:
        hp = np.full((1, 1, 1, 1), np.inf, dtype=np.float32)
    bend_geo = bend = None
    if bends:
        from .bend import bend_geometry, bend_table

        with stage("bend table"):
            bend_geo = bend_geometry(p, g.res)
            bend = bend_table((open_, zs, gx, gy, soil, z_raw), p, g.res, bend_geo, (channel, area))
    return SearchProblem(
        nr=g.shape[0], nc=g.shape[1], open_=open_, gx=gx, gy=gy, zs=zs, soil=soil,
        z_raw=z_raw, channel=channel, area=area, chdist=chdist,
        res=g.res, params=p,
        lat=lattice_arrays(p.N_theta, p.R_min / g.res, slots,
                           p.L_cross_max / g.res if crossings else 0.0, bend_geo, p.move_max_cells,
                           p.kink_max_deg),
        cost=build_cost_table(p), hp=hp, bend=bend, bend_geo=bend_geo, open_tables=open_.copy(),
    )


# --- kernel ------------------------------------------------------------------

@numba.njit(cache=True, inline="always")
def _bilinear(a, fr, fc, nr, nc):
    if fr < 0.0 or fc < 0.0 or fr > nr - 1 or fc > nc - 1:
        return math.nan
    r0 = min(int(fr), nr - 2) if nr > 1 else 0
    c0 = min(int(fc), nc - 2) if nc > 1 else 0
    wr = fr - r0
    wc = fc - c0
    return ((1 - wr) * ((1 - wc) * a[r0, c0] + wc * a[r0, c0 + 1])
            + wr * ((1 - wc) * a[r0 + 1, c0] + wc * a[r0 + 1, c0 + 1]))


@numba.njit(cache=True, inline="always")
def _table_cost(table, ds, dg, s, g):
    ns, ng = table.shape
    fs = s / ds
    if fs >= ns - 1:
        return math.inf
    fg = min(g / dg, ng - 1.0)
    i = int(fs)
    j = min(int(fg), ng - 2)
    ws = fs - i
    wg = fg - j
    a = table[i, j]
    b = table[i, j + 1]
    c = table[i + 1, j]
    d = table[i + 1, j + 1]
    if not (a < math.inf and b < math.inf and c < math.inf and d < math.inf):
        return math.inf
    return (1 - ws) * ((1 - wg) * a + wg * b) + ws * ((1 - wg) * c + wg * d)


@numba.njit(cache=True)
def _cell_open(open_, r, c, nr, nc):
    return 0 <= r < nr and 0 <= c < nc and open_[r, c] != 0


@numba.njit(cache=True)
def _step_ok(open_, pr, pc, r, c, nr, nc):
    """Moving between neighboring cells: both open, and a diagonal step
    needs one of the two side cells open (no squeezing through corners)."""
    if not _cell_open(open_, r, c, nr, nc):
        return False
    if pr != r and pc != c:
        return _cell_open(open_, pr, c, nr, nc) or _cell_open(open_, r, pc, nr, nc)
    return True


@numba.njit(cache=True)
def _eval_fill(r, c, p, lat, terr):
    """Fill crossing: straight macro-action over a channel on a linear design."""
    p_dr = lat[2]
    p_dc = lat[3]
    p_len = lat[4]
    (open_, gx, gy, zs, hp, table, ds, dg, res, g_max, C_base,
     z_raw, channel, area, chdist, cbreaks, ccosts, lam, w, f_rock, H_max, tol, bend, C_travel,
     C_vert, C_turn) = terr
    nr, nc = open_.shape
    Lc = p_len[p]
    if chdist[r, c] > Lc:
        return math.inf
    r1 = r + p_dr[p]
    c1 = c + p_dc[p]
    if not (_cell_open(open_, r, c, nr, nc) and _cell_open(open_, r1, c1, nr, nc)):
        return math.inf
    if channel[r, c] or channel[r1, c1]:
        return math.inf
    L = Lc * res
    # Design between raw-DEM elevations at the end nodes; the smoothed climb
    # is bounded too, which keeps the A* heuristic consistent.
    z0 = z_raw[r, c]
    z1 = z_raw[r1, c1]
    gl = g_max * L * (1.0 + 1e-12)
    if abs(z1 - z0) > gl or abs(zs[r1, c1] - zs[r, c]) > gl:
        return math.inf
    n = max(2, int(math.ceil(2.0 * Lc)))
    wseg = L / n
    vol = 0.0
    amax = -1.0
    pr = r
    pc = c
    for k in range(n):
        f = (k + 0.5) / n
        fr = r + f * p_dr[p]
        fc = c + f * p_dc[p]
        cr = int(math.floor(fr + 0.5))
        cc = int(math.floor(fc + 0.5))
        if cr != pr or cc != pc:
            if abs(cr - pr) > 1 or abs(cc - pc) > 1 or not _step_ok(open_, pr, pc, cr, cc, nr, nc):
                return math.inf
            pr = cr
            pc = cc
        if channel[cr, cc]:
            amax = max(amax, area[cr, cc])
        zg = _bilinear(z_raw, fr, fc, nr, nc)
        h = z0 + f * (z1 - z0) - zg
        if not (h >= -tol) or h > H_max:
            return math.inf
        if h > 0.0:
            vol += (w * h + h * h / f_rock) * wseg
    if (r1 != pr or c1 != pc) and (
            abs(r1 - pr) > 1 or abs(c1 - pc) > 1 or not _step_ok(open_, pr, pc, r1, c1, nr, nc)):
        return math.inf
    if amax < 0.0:
        return math.inf
    return ((C_base + C_travel) * L + C_vert * abs(zs[r1, c1] - zs[r, c]) + lam * vol
            + culvert_cost(amax, cbreaks, ccosts))


@numba.njit(cache=True)
def fill_crossing_detail(r, c, p, lat, terr):
    """(fill volume, max fill height, contributing area) of a fill crossing."""
    p_dr = lat[2]
    p_dc = lat[3]
    Lc = lat[4][p]
    zs = terr[3]
    res = terr[8]
    z_raw = terr[11]
    channel = terr[12]
    area = terr[13]
    w = terr[18]
    f_rock = terr[19]
    nr, nc = zs.shape
    r1 = r + p_dr[p]
    c1 = c + p_dc[p]
    n = max(2, int(math.ceil(2.0 * Lc)))
    wseg = Lc * res / n
    vol = 0.0
    hmax = 0.0
    amax = 0.0
    for k in range(n):
        f = (k + 0.5) / n
        fr = r + f * p_dr[p]
        fc = c + f * p_dc[p]
        cr = int(math.floor(fr + 0.5))
        cc = int(math.floor(fc + 0.5))
        if channel[cr, cc]:
            amax = max(amax, area[cr, cc])
        h = z_raw[r, c] + f * (z_raw[r1, c1] - z_raw[r, c]) - _bilinear(z_raw, fr, fc, nr, nc)
        if h > 0.0:
            vol += (w * h + h * h / f_rock) * wseg
            hmax = max(hmax, h)
    return vol, hmax, amax


@numba.njit(cache=True)
def culvert_charges(r, c, p, lat, terr):
    """At-grade culvert charge of a primitive edge, and the largest area crossed."""
    p_dr = lat[2]
    p_dc = lat[3]
    p_s0 = lat[6]
    p_ns = lat[7]
    sdr = lat[8]
    sdc = lat[9]
    channel = terr[12]
    area = terr[13]
    cbreaks = terr[15]
    ccosts = terr[16]
    r1 = r + p_dr[p]
    c1 = c + p_dc[p]
    total = 0.0
    amax = 0.0
    in_run, run_max, run_ts, ch = _run_step(channel[r, c] != 0, area[r, c], False, 0.0, False, True,
                                            cbreaks, ccosts)
    if channel[r, c]:
        amax = area[r, c]
    for k in range(p_s0[p], p_s0[p] + p_ns[p]):
        cr = int(math.floor(r + sdr[k] + 0.5))
        cc = int(math.floor(c + sdc[k] + 0.5))
        in_run, run_max, run_ts, ch = _run_step(channel[cr, cc] != 0, area[cr, cc], in_run, run_max,
                                                run_ts, False, cbreaks, ccosts)
        total += ch
        if channel[cr, cc]:
            amax = max(amax, area[cr, cc])
    in_run, run_max, run_ts, ch = _run_step(channel[r1, c1] != 0, area[r1, c1], in_run, run_max,
                                            run_ts, False, cbreaks, ccosts)
    total += ch
    if channel[r1, c1]:
        amax = max(amax, area[r1, c1])
    if in_run:
        total += culvert_cost(run_max, cbreaks, ccosts) * (1.0 - 0.5 * run_ts - 0.5)
    return total, amax


@numba.njit(cache=True)
def _eval_action(r, c, p, lat, terr):
    """Cost of action p from node (r, c), in its own direction; inf if infeasible."""
    (p_h0, p_h1, p_dr, p_dc, p_len, p_rev, p_s0, p_ns, sdr, sdc, stx, sty, by_heading, p_kind,
     p_slot, p_turn) = lat
    (open_, gx, gy, zs, hp, table, ds, dg, res, g_max, C_base,
     z_raw, channel, area, chdist, cbreaks, ccosts, lam, w, f_rock, H_max, tol, bend, C_travel,
     C_vert, C_turn) = terr
    nr, nc = open_.shape
    kind = p_kind[p]
    if kind == ACTION_HP_LEFT or kind == ACTION_BEND_L:
        # Tables are built once; the end nodes are checked against the current
        # mask so cells blocked later (self-overlap avoidance) are respected.
        if not (_cell_open(open_, r, c, nr, nc) and _cell_open(open_, r + p_dr[p], c + p_dc[p], nr, nc)):
            return math.inf
        tab = hp if kind == ACTION_HP_LEFT else bend
        if r < tab.shape[0] and c < tab.shape[1]:
            return float(tab[r, c, p_h0[p], p_slot[p]])
        return math.inf
    if kind == ACTION_FILL:
        return _eval_fill(r, c, p, lat, terr)
    r1 = r + p_dr[p]
    c1 = c + p_dc[p]
    if not (_cell_open(open_, r, c, nr, nc) and _cell_open(open_, r1, c1, nr, nc)):
        return math.inf
    L = p_len[p] * res
    n = p_ns[p]
    w_s = L / n
    gtol = g_max * (1.0 + 1e-12)
    zprev = zs[r, c]
    step = 0.5 * w_s
    total = 0.0
    vert = 0.0
    pr = r
    pc = c
    for k in range(p_s0[p], p_s0[p] + n):
        fr = r + sdr[k]
        fc = c + sdc[k]
        cr = int(math.floor(fr + 0.5))
        cc = int(math.floor(fc + 0.5))
        if cr != pr or cc != pc:
            if abs(cr - pr) > 1 or abs(cc - pc) > 1 or not _step_ok(open_, pr, pc, cr, cc, nr, nc):
                return math.inf
            pr = cr
            pc = cc
        zk = _bilinear(zs, fr, fc, nr, nc)
        ax = _bilinear(gx, fr, fc, nr, nc)
        ay = _bilinear(gy, fr, fc, nr, nc)
        if not (abs(ax) < math.inf and abs(ay) < math.inf and abs(zk) < math.inf):
            return math.inf
        if abs(zk - zprev) > gtol * step:
            return math.inf
        vert += abs(zk - zprev)
        zprev = zk
        step = w_s
        # Section geometry from the smoothed gradient. Grade enters the
        # section only through soil depth (S = hypot(s, g)).
        g = min(abs(ax * stx[k] + ay * sty[k]), g_max)
        s = abs(-ax * sty[k] + ay * stx[k])
        cpm = _table_cost(table, ds, dg, s, g)
        if cpm == math.inf:
            return math.inf
        total += cpm * w_s
    if abs(zs[r1, c1] - zprev) > gtol * 0.5 * w_s:
        return math.inf
    vert += abs(zs[r1, c1] - zprev)
    if (r1 != pr or c1 != pc) and (
            abs(r1 - pr) > 1 or abs(c1 - pc) > 1 or not _step_ok(open_, pr, pc, r1, c1, nr, nc)):
        return math.inf
    return total + C_travel * L + C_vert * vert + culvert_charges(r, c, p, lat, terr)[0]


@numba.njit(cache=True)
def edge_cost(r, c, p, lat, terr):
    """Cost of action p from (r, c), evaluated in canonical direction, plus its turning
    cost (C_turn per radian of heading change: the same both ways, never negative)."""
    p_dr = lat[2]
    p_dc = lat[3]
    q = lat[5][p]
    if q < p:
        cost = _eval_action(r + p_dr[p], c + p_dc[p], q, lat, terr)
    else:
        cost = _eval_action(r, c, p, lat, terr)
    return cost + terr[25] * lat[15][p]


@numba.njit(cache=True, inline="always")
def _less(k1, i1, k2, i2):
    return k1 < k2 or (k1 == k2 and i1 < i2)


@numba.njit(cache=True)
def _heap_push(hk, hi, size, key, idx):
    j = size
    hk[j] = key
    hi[j] = idx
    while j > 0:
        parent = (j - 1) >> 1
        if _less(hk[j], hi[j], hk[parent], hi[parent]):
            hk[j], hk[parent] = hk[parent], hk[j]
            hi[j], hi[parent] = hi[parent], hi[j]
            j = parent
        else:
            break
    return size + 1


@numba.njit(cache=True)
def _heap_pop(hk, hi, size):
    key = hk[0]
    idx = hi[0]
    size -= 1
    hk[0] = hk[size]
    hi[0] = hi[size]
    j = 0
    while True:
        l = 2 * j + 1
        if l >= size:
            break
        m = l
        if l + 1 < size and _less(hk[l + 1], hi[l + 1], hk[l], hi[l]):
            m = l + 1
        if _less(hk[m], hi[m], hk[j], hi[j]):
            hk[j], hk[m] = hk[m], hk[j]
            hi[j], hi[m] = hi[m], hi[j]
            j = m
        else:
            break
    return key, idx, size


@numba.njit(cache=True)
def _heuristic(r, c, goal_r, goal_c, goal_z, zs, res, g_max, C_len, C_vert):
    """Lower bound on the cost to the goal: every meter of road costs at least
    C_base + C_travel, the road is at least as long as the straight distance
    and as |dz| / g_max, and its vertical travel is at least |dz|."""
    dist = math.hypot(r - goal_r, c - goal_c) * res
    dz = abs(goal_z - zs[r, c])
    return C_len * max(dist, dz / g_max) + C_vert * dz


@numba.njit(cache=True)
def _search(nh, lat, terr, sources, goal_r, goal_c, astar, max_cost):
    """Dijkstra (astar=False) or A* (astar=True, stops at the goal cell).

    sources: state ids starting at cost 0. Returns (G, parent action,
    closed flags, expansions, goal state or -1).
    """
    p_h1 = lat[1]
    p_dr = lat[2]
    p_dc = lat[3]
    by_heading = lat[12]
    open_ = terr[0]
    zs = terr[3]
    res = terr[8]
    g_max = terr[9]
    C_len = terr[10] + terr[23]
    C_vert = terr[24]
    nr, nc = open_.shape
    n = nr * nc * nh
    G = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int16)
    closed = np.zeros(n, dtype=np.uint8)
    cap = 1 << 16
    hk = np.empty(cap)
    hi = np.empty(cap, dtype=np.int64)
    size = 0
    goal_z = zs[goal_r, goal_c] if goal_r >= 0 else 0.0
    for s in sources:
        G[s] = 0.0
        key = 0.0
        if astar:
            cell = s // nh
            key = _heuristic(cell // nc, cell % nc, goal_r, goal_c, goal_z, zs, res, g_max, C_len, C_vert)
        size = _heap_push(hk, hi, size, key, s)
    n_exp = 0
    goal_state = -1
    while size > 0:
        key, u, size = _heap_pop(hk, hi, size)
        if closed[u]:
            continue
        if key > max_cost:
            break
        closed[u] = 1
        n_exp += 1
        cell = u // nh
        h = u - cell * nh
        r = cell // nc
        c = cell - r * nc
        if r == goal_r and c == goal_c and (goal_state < 0 or G[u] < G[goal_state]):
            goal_state = u
            if astar:
                break
        gu = G[u]
        for j in range(by_heading.shape[1]):
            p = by_heading[h, j]
            if p < 0:
                break
            r2 = r + p_dr[p]
            c2 = c + p_dc[p]
            if r2 < 0 or r2 >= nr or c2 < 0 or c2 >= nc:
                continue
            v = (r2 * nc + c2) * nh + p_h1[p]
            if closed[v]:
                continue
            w = edge_cost(r, c, p, lat, terr)
            if w == math.inf:
                continue
            ng = gu + w
            if ng < G[v]:
                G[v] = ng
                parent[v] = p
                if size == cap:
                    cap *= 2
                    nk = np.empty(cap)
                    ni = np.empty(cap, dtype=np.int64)
                    nk[:size] = hk[:size]
                    ni[:size] = hi[:size]
                    hk = nk
                    hi = ni
                f = ng
                if astar:
                    f += _heuristic(r2, c2, goal_r, goal_c, goal_z, zs, res, g_max, C_len, C_vert)
                size = _heap_push(hk, hi, size, f, v)
    return G, parent, closed, n_exp, goal_state


# --- Python API ----------------------------------------------------------------

@dataclass
class SearchResult:
    G: np.ndarray          # (nr, nc, nh) cost-to-come
    parent: np.ndarray     # (nr, nc, nh) action used to arrive, -1 at sources/unreached
    closed: np.ndarray     # (nr, nc, nh) bool
    n_expanded: int
    goal_state: tuple[int, int, int] | None

    @property
    def goal_cost(self) -> float:
        return float(self.G[self.goal_state]) if self.goal_state else math.inf

    def min_over_headings(self, closed_only=False) -> np.ndarray:
        G = np.where(self.closed, self.G, np.inf) if closed_only else self.G
        return G.min(axis=2)


def _default_engine() -> str:
    try:
        import roadsearch_core  # noqa: F401
        return "rust"
    except ImportError:
        return "numba"


# Which kernel searches: "rust" (the roadsearch_core package built from rust/, used when it is
# installed) or "numba" (this module). Both give the same costs and parents, bit for bit; the
# Rust one is about ten times faster.
ENGINE = os.environ.get("ROADSEARCH_ENGINE") or _default_engine()


def rust_args(prob: SearchProblem):
    """The lattice and terrain tuples with the dtypes and memory layout roadsearch_core expects
    (arrays that already have them are passed through, not copied)."""
    f8, i8 = np.float64, np.int64
    lat = tuple(np.ascontiguousarray(a, dtype=f8 if i in (4, 8, 9, 10, 11, 15) else i8)
                for i, a in enumerate(prob.lat_tuple()))
    t = list(prob.terr_tuple())
    for i, dt in {0: np.uint8, 1: f8, 2: f8, 3: f8, 4: np.float32, 5: f8, 11: f8, 12: np.uint8, 13: f8,
                  14: f8, 15: f8, 16: f8, 22: np.float32}.items():
        t[i] = np.ascontiguousarray(t[i], dtype=dt)
    for i in (6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 23, 24, 25):
        t[i] = float(t[i])
    return lat, tuple(t)


def _search_rust(prob: SearchProblem, sources, gr, gc, astar, max_cost):
    """roadsearch_core.search, its per-cell results spread back onto the whole grid."""
    import roadsearch_core

    nh = prob.n_theta
    lat, terr = rust_args(prob)
    cells, g, par, cl, n_exp, gs = roadsearch_core.search(nh, lat, terr, sources, gr, gc, astar, float(max_cost))
    n = prob.nr * prob.nc * nh
    idx = (cells[:, None] * nh + np.arange(nh)).reshape(-1)
    G = np.full(n, np.inf)
    G[idx] = g
    parent = np.full(n, -1, dtype=np.int16)
    parent[idx] = par
    closed = np.zeros(n, dtype=np.uint8)
    closed[idx] = cl
    return G, parent, closed, n_exp, gs


def run_search(prob: SearchProblem, source_rc, goal_rc=None, astar=False, max_cost=math.inf,
               source_heading: int | None = None, engine: str | None = None) -> SearchResult:
    """Search from cell source_rc (all headings, or only source_heading)."""
    nh = prob.n_theta
    r0, c0 = source_rc
    hs = range(nh) if source_heading is None else [source_heading]
    sources = np.array([(r0 * prob.nc + c0) * nh + h for h in hs], dtype=np.int64)
    gr, gc = goal_rc if goal_rc is not None else (-1, -1)
    if astar and goal_rc is None:
        raise ValueError("A* needs a goal")
    engine = engine or ENGINE
    if engine == "rust":
        G, parent, closed, n_exp, gs = _search_rust(prob, sources, gr, gc, astar, max_cost)
    elif engine == "numba":
        G, parent, closed, n_exp, gs = _search(nh, prob.lat_tuple(), prob.terr_tuple(),
                                               sources, gr, gc, astar, max_cost)
    else:
        raise ValueError(f"unknown search engine {engine!r} (numba or rust)")
    shape = (prob.nr, prob.nc, nh)
    goal = None
    if gs >= 0:
        cell, h = divmod(int(gs), nh)
        goal = (cell // prob.nc, cell % prob.nc, h)
    return SearchResult(G.reshape(shape), parent.reshape(shape), closed.reshape(shape).astype(bool),
                        int(n_exp), goal)


def reconstruct(res: SearchResult, lat: LatticeArrays, state) -> list[tuple[int, int, int]]:
    """Edges (row, col, action) from a source to `state`, in travel order."""
    r, c, h = state
    edges = []
    while True:
        p = int(res.parent[r, c, h])
        if p < 0:
            break
        r, c, h = r - int(lat.dr[p]), c - int(lat.dc[p]), int(lat.h0[p])
        edges.append((r, c, p))
    edges.reverse()
    return edges


def reverse_edges(edges, lat: LatticeArrays) -> list[tuple[int, int, int]]:
    """The same route traveled the other way."""
    out = []
    for r, c, p in reversed(edges):
        out.append((r + int(lat.dr[p]), c + int(lat.dc[p]), int(lat.rev[p])))
    return out


def single_edge_cost(prob: SearchProblem, r, c, p) -> float:
    return edge_cost(r, c, p, prob.lat_tuple(), prob.terr_tuple())


def path_cost(prob: SearchProblem, edges) -> float:
    lt, tt = prob.lat_tuple(), prob.terr_tuple()
    return float(sum(edge_cost(r, c, p, lt, tt) for r, c, p in edges))


def canonical_hairpin(prob: SearchProblem, r, c, p):
    """(row, col, heading, slot) of the left hairpin equivalent to action p from (r, c)."""
    L = prob.lat
    if L.kind[p] == ACTION_HP_LEFT:
        return r, c, int(L.h0[p]), int(L.slot[p])
    return r + int(L.dr[p]), c + int(L.dc[p]), int(L.h0[p]), int(L.slot[p])


def hairpin_geometry(prob: SearchProblem, r, c, p):
    """(detail, points (row, col), design z) of hairpin action p from (r, c), in travel order."""
    from .hairpin import hairpin_centerline, hairpin_detail

    hr, hc, h, slot = canonical_hairpin(prob, r, c, p)
    det = hairpin_detail(hr, hc, h, slot, prob.macro_terrain_arrays, prob.params, prob.res,
                         (prob.channel, prob.area))
    if not det["cost"] < math.inf:
        raise ValueError(f"hairpin at row {hr}, col {hc}, heading {h}, slot {slot} is infeasible")
    pts, z = hairpin_centerline(hr, hc, h, det, prob.params, prob.res)
    if prob.lat.kind[p] != ACTION_HP_LEFT:
        pts, z = pts[::-1], z[::-1]
    return det, pts, z


def canonical_bend(prob: SearchProblem, r, c, p):
    """(row, col, heading, slot) of the left bend equivalent to bend action p from (r, c)."""
    L = prob.lat
    if L.kind[p] == ACTION_BEND_L:
        return r, c, int(L.h0[p]), int(L.slot[p])
    q = int(L.rev[p])
    return r + int(L.dr[p]), c + int(L.dc[p]), int(L.h0[q]), int(L.slot[q])


def bend_geometry_of(prob: SearchProblem, r, c, p):
    """(detail, points (row, col), design z) of bend action p from (r, c), in travel order."""
    from .bend import bend_centerline, bend_detail

    br, bc, h, slot = canonical_bend(prob, r, c, p)
    det = bend_detail(br, bc, h, slot, prob.macro_terrain_arrays, prob.params, prob.res, prob.bend_geo,
                      (prob.channel, prob.area))
    if not det["cost"] < math.inf:
        raise ValueError(f"bend at row {br}, col {bc}, heading {h}, slot {slot} is infeasible")
    pts, z = bend_centerline(br, bc, h, slot, det, prob.params, prob.res, prob.bend_geo)
    if prob.lat.kind[p] != ACTION_BEND_L:
        pts, z = pts[::-1], z[::-1]
    return det, pts, z


def edge_points(prob: SearchProblem, r, c, p) -> np.ndarray:
    """(row, col) float points of one edge, from its start node to its end node."""
    L = prob.lat
    if L.kind[p] == ACTION_FILL:
        n = max(2, int(np.ceil(2 * L.length[p])))
        f = np.linspace(0.0, 1.0, n + 1)
        return np.column_stack([r + f * L.dr[p], c + f * L.dc[p]])
    if L.kind[p] == ACTION_PRIM:
        s0, n = L.s0[p], L.ns[p]
        mid = np.column_stack([r + L.sdr[s0:s0 + n], c + L.sdc[s0:s0 + n]])
        return np.vstack([[r, c], mid, [r + L.dr[p], c + L.dc[p]]]).astype(float)
    if L.kind[p] in (ACTION_BEND_L, ACTION_BEND_R):
        return bend_geometry_of(prob, r, c, p)[1]
    return hairpin_geometry(prob, r, c, p)[1]


def path_xy(prob: SearchProblem, edges, start_rc) -> np.ndarray:
    """Polyline of the route in grid (row, col) float coordinates."""
    pts = [np.array([start_rc], dtype=float)]
    for r, c, p in edges:
        pts.append(edge_points(prob, r, c, p)[1:])
    return np.vstack(pts)
