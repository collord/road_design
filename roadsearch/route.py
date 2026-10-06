"""Exact evaluation of a found route, segment by segment, and stacked-leg
conflict detection (brief sections 6 and 10).

The search prices edges with an interpolated cost table; here every segment
is re-evaluated with the exact section function at the same sample points,
so the reported volumes and costs need no recomputation downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString

from .lattice import ACTION_BEND_L, ACTION_BEND_R, ACTION_FILL, ACTION_HP_LEFT, ACTION_HP_RIGHT, ACTION_PRIM
from .search import (
    SearchProblem, bend_geometry_of, culvert_charges, edge_points, fill_crossing_detail, hairpin_geometry,
    single_edge_cost,
)
from .section import SECTION_NAMES, section_arrays, section_params, soil_depth

# Severity order used to label a segment whose samples have mixed section types.
_SEVERITY = {"flat": 0, "half_bench": 1, "full_bench": 2}


def _bil(a, fr, fc):
    nr, nc = a.shape
    r0 = np.clip(np.floor(fr).astype(int), 0, nr - 2)
    c0 = np.clip(np.floor(fc).astype(int), 0, nc - 2)
    wr, wc = fr - r0, fc - c0
    return ((1 - wr) * ((1 - wc) * a[r0, c0] + wc * a[r0, c0 + 1])
            + wr * ((1 - wc) * a[r0 + 1, c0] + wc * a[r0 + 1, c0 + 1]))


@dataclass
class Segment:
    seq: int
    r: int
    c: int
    action: int
    section_type: str
    pts_rc: np.ndarray          # (n, 2) row/col float points, start node to end node
    z: np.ndarray               # road elevation at those points
    length: float
    grade: float                # mean grade in the direction of travel (rise/run)
    sidehill_slope: float
    terrain_slope: float
    soil_depth: float
    cut_width: float
    fill_width: float
    cut_catch: float
    fill_toe: float
    cut_soil_m3: float
    cut_rock_m3: float
    fill_m3: float
    spoil_m3: float
    borrow_m3: float
    cost_base: float
    cost_earthwork: float       # excavation (+ borrow for hairpins; + fill at lam for crossings)
    cost_surrogate: float       # the search's edge cost
    cost_culvert: float = 0.0
    culvert_area_ha: float = 0.0
    vertical_travel: float = 0.0    # rise + fall along the profile (m)
    turning: float = 0.0            # change in heading (radians)
    cost_operating: float = 0.0     # C_travel * length + C_vertical * vertical_travel + C_turn * turning
    extra: dict = field(default_factory=dict)

    @property
    def cost_construction(self) -> float:
        return self.cost_base + self.cost_earthwork + self.cost_culvert

    @property
    def cost_total(self) -> float:
        return self.cost_construction + self.cost_operating


def evaluate_edge(prob: SearchProblem, seq: int, r: int, c: int, a: int) -> Segment:
    """Exact construction quantities and costs of one edge, plus its operating cost."""
    seg = _construction(prob, seq, r, c, a)
    seg.vertical_travel = vertical_travel(prob, r, c, a)
    seg.turning = float(prob.lat.turn[a])
    p = prob.params
    seg.cost_operating = p.C_travel * seg.length + p.C_vertical * seg.vertical_travel + p.C_turn * seg.turning
    return seg


def vertical_travel(prob: SearchProblem, r: int, c: int, a: int) -> float:
    """Rise + fall of the smoothed-DEM profile, as the search prices it: over
    the samples of a primitive, end to end for macro-actions (whose design is
    monotone)."""
    L = prob.lat
    r1, c1 = r + int(L.dr[a]), c + int(L.dc[a])
    if L.kind[a] != ACTION_PRIM:
        return float(abs(prob.zs[r1, c1] - prob.zs[r, c]))
    s0, n = L.s0[a], L.ns[a]
    z = _bil(prob.zs, r + L.sdr[s0:s0 + n], c + L.sdc[s0:s0 + n])
    prof = np.concatenate([[prob.zs[r, c]], z, [prob.zs[r1, c1]]])
    return float(np.abs(np.diff(prof)).sum())


def _construction(prob: SearchProblem, seq: int, r: int, c: int, a: int) -> Segment:
    p = prob.params
    L = prob.lat
    pts = edge_points(prob, r, c, a)
    z_road = _bil(prob.z_raw, pts[:, 0], pts[:, 1])
    surrogate = single_edge_cost(prob, r, c, a)
    if L.kind[a] in (ACTION_HP_LEFT, ACTION_HP_RIGHT):
        det, pts, z_road = hairpin_geometry(prob, r, c, a)
        sign = 1.0 if L.kind[a] == ACTION_HP_LEFT else -1.0
        earth = (p.C_soil * det["cut_soil_m3"] + p.C_rock * det["cut_rock_m3"]
                 + p.C_borrow * det["borrow_m3"] + (p.C_wall * det["wall_m2"] if det["wall_m2"] > 0 else 0.0))
        return Segment(
            seq, r, c, a, "hairpin", pts, z_road, det["length"],
            sign * (det["z1"] - det["z0"]) / det["length"], math.nan, math.nan, math.nan,
            p.w_hp, 0.0, det["max_catch"], det["max_toe"],
            det["cut_soil_m3"], det["cut_rock_m3"], det["fill_m3"], det["spoil_m3"], det["borrow_m3"],
            p.C_base * det["length"], earth, surrogate, det["culvert"], det["culvert_area"],
            extra={k: det[k] for k in ("leg", "radius", "arc_grade", "inner_edge_grade", "conflict",
                                       "max_cut", "max_fill", "leg_spacing", "alpha", "rho", "wall_m2")},
        )
    if L.kind[a] in (ACTION_BEND_L, ACTION_BEND_R):
        det, pts, z_road = bend_geometry_of(prob, r, c, a)
        sign = 1.0 if L.kind[a] == ACTION_BEND_L else -1.0
        earth = p.C_soil * det["cut_soil_m3"] + p.C_rock * det["cut_rock_m3"] + p.C_borrow * det["borrow_m3"]
        return Segment(
            seq, r, c, a, "bend", pts, z_road, det["length"], sign * det["grade"],
            math.nan, math.nan, math.nan, p.w, 0.0, det["max_catch"], det["max_toe"],
            det["cut_soil_m3"], det["cut_rock_m3"], det["fill_m3"], det["spoil_m3"], det["borrow_m3"],
            p.C_base * det["length"], earth, surrogate, det["culvert"], det["culvert_area"],
            extra={k: det[k] for k in ("angle_deg", "radius", "max_cut", "max_fill")},
        )
    lt, tt = prob.lat_tuple(), prob.terr_tuple()
    if L.kind[a] == ACTION_FILL:
        vol, hmax, amax = fill_crossing_detail(r, c, a, lt, tt)
        length = L.length[a] * prob.res
        z0, z1 = prob.z_raw[r, c], prob.z_raw[r + L.dr[a], c + L.dc[a]]
        z_road = z0 + (z1 - z0) * np.linspace(0.0, 1.0, len(pts))
        lam = tt[17]
        culv = p.culvert_cost(amax)
        return Segment(
            seq, r, c, a, "fill_crossing", pts, z_road, length, (z1 - z0) / length,
            math.nan, math.nan, math.nan, p.w, 0.0, 0.0, hmax / p.f_rock,
            0.0, 0.0, vol, 0.0, 0.0, p.C_base * length, lam * vol, surrogate, culv, amax,
            extra={"max_fill": hmax, "lam_fill": lam},
        )
    s0, n = L.s0[a], L.ns[a]
    fr = r + L.sdr[s0:s0 + n]
    fc = c + L.sdc[s0:s0 + n]
    ax, ay = _bil(prob.gx, fr, fc), _bil(prob.gy, fr, fc)
    tx, ty = L.tx[s0:s0 + n], L.ty[s0:s0 + n]
    g = ax * tx + ay * ty
    s = np.abs(-ax * ty + ay * tx)
    S = np.hypot(s, g)
    d = soil_depth(S, p)
    sec = section_arrays(s, d, section_params(p))
    length = L.length[a] * prob.res
    w = length / n
    kinds = [SECTION_NAMES[int(k)] for k in sec[:, 0]]
    counts = {k: kinds.count(k) for k in set(kinds)}
    label = max(counts, key=lambda k: (counts[k], _SEVERITY.get(k, 9)))
    zs = prob.zs
    culv, amax = culvert_charges(r, c, a, lt, tt)
    return Segment(
        seq, r, c, a, label, pts, z_road, length,
        (zs[r + L.dr[a], c + L.dc[a]] - zs[r, c]) / length,
        float(s.mean()), float(S.mean()), float(d.mean()),
        float(sec[:, 1].mean()), float(sec[:, 2].mean()), float(sec[:, 7].max()), float(sec[:, 8].max()),
        float(sec[:, 3].sum() * w), float(sec[:, 4].sum() * w), float(sec[:, 5].sum() * w),
        float(sec[:, 6].sum() * w), 0.0,
        p.C_base * length, float(((sec[:, 10] - p.C_base).sum()) * w), surrogate, culv, amax,
    )


def evaluate_route(prob: SearchProblem, edges) -> list[Segment]:
    return [evaluate_edge(prob, i, r, c, a) for i, (r, c, a) in enumerate(edges)]


@dataclass
class Conflict:
    upper: int          # seq of the higher segment
    lower: int
    separation: float   # m, horizontal
    required: float     # upper fill toe + lower cut catch
    row: float          # midpoint of the closest approach, grid row (float)
    col: float          # grid col (float)


def stacked_leg_conflicts(segs: list[Segment], res: float, min_ratio: float = 1.3) -> list[Conflict]:
    """Pairs of non-adjacent segments closer than upper fill toe + lower cut catch.

    Pairs that are close only because they are near each other along the
    route (separation comparable to the distance along the road between
    them) are not stacked legs; a pair counts only when the road between
    them is at least min_ratio times their separation, i.e. it doubles back.
    """
    from shapely import STRtree
    from shapely.ops import nearest_points

    lines = [LineString(sg.pts_rc[:, ::-1] * res) for sg in segs]
    station = np.concatenate([[0.0], np.cumsum([sg.length for sg in segs])])
    zmean = np.array([float(np.mean(sg.z)) for sg in segs])
    reach = max((sg.cut_catch for sg in segs), default=0) + max((sg.fill_toe for sg in segs), default=0)
    tree = STRtree(lines)
    out = []
    for i, li in enumerate(lines):
        for j in tree.query(li.buffer(reach)):
            j = int(j)
            if j <= i + 1:
                continue
            sep = li.distance(lines[j])
            upper, lower = (i, j) if zmean[i] >= zmean[j] else (j, i)
            need = segs[upper].fill_toe + segs[lower].cut_catch
            gap = station[j] - station[i + 1]
            if sep >= need or gap < min_ratio * sep:
                continue
            a, b = nearest_points(li, lines[j])
            out.append(Conflict(segs[upper].seq, segs[lower].seq, sep, need,
                                0.5 * (a.y + b.y) / res, 0.5 * (a.x + b.x) / res))
    return out


def self_overlaps(polylines: list[np.ndarray], res: float, width: float,
                  min_ratio: float = 1.3) -> list[tuple[int, int, float]]:
    """Pairs (i, j, separation m) of non-adjacent route pieces whose road
    surfaces overlap: centerlines closer than the road width, with the road
    between them at least min_ratio times longer than that (the route comes
    back on itself, as opposed to an ordinary curve).

    polylines: (n, 2) row/col points per route piece, in travel order.
    """
    from shapely import STRtree

    lines = [LineString(pl[:, ::-1] * res) for pl in polylines]
    station = np.concatenate([[0.0], np.cumsum([ln.length for ln in lines])])
    tree = STRtree(lines)
    out = []
    for i, li in enumerate(lines):
        for j in tree.query(li.buffer(width)):
            j = int(j)
            if j <= i + 1:
                continue
            sep = li.distance(lines[j])
            gap = station[j] - station[i + 1]
            if sep < width and gap >= min_ratio * sep and gap > width:
                out.append((i, j, sep))
    return out
