"""Costing and profiling a given centerline, such as an as-built road, for reference.

evaluate_polyline() prices a line with the same rules evaluator.js uses for a
drawn route, which are the search's own: samples every half cell or less, the
exact half-bench section from the smoothed DEM's gradient, the grade checked on
the profile, one culvert per run of channel cells, and operating costs from length
and rise + fall. Nothing is refused; steep grades, excluded ground and side slopes
too steep to build are measured and reported.

profile_metrics() describes the line itself: grades averaged over set distances
from the working DEM, curve radii, switchbacks.

Caveat for an as-built road: the DEM shows the road already built, so the ground
along it is the finished bench, not the original hillside. Smoothing at the road
width recovers part of the hillside slope; the earthwork cost is still a lower bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio

from .params import Params, load_params
from .section import INFEASIBLE, FULL_BENCH, section_arrays, section_params, soil_depth


def _bilinear(a: np.ndarray, fr: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """search._bilinear, vectorized: NaN outside the grid."""
    nr, nc = a.shape
    ok = (fr >= 0) & (fc >= 0) & (fr <= nr - 1) & (fc <= nc - 1)
    r0 = np.clip(np.floor(np.where(ok, fr, 0)).astype(int), 0, max(nr - 2, 0))
    c0 = np.clip(np.floor(np.where(ok, fc, 0)).astype(int), 0, max(nc - 2, 0))
    wr, wc = fr - r0, fc - c0
    r1, c1 = np.minimum(r0 + 1, nr - 1), np.minimum(c0 + 1, nc - 1)
    v = ((1 - wr) * ((1 - wc) * a[r0, c0] + wc * a[r0, c1]) + wr * ((1 - wc) * a[r1, c0] + wc * a[r1, c1]))
    return np.where(ok, v, np.nan)


class RunTerrain:
    """The working rasters a line is costed on: a run's, or any grid's."""

    def __init__(self, zs, z_raw, code, area, transform, params: Params):
        self.params = params
        self.transform = transform
        self.zs, self.z_raw = np.asarray(zs, float), np.asarray(z_raw, float)
        self.code, self.area = np.nan_to_num(np.asarray(code, float), nan=1.0), np.nan_to_num(np.asarray(area, float))
        d_row, d_col = np.gradient(self.zs)
        self.res = abs(transform.a)
        self.gx, self.gy = d_col / transform.a, d_row / transform.e
        self.x0, self.y0 = transform.c, transform.f

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "RunTerrain":
        run_dir = Path(run_dir)
        rec = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))

        def read(name):
            with rasterio.open(run_dir / name) as ds:
                return ds.read(1).astype(np.float64), ds.transform

        zs, tr = read("dem_smoothed.tif")
        return cls(zs, read("dem_working.tif")[0], read("exclusion_code.tif")[0],
                   read("contributing_area_ha.tif")[0], tr, load_params(overrides=rec["params"]))

    def rc(self, xy: np.ndarray):
        """World (x, y) -> grid (row, col) floats, cell centers at integers."""
        return (self.y0 - xy[:, 1]) / self.res - 0.5, (xy[:, 0] - self.x0) / self.res - 0.5


def sample_polyline(xy: np.ndarray, spacing: float):
    """Midpoints of equal pieces no longer than `spacing`, with tangents, weights and stations."""
    pts, tan, w, st = [], [], [], []
    station = 0.0
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-9:
            continue
        n = max(1, math.ceil(L / spacing - 1e-9))
        f = (np.arange(n) + 0.5) / n
        pts.append(np.column_stack([x0 + f * (x1 - x0), y0 + f * (y1 - y0)]))
        tan.append(np.tile([(x1 - x0) / L, (y1 - y0) / L], (n, 1)))
        w.append(np.full(n, L / n))
        st.append(station + f * L)
        station += L
    return np.vstack(pts), np.vstack(tan), np.concatenate(w), np.concatenate(st), station


def evaluate_polyline(T: RunTerrain, xy, p: Params | None = None) -> dict:
    """Totals of costing the line `xy` ((n, 2), world coordinates of the run's grid)."""
    p = p or T.params
    xy = np.asarray(xy, dtype=float)
    pts, tan, w, st, length = sample_polyline(xy, 0.5 * T.res)
    fr, fc = T.rc(pts)
    z = _bilinear(T.zs, fr, fc)
    ax, ay = _bilinear(T.gx, fr, fc), _bilinear(T.gy, fr, fc)
    outside = ~(np.isfinite(z) & np.isfinite(ax) & np.isfinite(ay))
    # Profile: node, samples, node (as the search's edges), through finite samples only.
    r0, c0 = T.rc(xy[:1]); r1, c1 = T.rc(xy[-1:])
    zp = np.concatenate([_bilinear(T.zs, r0, c0), z, _bilinear(T.zs, r1, c1)])
    sp = np.concatenate([[0.0], st, [length]])
    keep = np.isfinite(zp)
    dz, ds = np.diff(zp[keep]), np.diff(sp[keep])
    over = np.abs(dz) > p.g_max * (1 + 1e-12) * ds
    wk = np.concatenate([[0.0], w, [0.0]])[keep][1:]
    # Sections.
    g = ax * tan[:, 0] + ay * tan[:, 1]
    s = np.abs(-ax * tan[:, 1] + ay * tan[:, 0])
    d = soil_depth(np.hypot(s, g), p)
    sec = section_arrays(np.where(outside, 0.0, s), np.where(outside, 0.0, d), section_params(p))
    kind = np.where(outside, INFEASIBLE, sec[:, 0])
    bad = (kind == INFEASIBLE) & ~outside
    good = ~outside & ~bad
    r = np.floor(fr + 0.5).astype(int); c = np.floor(fc + 0.5).astype(int)
    ins = (r >= 0) & (r < T.zs.shape[0]) & (c >= 0) & (c < T.zs.shape[1])
    cell = np.where(ins, r * T.zs.shape[1] + c, -1)
    excluded = ins & (T.code.ravel()[np.maximum(cell, 0)] > 0)
    # Culverts: one per run of channel cells along the cell sequence (end nodes included).
    def cell_of(rr, cc):
        rr, cc = int(np.floor(rr[0] + 0.5)), int(np.floor(cc[0] + 0.5))
        return rr * T.zs.shape[1] + cc if 0 <= rr < T.zs.shape[0] and 0 <= cc < T.zs.shape[1] else -1
    seq = [cell_of(r0, c0)]
    for j in cell:
        if j != seq[-1]:
            seq.append(int(j))
    end = cell_of(r1, c1)
    if end != seq[-1]:
        seq.append(end)
    area = T.area.ravel()
    culverts, culvert_cost, run_max, in_run = 0, 0.0, 0.0, False
    for j in seq + [-1]:
        ch = j >= 0 and area[j] >= p.A_min
        if ch:
            run_max = max(run_max, area[j]) if in_run else area[j]
            in_run = True
        elif in_run:
            culverts += 1
            culvert_cost += p.culvert_cost(run_max)
            in_run = False
    base = p.C_base * float(w[~outside].sum())
    earthwork = float(((sec[:, 10] - p.C_base) * w)[good].sum())
    vertical = float(np.abs(dz).sum())
    tot = {
        "length": length, "base": base, "earthwork": earthwork, "culverts": culverts, "culvert_cost": culvert_cost,
        "soil_m3": float((sec[:, 3] * w)[good].sum()), "rock_m3": float((sec[:, 4] * w)[good].sum()),
        "fill_m3": float((sec[:, 5] * w)[good].sum()), "spoil_m3": float((sec[:, 6] * w)[good].sum()),
        "vertical": vertical, "climb": float(dz[dz > 0].sum()), "descent": float(-dz[dz < 0].sum()),
        "over_grade_m": float(wk[over].sum()), "excluded_m": float(w[excluded].sum()),
        "infeasible_m": float(w[bad].sum()), "outside_m": float(w[outside].sum()),
        "full_bench_m": float(w[good & (sec[:, 0] == FULL_BENCH)].sum()),
        "max_catch": float(sec[good, 7].max()) if good.any() else 0.0,
    }
    tot["construction"] = tot["base"] + tot["earthwork"] + tot["culvert_cost"]
    tot["operating"] = p.C_travel * length + p.C_vertical * vertical
    tot["total"] = tot["construction"] + tot["operating"]
    return tot


def profile_metrics(T: RunTerrain, xy, windows=(100.0, 200.0), limits=(0.06, 0.08), radii=(10.0, 20.0, 30.0),
                    step: float = 2.0) -> dict:
    """Grades (from the working DEM, averaged over each window), curve radii and switchbacks of a line."""
    xy = np.asarray(xy, dtype=float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1], step)
    pts = np.column_stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])
    fr, fc = T.rc(pts)
    z = _bilinear(T.z_raw, fr, fc)
    out = {"length": float(cum[-1]), "z_start": float(z[0]), "z_end": float(z[-1]),
           "z_min": float(np.nanmin(z)), "z_max": float(np.nanmax(z)), "grades": {}}
    for win in windows:
        k = max(1, int(round(win / step)))
        gr = np.abs(z[k:] - z[:-k]) / (k * step)
        out["grades"][f"{win:g}"] = {"max": float(np.nanmax(gr)), "mean": float(np.nanmean(gr)),
                                     **{f"over_{lim:g}_m": float(np.nansum(gr > lim) * step) for lim in limits}}
    # Rise and fall of the profile smoothed over 50 m (shorter wiggles are DEM and digitizing noise).
    k = max(1, int(round(25.0 / step)))
    zf = np.convolve(np.nan_to_num(z, nan=np.nanmean(z)), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    dzf = np.diff(zf)
    out["climb_50m"], out["descent_50m"] = float(dzf[dzf > 0].sum()), float(-dzf[dzf < 0].sum())
    # Curvature: heading change across 40 m of road; radius = arc / turn.
    h = np.unwrap(np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0])))
    k = max(1, int(round(20.0 / step)))
    turn = np.abs(h[2 * k:] - h[:-2 * k])
    R = (2 * k * step) / np.maximum(turn, 1e-9)
    out["min_radius_40m"] = float(R.min())
    out["tighter_than_m"] = {f"{r:g}": float((R < r).sum() * step) for r in radii}
    # Switchbacks: the road turns more than 150 degrees within 150 m.
    k = max(1, int(round(75.0 / step)))
    big = np.flatnonzero(np.abs(h[2 * k:] - h[:-2 * k]) > math.radians(150))
    out["switchbacks"] = int(len(np.split(big, np.flatnonzero(np.diff(big) > 10) + 1))) if len(big) else 0
    return out
