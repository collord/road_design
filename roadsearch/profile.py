"""Cost and profile an existing or proposed road (any centerline file) on a DEM, beside a run's routes.

    python -m roadsearch profile LINE --dem DEM --params P --resolution 15 --label "FEIS road"
        [--run out/runs/RUN_ID] [--out DIR] [--gpkg results_x.gpkg --prefix x_feis]

The line is costed the way the viewer's "Draw a route" tab costs a drawn line (polyline.py):
earthwork from the section model along the line, culverts, truck travel, climb and turning.
A run's routes are costed the same way on the same terrain, so the numbers compare directly
(their search costs, which price hairpins as designed macros, are listed too).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .params import Params
from .polyline import RunTerrain, _bilinear, evaluate_polyline, profile_metrics

PROFILE_STEP_M = 10.0
GRADE_WINDOW_M = 100.0


def load_centerline(path, crs, tol: float = 150.0) -> tuple[np.ndarray, list[dict]]:
    """One (x, y) polyline from a file of one or more line parts, in `crs`.

    Parts are chained end to end, nearest end first. Where the next part starts back along the
    end of the chain (digitized sections often overlap by tens of meters), the overlap is cut
    from it; a gap is bridged with a straight piece. Returns the coordinates and a record of
    each joint."""
    gdf = gpd.read_file(path).to_crs(crs)
    parts = []
    for g in gdf.geometry:
        parts += [LineString(np.asarray(p.coords)[:, :2]) for p in getattr(g, "geoms", [g]) if p.length > 0]
    if not parts:
        raise ValueError(f"{path}: no line geometry")
    # Start from the part whose free end is farthest from all the others.
    def far_end(i):
        best = (-1.0, None)
        for flip in (False, True):
            end = Point(parts[i].coords[-1 if flip else 0])
            d = min((end.distance(q) for j, q in enumerate(parts) if j != i), default=math.inf)
            best = max(best, (d, flip), key=lambda t: t[0])
        return best
    i0, (_, flip) = max(((i, far_end(i)) for i in range(len(parts))), key=lambda t: t[1][0])
    chain = parts[i0] if not flip else LineString(parts[i0].coords[::-1])
    left = [p for j, p in enumerate(parts) if j != i0]
    joints = []
    while left:
        end = Point(chain.coords[-1])
        k, rev = min(((j, r) for j in range(len(left)) for r in (False, True)),
                     key=lambda t: Point(left[t[0]].coords[-1 if t[1] else 0]).distance(end))
        nxt = left.pop(k)
        if rev:
            nxt = LineString(nxt.coords[::-1])
        t = nxt.project(end)
        if nxt.distance(end) < 1.0 and 0.0 < t < min(tol, nxt.length):
            nxt = substring(nxt, t, nxt.length)              # starts back along the chain: cut the overlap
            joints.append({"overlap_m": round(t, 1)})
        else:
            joints.append({"gap_m": round(Point(nxt.coords[0]).distance(end), 1)})
        chain = LineString(list(chain.coords) + list(nxt.coords)[1:])
    return np.asarray(chain.coords), joints


def line_terrain(dem, p: Params, res: float, bounds) -> RunTerrain:
    """The rasters a line is costed on (as a run's preprocessing makes them), over `bounds`."""
    from .exclusion import combine_codes
    from .flow import contributing_area_ha
    from .preprocess import gradient, slope_mask, smooth_dem
    from .terrain import load_grid

    grid, _ = load_grid(dem, p, res, bounds)
    zs = smooth_dem(grid.z, p.sigma_m / grid.res)
    dzdx, dzdy = gradient(zs, grid)
    _, slope_ex, _ = slope_mask(np.hypot(dzdx, dzdy), grid, p)
    code = combine_codes(slope_ex, np.zeros(grid.shape, dtype=bool))
    return RunTerrain(zs, grid.z, code, contributing_area_ha(grid), grid.transform, p), grid


def _profile(T: RunTerrain, xy: np.ndarray, step: float = PROFILE_STEP_M, window: float = GRADE_WINDOW_M):
    """Station, elevation (working DEM) and grade over `window` (centered) every `step` meters."""
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    s = np.arange(0.0, cum[-1] + 1e-9, step)
    pts = np.column_stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])
    z = _bilinear(T.z_raw, *T.rc(pts))
    k = max(1, int(round(window / step / 2)))
    g = np.full(z.shape, np.nan)
    g[k:-k] = (z[2 * k:] - z[:-2 * k]) / (2 * k * step)
    return s, pts, z, g


def profile_report(dem, line_path, p: Params, res: float, label: str, out_dir, run_dir=None,
                   gpkg=None, prefix=None, margin: float = 500.0) -> dict:
    """Cost and profile the line (and a run's routes); write profile.png, summary.json and
    summary.md to out_dir, and optionally the line and its profile points to a GeoPackage."""
    import rasterio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem) as ds:
        crs = ds.crs
    xy, joints = load_centerline(line_path, crs)
    lines = {label: xy}
    search_costs = {}
    if run_dir is not None:
        run_dir = Path(run_dir)
        rec = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
        summ = gpd.read_parquet(run_dir.parent.parent / "route_summary.parquet")
        summ = summ[summ.run_id == rec["run_id"]].sort_values("rank").to_crs(crs)
        for _, r in summ.iterrows():
            name = f"route {int(r['rank'])} (run {rec['run_id']})"
            lines[name] = np.asarray(r.geometry.coords)[:, :2]
            search_costs[name] = {"cost_true": float(r["cost_true"]), "length": float(r["length"]),
                                  "hairpins": int(r["hairpin_count"])}
    # Every line runs from the end nearest the first line's south end, so profiles share an origin.
    south = xy[0] if xy[0, 1] < xy[-1, 1] else xy[-1]
    for name, l in lines.items():
        if np.hypot(*(l[-1] - south)) < np.hypot(*(l[0] - south)):
            lines[name] = l[::-1]
    allxy = np.vstack(list(lines.values()))
    bounds = (allxy[:, 0].min() - margin, allxy[:, 1].min() - margin, allxy[:, 0].max() + margin,
              allxy[:, 1].max() + margin)
    T, grid = line_terrain(dem, p, res, bounds)
    results = {}
    for name, l in lines.items():
        cost = evaluate_polyline(T, l)
        prof = profile_metrics(T, l, limits=sorted({p.g_hp, p.g_max}), radii=sorted({p.R_min, 2 * p.R_min}))
        results[name] = {"cost": cost, "profile": prof, **({"search": search_costs[name]} if name in search_costs else {})}
    summary = {"label": label, "file": Path(line_path).name, "dem": str(dem), "resolution_m": res,
               "joints": joints, "params": {k: getattr(p, k) for k in ("w", "g_max", "g_hp", "R_min", "C_travel",
                                                                      "C_vertical", "C_turn")},
               "lines": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    _figure(out_dir / "profile.png", T, grid, lines, label, p)
    if gpkg:
        _write_gpkg(gpkg, prefix or "line", T, lines[label], label, crs)
    return summary


def _summary_md(s: dict) -> str:
    names = list(s["lines"])
    rows = [
        ("length (km)", lambda r: f"{r['cost']['length'] / 1000:.2f}"),
        ("total cost", lambda r: f"{r['cost']['total']:,.0f}"),
        ("construction", lambda r: f"{r['cost']['construction']:,.0f}"),
        ("  earthwork", lambda r: f"{r['cost']['earthwork']:,.0f}"),
        ("  culverts", lambda r: f"{r['cost']['culverts']} ({r['cost']['culvert_cost']:,.0f})"),
        ("operating", lambda r: f"{r['cost']['operating']:,.0f}"),
        ("climb / descent (m)", lambda r: f"{r['cost']['climb']:,.0f} / {r['cost']['descent']:,.0f}"),
        ("elevation range (m)", lambda r: f"{r['profile']['z_min']:,.0f} to {r['profile']['z_max']:,.0f}"),
        ("max grade, 100 m", lambda r: f"{100 * r['profile']['grades']['100']['max']:.1f}%"),
        ("mean grade, 100 m", lambda r: f"{100 * r['profile']['grades']['100']['mean']:.1f}%"),
        *[(f"length over {100 * g:g}%, 100 m (m)",
           (lambda g: lambda r: f"{r['profile']['grades']['100'].get(f'over_{g:g}_m', 0):,.0f}")(g))
          for g in sorted({s['params']['g_hp'], s['params']['g_max']})],
        ("length steeper than g_max, sampled (m)", lambda r: f"{r['cost']['over_grade_m']:,.0f}"),
        ("on slope-masked ground (m)", lambda r: f"{r['cost']['excluded_m']:,.0f}"),
        ("section infeasible (m)", lambda r: f"{r['cost']['infeasible_m']:,.0f}"),
        ("tightest radius over 40 m (m)", lambda r: f"{r['profile']['min_radius_40m']:,.0f}"),
        (f"curves tighter than R_min (m)",
         lambda r: f"{r['profile']['tighter_than_m'].get(f'{s['params']['R_min']:g}', 0):,.0f}"),
        ("switchbacks found from the line's vertices", lambda r: f"{r['profile']['switchbacks']}"),
        ("hairpins designed by the search", lambda r: f"{r['search']['hairpins']}" if "search" in r else ""),
        ("search cost (macros designed)", lambda r: f"{r['search']['cost_true']:,.0f}" if "search" in r else ""),
    ]
    out = [f"# {s['label']}: cost and profile", "",
           f"Costed on `{Path(s['dem']).name}` at {s['resolution_m']:g} m with w {s['params']['w']} m, "
           f"g_max {100 * s['params']['g_max']:g}%, hairpin grade {100 * s['params']['g_hp']:g}%, "
           f"R_min {s['params']['R_min']:g} m, C_travel {s['params']['C_travel']:g}, "
           f"C_vertical {s['params']['C_vertical']:g}, C_turn {s['params']['C_turn']:g}.", "",
           "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for lab, f in rows:
        out.append(f"| {lab} | " + " | ".join(f(s["lines"][n]) for n in names) + " |")
    out += ["", "Notes:",
            "- Grades are of the ground (the working DEM) under each centerline, over 100 m, not of a designed "
            "road: cut and fill would ease them. Through a search route's hairpins the ground under the loop "
            "is steep while the designed road holds the hairpin grade.",
            "- The cost is the drawn-line cost (as the viewer's Draw tab): earthwork for a road on the ground "
            "profile, culverts, and operating costs. Length steeper than g_max is reported, not priced, so a "
            "line with much of it would cost more as a road built to grade.",
            "- The switchback count finds turns of more than 150 degrees within 200 m among the line's vertices; "
            "a search route's small lattice zigzags inflate it, so its designed hairpins are listed too."]
    if s["joints"]:
        out += ["", "Joints between the file's parts: " + ", ".join(
            f"overlap {j['overlap_m']:g} m cut" if "overlap_m" in j else f"gap {j['gap_m']:g} m bridged"
            for j in s["joints"]) + "."]
    return "\n".join(out) + "\n"


def _figure(path, T, grid, lines, label, p: Params):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    colors = ["#0369a1", "#b91c1c", "#ea580c", "#7c3aed"]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1.6, 1])
    ax_z, ax_g, ax_m = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for i, (name, xy) in enumerate(lines.items()):
        s, _, z, g = _profile(T, xy)
        col, lw = colors[i % len(colors)], 2.0 if i == 0 else 1.2
        ax_z.plot(s / 1000, z, color=col, lw=lw, label=name)
        ax_g.plot(s / 1000, 100 * g, color=col, lw=lw if i == 0 else 0.8, alpha=1.0 if i == 0 else 0.6)
    for gl, ls in ((p.g_max, "--"), (p.g_hp, ":")):
        for sgn in (1, -1):
            ax_g.axhline(100 * sgn * gl, color="k", lw=0.8, ls=ls)
    ax_g.text(0.002, 100 * p.g_max, f" {100 * p.g_max:g}% (g_max)", va="bottom", fontsize=8, transform=ax_g.get_yaxis_transform())
    ax_g.text(0.002, 100 * p.g_hp, f" {100 * p.g_hp:g}% (hairpins)", va="bottom", fontsize=8, transform=ax_g.get_yaxis_transform())
    ax_z.set_ylabel("elevation (m)")
    ax_g.set_ylabel(f"grade over {GRADE_WINDOW_M:g} m (%)")
    ax_g.set_xlabel("distance from the south end (km)")
    ax_z.legend(fontsize=9, loc="upper left")
    ax_z.set_title(f"{label}: elevation profile", fontsize=11)
    ax_z.grid(alpha=0.3)
    ax_g.grid(alpha=0.3)
    # Plan view: hillshade at display resolution, lines over it.
    h, w = T.zs.shape
    k = max(1, -(-max(h, w) // 1000))
    zz = T.zs[::k, ::k]
    hs = LightSource(315, 45).hillshade(np.where(np.isnan(zz), np.nanmin(zz), zz), dx=T.res * k, dy=T.res * k)
    x0, y0 = T.x0, T.y0
    ax_m.imshow(hs, cmap="gray", extent=(x0, x0 + w * T.res, y0 - h * T.res, y0), interpolation="nearest")
    for i, (name, xy) in enumerate(lines.items()):
        ax_m.plot(xy[:, 0], xy[:, 1], color=colors[i % len(colors)], lw=2.0 if i == 0 else 1.2)
        ax_m.plot(*xy[0], "o", ms=6, mfc="w", mec=colors[i % len(colors)])
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.set_title("plan (circles: south ends)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_gpkg(gpkg, prefix, T, xy, label, crs):
    """The line (with its elevations) and profile points every PROFILE_STEP_M with elevation and grade."""
    s, pts, z, g = _profile(T, xy)
    line = gpd.GeoDataFrame({"label": [label], "length_m": [float(s[-1])]},
                            geometry=[LineString(np.column_stack([pts, np.nan_to_num(z)]))], crs=crs)
    prof = gpd.GeoDataFrame({"label": label, "station_m": s, "elevation_m": z, f"grade_{GRADE_WINDOW_M:g}m": g},
                            geometry=gpd.points_from_xy(pts[:, 0], pts[:, 1]), crs=crs)
    line.to_file(gpkg, layer=f"{prefix}_line", driver="GPKG")
    prof.to_file(gpkg, layer=f"{prefix}_profile", driver="GPKG")
