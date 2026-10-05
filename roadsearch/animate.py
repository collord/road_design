"""Wavefront animation and run comparison, as a small static site.

No search trace is needed: Dijkstra settles states in order of cost, so a
cell is reached when the search frontier passes its cost-to-come (minimum
over headings), and A* settles in order of cost-to-come plus the heuristic,
which depends on the cell only. The page thresholds these saved fields frame
by frame:

  1. A*: the fill-price search racing to the goal (explored_astar.tif + h).
  2. Two-way Dijkstra: fronts from both ends (cost_to_come_forward/backward).
  3. Corridor: cells by the cost of the cheapest route through them, and the
     ranked routes (corridor_field.tif, routes.parquet).

write_viewer() puts several runs in one site: index.html carries a manifest
(parameters, cost breakdown and routes of every run), a shared hillshade for
the comparison view and the method write-up (reference.md, rendered into the
page's "How the search works" tab); data/<run_id>.json holds one run's
animation fields, the smoothed DEM and parameters the "Draw a route" tab
costs a drawn line with (evaluator.js, a port of the section model), and is
loaded when that run is opened; data/<run_id>_tree.json holds its corridor
tree (explore.py) for "best route through here", loaded on demand.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.colors import LightSource

NEVER = 65535
TEMPLATE_PATH = Path(__file__).with_name("viewer_template.html")
REFERENCE_PATH = Path(__file__).with_name("reference.md")
EVALUATOR_PATH = Path(__file__).with_name("evaluator.js")
FIGURES = {f.stem[len("reference_"):]: f for f in Path(__file__).parent.glob("reference_*.svg")}


def _read(path: Path):
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float64), ds.transform


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _quantize(v: np.ndarray, vmax: float) -> np.ndarray:
    """Values 0..vmax to 0..65534; NaN, inf and anything above vmax to NEVER."""
    out = np.full(v.shape, NEVER, dtype=np.uint16)
    ok = np.isfinite(v) & (v <= vmax)
    out[ok] = np.round(np.clip(v[ok], 0, vmax) / vmax * 65534).astype(np.uint16)
    return out


def _hillshade(z: np.ndarray, res: float) -> np.ndarray:
    zfill = np.where(np.isfinite(z), z, np.nanmin(z))
    return np.round(LightSource(315, 45).hillshade(zfill, dx=res, dy=res) * 255).astype(np.uint8)


def _routes(run_id: str, out_root: Path, F_min: float, width: float = 0.0) -> list[dict]:
    """Ranked routes of a run, with world coordinates and cost breakdown, and how many
    times each still crosses or runs over itself (self-overlap avoidance gives up after
    a few rounds)."""
    from .route import self_overlaps
    rp, sp = out_root / "routes.parquet", out_root / "route_summary.parquet"
    if not (rp.exists() and sp.exists()):
        return []
    segs = gpd.read_parquet(rp)
    segs = segs[segs["run_id"] == run_id]
    summ = gpd.read_parquet(sp)
    summ = summ[summ["run_id"] == run_id].sort_values("rank")
    out = []
    for _, s in summ.iterrows():
        rs = segs[segs["route_id"] == s["route_id"]].sort_values("seq")
        xy = [[round(x, 1), round(y, 1)] for geom in rs.geometry for x, y, *_ in geom.coords]
        pieces = [np.asarray(geom.coords)[:, :2][:, ::-1] for geom in rs.geometry]   # (y, x): distances only
        overlaps = len(self_overlaps(pieces, 1.0, width)) if width > 0 and len(pieces) > 2 else 0

        def get(k, d=0.0):
            return float(s[k]) if k in s.index and s[k] == s[k] else d

        out.append({
            "rank": int(s["rank"]), "cost": get("cost_true"),
            "construction": get("cost_construction", get("cost_true")), "operating": get("cost_operating"),
            "length": get("length"), "vertical": get("vertical_travel", get("total_climb") + get("total_descent")),
            "climb": get("total_climb"), "descent": get("total_descent"),
            "hairpins": int(s["hairpin_count"]), "bends": int(get("bend_count")),
            "crossings": int(s["fill_crossing_count"] + s["at_grade_crossing_count"]),
            "rel": (get("cost_surrogate") - F_min) / F_min, "self_overlaps": overlaps,
            "xy": xy,
        })
    return out


def run_data(run_dir: str | Path, out_root: str | Path) -> dict:
    """Animation fields and routes of one finished run."""
    run_dir, out_root = Path(run_dir), Path(out_root)
    rec = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    run_id = rec["run_id"]
    p = rec["params"]
    zs, tr = _read(run_dir / "dem_smoothed.tif")
    res = abs(tr.a)
    h, w = zs.shape

    code, _ = _read(run_dir / "exclusion_code.tif")
    area, _ = _read(run_dir / "contributing_area_ha.tif")
    flags = np.zeros((h, w), dtype=np.uint8)
    flags[np.nan_to_num(code) > 0] |= 1
    channel = np.nan_to_num(area) >= p["A_min"]
    flags[channel] |= 2
    # Bits 3-4: culvert size class of channel cells (index into culvert_costs).
    cls = np.searchsorted(np.asarray(p["culvert_area_breaks"], float), np.nan_to_num(area), side="right")
    flags[channel] |= (np.minimum(cls, 3).astype(np.uint8) << 3)[channel]
    if (run_dir / "clip_region.tif").exists():
        clip, _ = _read(run_dir / "clip_region.tif")
        flags[np.nan_to_num(clip, nan=0) == 0] |= 4

    sr, sc = rec["preprocess"]["start_rc"]
    gr, gc = rec["preprocess"]["goal_rc"]

    # A*: settled in order of f = g + h, with h exactly as the kernel computes it.
    g_ast, _ = _read(run_dir / "explored_astar.tif")
    rr, cc = np.indices((h, w))
    dist = np.hypot(rr - gr, cc - gc) * res
    dz = np.abs(zs[gr, gc] - zs)
    c_len = p["C_base"] + p.get("C_travel", 0.0)
    hval = c_len * np.maximum(dist, dz / p["g_max"]) + p.get("C_vertical", 0.0) * dz
    f_ast = np.where(np.isfinite(g_ast), g_ast + hval, np.nan)
    f_goal = float(np.nanmax(f_ast))
    f0 = float(np.nanmin(f_ast))                    # the start cell's lower bound: A* begins here

    g_f, _ = _read(run_dir / "cost_to_come_forward.tif")
    g_b, _ = _read(run_dir / "cost_to_come_backward.tif")
    F, _ = _read(run_dir / "corridor_field.tif")
    F_min = float(np.nanmin(F))
    # The full searches run far past the best route; the interesting part is
    # the fronts meeting (at F_min / 2) and covering the corridor.
    g_max = min(float(np.nanmax([np.nanmax(g_f), np.nanmax(g_b)])), 1.3 * F_min)
    rel_max = 1.0                                   # corridor shown up to 2 x the best cost
    F_rel = (F - F_min) / F_min

    from .params import load_params
    from .section import section_params

    pp = load_params(overrides=p)
    evaluator = {"sp": section_params(pp)._asdict(), "d0": pp.d0, "S0": pp.S0, "g_max": pp.g_max, "R_min": pp.R_min,
                 "w": pp.w, "C_travel": pp.C_travel, "C_vertical": pp.C_vertical,
                 "culvert_area_breaks": list(pp.culvert_area_breaks), "culvert_costs": list(pp.culvert_costs)}
    return {
        "run_id": run_id, "w": w, "h": h, "res": res, "x0": tr.c, "y0": tr.f,
        "zs": _b64(np.where(np.isfinite(zs), zs, np.nan).astype(np.float32)), "P": evaluator,
        "start": [sc, sr], "goal": [gc, gr],
        "f_goal": f_goal, "f0": f0, "g_max": g_max, "F_min": F_min, "rel_max": rel_max,
        "eps_alt": p["eps_alt"],
        "n_astar": int(np.isfinite(g_ast).sum()), "n_fwd": int(np.isfinite(g_f).sum()),
        "n_bwd": int(np.isfinite(g_b).sum()), "cells": int(h * w),
        "n_corridor": int((np.nan_to_num(F_rel, nan=np.inf) <= p["eps_alt"]).sum()),
        "params": {k: p[k] for k in ("w", "g_max", "R_min", "R_hp", "N_theta", "C_base", "C_travel", "C_vertical")
                   if k in p},
        "hill": _b64(_hillshade(zs, res)), "flags": _b64(flags),
        "astar": _b64(_quantize(f_ast - f0, f_goal - f0)),
        "fwd": _b64(_quantize(g_f, g_max)), "bwd": _b64(_quantize(g_b, g_max)),
        "corr": _b64(_quantize(F_rel, rel_max)),
        "routes": _routes(run_id, out_root, F_min, p["w"]),
    }


def tree_data(run_dir: str | Path) -> dict | None:
    """The run's corridor tree for the viewer (explore.viewer_tree), or None if it has none."""
    from .explore import read_corridor_tree, viewer_tree

    run_dir = Path(run_dir)
    t = read_corridor_tree(run_dir)
    if t is None:
        return None
    F, tr = _read(run_dir / "corridor_field.tif")
    v = viewer_tree(t, np.where(np.isfinite(F), F, np.inf), abs(tr.a))
    out = {"n_theta": v["n_theta"], "step": v["step"], "best_h": _b64(v["best_h"])}
    for d in ("fwd", "bwd"):
        a = v[d]
        out[d] = {"n": int(a["cell"].size), "cell": _b64(a["cell"]), "h": _b64(a["h"]), "parent": _b64(a["parent"]),
                  "ng": _b64(a["ng"]), "geo": _b64(a["geo"])}
    return out


def reference_data(path: str | Path, label: str, run_dirs: list[Path]) -> dict:
    """A reference centerline (e.g. the as-built road): costed on each run's terrain with
    that run's parameters, plus its grade and curve profile."""
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    from .polyline import RunTerrain, evaluate_polyline, profile_metrics

    recs = [json.loads((d / "params.json").read_text(encoding="utf-8")) for d in run_dirs]
    gdf = gpd.read_file(path).to_crs(recs[0]["grid"]["working_crs"])
    geom = linemerge(list(gdf.geometry)) if len(gdf) > 1 else gdf.geometry.iloc[0]
    if isinstance(geom, MultiLineString):
        geom = max(geom.geoms, key=lambda g: g.length)
    xy = np.asarray(LineString(geom).coords)[:, :2]
    start = np.asarray(recs[0]["inputs"]["start"], float)
    if np.hypot(*(xy[-1] - start)) < np.hypot(*(xy[0] - start)):
        xy = xy[::-1]                                   # run it from the start toward the goal
    terrains = [RunTerrain.from_run(d) for d in run_dirs]
    g_lims = sorted({t.params.g_max for t in terrains})
    radii = sorted({t.params.R_min for t in terrains})
    rnd = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}   # noqa: E731
    return {
        "label": label, "file": Path(path).name, "xy": [[round(x, 1), round(y, 1)] for x, y in xy],
        "profile": profile_metrics(terrains[0], xy, limits=g_lims, radii=radii),
        "runs": {rec["run_id"]: rnd(evaluate_polyline(t, xy)) for rec, t in zip(recs, terrains)},
        "g_max": g_lims, "R_min": radii,
    }


def _basemap(run_dirs: list[Path], res: float) -> dict:
    """Hillshade over the union of the runs' grids, from the DEM itself."""
    from .params import load_params
    from .preprocess import smooth_dem
    from .terrain import load_grid

    recs = [json.loads((d / "params.json").read_text(encoding="utf-8")) for d in run_dirs]
    bounds = np.array([r["grid"]["working_bounds"] for r in recs])
    box = (bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max())
    p = load_params(overrides=recs[0]["params"])
    grid, _ = load_grid(recs[0]["inputs"]["dem"], p, res, box)
    zs = smooth_dem(grid.z, p.sigma_m / res)
    pre = recs[0]["preprocess"]
    with rasterio.open(run_dirs[0] / "dem_smoothed.tif") as ds:
        tr0 = ds.transform
    sx, sy = tr0 * (pre["start_rc"][1] + 0.5, pre["start_rc"][0] + 0.5)
    gx, gy = tr0 * (pre["goal_rc"][1] + 0.5, pre["goal_rc"][0] + 0.5)
    return {"w": grid.shape[1], "h": grid.shape[0], "res": res, "x0": grid.transform.c, "y0": grid.transform.f,
            "hill": _b64(_hillshade(zs, res)), "start": [sx, sy], "goal": [gx, gy], "crs": recs[0]["grid"]["working_crs"]}


def reference_html() -> str:
    """The method write-up as HTML, with its figures inlined."""
    import mistune

    html = mistune.create_markdown(plugins=["table"])(REFERENCE_PATH.read_text(encoding="utf-8"))
    for name, path in FIGURES.items():
        html = html.replace(f"<p>[[figure:{name}]]</p>", f'<figure>{path.read_text(encoding="utf-8")}</figure>')
    return html


def write_viewer(run_ids: list[str], out_root: str | Path, dest: str | Path, labels: list[str] | None = None,
                 factors: list[float] | None = None, title: str = "Road Search",
                 heading: str | None = None, column: str | None = None,
                 reference: tuple[str | Path, str] | None = None) -> Path:
    """index.html plus data/<run_id>.json for each run.

    factors: operating-cost weights k of the runs (a truck-cost sweep; None for a run outside it);
    labels and column: a sweep over something else, labeled per run under that column heading;
    reference: (file, label) of a centerline to show and cost for comparison."""
    """index.html plus data/<run_id>.json for each run."""
    out_root, dest = Path(out_root), Path(dest)
    (dest / "data").mkdir(parents=True, exist_ok=True)
    run_dirs = [out_root / "runs" / r for r in run_ids]
    runs = []
    for i, (rid, rd) in enumerate(zip(run_ids, run_dirs)):
        d = run_data(rd, out_root)
        tree = tree_data(rd)
        if tree is not None:
            d["tree_file"] = f"data/{rid}_tree.json"
            (dest / "data" / f"{rid}_tree.json").write_text(json.dumps(tree, separators=(",", ":")), encoding="utf-8")
        (dest / "data" / f"{rid}.json").write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        runs.append({
            "run_id": rid,
            "label": labels[i] if labels else (f"C_travel {d['params'].get('C_travel', 0):g}, "
                                               f"C_vertical {d['params'].get('C_vertical', 0):g}"),
            "factor": factors[i] if factors and factors[i] is not None else None,
            "params": d["params"], "res": d["res"], "F_min": d["F_min"],
            "routes": d["routes"],
        })
    sweep = bool(factors) and any(f is not None and f > 0 for f in factors)
    heading = heading or ("Where the road goes as truck costs rise" if sweep else title)
    if sweep:
        explain = None                     # the page's own truck-cost text
    elif column:
        explain = (f"Each run uses a different {column[:1].lower() + column[1:]}; everything else is the same. "
                   "Hover a row to pick out its route; select it to watch that search.")
    else:
        column = "Run"
        explain = ("The cheapest route found, with its ranked alternatives: highlight a row to see them, dashed. "
                   "Select it to watch the search.")
    manifest = {"runs": runs, "base": _basemap(run_dirs, runs[0]["res"]), "sweep": sweep, "heading": heading,
                "column": column, "explain": explain,
                "reference": reference_data(reference[0], reference[1], run_dirs) if reference else None}
    page = (TEMPLATE_PATH.read_text(encoding="utf-8")
            .replace("__TITLE__", html.escape(title)).replace("__HEADING__", html.escape(heading))
            .replace("__MANIFEST__", json.dumps(manifest, separators=(",", ":")))
            .replace("__METHOD__", reference_html())
            .replace("/*__EVALUATOR__*/", EVALUATOR_PATH.read_text(encoding="utf-8")))
    (dest / "index.html").write_text(page, encoding="utf-8")
    return dest / "index.html"
