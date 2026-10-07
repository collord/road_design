"""Output files (brief section 10).

Vector tables accumulate in the output root and are keyed by run_id: a run
replaces its own rows and keeps every other run's, so routes from parameter
sweeps can be overlaid. Rasters, params.json and the quick-look PNG go in
runs/<run_id>/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .lattice import heading_vectors
from .route import Segment, stacked_leg_conflicts
from .terrain import Grid, write_raster

ROUTE_COLUMNS = [
    "run_id", "route_id", "rank", "seq", "station_start", "length", "grade", "sidehill_slope",
    "terrain_slope", "soil_depth", "section_type", "cut_width", "fill_width", "cut_catch", "fill_toe",
    "cut_soil_m3", "cut_rock_m3", "fill_m3", "spoil_m3", "borrow_m3", "cost_base", "cost_earthwork",
    "cost_culvert", "culvert_area_ha", "cost_construction", "vertical_travel", "turning_deg", "cost_operating", "cost_total",
    "cost_surrogate",
]


def rc_to_xy(grid: Grid, rc: np.ndarray) -> np.ndarray:
    t = grid.transform
    x = t.c + (rc[:, 1] + 0.5) * t.a
    y = t.f + (rc[:, 0] + 0.5) * t.e
    return np.column_stack([x, y])


def _line3d(grid: Grid, seg: Segment) -> LineString:
    xy = rc_to_xy(grid, seg.pts_rc)
    return LineString(np.column_stack([xy, seg.z]))


def segment_rows(grid: Grid, run_id: str, route_id: int, rank: int, segs: list[Segment], sink_cost: dict):
    rows, geoms = [], []
    station = 0.0
    for s in segs:
        earth = s.cost_earthwork
        if s.section_type == "fill_crossing":
            earth = sink_cost.get(s.seq, 0.0)   # true haul + borrow for this crossing's fill
        rows.append({
            "run_id": run_id, "route_id": route_id, "rank": rank, "seq": s.seq,
            "station_start": station, "length": s.length, "grade": s.grade,
            "sidehill_slope": s.sidehill_slope, "terrain_slope": s.terrain_slope, "soil_depth": s.soil_depth,
            "section_type": s.section_type, "cut_width": s.cut_width, "fill_width": s.fill_width,
            "cut_catch": s.cut_catch, "fill_toe": s.fill_toe, "cut_soil_m3": s.cut_soil_m3,
            "cut_rock_m3": s.cut_rock_m3, "fill_m3": s.fill_m3, "spoil_m3": s.spoil_m3,
            "borrow_m3": s.borrow_m3, "cost_base": s.cost_base, "cost_earthwork": earth,
            "cost_culvert": s.cost_culvert, "culvert_area_ha": s.culvert_area_ha,
            "cost_construction": s.cost_base + earth + s.cost_culvert, "vertical_travel": s.vertical_travel,
            "turning_deg": math.degrees(s.turning),
            "cost_operating": s.cost_operating,
            "cost_total": s.cost_base + earth + s.cost_culvert + s.cost_operating,
            "cost_surrogate": s.cost_surrogate,
        })
        geoms.append(_line3d(grid, s))
        station += s.length
    return rows, geoms


def sink_costs(route) -> dict:
    """True cost (haul + borrow) of each fill crossing's fill, from the LP."""
    p_borrow = route.params.C_borrow
    m = route.cost.mass
    out = {seq: v["borrow_m3"] * p_borrow for seq, v in m.sink_source.items()}
    for src, snk, vol, dist, cpm in m.assignments:
        out[snk] = out.get(snk, 0.0) + vol * cpm
    return out


def summary_row(run_id, route_id, rank, route, conflicts, via_xy) -> dict:
    segs = route.segments
    z = np.concatenate([s.z for s in segs])
    dz = np.diff(z)
    tc = route.cost
    count = lambda k: sum(s.section_type == k for s in segs)  # noqa: E731
    return {
        "run_id": run_id, "route_id": route_id, "rank": rank,
        "via_x": via_xy[0], "via_y": via_xy[1],
        "length": sum(s.length for s in segs),
        "total_climb": float(dz[dz > 0].sum()), "total_descent": float(-dz[dz < 0].sum()),
        "cut_soil_m3": sum(s.cut_soil_m3 for s in segs), "cut_rock_m3": sum(s.cut_rock_m3 for s in segs),
        "fill_m3": sum(s.fill_m3 for s in segs), "spoil_m3": sum(s.spoil_m3 for s in segs),
        "hairpin_borrow_m3": sum(s.borrow_m3 for s in segs),
        "crossing_fill_m3": tc.mass.fill_m3, "crossing_fill_reused_m3": tc.mass.reused_m3,
        "crossing_fill_borrow_m3": tc.mass.borrow_m3,
        "hairpin_count": count("hairpin"), "bend_count": count("bend"),
        "fill_crossing_count": count("fill_crossing"),
        "at_grade_crossing_count": sum(s.cost_culvert > 0 and s.section_type != "fill_crossing" for s in segs),
        "full_bench_length": sum(s.length for s in segs if s.section_type == "full_bench"),
        "conflict_count": len(conflicts),
        "cost_surrogate": tc.surrogate, "cost_true": tc.total,
        "cost_construction": tc.construction, "cost_operating": tc.operating,
        "vertical_travel": sum(s.vertical_travel for s in segs),
        "turning_deg": math.degrees(sum(s.turning for s in segs)),
        "cost_base": tc.base, "cost_excavation": tc.excavation, "cost_culverts": tc.culverts,
        "cost_crossing_fill": tc.crossing_fill, "cost_haul": tc.mass.haul_cost,
        "cost_borrow_crossings": tc.mass.borrow_cost, "lam_fill": route.lam_fill,
    }


def feature_rows(grid: Grid, run_id, route_id, rank, route, conflicts, prob):
    rows = []
    segs = route.segments
    by_seq = {s.seq: s for s in segs}
    src = route.cost.mass.sink_source
    for s in segs:
        mid = s.pts_rc[len(s.pts_rc) // 2][None]
        if s.section_type == "hairpin":
            rows.append(dict(feature_type="hairpin", seq=s.seq, cost=s.cost_total, radius=s.extra["radius"],
                             leg=s.extra["leg"], inner_edge_grade=s.extra["inner_edge_grade"],
                             leg_spacing=s.extra["leg_spacing"], alpha_deg=float(np.degrees(s.extra["alpha"])),
                             shape="keyhole" if s.extra["alpha"] > 0 else "semicircle",
                             wall_m2=s.extra["wall_m2"],
                             cut_m3=s.cut_soil_m3 + s.cut_rock_m3, fill_m3=s.fill_m3,
                             internal_conflict=bool(s.extra["conflict"]), pt=mid))
        if s.section_type == "bend":
            rows.append(dict(feature_type="bend", seq=s.seq, cost=s.cost_total, angle_deg=s.extra["angle_deg"],
                             radius=s.extra["radius"], cut_m3=s.cut_soil_m3 + s.cut_rock_m3, fill_m3=s.fill_m3,
                             pt=mid))
        if s.section_type == "fill_crossing":
            rows.append(dict(feature_type="crossing", crossing_type="fill", seq=s.seq,
                             contributing_area_ha=s.culvert_area_ha, fill_m3=s.fill_m3,
                             fill_from_spoil_m3=src.get(s.seq, {}).get("spoil_m3", 0.0),
                             fill_borrow_m3=src.get(s.seq, {}).get("borrow_m3", 0.0),
                             culvert_cost=s.cost_culvert, pt=mid))
        elif s.cost_culvert > 0:
            ch = np.floor(s.pts_rc + 0.5).astype(int)
            on = prob.channel[ch[:, 0], ch[:, 1]].astype(bool)
            at = s.pts_rc[np.flatnonzero(on)[len(np.flatnonzero(on)) // 2]][None] if on.any() else mid
            rows.append(dict(feature_type="crossing", crossing_type="at_grade", seq=s.seq,
                             contributing_area_ha=s.culvert_area_ha, fill_m3=0.0, fill_from_spoil_m3=0.0,
                             fill_borrow_m3=0.0, culvert_cost=s.cost_culvert, pt=at))
    for c in conflicts:
        rows.append(dict(feature_type="conflict", seq=c.upper, lower_seq=c.lower, separation=c.separation,
                         required=c.required, upper_type=by_seq[c.upper].section_type,
                         lower_type=by_seq[c.lower].section_type, pt=np.array([[c.row, c.col]])))
    for r in rows:
        r.update(run_id=run_id, route_id=route_id, rank=rank)
    return rows


def hairpin_candidate_rows(grid: Grid, prob, run_id) -> list[dict]:
    """One point per node with any feasible hairpin: cheapest cost, its
    heading (degrees, east = 0, CCW) and side, and how many are feasible."""
    if prob.hp.shape[0] == 0 or prob.macro_lookup is None:
        return []
    hp = prob.hp.min(axis=2)                      # (table cells, headings)
    rows = np.flatnonzero(np.isfinite(hp).any(axis=1))
    if not len(rows):
        return []
    # Table rows follow the cells with a lookup entry, in row-major order.
    cells = np.argwhere(prob.macro_lookup >= 0)[rows]
    sub = hp[rows]
    h = np.argmin(sub, axis=1)
    vec = heading_vectors(prob.n_theta)
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    return [dict(feature_type="hairpin_candidate", run_id=run_id, cost=float(sub[i, h[i]]),
                 heading_deg=float(ang[h[i]]), side="left", n_feasible=int(np.isfinite(sub[i]).sum()),
                 pt=np.array([[r, c]], dtype=float))
            for i, (r, c) in enumerate(cells)]


def _merge_write(path: Path, gdf: gpd.GeoDataFrame, run_id: str, sort_cols) -> None:
    if path.exists():
        old = gpd.read_parquet(path)
        old = old[old["run_id"] != run_id]
        if len(old):
            gdf = pd.concat([old, gdf], ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=old.crs)
    gdf = gdf.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    gdf.to_parquet(path, index=False)


def write_vectors(out: Path, grid: Grid, run_id: str, routes, prob) -> dict:
    """routes: ranked list of objects with .route_id, .segments, .cost, .lam_fill, .via, .params."""
    seg_rows, seg_geom, sum_rows, sum_geom, feat = [], [], [], [], []
    counts = {}
    for rank, rt in enumerate(routes, start=1):
        conflicts = stacked_leg_conflicts(rt.segments, grid.res)
        rows, geoms = segment_rows(grid, run_id, rt.route_id, rank, rt.segments, sink_costs(rt))
        seg_rows += rows
        seg_geom += geoms
        via_xy = rc_to_xy(grid, np.array([rt.via], dtype=float))[0]
        sum_rows.append(summary_row(run_id, rt.route_id, rank, rt, conflicts, via_xy))
        sum_geom.append(LineString(np.vstack([np.column_stack([rc_to_xy(grid, s.pts_rc), s.z])
                                              for s in rt.segments])))
        feat += feature_rows(grid, run_id, rt.route_id, rank, rt, conflicts, prob)
        counts[rt.route_id] = len(conflicts)
    feat += hairpin_candidate_rows(grid, prob, run_id)
    crs = grid.crs.to_wkt()
    routes_gdf = gpd.GeoDataFrame(pd.DataFrame(seg_rows, columns=ROUTE_COLUMNS), geometry=seg_geom, crs=crs)
    _merge_write(out / "routes.parquet", routes_gdf, run_id, ["run_id", "rank", "seq"])
    _merge_write(out / "route_summary.parquet",
                 gpd.GeoDataFrame(pd.DataFrame(sum_rows), geometry=sum_geom, crs=crs), run_id, ["run_id", "rank"])
    pts = [Point(*rc_to_xy(grid, r.pop("pt"))[0]) for r in feat]
    fdf = pd.DataFrame(feat)
    for col in ("run_id", "feature_type"):
        if col not in fdf:
            fdf[col] = pd.Series(dtype=object)
    fdf["run_id"] = run_id
    for col in ("route_id", "rank", "seq"):
        fdf[col] = (fdf[col] if col in fdf else pd.Series([pd.NA] * len(fdf))).astype("Int64")
    _merge_write(out / "features.parquet", gpd.GeoDataFrame(fdf, geometry=pts, crs=crs), run_id,
                 ["run_id", "feature_type", "rank", "seq"])
    return counts


def write_field_rasters(run_dir: Path, grid: Grid, fields, explored) -> None:
    inf_to_nan = lambda a: np.where(np.isfinite(a), a, np.nan)  # noqa: E731
    write_raster(run_dir / "corridor_field.tif", inf_to_nan(fields.F), grid)
    write_raster(run_dir / "cost_to_come_forward.tif", inf_to_nan(fields.fwd.min_over_headings()), grid)
    write_raster(run_dir / "cost_to_come_backward.tif", inf_to_nan(fields.bwd.min_over_headings()), grid)
    if explored is not None:
        write_raster(run_dir / "explored_astar.tif", inf_to_nan(explored), grid)


def write_params_json(run_dir: Path, record: dict) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not math.isfinite(o) else float(o)
        return str(o)
    (run_dir / "params.json").write_text(json.dumps(record, indent=2, default=default), encoding="utf-8")


VECTOR_FILES = {"routes.parquet": ["run_id", "rank", "seq"], "route_summary.parquet": ["run_id", "rank"],
                "features.parquet": ["run_id", "feature_type", "rank", "seq"]}


def merge_outputs(sources: list[Path], dest: Path) -> list[str]:
    """Combine output roots (e.g. one per CI job) into dest: run folders are
    copied, and each run's rows in the GeoParquet files replace any it had.
    Returns the run ids merged."""
    import shutil

    dest = Path(dest)
    (dest / "runs").mkdir(parents=True, exist_ok=True)
    merged = []
    for src in map(Path, sources):
        for rd in sorted((src / "runs").iterdir()) if (src / "runs").exists() else []:
            if rd.is_dir() and not (dest / "runs" / rd.name).exists():
                shutil.copytree(rd, dest / "runs" / rd.name)
            merged.append(rd.name)
        for name, sort_cols in VECTOR_FILES.items():
            if not (src / name).exists():
                continue
            gdf = gpd.read_parquet(src / name)
            for run_id, part in gdf.groupby("run_id", sort=True):
                _merge_write(dest / name, gpd.GeoDataFrame(part, geometry="geometry", crs=gdf.crs), run_id, sort_cols)
    return sorted(set(merged))
