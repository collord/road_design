"""End-to-end run: inspect, clip, preprocess, search, alternatives, outputs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from scipy import ndimage

from . import memguard
from .exclusion import ExclusionError, normalize_spec, split_spec
from .explore import write_corridor_tree
from .masshaul import TrueCost, true_cost
from .outputs import write_field_rasters, write_params_json, write_vectors
from .params import Params
from .pipeline import alternatives, avoid_self_overlap, corridor_fields, fill_price_iteration
from .preprocess import ConnectivityError, EndpointError, preprocess
from .quicklook import quicklook
from .route import Segment, evaluate_route, stacked_leg_conflicts
from .search import make_problem, path_xy, reconstruct, run_search
from .timing import stage
from .terrain import NICE_RESOLUTIONS, Grid, inspect_dem, load_grid, parse_point, write_raster


@dataclass
class RankedRoute:
    route_id: int
    via: tuple[int, int]
    edges: list
    segments: list[Segment]
    cost: TrueCost
    lam_fill: float
    params: Params


# Parameters added after runs were made: left out of the run id at their default, so the
# ids (and the reuse of finished runs) of runs made before them don't change.
_LATER_PARAMS = {"C_turn": 0.0}


def _id_params(p: Params) -> dict:
    d = p.to_dict()
    for k, v in _LATER_PARAMS.items():
        if d.get(k) == v:
            d.pop(k, None)
    if d.get("alt_method") == "via":       # the original method: ids as before it had a name
        d.pop("alt_method", None)
        d.pop("alt_plateau", None)
    return d


def make_run_id(payload: dict) -> str:
    """Deterministic id from parameters and inputs, so identical runs coincide."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


BYTES_PER_STATE = 26   # two Dijkstra fields (cost, parent, closed) + hairpin table; picks resolutions
# What a run holds, for the memory check: per searched state, the two corridor searches and the
# last A* (cost, parent, closed: 11 bytes each) and the hairpin and bend tables (16); per grid
# cell, the terrain rasters and lookups.
SEARCH_BYTES_PER_STATE = 50
BYTES_PER_CELL = 150


def resolution_for(bounds, p: Params, region_area: float | None = None, floor: float = 1.0) -> float:
    """Smallest nice resolution with the searchable states (cells in the
    region, or the whole box) under max_states and the whole grid's search
    memory under max_memory_gb."""
    box = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    region = box if region_area is None else region_area
    for r in NICE_RESOLUTIONS:
        if (r >= floor and region / r**2 * p.N_theta <= p.max_states
                and box / r**2 * p.N_theta * BYTES_PER_STATE <= p.max_memory_gb * 1e9):
            return float(r)
    return float(NICE_RESOLUTIONS[-1])


def check_memory(grid: Grid, p: Params, what: str = "working grid", n_search: int | None = None) -> None:
    """Refuse a grid over the max_memory_gb budget (the measure resolution_for chooses by), or
    one that would not fit in the memory the machine has free. Searches and tables hold only
    the searched cells (n_search; default all of them)."""
    n_cells = grid.shape[0] * grid.shape[1]
    budget = n_cells * p.N_theta * BYTES_PER_STATE / 1e9
    if budget > p.max_memory_gb:
        raise MemoryError(f"{what} needs ~{budget:.1f} GB of search memory, over max_memory_gb = "
                          f"{p.max_memory_gb:g}; use a coarser --resolution, a tighter clip, or raise it")
    n_search = n_cells if n_search is None else n_search
    need = (n_search * p.N_theta * SEARCH_BYTES_PER_STATE + n_cells * BYTES_PER_CELL) / memguard.GB
    memguard.require(need, what, memguard.FLOOR_GB)


def _region_bounds(grid: Grid, mask: np.ndarray, pad: float):
    rows, cols = np.nonzero(mask)
    t = grid.transform
    xmin = t.c + cols.min() * t.a - pad
    xmax = t.c + (cols.max() + 1) * t.a + pad
    ymax = t.f + rows.min() * t.e + pad
    ymin = t.f + (rows.max() + 1) * t.e - pad
    return (xmin, ymin, xmax, ymax)


def _resample_mask(src: Grid, mask: np.ndarray, dst: Grid) -> np.ndarray:
    """Nearest-neighbor lookup of a coarse mask at fine cell centers."""
    rr, cc = np.mgrid[0:dst.shape[0], 0:dst.shape[1]]
    x = dst.transform.c + (cc + 0.5) * dst.transform.a
    y = dst.transform.f + (rr + 0.5) * dst.transform.e
    col = np.floor((x - src.transform.c) / src.transform.a).astype(int)
    row = np.floor((y - src.transform.f) / src.transform.e).astype(int)
    ok = (row >= 0) & (row < src.shape[0]) & (col >= 0) & (col < src.shape[1])
    out = np.zeros(dst.shape, dtype=bool)
    out[ok] = mask[row[ok], col[ok]]
    return out


def coarse_corridor(dem, p: Params, start, goal, exclude, coarse_res, log):
    """Coarse search over the endpoint box plus clip_coarse_margin, at the
    finest resolution that fits max_states (or coarse_res). The margin is
    doubled once if the coarse route reaches the edge of the box.
    Returns (grid, region mask, info, fields)."""
    margin = p.clip_coarse_margin
    for attempt in range(2):
        bounds = (min(start[0], goal[0]) - margin, min(start[1], goal[1]) - margin,
                  max(start[0], goal[0]) + margin, max(start[1], goal[1]) + margin)
        res = coarse_res or resolution_for(bounds, p)
        with stage("load and preprocess"):
            grid, ginfo = load_grid(dem, p, res, bounds)
            check_memory(grid, p, "coarse grid")
            t = preprocess(grid, start, goal, p, exclude)
        with stage("hairpin and bend tables"):
            prob = make_problem(t, p)
        with stage("A* search", resolution_m=float(grid.res)) as row:
            a = run_search(prob, t.start_rc, t.goal_rc, astar=True)
            row["settled"] = int(a.n_expanded)
        if a.goal_state is None:
            raise ConnectivityError(
                f"coarse search ({res:g} m, margin {margin:g} m) found no route; "
                "try a finer --coarse-res or --clip box")
        edges = reconstruct(a, prob.lat, a.goal_state)
        if not touches_boundary(prob, edges, t.start_rc, ~np.isnan(grid.z)) or attempt == 1:
            break
        log(f"coarse route reaches the edge of the {margin:g} m box; doubling the margin")
        margin *= 2
    thr = (1 + p.eps_alt) * (1 + p.clip_slack) * a.goal_cost
    with stage("corridor fields"):
        fields = corridor_fields(prob, t.start_rc, t.goal_rc, max_cost=thr * (1 + 1e-9))
    region = fields.F <= thr
    dist = ndimage.distance_transform_edt(~region) * grid.res
    region = dist <= p.clip_buffer
    info = {"coarse_res_m": grid.res, "coarse_shape": list(grid.shape), "coarse_cost": a.goal_cost,
            "coarse_margin_m": margin, "coarse_bounds": list(bounds),
            "coarse_threshold": thr, "region_cells": int(region.sum())}
    log(f"coarse pass at {grid.res:g} m: cost {a.goal_cost:,.0f}, corridor {region.sum() * grid.res**2 / 1e6:.1f} km2")
    return grid, region, info, fields


def touches_boundary(prob, edges, start_rc, allowed: np.ndarray) -> bool:
    """True if the route comes within one cell of the edge of the allowed region."""
    pts = path_xy(prob, edges, start_rc)
    cells = np.unique(np.floor(pts + 0.5).astype(int), axis=0)
    edge = ~ndimage.binary_erosion(allowed, structure=np.ones((3, 3)), border_value=0)
    return bool(edge[cells[:, 0], cells[:, 1]].any())


def run(dem, start_txt: str, goal_txt: str, p: Params, exclude=(), out="out", resolution=None,
        clip="none", clip_margin=1000.0, coarse_res=None, run_id=None, endpoint_crs=None,
        log=print, reuse=False, tree=False) -> dict:
    """One full run. With reuse, a run with the same inputs that already finished is returned as is.
    With tree, the corridor tree is saved too (display only: the viewer's "best route through a
    point"); `roadsearch tree RUN_ID` adds it to a finished run.

    Stages are timed (roadsearch.timing): the profile is saved in params.json and logged.
    Memory is guarded (roadsearch.memguard): if the machine runs short, or the kernel's non-paged
    pool balloons, the run stops, records why, and raises MemoryGuardError."""
    from . import timing

    s0 = memguard.snapshot()
    if s0.get("kernel_nonpaged_gb", 0.0) > memguard.POOL_WARN_GB:
        log(f"memory guard: the kernel's non-paged pool is already {s0['kernel_nonpaged_gb']:.1f} GB "
            f"(top tags {memguard.describe_tags(memguard.pool_tags(3))}); a restart clears it")
    ctx = {}

    def memory_record():
        return {"start": s0, **wd.peak, "pool_events": wd.pool_events}

    def record_abort(why):
        rec = ctx.get("record")
        if rec is not None:
            rec["error"] = f"MemoryGuardError: {why}"
            rec["memory"] = memory_record()
            rec["profile"] = prof.rows
            write_params_json(ctx["run_dir"], rec)
        log(f"memory guard stopped the run: {why}")

    prof = timing.start()
    wd = memguard.Watchdog(log=log, on_hard_stop=record_abort)
    try:
        with wd, timing.stage("run"):
            rec = _run(dem, start_txt, goal_txt, p, exclude, out, resolution, clip, clip_margin, coarse_res,
                       run_id, endpoint_crs, log, reuse, ctx, tree)
    except BaseException:
        # The interrupt can surface as something else (numba turns it into SystemError) or be
        # swallowed below, so the watchdog's reason, not the exception type, decides.
        if wd.reason is None:
            raise
        record_abort(wd.reason)
        raise memguard.MemoryGuardError(wd.reason) from None
    finally:
        timing.stop()
    if wd.reason is not None:
        record_abort(wd.reason)
        raise memguard.MemoryGuardError(wd.reason)
    if "timings" in rec and "profile" not in rec:
        rec["profile"] = prof.rows
        rec["memory"] = memory_record()
        write_params_json(Path(out) / "runs" / rec["run_id"], rec)
        log("time by stage:" + chr(10) + prof.table())
        log("memory: " + ", ".join(f"{k} {v}" for k, v in wd.peak.items()))
    return rec


def _run(dem, start_txt: str, goal_txt: str, p: Params, exclude=(), out="out", resolution=None,
         clip="none", clip_margin=1000.0, coarse_res=None, run_id=None, endpoint_crs=None,
         log=print, reuse=False, ctx=None, tree=False) -> dict:
    t_start = time.time()
    out = Path(out)
    timings = {}
    dem_info = inspect_dem(dem, p)
    with rasterio.open(dem) as ds:
        dem_crs = CRS.from_user_input(ds.crs)
    wcrs = CRS.from_user_input(dem_info["working_crs_epsg"] or dem_info["working_crs_name"])
    pt_crs = CRS.from_user_input(endpoint_crs) if endpoint_crs else dem_crs
    start = parse_point(start_txt, pt_crs, wcrs)
    goal = parse_point(goal_txt, pt_crs, wcrs)
    exclude = [normalize_spec(e) for e in exclude]
    inputs = {"dem": dem_info["path"], "start": start, "goal": goal, "exclude": exclude,
              "resolution": resolution, "clip": clip, "clip_margin": clip_margin,
              "coarse_res": coarse_res, "params": _id_params(p)}
    if exclude:
        # Zone files are identified by their contents too, so an edited file is a new run.
        inputs["exclude_sha256"] = [hashlib.sha256(Path(split_spec(e)[0]).read_bytes()).hexdigest()[:16]
                                    for e in exclude]
    run_id = run_id or make_run_id(inputs)
    run_dir = out / "runs" / run_id
    if reuse and (run_dir / "params.json").exists():
        prev = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
        if "error" not in prev and "total_s" in prev.get("timings", {}):
            log(f"run {run_id}: already finished, reusing it")
            return prev
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "inputs": inputs, "params": p.to_dict(), "dem": dem_info}
    if ctx is not None:
        ctx.update(record=record, run_dir=run_dir)
    log(f"run {run_id}")

    def fail(exc):
        record["error"] = f"{type(exc).__name__}: {exc}"
        write_params_json(run_dir, record)
        log(f"run {run_id}: {record['error']}")
        return record

    buffer = p.clip_buffer
    attempts = 0
    while True:
        attempts += 1
        region_mask = None
        try:
            if clip == "none":
                bounds = None
                res = resolution or dem_info["proposed_res_m"]
            elif clip == "box":
                bounds = (min(start[0], goal[0]) - clip_margin, min(start[1], goal[1]) - clip_margin,
                          max(start[0], goal[0]) + clip_margin, max(start[1], goal[1]) + clip_margin)
                res = resolution or resolution_for(bounds, p)
            elif clip == "auto":
                t0 = time.time()
                pc = dataclasses.replace(p, clip_buffer=buffer)
                with stage('coarse pass'):
                    cgrid, cregion, cinfo, cfields = coarse_corridor(dem, pc, start, goal, exclude, coarse_res, log)
                timings["coarse_s"] = time.time() - t0
                record["coarse"] = cinfo
                write_raster(run_dir / "coarse_corridor_field.tif",
                             np.where(np.isfinite(cfields.F), cfields.F, np.nan), cgrid)
                del cfields
                bounds = _region_bounds(cgrid, cregion, 0.0)
                res = resolution or resolution_for(bounds, p, float(cregion.sum()) * cgrid.res**2)
            else:
                raise ValueError(f"unknown clip mode {clip!r}")
            t0 = time.time()
            with stage('fine: load the grid'):
                grid, ginfo = load_grid(dem, p, res, bounds)
            if clip == "auto":
                region_mask = _resample_mask(cgrid, cregion, grid)
            check_memory(grid, p, n_search=int(region_mask.sum()) if region_mask is not None else None)
            record["grid"] = ginfo
            n_search = int(region_mask.sum()) if region_mask is not None else grid.shape[0] * grid.shape[1]
            log(f"working grid {grid.shape[0]} x {grid.shape[1]} at {grid.res:g} m "
                f"({n_search * p.N_theta:,} searchable states)")
            with stage('fine: preprocess'):
                terrain = preprocess(grid, start, goal, p, exclude, run_dir)
            if region_mask is not None:
                terrain.clip = region_mask
                write_raster(run_dir / "clip_region.tif", region_mask, grid, nodata=255)
            record["preprocess"] = terrain.info
            timings["preprocess_s"] = time.time() - t0
        except (EndpointError, ExclusionError, ConnectivityError, MemoryError) as exc:
            return fail(exc)

        t0 = time.time()
        with stage('fine: hairpin and bend tables'):
            prob = make_problem(terrain, p)
        timings["hairpin_table_s"] = time.time() - t0
        t0 = time.time()
        with stage('fine: A* with fill-price repeats'):
            it = fill_price_iteration(prob, terrain.start_rc, terrain.goal_rc)
        timings["fill_price_iteration_s"] = time.time() - t0
        record["fill_price_iteration"] = [h.__dict__ for h in it.history]
        with stage("self-overlap avoidance") as row:
            it, overlap_rounds, overlap_blocked = avoid_self_overlap(prob, terrain.start_rc, terrain.goal_rc, it,
                                                                     log=log)
            row["rounds"] = len(overlap_rounds)
        record["self_overlap_rounds"] = overlap_rounds
        if overlap_blocked.any():
            write_raster(run_dir / "overlap_blocked.tif", overlap_blocked, grid, nodata=255)
        explored = it.last_search.min_over_headings(closed_only=True) if it.last_search else None
        it.last_search = None
        if it.best is None:
            return fail(ConnectivityError("no feasible route on the lattice (all edges infeasible somewhere "
                                          "between the endpoints)"))
        allowed = region_mask if region_mask is not None else np.ones(prob.open_.shape, bool)
        if clip != "none" and touches_boundary(prob, it.best.edges, terrain.start_rc, allowed):
            record.setdefault("clip_warnings", []).append(
                f"attempt {attempts}: best route reaches the clip boundary (buffer {buffer:g} m)")
            log(record["clip_warnings"][-1])
            if clip == "auto" and attempts < 2:
                buffer *= 2
                continue
        break

    t0 = time.time()
    with stage('corridor fields (two full Dijkstra)'):
        fields = corridor_fields(prob, terrain.start_rc, terrain.goal_rc)
    timings["corridor_fields_s"] = time.time() - t0
    with stage('alternatives'):
        alts = alternatives(prob, fields, terrain.start_rc, log=log)
    routes = []
    with stage("true costs of the routes", routes=len(alts)):
        for k, a in enumerate(alts):
            segs = evaluate_route(prob, a.edges)
            routes.append(RankedRoute(k, a.via, a.edges, segs, true_cost(segs, p), prob.lam_fill, p))
    routes.sort(key=lambda r: (r.cost.total, r.route_id))
    timings["alternatives_s"] = time.time() - t0 - timings["corridor_fields_s"]

    with stage('write field rasters'):
        write_field_rasters(run_dir, grid, fields, explored)
    if tree:
        with stage('write corridor tree'):
            write_corridor_tree(run_dir, prob, fields)
    with stage('write vectors (and conflict checks)'):
        conflicts = write_vectors(out, grid, run_id, routes, prob)
    with stage("quick-look image"):
        quicklook(run_dir / "quicklook.png", terrain, fields, routes, prob, p,
                  f"run {run_id}: {grid.res:g} m, N_theta={p.N_theta}, clip={clip}")
    record["routes"] = [{"rank": i + 1, "route_id": r.route_id, "via_rc": list(r.via),
                         "cost_true": r.cost.total, "cost_surrogate": r.cost.surrogate,
                         "length_m": sum(s.length for s in r.segments),
                         "hairpins": sum(s.section_type == "hairpin" for s in r.segments),
                         "conflicts": conflicts[r.route_id]} for i, r in enumerate(routes)]
    record["corridor_F_min"] = fields.F_min
    record["lam_fill_final"] = prob.lam_fill
    timings["total_s"] = time.time() - t_start
    record["timings"] = timings
    write_params_json(run_dir, record)
    for r in record["routes"]:
        log(f"  rank {r['rank']}: true cost {r['cost_true']:,.0f} (surrogate {r['cost_surrogate']:,.0f}), "
            f"{r['length_m']:,.0f} m, {r['hairpins']} hairpins, {r['conflicts']} conflicts")
    return record
