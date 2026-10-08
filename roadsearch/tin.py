"""The coarse stage on a sparse graph over the whole DEM (`--clip tin`; rust/tin-graph).

Nodes are where roads plausibly go or turn: the vertices of a triangulation of the smoothed DEM
(to tin_tolerance), channel nodes, saddles and a sparse background grid. Each node is joined to
its tin_k nearest neighbors within tin_edge_max, and every edge is costed on the rasters with the
road cost model (earthwork, culverts, travel and climb, a switchback allowance where an edge is
steeper than g_max). Two exact searches give the best route and the corridor field; distinct
corridors come from saddles and a penalty loop. The corridors, buffered by tin_buffer, are the
region the fine lattice search covers.

The graph covers the endpoints' box plus tin_margin (by default 10 km, room for detours down
valleys), within the DEM: a whole USGS tile is never loaded. If that window at tin_res would
exceed tin_max_cells, the graph is built coarser, and the log says so.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import LineString
from shapely.ops import unary_union

from .params import Params
from .preprocess import ConnectivityError, preprocess
from .terrain import load_grid
from . import events
from .timing import stage

KINDS = {0: "triangulation", 1: "channel", 2: "saddle", 3: "background", 4: "endpoint", 5: "sampled"}

# Recorded in a --clip tin run's inputs (so in its id): bump it when the graph or its corridor
# search changes what corridors it finds. (The first version, with no turning cost on the graph,
# recorded nothing.) tin-2: turning cost on the graph, penalties graded by the corridor field.
# tin-3: the graph covers the endpoints' box plus tin_margin, not the whole DEM.
COARSE_ENGINE = "tin-3"

# What building the graph holds per cell of its window: the rasters (working and smoothed DEM,
# gradients, slope, soil, area, codes...) and the graph's share. For the memory check.
BYTES_PER_GRAPH_CELL = 160


def graph_window(dem, p: Params, start, goal, log=print):
    """(bounds, resolution) of the graph: the endpoints' box plus tin_margin, within the DEM's
    extent (in the working CRS), at tin_res or coarser if it would exceed tin_max_cells."""
    import rasterio
    from rasterio.warp import transform_bounds

    from .terrain import NICE_RESOLUTIONS, working_crs

    from pyproj import CRS

    with rasterio.open(dem) as ds:
        src = CRS.from_user_input(ds.crs)
        wcrs = working_crs(src, ds.bounds)
        dx0, dy0, dx1, dy1 = transform_bounds(src, wcrs, *ds.bounds)
    m = p.tin_margin
    bounds = (max(min(start[0], goal[0]) - m, dx0), max(min(start[1], goal[1]) - m, dy0),
              min(max(start[0], goal[0]) + m, dx1), min(max(start[1], goal[1]) + m, dy1))
    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    res = p.tin_res
    if area / res**2 > p.tin_max_cells:
        res = next((r for r in NICE_RESOLUTIONS if r >= res and area / r**2 <= p.tin_max_cells),
                   math.sqrt(area / p.tin_max_cells))
        log(f"coarse graph: a {area / 1e6:,.0f} km2 window at {p.tin_res:g} m would be "
            f"{area / p.tin_res**2 / 1e6:,.1f} million cells (tin_max_cells {p.tin_max_cells / 1e6:g} million); "
            f"building it at {res:g} m")
    return bounds, float(res)


def hairpin_allowance(p: Params) -> float:
    """What a hairpin implied by a steep graph edge is charged: extra road around the loop and the
    turn, unless tin_hairpin_cost sets it."""
    if p.tin_hairpin_cost >= 0:
        return p.tin_hairpin_cost
    return (p.C_base + p.C_travel) * (math.pi + 2.0) * p.R_hp + p.C_turn * math.pi


def tin_corridor(dem, p: Params, start, goal, exclude, log=print, buffer: float | None = None):
    """Returns (grid, region mask, info, corridors as a GeoDataFrame) on the tin_res grid."""
    import roadsearch_core

    from .search import build_cost_table

    from . import memguard

    buffer = p.tin_buffer if buffer is None else buffer
    bounds, gres = graph_window(dem, p, start, goal, log)
    cells = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / gres**2
    memguard.require(cells * BYTES_PER_GRAPH_CELL / memguard.GB, "coarse graph", memguard.FLOOR_GB)
    with stage("load and preprocess", resolution_m=gres):
        grid, _ = load_grid(dem, p, gres, bounds)
        t = preprocess(grid, start, goal, p, exclude)
    res = grid.res
    f8 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
    rasters = (f8(grid.z), f8(t.z_smooth), f8(t.dzdx), f8(t.dzdy), np.ascontiguousarray(t.code == 0, dtype=np.uint8),
               f8(np.nan_to_num(t.area_ha)))
    table = build_cost_table(p)
    edge = (p.g_max, p.C_travel, p.C_vertical, p.A_min, hairpin_allowance(p), p.C_turn,
            np.array(p.culvert_area_breaks, dtype=np.float64), np.array(p.culvert_costs, dtype=np.float64))
    tin = (p.tin_tolerance, p.tin_k, p.tin_edge_max / res, p.tin_block / res, p.tin_channel_ha, p.tin_alternatives,
           p.tin_penalty, p.tin_stretch, p.tin_max_shared)
    near = buffer / res
    cost_s = (p.cost_samples, p.cost_sigma, p.cost_sigma, p.cost_sigma_edge, 20261007)
    open_km2 = float((t.code == 0).sum()) * res**2 / 1e6
    node_s = (p.tin_node_samples, round(p.tin_node_density * open_km2), 20261008)

    def sampled(i, which):
        events.emit("node_sample", sample=int(i) + 1, of=int(p.tin_node_samples), corridor=int(which))

    with stage("graph and corridors") as row:
        rows, cols, kinds, field, routes, cost_none, node_none, n_edges = roadsearch_core.tin_corridors(
            rasters, np.ascontiguousarray(table.table), table.ds, table.dg, res, tuple(map(int, t.start_rc)),
            tuple(map(int, t.goal_rc)), edge, tin, cost_s, node_s, near, sampled)
        row.update(nodes=int(rows.size), edges=int(n_edges), corridors=len(routes))
    if not routes:
        raise ConnectivityError("no route on the coarse graph between the endpoints")
    nan = lambda v: None if v != v else v  # noqa: E731
    lines = [LineString([grid.xy(float(r_), float(c_)) for r_, c_ in zip(r[0], r[1])]) for r in routes]
    recs = [{"rank": k + 1, "cost": r[2], "construction": r[3], "operating": r[4], "length_m": r[5], "hairpins": r[6],
             "turning_deg": math.degrees(r[7]), "found_by": r[8], "cost_sample_share": nan(r[9]),
             "node_sample_share": nan(r[10])} for k, r in enumerate(routes)]
    gdf = gpd.GeoDataFrame(recs, geometry=lines, crs=grid.crs)
    region = features.rasterize([(unary_union([l.buffer(buffer) for l in lines]), 1)], out_shape=grid.shape,
                                transform=grid.transform, fill=0, dtype="uint8").astype(bool)
    info = {"tin_res_m": res, "window_bounds": [round(v, 1) for v in bounds], "nodes": int(rows.size), "node_kinds": {KINDS[k]: int((kinds == k).sum()) for k in KINDS},
            "edges": int(n_edges), "buffer_m": buffer, "region_cells": int(region.sum()), "corridors": recs,
            "cost_samples": int(p.cost_samples), "cost_sample_share_none": nan(cost_none),
            "node_samples": int(p.tin_node_samples), "node_sample_nodes": int(node_s[1]),
            "node_sample_share_new": nan(node_none)}
    best = recs[0]
    log(f"coarse graph at {res:g} m: {rows.size:,} nodes, {n_edges:,} edges; {len(routes)} corridors, best "
        f"{best['cost']:,.0f} ({best['length_m'] / 1000:.1f} km, {best['hairpins']} hairpins); "
        f"region {region.sum() * res**2 / 1e6:.1f} km2")
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    if p.tin_node_samples > 0:
        new = sum(r["found_by"] == "sampled" for r in recs)
        log(f"node samples ({int(p.tin_node_samples)}, {node_s[1]:,} random nodes each): corridor shares "
            + ", ".join(f"{r['rank']}: {pct(r['node_sample_share'])}" for r in recs if r["found_by"] != "sampled")
            + f"; new corridors {new} ({pct(nan(node_none))} of samples)")
    if p.cost_samples > 0:
        log(f"cost samples ({int(p.cost_samples)}): corridor shares "
            + ", ".join(f"{r['rank']}: {pct(r['cost_sample_share'])}" for r in recs if r["found_by"] != "sampled")
            + f", none of them: {pct(nan(cost_none))}")
    return grid, region, info, gdf
