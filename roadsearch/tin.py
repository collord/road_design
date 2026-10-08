"""The coarse stage on a sparse graph over the whole DEM (`--clip tin`; rust/tin-graph).

Nodes are where roads plausibly go or turn: the vertices of a triangulation of the smoothed DEM
(to tin_tolerance), channel nodes, saddles and a sparse background grid. Each node is joined to
its tin_k nearest neighbors within tin_edge_max, and every edge is costed on the rasters with the
road cost model (earthwork, culverts, travel and climb, a switchback allowance where an edge is
steeper than g_max). Two exact searches give the best route and the corridor field; distinct
corridors come from saddles and a penalty loop. The corridors, buffered by tin_buffer, are the
region the fine lattice search covers.

Unlike `--clip auto` there is no search window: the whole DEM is in play.
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
from .timing import stage

KINDS = {0: "triangulation", 1: "channel", 2: "saddle", 3: "background", 4: "endpoint"}


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

    buffer = p.tin_buffer if buffer is None else buffer
    with stage("load and preprocess"):
        grid, _ = load_grid(dem, p, p.tin_res, None)
        t = preprocess(grid, start, goal, p, exclude)
    res = grid.res
    f8 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
    rasters = (f8(grid.z), f8(t.z_smooth), f8(t.dzdx), f8(t.dzdy), np.ascontiguousarray(t.code == 0, dtype=np.uint8),
               f8(np.nan_to_num(t.area_ha)))
    table = build_cost_table(p)
    edge = (p.g_max, p.C_travel, p.C_vertical, p.A_min, hairpin_allowance(p),
            np.array(p.culvert_area_breaks, dtype=np.float64), np.array(p.culvert_costs, dtype=np.float64))
    tin = (p.tin_tolerance, p.tin_k, p.tin_edge_max / res, p.tin_block / res, p.tin_channel_ha, p.tin_alternatives,
           p.tin_penalty, p.tin_stretch, p.tin_max_shared)
    with stage("graph and corridors") as row:
        rows, cols, kinds, field, routes, n_edges = roadsearch_core.tin_corridors(
            rasters, np.ascontiguousarray(table.table), table.ds, table.dg, res, tuple(map(int, t.start_rc)),
            tuple(map(int, t.goal_rc)), edge, tin)
        row.update(nodes=int(rows.size), edges=int(n_edges), corridors=len(routes))
    if not routes:
        raise ConnectivityError("no route on the coarse graph between the endpoints")
    xy = lambda idx: [grid.xy(float(rows[i]), float(cols[i])) for i in idx]  # noqa: E731
    lines = [LineString(xy(r[0])) for r in routes]
    recs = [{"rank": k + 1, "cost": r[1], "construction": r[2], "operating": r[3], "length_m": r[4], "hairpins": r[5],
             "found_by": r[6]} for k, r in enumerate(routes)]
    gdf = gpd.GeoDataFrame(recs, geometry=lines, crs=grid.crs)
    region = features.rasterize([(unary_union([l.buffer(buffer) for l in lines]), 1)], out_shape=grid.shape,
                                transform=grid.transform, fill=0, dtype="uint8").astype(bool)
    info = {"tin_res_m": res, "nodes": int(rows.size), "node_kinds": {KINDS[k]: int((kinds == k).sum()) for k in KINDS},
            "edges": int(n_edges), "buffer_m": buffer, "region_cells": int(region.sum()), "corridors": recs}
    best = recs[0]
    log(f"coarse graph at {res:g} m: {rows.size:,} nodes, {n_edges:,} edges; {len(routes)} corridors, best "
        f"{best['cost']:,.0f} ({best['length_m'] / 1000:.1f} km, {best['hairpins']} hairpins); "
        f"region {region.sum() * res**2 / 1e6:.1f} km2")
    return grid, region, info, gdf
