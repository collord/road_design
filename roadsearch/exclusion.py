"""Manual exclusion zones from polygon files (brief section 3.6)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .terrain import Grid

# Coded exclusion raster values. The search treats any non-zero as excluded.
OPEN, SLOPE, MANUAL, BOTH = 0, 1, 2, 3


class ExclusionError(ValueError):
    pass


@dataclass
class ZoneFile:
    path: str
    gdf: gpd.GeoDataFrame          # repaired, buffered, working CRS; index = source row
    n_features_read: int
    n_dropped: int
    raster: np.ndarray = field(repr=False)

    def record(self, grid: Grid) -> dict:
        return {
            "path": self.path,
            "feature_count": int(len(self.gdf)),
            "features_read": self.n_features_read,
            "features_dropped": self.n_dropped,
            "polygon_area_m2": float(unary_union(list(self.gdf.geometry)).area) if len(self.gdf) else 0.0,
            "excluded_cells": int(self.raster.sum()),
            "excluded_area_m2": float(self.raster.sum() * grid.cell_area),
        }

    def describe(self, idx) -> str:
        row = self.gdf.loc[idx]
        label = next((f"{c}={row[c]!r}" for c in ("name", "NAME", "Name", "id", "ID", "fid") if c in row.index), None)
        return f"feature {idx}" + (f" ({label})" if label else "")


def _polygonal(geom):
    """Polygonal part of a geometry, or None."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        parts = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        return unary_union(parts) if parts else None
    return None


def load_zone_file(path: str | Path, grid: Grid, buffer_m: float) -> ZoneFile:
    path = str(path)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ExclusionError(
            f"exclusion file {path} has no CRS; define one (e.g. a .prj next to a shapefile) and retry"
        )
    n_read = len(gdf)
    gdf = gdf.to_crs(grid.crs)
    invalid = ~gdf.geometry.is_valid & gdf.geometry.notna()
    if invalid.any():
        warnings.warn(f"{path}: repairing {int(invalid.sum())} invalid geometries", stacklevel=2)
        gdf.loc[invalid, gdf.geometry.name] = gdf.geometry[invalid].make_valid()
    poly = gdf.geometry.apply(_polygonal)
    dropped = int(poly.isna().sum())
    if dropped:
        types = sorted(set(gdf.geometry[poly.isna()].geom_type.dropna()))
        warnings.warn(f"{path}: ignoring {dropped} non-polygon or empty features ({', '.join(types) or 'empty'})",
                      stacklevel=2)
    gdf = gdf.set_geometry(poly).loc[poly.notna()]
    if buffer_m:
        gdf = gdf.set_geometry(gdf.geometry.buffer(buffer_m))
    gdf = gdf.loc[~gdf.geometry.is_empty]
    if len(gdf):
        raster = rasterize(
            ((g, 1) for g in gdf.geometry), out_shape=grid.shape, transform=grid.transform,
            fill=0, all_touched=True, dtype="uint8",
        ).astype(bool)
    else:
        raster = np.zeros(grid.shape, dtype=bool)
    return ZoneFile(path, gdf, n_read, dropped, raster)


def check_endpoints(zones: list[ZoneFile], grid: Grid, endpoints: dict[str, tuple[float, float]]) -> None:
    """Stop if an endpoint lies in a manual zone (or in a cell it rasterizes to)."""
    problems = []
    for name, (x, y) in endpoints.items():
        pt = Point(x, y)
        r, c = grid.rowcol(x, y)
        for z in zones:
            hits = z.gdf.index[z.gdf.geometry.covers(pt)]
            for idx in hits:
                problems.append(f"{name} ({x:.2f}, {y:.2f}) lies inside {z.describe(idx)} of {z.path}")
            if not len(hits) and grid.contains(r, c) and z.raster[r, c]:
                near = z.gdf.distance(pt).idxmin()
                problems.append(
                    f"{name} ({x:.2f}, {y:.2f}) is in a cell touched by {z.describe(near)} of {z.path} "
                    f"({z.gdf.distance(pt)[near]:.2f} m away; rasterization includes all touched cells)"
                )
    if problems:
        raise ExclusionError("endpoint in manual exclusion zone:\n  " + "\n  ".join(problems))


def combine_codes(slope_excluded: np.ndarray, manual: np.ndarray) -> np.ndarray:
    code = np.zeros(slope_excluded.shape, dtype=np.uint8)
    code[slope_excluded] |= SLOPE
    code[manual] |= MANUAL
    return code
