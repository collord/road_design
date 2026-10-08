"""Terrain preprocessing (brief section 3): derived rasters, masks, connectivity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .engine import use_rust
from .exclusion import ZoneFile, check_endpoints, combine_codes, load_zone_file
from .flow import contributing_area_ha
from .params import Params
from .section import max_catch_at_mask, soil_depth
from .terrain import Grid, write_raster

# 4-connectivity: the search never squeezes diagonally between two excluded
# cells that touch at a corner, so neither does the flood fill.
FOUR = ndimage.generate_binary_structure(2, 1)


class ConnectivityError(RuntimeError):
    pass


class EndpointError(ValueError):
    pass


@dataclass
class Terrain:
    grid: Grid                  # raw working DEM (elevations come from here)
    z_smooth: np.ndarray
    dzdx: np.ndarray            # gradients of the smoothed surface
    dzdy: np.ndarray
    slope: np.ndarray           # |grad z|
    soil: np.ndarray            # vertical soil depth
    area_ha: np.ndarray         # contributing area
    channel: np.ndarray         # area_ha >= A_min
    slope_excluded: np.ndarray  # slope mask after dilation, plus nodata
    manual: np.ndarray          # union of manual zones
    code: np.ndarray            # 0 open, 1 slope, 2 manual, 3 both
    reachable: np.ndarray       # 4-connected open region containing the start
    start_rc: tuple[int, int]
    goal_rc: tuple[int, int]
    zones: list[ZoneFile] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    clip: np.ndarray | bool = True   # cells the search may use (clip region); True = all

    @property
    def excluded(self) -> np.ndarray:
        return self.code != 0


def smooth_dem(z: np.ndarray, sigma_cells: float, engine: str | None = None) -> np.ndarray:
    """Gaussian smoothing that ignores NaN cells (normalized convolution). The Rust core's gives
    the same values as scipy's."""
    if use_rust(engine):
        import roadsearch_core
        return roadsearch_core.terrain_smooth(np.ascontiguousarray(z, dtype=np.float64), float(sigma_cells))
    valid = ~np.isnan(z)
    zf = np.where(valid, z, 0.0)
    num = ndimage.gaussian_filter(zf, sigma_cells, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(float), sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def gradient(z: np.ndarray, grid: Grid, engine: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """dz/dx (east) and dz/dy (north) by central differences."""
    if use_rust(engine):
        import roadsearch_core
        return roadsearch_core.terrain_gradient(np.ascontiguousarray(z, dtype=np.float64), grid.transform.a,
                                                grid.transform.e)
    d_row, d_col = np.gradient(z)
    return d_col / grid.transform.a, d_row / grid.transform.e


def slope_mask(slope: np.ndarray, grid: Grid, p: Params,
               engine: str | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Cells too steep for any road at grade <= g_max, dilated by the catch distance.

    Returns (raw mask, dilated mask, dilation radius in m). NaN slope (nodata)
    counts as excluded.
    """
    if use_rust(engine):
        import roadsearch_core
        radius = max_catch_at_mask(p)
        raw, dilated = roadsearch_core.terrain_slope_mask(np.ascontiguousarray(slope, dtype=np.float64), p.g_max,
                                                          p.s_mask, radius, abs(grid.transform.e), grid.transform.a)
        return raw, dilated, radius
    with np.errstate(invalid="ignore"):
        side_min = np.sqrt(np.maximum(slope**2 - p.g_max**2, 0.0))
        raw = (side_min >= p.s_mask) | np.isnan(slope)
    radius = max_catch_at_mask(p)
    if raw.any():
        dist = ndimage.distance_transform_edt(~raw, sampling=(abs(grid.transform.e), grid.transform.a))
        dilated = dist <= radius
    else:
        dilated = raw.copy()
    return raw, dilated, radius


def reachable_from(open_cells: np.ndarray, rc: tuple[int, int], engine: str | None = None) -> np.ndarray:
    if use_rust(engine):
        import roadsearch_core
        return roadsearch_core.terrain_reachable(np.ascontiguousarray(open_cells, dtype=bool), int(rc[0]), int(rc[1]))
    labels, _ = ndimage.label(open_cells, structure=FOUR)
    lab = labels[rc]
    return (labels == lab) if lab else np.zeros_like(open_cells)


def preprocess(
    grid: Grid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    p: Params,
    exclude: list[str | Path] = (),
    out_dir: str | Path | None = None,
    engine: str | None = None,
) -> Terrain:
    """Build every preprocessing raster; write them to out_dir if given.

    Raises EndpointError if an endpoint is off-grid or excluded, ExclusionError
    for bad polygon files, and ConnectivityError if the goal is unreachable
    (after writing the reachable-region raster).
    """
    out = Path(out_dir) if out_dir is not None else None
    endpoints = {"start": start_xy, "goal": goal_xy}
    rcs = {}
    for name, (x, y) in endpoints.items():
        rc = grid.rowcol(x, y)
        if not grid.contains(*rc):
            raise EndpointError(f"{name} ({x:.2f}, {y:.2f}) is outside the working grid")
        if np.isnan(grid.z[rc]):
            raise EndpointError(f"{name} ({x:.2f}, {y:.2f}) is on a nodata cell")
        rcs[name] = rc

    sigma_cells = p.sigma_m / grid.res
    zs = smooth_dem(grid.z, sigma_cells, engine)
    dzdx, dzdy = gradient(zs, grid, engine)
    S = np.hypot(dzdx, dzdy)
    soil = np.where(np.isnan(S), np.nan, soil_depth(np.nan_to_num(S, nan=np.inf), p))
    area = contributing_area_ha(grid, engine)
    channel = np.nan_to_num(area) >= p.A_min

    raw_slope, slope_ex, radius = slope_mask(S, grid, p, engine)

    zones = [load_zone_file(f, grid, p.excl_buffer) for f in exclude]
    check_endpoints(zones, grid, endpoints)
    manual = np.zeros(grid.shape, dtype=bool)
    for z in zones:
        manual |= z.raster
    code = combine_codes(slope_ex, manual)

    info = {
        "smooth_sigma_m": p.sigma_m,
        "smooth_sigma_cells": sigma_cells,
        "slope_mask_threshold_sidehill": p.s_mask,
        "slope_mask_dilation_m": radius,
        "slope_excluded_raw_cells": int(raw_slope.sum()),
        "slope_excluded_cells": int(slope_ex.sum()),
        "manual_excluded_cells": int(manual.sum()),
        "nodata_cells": int(np.isnan(grid.z).sum()),
        "channel_cells": int(channel.sum()),
        "exclusion_files": [z.record(grid) for z in zones],
        "start_rc": list(rcs["start"]),
        "goal_rc": list(rcs["goal"]),
    }

    for name, rc in rcs.items():
        if slope_ex[rc]:
            reason = "too steep" if raw_slope[rc] else f"within {radius:.1f} m of too-steep ground"
            raise EndpointError(
                f"{name} {endpoints[name]} is slope-excluded ({reason}; terrain slope "
                f"{S[rc]:.3f}, threshold sidehill {p.s_mask:.3f} at grade {p.g_max})"
            )

    reachable = reachable_from(code == 0, rcs["start"], engine)
    terrain = Terrain(grid, zs, dzdx, dzdy, S, soil, area, channel, slope_ex, manual, code,
                      reachable, rcs["start"], rcs["goal"], zones, info)

    if out is not None:
        write_rasters(terrain, out)

    if not reachable[rcs["goal"]]:
        without_manual = reachable_from(~slope_ex, rcs["start"], engine)[rcs["goal"]]
        info["goal_reachable_ignoring_manual"] = bool(without_manual)
        where = f" Reachable region written to {out / 'reachable.tif'}." if out is not None else ""
        cause = (
            "The manual exclusion zones cause the disconnection: the goal is reachable "
            "when they are ignored."
            if without_manual and zones else
            "The terrain (slope mask) causes the disconnection"
            + (": the goal is unreachable even with manual zones ignored." if zones else ".")
        )
        raise ConnectivityError(
            f"goal {goal_xy} is not reachable from start {start_xy} through open cells. {cause}{where}"
        )
    return terrain


def write_rasters(t: Terrain, out: Path) -> None:
    g = t.grid
    write_raster(out / "dem_working.tif", g.z, g)
    write_raster(out / "dem_smoothed.tif", t.z_smooth, g)
    write_raster(out / "dzdx.tif", t.dzdx, g)
    write_raster(out / "dzdy.tif", t.dzdy, g)
    write_raster(out / "slope.tif", t.slope, g)
    write_raster(out / "soil_depth.tif", t.soil, g)
    write_raster(out / "contributing_area_ha.tif", t.area_ha, g)
    write_raster(out / "exclusion_code.tif", t.code, g, nodata=255)
    write_raster(out / "reachable.tif", t.reachable, g, nodata=255)
