"""DEM inspection, reprojection to a metric working grid, and raster I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from .params import Z_UNIT_SCALE, Params

NICE_RESOLUTIONS = (1, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100)


@dataclass
class Grid:
    """A north-up raster on a projected metric CRS. z is float64 with NaN nodata."""

    z: np.ndarray
    transform: Affine
    crs: CRS
    # Roughness (m, roughness.py) on the same cells, from the DEM's native pixels; None: unknown.
    rough: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def res(self) -> float:
        return abs(self.transform.a)

    @property
    def cell_area(self) -> float:
        return abs(self.transform.a * self.transform.e)

    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        col, row = ~self.transform @ (x, y)
        return int(math.floor(row)), int(math.floor(col))

    def xy(self, row: float, col: float) -> tuple[float, float]:
        """Coordinates of a cell center (fractional indices allowed)."""
        return self.transform @ (col + 0.5, row + 0.5)

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]


def _is_metric(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    units = {ax.unit_name for ax in crs.axis_info[:2]}
    # PROJ spells the unit "metre"; keep that spelling here (it is data, not prose).
    return units <= {"metre", "meter"}


def utm_crs_for(crs: CRS, bounds: tuple[float, float, float, float]) -> CRS:
    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", *bounds)
    infos = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(lon0, lat0, lon1, lat1),
    )
    if not infos:
        raise ValueError("no UTM zone found for the DEM extent")
    # Prefer the zone containing the center of the DEM.
    lonc, latc = 0.5 * (lon0 + lon1), 0.5 * (lat0 + lat1)
    zone = int((lonc + 180) // 6) + 1
    epsg = (32600 if latc >= 0 else 32700) + zone
    codes = {int(i.code) for i in infos}
    return CRS.from_epsg(epsg if epsg in codes else int(infos[0].code))


def working_crs(src_crs: CRS, bounds) -> CRS:
    return src_crs if _is_metric(src_crs) else utm_crs_for(src_crs, bounds)


def propose_resolution(native_res: float, rows: int, cols: int, p: Params) -> tuple[float, int]:
    """Native resolution if it fits, else the smallest 'nice' coarser resolution
    keeping rows*cols*N_theta under max_states."""
    area = rows * cols * native_res**2
    if rows * cols * p.N_theta <= p.max_states:
        return float(native_res), rows * cols * p.N_theta
    for r in NICE_RESOLUTIONS:
        if r + 1e-9 < native_res:
            continue
        states = int(area / r**2) * p.N_theta
        if states <= p.max_states:
            return float(r), states
    r = native_res * math.sqrt(rows * cols * p.N_theta / p.max_states)
    return r, int(area / r**2) * p.N_theta


def inspect_dem(path: str | Path, p: Params) -> dict:
    """Report CRS, resolution, extent, nodata, units and elevation range."""
    with rasterio.open(path) as ds:
        crs = CRS.from_user_input(ds.crs) if ds.crs else None
        if crs is None:
            raise ValueError(f"{path} has no CRS")
        band = ds.read(1, masked=True)
        valid = band.compressed().astype(float)
        info = {
            "path": str(Path(path).resolve()),
            "crs_name": crs.name,
            "crs_epsg": crs.to_epsg(),
            "crs_is_geographic": crs.is_geographic,
            "horizontal_units": [ax.unit_name for ax in crs.axis_info[:2]],
            "width": ds.width,
            "height": ds.height,
            "res": [abs(ds.res[0]), abs(ds.res[1])],
            "bounds": list(ds.bounds),
            "nodata": ds.nodata,
            "dtype": ds.dtypes[0],
            "band_units": ds.units[0] or None,
            "vertical_crs": crs.sub_crs_list[1].name if crs.is_compound else None,
            "nodata_fraction": float(band.mask.sum() / band.size) if np.ma.is_masked(band) else 0.0,
            "z_min": float(valid.min()),
            "z_max": float(valid.max()),
            "z_mean": float(valid.mean()),
        }
        wcrs = working_crs(crs, ds.bounds)
        if wcrs == crs:
            native = abs(ds.res[0])
            w_rows, w_cols = ds.height, ds.width
        else:
            tr, w_cols, w_rows = calculate_default_transform(crs, wcrs, ds.width, ds.height, *ds.bounds)
            native = abs(tr.a)
    res, states = propose_resolution(native, w_rows, w_cols, p)
    info.update({
        "z_units": p.z_units,
        "working_crs_name": wcrs.name,
        "working_crs_epsg": wcrs.to_epsg(),
        "reprojection_needed": wcrs != crs,
        "native_res_m": native,
        "native_states": w_rows * w_cols * p.N_theta,
        "proposed_res_m": res,
        "proposed_states": states,
    })
    return info


def format_inspection(info: dict) -> str:
    b = info["bounds"]
    lines = [
        f"DEM: {info['path']}",
        f"  CRS: {info['crs_name']} (EPSG:{info['crs_epsg']}), geographic={info['crs_is_geographic']}",
        f"  size: {info['width']} x {info['height']} cells, res {info['res'][0]:.4g} x {info['res'][1]:.4g}",
        f"  extent: x {b[0]:.3f} .. {b[2]:.3f}, y {b[1]:.3f} .. {b[3]:.3f}",
        f"  nodata: {info['nodata']} ({100 * info['nodata_fraction']:.2f}% of cells), dtype {info['dtype']}",
        f"  horizontal units (from CRS): {info['horizontal_units'][0]}"
        + (" -> reprojected to meters" if info["reprojection_needed"] else ""),
        f"  vertical units (z_units parameter): {info['z_units']}"
        f"  [file: band units={info['band_units'] or 'unset'}, vertical CRS={info['vertical_crs'] or 'none'}]",
        f"  elevation: {info['z_min']:.2f} .. {info['z_max']:.2f} (mean {info['z_mean']:.2f})",
        f"  working CRS: {info['working_crs_name']} (EPSG:{info['working_crs_epsg']})"
        + (" - reprojection required" if info["reprojection_needed"] else ""),
        f"  native res in working CRS: {info['native_res_m']:.3f} m -> {info['native_states']:,} states",
        f"  proposed working res: {info['proposed_res_m']:g} m -> {info['proposed_states']:,} states",
    ]
    return "\n".join(lines)


def load_grid(path: str | Path, p: Params, resolution: float | None = None,
              bounds: tuple[float, float, float, float] | None = None) -> tuple[Grid, dict]:
    """Read the DEM onto the working grid (metric CRS, square cells, meters).

    bounds (xmin, ymin, xmax, ymax) in the working CRS clips the grid; the
    window is snapped outward to whole cells of the working resolution.
    Upsampling uses cubic convolution (smooth slopes for a fine lattice),
    downsampling averages, near-native resolution is bilinear.
    Returns the grid and a dict describing what was done.
    """
    z_scale = Z_UNIT_SCALE[p.z_units]
    with rasterio.open(path) as ds:
        src_crs = CRS.from_user_input(ds.crs)
        dst_crs = working_crs(src_crs, ds.bounds)
        same_grid = (
            bounds is None
            and dst_crs == src_crs
            and abs(abs(ds.res[0]) - abs(ds.res[1])) < 1e-9
            and (resolution is None or abs(resolution - abs(ds.res[0])) < 1e-9)
            and ds.transform.b == 0 and ds.transform.d == 0 and ds.transform.e < 0
        )
        if same_grid:
            band = ds.read(1, masked=True).astype(np.float64)
            z = band.filled(np.nan)
            transform = ds.transform
            method = "none"
        else:
            native, nw, nh = calculate_default_transform(src_crs, dst_crs, ds.width, ds.height, *ds.bounds)
            src_res = abs(native.a)
            if resolution is None:
                resolution = src_res
            if bounds is None:
                transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, ds.width, ds.height, *ds.bounds, resolution=resolution
                )
            else:
                xmin, ymin, xmax, ymax = bounds
                x0 = math.floor(xmin / resolution) * resolution
                y1 = math.ceil(ymax / resolution) * resolution
                width = int(math.ceil((xmax - x0) / resolution))
                height = int(math.ceil((y1 - ymin) / resolution))
                transform = Affine(resolution, 0, x0, 0, -resolution, y1)
            if resolution > 1.5 * src_res:
                resampling = Resampling.average
            elif resolution < src_res / 1.5:
                resampling = Resampling.cubic
            else:
                resampling = Resampling.bilinear
            z = np.full((height, width), np.nan, dtype=np.float64)
            reproject(
                source=rasterio.band(ds, 1), destination=z,
                src_transform=ds.transform, src_crs=src_crs, src_nodata=ds.nodata,
                dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan,
                resampling=resampling,
            )
            method = resampling.name
        rough, rough_info = None, None
        if p.rough_r0 > 0:
            from .roughness import native_roughness
            native_m = abs(ds.res[0]) if same_grid else src_res
            rough, rough_info = native_roughness(ds, z_scale, transform, z.shape, dst_crs, native_m,
                                                 p.rough_max_res)
    z = z * z_scale
    grid = Grid(z=z, transform=transform, crs=dst_crs, rough=rough)
    return grid, {
        "roughness": rough_info,
        "working_crs": dst_crs.to_string(),
        "working_res_m": grid.res,
        "working_shape": list(grid.shape),
        "working_bounds": list(rasterio.transform.array_bounds(*grid.shape, transform)),
        "clip_bounds": list(bounds) if bounds is not None else None,
        "resampling": method,
        "z_scale_to_m": z_scale,
    }


def parse_point(text: str, src_crs: CRS, dst_crs: CRS) -> tuple[float, float]:
    """Parse 'x,y' in src_crs (use EPSG:4326 for lon,lat) into dst_crs."""
    try:
        a, b = (float(v) for v in text.replace(" ", "").split(","))
    except ValueError as exc:
        raise ValueError(f"expected 'x,y', got {text!r}") from exc
    if src_crs == dst_crs:
        return a, b
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform(a, b)


def write_raster(path: str | Path, arr: np.ndarray, grid: Grid, nodata=None) -> None:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    if arr.dtype.kind == "f":
        arr = arr.astype(np.float32)
        nodata = np.nan if nodata is None else nodata
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
        "dtype": arr.dtype.name, "crs": grid.crs.to_wkt(), "transform": grid.transform,
        "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
