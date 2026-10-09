"""Terrain roughness from the DEM at its native resolution, for soil depth.

Rough ground (outcrop, talus, blocky rock) has little soil, whatever its slope. Roughness is the
detrended relief: the DEM minus a light smoothing of itself (sigma ROUGH_SIGMA_PX native pixels),
as an RMS over each pixel's 3 x 3 neighborhood, in meters. It is computed on the native pixels,
since the working grid has already averaged that detail away, then averaged onto the working grid.

Soil depth then thins with roughness as if it were extra slope: d0 exp(-(S + (S0/rough_r0) rho)/S0)
= d0 exp(-S/S0) exp(-rho/rough_r0), none once that effective slope reaches c_soil; so rough ground
is excavated at the rock price.

Roughness means rock most clearly at lidar-like resolution: DEMs with pixels coarser than
rough_max_res (default 3 m) are left without it, since their few-pixel relief is largely the
landform's curvature (meters of it on mountain DEMs of 10 m).
"""

from __future__ import annotations

import math

import numpy as np
from affine import Affine
from scipy import ndimage

from .engine import use_rust
from .params import Params

ROUGH_SIGMA_PX = 2.0
# Native pixels read at once for roughness (about 1 GB of intermediates): larger areas go in bands.
ROUGH_MAX_PX = 20_000_000


def rough_k(p: Params) -> float:
    """Slope added to the soil-thinning slope per meter of roughness (S0 / rough_r0); 0: off."""
    return p.S0 / p.rough_r0 if p.rough_r0 > 0 else 0.0


def roughness(z: np.ndarray, sigma_px: float = ROUGH_SIGMA_PX, engine: str | None = None) -> np.ndarray:
    """Detrended relief (m): RMS over each pixel's 3 x 3 neighborhood (within the raster) of z minus
    its NaN-aware Gaussian smoothing. NaN where z is NaN."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    if use_rust(engine):
        import roadsearch_core
        return roadsearch_core.terrain_roughness(z, float(sigma_px))
    from .preprocess import smooth_dem

    valid = ~np.isnan(z)
    resid = np.where(valid, z - smooth_dem(z, sigma_px, "numba"), 0.0)
    box = np.ones((3, 3))
    total = ndimage.convolve(resid * resid, box, mode="constant", cval=0.0)
    count = ndimage.convolve(valid.astype(np.float64), box, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(total / count)
    out[~valid] = np.nan
    return out


def native_roughness(ds, z_scale: float, transform, shape: tuple[int, int], dst_crs, native_m: float,
                     max_res: float = 3.0) -> tuple[np.ndarray | None, dict]:
    """Roughness on the working grid (transform, shape, dst_crs) from the open DEM dataset `ds` at
    its native pixels (native_m m), averaged onto the working cells: read in bands of the working
    grid of at most ROUGH_MAX_PX native pixels each, so memory stays bounded at any resolution.
    None (with the reason) when the pixels are coarser than max_res: the detrended relief would
    then be mostly landform, not rock. Returns the raster and a record of what was read."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    from . import memguard

    if native_m > max_res:
        return None, {"skipped": f"the DEM's {native_m:.1f} m pixels are coarser than rough_max_res {max_res:g} m: "
                                 f"their detrended relief is mostly landform, not rock; roughness is ignored",
                      "pixel_size": native_m}
    h, w = shape
    res = abs(transform.a)
    per_row = max(1.0, w * (res / native_m) ** 2)
    band = max(1, int(ROUGH_MAX_PX // per_row))
    pad = 4 * ROUGH_SIGMA_PX + 2
    rho = np.full(shape, np.nan, dtype=np.float64)
    pixels, bands = 0, 0
    for i0 in range(0, h, band):
        bh = min(band, h - i0)
        bt = transform @ Affine.translation(0, i0)
        xmin, ymax = bt @ (0, 0)
        xmax, ymin = bt @ (w, bh)
        win = from_bounds(*transform_bounds(dst_crs, ds.crs, xmin, ymin, xmax, ymax), transform=ds.transform)
        win = Window(math.floor(win.col_off - pad), math.floor(win.row_off - pad),
                     math.ceil(win.width + 2 * pad), math.ceil(win.height + 2 * pad))
        try:
            win = win.intersection(Window(0, 0, ds.width, ds.height))
        except Exception:          # the band lies outside the DEM
            continue
        n = int(win.width) * int(win.height)
        memguard.require(n * 8 * 6 / memguard.GB, "roughness", memguard.FLOOR_GB)
        z = ds.read(1, window=win, masked=True).astype(np.float64).filled(np.nan) * z_scale
        reproject(source=roughness(z), destination=rho[i0:i0 + bh], src_transform=ds.window_transform(win),
                  src_crs=ds.crs, src_nodata=np.nan, dst_transform=bt, dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=Resampling.average)
        pixels += n
        bands += 1
    return rho, {"pixels": pixels, "bands": bands, "pixel_size": native_m,
                 "median_m": float(np.nanmedian(rho)) if np.isfinite(rho).any() else None}
