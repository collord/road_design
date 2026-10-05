"""D8 flow accumulation via pysheds (brief section 3.4)."""

from __future__ import annotations

import warnings

import numpy as np

# pysheds 0.5 calls np.in1d, which NumPy 2.4 removed; np.isin is the
# documented replacement with the same semantics for 1-D input.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import pyproj  # noqa: E402
from pysheds.grid import Grid as PyshedsGrid  # noqa: E402
from pysheds.sview import Raster, ViewFinder  # noqa: E402

from .terrain import Grid  # noqa: E402


def contributing_area_ha(grid: Grid) -> np.ndarray:
    """Upslope contributing area in hectares, including the cell itself.

    Depressions are filled and flats resolved before D8 routing, so every
    cell drains to the grid edge or a nodata boundary. NaN where z is NaN.
    """
    z = grid.z.astype(np.float64)
    nodata = np.isnan(z)
    vf = ViewFinder(affine=grid.transform, shape=z.shape, nodata=np.nan,
                    crs=pyproj.Proj(grid.crs.to_wkt()))
    dem = Raster(z, viewfinder=vf)
    pg = PyshedsGrid(viewfinder=vf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filled = pg.fill_pits(dem)
        filled = pg.fill_depressions(filled)
        inflated = pg.resolve_flats(filled)
        fdir = pg.flowdir(inflated, nodata_out=np.int64(0))
        acc = pg.accumulation(fdir, nodata_out=np.nan)
    area = np.asarray(acc, dtype=np.float64) * grid.cell_area / 1e4
    area[nodata] = np.nan
    return area
