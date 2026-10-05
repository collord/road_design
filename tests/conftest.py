import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS

from roadsearch.params import load_params
from roadsearch.terrain import Grid

UTM = CRS.from_epsg(32611)
X0, Y0 = 500000.0, 4950000.0  # top-left corner of synthetic grids


def make_grid(z, res=5.0, crs=UTM) -> Grid:
    return Grid(np.asarray(z, dtype=float), Affine(res, 0, X0, 0, -res, Y0), crs)


def write_dem(path, z, transform, crs, nodata=-9999.0):
    z = np.asarray(z, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs=crs.to_wkt() if hasattr(crs, "to_wkt") else crs,
                       transform=transform, nodata=nodata) as dst:
        dst.write(z, 1)
    return path


def cell_xy(grid: Grid, r, c):
    return grid.xy(r, c)


@pytest.fixture
def params():
    return load_params()
