"""Roughness from the native DEM (roughness.py): the engines agree, a plane has none, and it thins
the soil, read at the DEM's own pixels."""

import numpy as np
import pytest
from rasterio.transform import from_origin

from roadsearch.params import load_params
from roadsearch.preprocess import preprocess
from roadsearch.roughness import rough_k, roughness
from roadsearch.search import build_cost_table
from roadsearch.terrain import load_grid

from .conftest import UTM, make_grid, write_dem
from .test_search import random_terrain


def rough_dem(n=120, seed=3):
    """Smooth hills with a blocky patch (rubble: meter-scale bumps) in one quarter."""
    z = random_terrain(seed, n=n, relief=1.0)
    rng = np.random.default_rng(seed)
    z[: n // 2, : n // 2] += rng.uniform(-1.0, 1.0, (n // 2, n // 2))
    return z


def test_rust_matches_scipy():
    pytest.importorskip("roadsearch_core")
    z = rough_dem()
    z[40:44, 70:75] = np.nan
    a, b = roughness(z, engine="numba"), roughness(z, engine="rust")
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-9)


def test_a_plane_has_none_and_rubble_has_some():
    r, c = np.mgrid[0:60, 0:60]
    plane = roughness(100.0 + 0.3 * r - 0.2 * c)
    assert np.abs(plane[10:-10, 10:-10]).max() < 1e-9
    rho = roughness(rough_dem())
    assert np.nanmedian(rho[5:55, 5:55]) > 0.3 > 10 * np.nanmedian(rho[65:115, 65:115])


def test_the_soil_thins_as_with_extra_slope():
    p = load_params()
    g = make_grid(random_terrain(4, n=60, relief=6.0), 5.0)
    t0 = preprocess(g, g.xy(5, 5), g.xy(54, 54), p)
    g.rough = np.full(g.z.shape, 0.5 * p.rough_r0)
    t = preprocess(g, g.xy(5, 5), g.xy(54, 54), p)
    k = rough_k(p)
    assert k == pytest.approx(p.S0 / p.rough_r0)
    # Where soil remains, rho = r0/2 thins it by e^-1/2.
    both = (t.soil > 0) & (t0.soil > 0)
    assert both.any()
    assert np.allclose(t.soil[both], t0.soil[both] * np.exp(-0.5))
    # And none at all once the effective slope passes c_soil.
    assert np.array_equal(t.soil == 0, (np.hypot(t0.dzdx, t0.dzdy) + k * 0.5 * p.rough_r0 >= p.c_soil) | (t0.soil == 0))


def test_the_cost_table_runs_on_to_c_soil():
    p0, p = load_params(overrides={"rough_r0": 0.0}), load_params(overrides={"rough_r0": 0.5})
    a, b = build_cost_table(p0), build_cost_table(p)
    assert (a.ds, a.dg) == (b.ds, b.dg)
    assert np.array_equal(b.table[:, : a.table.shape[1]], a.table)
    assert (b.table.shape[1] - 2) * b.dg >= p.c_soil > (b.table.shape[1] - 3) * b.dg


def test_load_grid_reads_it_at_the_native_pixels(tmp_path):
    """A 1 m DEM loaded onto 5 m cells: the roughness comes from the 1 m pixels, so the rubble
    shows though the 5 m DEM has averaged it away."""
    z = rough_dem(n=300, seed=5)
    path = write_dem(tmp_path / "dem.tif", z, from_origin(500_000.0, 5_000_300.0, 1.0, 1.0), UTM)
    grid, info = load_grid(path, load_params(), resolution=5.0)
    assert grid.rough is not None and grid.rough.shape == grid.z.shape
    assert info["roughness"]["pixel_size"] == pytest.approx(1.0)
    q = grid.z.shape[0] // 2
    assert np.nanmedian(grid.rough[2: q - 2, 2: q - 2]) > 0.3 > 10 * np.nanmedian(grid.rough[q + 2: -2, q + 2: -2])
    assert np.nanmedian(roughness(grid.z)[2: q - 2, 2: q - 2]) < 0.5 * np.nanmedian(grid.rough[2: q - 2, 2: q - 2])


def test_coarse_dems_have_none(tmp_path):
    """At 10 m pixels a few pixels' relief is landform: no roughness, and the reason recorded."""
    z = rough_dem(n=100, seed=5)
    path = write_dem(tmp_path / "dem.tif", z, from_origin(500_000.0, 5_001_000.0, 10.0, 10.0), UTM)
    grid, info = load_grid(path, load_params(), resolution=10.0)
    assert grid.rough is None and "landform" in info["roughness"]["skipped"]


def test_bands_match_one_read(tmp_path, monkeypatch):
    """Read in bands (as for large lidar areas), the roughness is that of one whole read."""
    import roadsearch.roughness as rmod
    z = rough_dem(n=300, seed=6)
    path = write_dem(tmp_path / "dem.tif", z, from_origin(500_000.0, 5_000_300.0, 1.0, 1.0), UTM)
    whole, _ = load_grid(path, load_params(), resolution=5.0)
    monkeypatch.setattr(rmod, "ROUGH_MAX_PX", 9_000)
    banded, info = load_grid(path, load_params(), resolution=5.0)
    assert info["roughness"]["bands"] > 5
    assert np.allclose(banded.rough, whole.rough, rtol=0, atol=1e-9, equal_nan=True)
