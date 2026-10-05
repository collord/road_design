"""Preprocessing: params, DEM loading, derived rasters, masks, connectivity.

The search-level parts of test 8 (detours, hairpin footprints, crossings)
come with the search code.
"""

import json
import warnings

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS
from shapely.geometry import Point, Polygon, box

from roadsearch.cli import main
from roadsearch.exclusion import BOTH, MANUAL, OPEN, SLOPE, ExclusionError
from roadsearch.params import load_params
from roadsearch.preprocess import ConnectivityError, EndpointError, preprocess, smooth_dem
from roadsearch.section import max_catch_at_mask
from roadsearch.terrain import inspect_dem, load_grid

from .conftest import UTM, X0, Y0, make_grid, write_dem

N = 200
RES = 5.0


def plane(gx=0.05, gy=0.0):
    rows, cols = np.mgrid[0:N, 0:N]
    return gx * cols * RES - gy * rows * RES + 1000.0


def with_wall(z, rows, cols, height=100.0):
    z = z.copy()
    z[rows, cols] += height
    return z


def xy(grid, r, c):
    return grid.xy(r, c)


# --- params ------------------------------------------------------------------

def test_params_overrides(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text("g_max = 0.12\nculvert_costs = [1, 2, 3]\n")
    p = load_params(f, {"w": "6", "N_theta": "8", "culvert_area_breaks": "5,20"})
    assert (p.g_max, p.w, p.N_theta) == (0.12, 6.0, 8)
    assert p.culvert_costs == (1.0, 2.0, 3.0) and p.culvert_area_breaks == (5.0, 20.0)
    assert p.culvert_cost(4.9) == 1.0 and p.culvert_cost(5.0) == 2.0 and p.culvert_cost(99) == 3.0
    with pytest.raises(ValueError, match="unknown"):
        load_params(overrides={"bogus": 1})
    with pytest.raises(ValueError, match="integer"):
        load_params(overrides={"N_theta": "8.5"})


def test_sigma_defaults_to_road_width(params):
    assert params.smooth_sigma == 0 and params.sigma_m == params.w
    assert load_params(overrides={"smooth_sigma": 3}).sigma_m == 3


# --- DEM loading ---------------------------------------------------------------

def test_load_metric_dem_unchanged(tmp_path, params):
    z = plane()
    path = write_dem(tmp_path / "dem.tif", z, Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    grid, info = load_grid(path, params)
    assert info["resampling"] == "none" and grid.crs == UTM and grid.res == RES
    np.testing.assert_allclose(grid.z, z.astype(np.float32))


def test_geographic_dem_reprojected_to_utm(tmp_path, params):
    z = np.add.outer(np.arange(100.0), np.arange(120.0)) + 1500
    tr = Affine(0.0005, 0, -115.6, 0, -0.0005, 44.6)
    path = write_dem(tmp_path / "geo.tif", z, tr, CRS.from_epsg(4326))
    info = inspect_dem(path, params)
    assert info["crs_is_geographic"] and info["reprojection_needed"]
    assert info["working_crs_epsg"] == 32611
    grid, ginfo = load_grid(path, params, resolution=50.0)
    assert grid.crs.to_epsg() == 32611 and grid.res == 50.0
    assert np.nanmin(grid.z) >= 1500 - 1e-3 and np.nanmax(grid.z) <= 1500 + 218 + 1e-3


def test_vertical_units_feet(tmp_path):
    p = load_params(overrides={"z_units": "ft"})
    path = write_dem(tmp_path / "dem.tif", np.full((10, 10), 1000.0), Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    grid, _ = load_grid(path, p)
    np.testing.assert_allclose(grid.z, 304.8)


def test_resolution_proposal_respects_state_cap(tmp_path):
    p = load_params(overrides={"max_states": 1e5})
    path = write_dem(tmp_path / "dem.tif", plane(), Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    info = inspect_dem(path, p)
    assert info["native_states"] == N * N * p.N_theta
    assert info["proposed_res_m"] > RES and info["proposed_states"] <= 1e5


# --- derived rasters -----------------------------------------------------------

def test_slope_and_soil_on_plane(params):
    g = make_grid(plane(0.3, 0.4))
    t = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params)
    inner = (slice(5, -5), slice(5, -5))
    np.testing.assert_allclose(t.dzdx[inner], 0.3, atol=1e-9)
    np.testing.assert_allclose(t.dzdy[inner], 0.4, atol=1e-9)  # z rises northward
    np.testing.assert_allclose(t.slope[inner], 0.5, atol=1e-9)
    np.testing.assert_allclose(t.soil[inner], params.d0 * np.exp(-0.5 / params.S0), rtol=1e-9)
    assert (t.code == OPEN).all()


def test_no_soil_where_slope_exceeds_soil_cut_slope(params):
    g = make_grid(plane(1.2))
    t = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params)
    assert np.all(t.soil[5:-5, 5:-5] == 0)


def test_smoothing_ignores_nodata():
    z = np.full((20, 20), 100.0)
    z[5, 5] = np.nan
    s = smooth_dem(z, 2.0)
    assert np.isnan(s[5, 5])
    np.testing.assert_allclose(s[~np.isnan(z)], 100.0)


def test_contributing_area_and_channels(params):
    rows, cols = np.mgrid[0:N, 0:N]
    z = 1000 - 0.05 * rows * RES + 0.1 * np.abs(cols - 100) * RES  # V-valley draining south
    g = make_grid(z)
    t = preprocess(g, xy(g, 50, 20), xy(g, 50, 180), params)
    # The outlet at the bottom of the valley collects close to the whole grid.
    assert t.area_ha[N - 1, 100] == pytest.approx(N * N * RES * RES / 1e4, rel=0.05)
    assert t.channel[N - 1, 100] and not t.channel[10, 10]
    # The channel runs down the valley axis.
    first = np.argmax(t.channel[:, 100])
    assert t.channel[first:, 100].all()


# --- slope mask ----------------------------------------------------------------

def test_cliff_band_masked_and_dilated(params):
    z = with_wall(plane(), slice(98, 103), slice(0, 150))
    g = make_grid(z)
    t = preprocess(g, xy(g, 20, 20), xy(g, 180, 20), params)
    radius = max_catch_at_mask(params)
    assert t.info["slope_mask_dilation_m"] == pytest.approx(radius)
    # The wall's steep faces (rows 97-98 and 102-103 after smoothing) are
    # excluded, then dilated by exactly radius/RES cells.
    k = int(radius // RES)
    assert (t.code[97 - k:104 + k, :150] == SLOPE).all()
    assert (t.code[: 97 - k, :] == OPEN).all() and (t.code[104 + k:, :] == OPEN).all()
    assert (t.code[:, 151 + k:] == OPEN).all()
    assert t.reachable[t.goal_rc]  # around the end of the band


def test_enclosing_band_fails_connectivity(tmp_path, params):
    z = plane()
    z[60:140, 60:64] += 100
    z[60:140, 136:140] += 100
    z[60:64, 60:140] += 100
    z[136:140, 60:140] += 100
    g = make_grid(z)
    with pytest.raises(ConnectivityError, match="terrain") as err:
        preprocess(g, xy(g, 100, 100), xy(g, 10, 10), params, out_dir=tmp_path)
    assert "reachable.tif" in str(err.value)
    with rasterio.open(tmp_path / "reachable.tif") as ds:
        reach = ds.read(1).astype(bool)
    assert reach[100, 100] and not reach[10, 10]


def test_endpoint_on_steep_ground_rejected(params):
    z = with_wall(plane(), slice(98, 103), slice(0, 150))
    g = make_grid(z)
    with pytest.raises(EndpointError, match="slope-excluded"):
        preprocess(g, xy(g, 100, 50), xy(g, 180, 20), params)


# --- manual exclusion zones ----------------------------------------------------

def zone_poly(g, r0, r1, c0, c1):
    """Rectangle through cell centers, so no edge lies on a cell boundary."""
    (xa, ya), (xb, yb) = g.xy(r0, c0), g.xy(r1, c1)
    return box(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))


def save(tmp_path, fname, geoms, crs, **cols):
    path = tmp_path / fname
    gpd.GeoDataFrame(cols or None, geometry=list(geoms), crs=crs).to_file(path)
    return str(path)


def test_manual_zone_rasterized_all_touched(tmp_path, params):
    g = make_grid(plane())
    poly = zone_poly(g, 40, 60, 20, 80)
    f = save(tmp_path, "zone.shp", [poly], UTM, name=["lake"])
    t = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [f])
    assert (t.code[40:61, 20:81] == MANUAL).all()
    assert t.code.sum() == MANUAL * 21 * 61
    rec = t.info["exclusion_files"][0]
    assert rec["feature_count"] == 1 and rec["excluded_cells"] == 21 * 61
    assert rec["polygon_area_m2"] == pytest.approx(poly.area)


def test_manual_zone_in_other_crs_gives_same_raster(tmp_path, params):
    g = make_grid(plane())
    poly = zone_poly(g, 40, 60, 20, 80).segmentize(1.0)
    utm = save(tmp_path, "utm.gpkg", [poly], UTM)
    ll = gpd.GeoSeries([poly], crs=UTM).to_crs(4326)
    geo = save(tmp_path, "ll.gpkg", ll, "EPSG:4326")
    a = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [utm])
    b = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [geo])
    np.testing.assert_array_equal(a.code, b.code)


def test_manual_zone_buffer_but_no_catch_dilation(tmp_path):
    g = make_grid(plane())
    f = save(tmp_path, "z.gpkg", [zone_poly(g, 40, 60, 20, 80)], UTM)
    p0 = load_params()
    p1 = load_params(overrides={"excl_buffer": 10.0})
    t0 = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), p0, [f])
    t1 = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), p1, [f])
    assert (t0.code == MANUAL).sum() == 21 * 61
    # 10 m = 2 cells each way; buffered corners are rounded, so check the edge midspans.
    assert (t1.code[38:63, 25:75] == MANUAL).all() and (t1.code[45:55, 18:83] == MANUAL).all()
    assert not (t1.code[36, 50] or t1.code[50, 16])


def test_both_code(tmp_path, params):
    z = with_wall(plane(), slice(98, 103), slice(0, 150))
    g = make_grid(z)
    f = save(tmp_path, "z.gpkg", [zone_poly(g, 90, 110, 30, 60)], UTM)
    t = preprocess(g, xy(g, 20, 20), xy(g, 180, 20), params, [f])
    assert t.code[100, 40] == BOTH
    assert t.code[88, 40] == SLOPE   # inside the dilation, outside the polygon
    assert t.code[100, 170] == OPEN
    assert set(np.unique(t.code)) == {OPEN, SLOPE, BOTH}


def test_endpoint_inside_zone_rejected(tmp_path, params):
    g = make_grid(plane())
    f = save(tmp_path, "z.gpkg", [zone_poly(g, 0, 10, 0, 10), zone_poly(g, 90, 110, 10, 30)], UTM,
             name=["a", "pond"])
    with pytest.raises(ExclusionError, match=r"start .* inside feature 1 \(name='pond'\) of .*z\.gpkg"):
        preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [f])


def test_endpoint_in_touched_cell_rejected(tmp_path, params):
    g = make_grid(plane())
    x, y = g.xy(100, 20)
    f = save(tmp_path, "z.gpkg", [box(x + 1.0, y - 10, x + 20, y + 10)], UTM)  # 1 m east of start
    with pytest.raises(ExclusionError, match="touched"):
        preprocess(g, (x, y), xy(g, 100, 180), params, [f])


def test_zone_without_crs_rejected(tmp_path, params):
    g = make_grid(plane())
    f = save(tmp_path, "nocrs.shp", [zone_poly(g, 40, 60, 20, 80)], None)
    with pytest.raises(ExclusionError, match="no CRS"):
        preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [f])


def test_invalid_and_non_polygon_features(tmp_path, params):
    g = make_grid(plane())
    (xa, ya), (xb, yb) = g.xy(40, 40), g.xy(60, 60)
    bowtie = Polygon([(xa, ya), (xb, yb), (xb, ya), (xa, yb)])
    assert not bowtie.is_valid
    f = save(tmp_path, "mixed.gpkg", [bowtie, Point(g.xy(10, 10))], UTM)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [f])
    msgs = " ".join(str(x.message) for x in w)
    assert "repairing 1 invalid" in msgs and "ignoring 1 non-polygon" in msgs
    assert t.code[50, 45] == MANUAL and t.code[50, 55] == MANUAL  # both bowtie lobes
    assert t.code[10, 10] == OPEN
    assert t.info["exclusion_files"][0]["feature_count"] == 1


def test_disconnection_by_manual_zone_is_attributed(tmp_path, params):
    g = make_grid(plane())
    ring = zone_poly(g, 60, 140, 60, 140).difference(zone_poly(g, 70, 130, 70, 130))
    f = save(tmp_path, "ring.gpkg", [ring], UTM)
    with pytest.raises(ConnectivityError, match="manual exclusion zones cause"):
        preprocess(g, xy(g, 100, 100), xy(g, 10, 10), params, [f], out_dir=tmp_path)


def test_multiple_zone_files(tmp_path, params):
    g = make_grid(plane())
    f1 = save(tmp_path, "a.gpkg", [zone_poly(g, 10, 20, 10, 20)], UTM)
    f2 = save(tmp_path, "b.shp", [zone_poly(g, 150, 160, 150, 160)], UTM)
    t = preprocess(g, xy(g, 100, 20), xy(g, 100, 180), params, [f1, f2])
    assert t.code[15, 15] == MANUAL and t.code[155, 155] == MANUAL
    assert [r["path"] for r in t.info["exclusion_files"]] == [f1, f2]


# --- CLI ---------------------------------------------------------------------------

def test_cli_preprocess_writes_rasters_and_params(tmp_path):
    path = write_dem(tmp_path / "dem.tif", plane(), Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    g = make_grid(plane())
    (sx, sy), (gx, gy) = g.xy(100, 20), g.xy(100, 180)
    args = ["preprocess", "--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}",
            "--out", str(tmp_path / "out"), "--g_max", "0.12"]
    assert main(args) == 0
    runs = list((tmp_path / "out" / "runs").iterdir())
    assert len(runs) == 1
    rec = json.loads((runs[0] / "params.json").read_text())
    assert rec["params"]["g_max"] == 0.12 and rec["run_id"] == runs[0].name
    for name in ["dem_working", "dem_smoothed", "dzdx", "dzdy", "slope", "soil_depth",
                 "contributing_area_ha", "exclusion_code", "reachable"]:
        assert (runs[0] / f"{name}.tif").exists(), name
    # Identical inputs give the same run id.
    assert main(args) == 0
    assert len(list((tmp_path / "out" / "runs").iterdir())) == 1


def test_cli_lonlat_endpoints(tmp_path):
    path = write_dem(tmp_path / "dem.tif", plane(), Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    g = make_grid(plane())
    ll = gpd.GeoSeries([Point(g.xy(100, 20)), Point(g.xy(100, 180))], crs=UTM).to_crs(4326)
    args = ["preprocess", "--dem", str(path), "--endpoint-crs", "EPSG:4326",
            "--start", f"{ll[0].x},{ll[0].y}", "--goal", f"{ll[1].x},{ll[1].y}",
            "--out", str(tmp_path / "out")]
    assert main(args) == 0
    rec = json.loads(next((tmp_path / "out" / "runs").iterdir()).joinpath("params.json").read_text())
    assert rec["preprocess"]["start_rc"] == [100, 20] and rec["preprocess"]["goal_rc"] == [100, 180]
