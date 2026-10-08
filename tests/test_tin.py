"""--clip tin: the coarse stage on a sparse graph over the whole DEM, then the fine search."""

import json

import geopandas as gpd
import pytest

pytest.importorskip("roadsearch_core")

from roadsearch.cli import main  # noqa: E402

from .test_run import args, dem_file  # noqa: E402


def test_clip_tin_runs_end_to_end(tmp_path):
    path, s, g = dem_file(tmp_path)
    out = tmp_path / "out"
    argv = args(path, s, g, out)
    argv[argv.index("--clip") + 1] = "tin"
    assert main(argv + ["--tin_res", "5", "--tin_buffer", "100", "--tin_block", "25", "--tin_edge_max", "60",
                        "--resolution", "5"]) == 0
    run_dir = next((out / "runs").iterdir())
    rec = json.loads((run_dir / "params.json").read_text())
    co = rec["coarse"]
    assert co["nodes"] > 50 and co["edges"] > co["nodes"] and co["corridors"][0]["found_by"] == "best"
    assert co["node_kinds"]["endpoint"] == 2
    corr = gpd.read_file(run_dir / "coarse_corridors.geojson")
    assert len(corr) == len(co["corridors"]) and corr.crs is not None
    # The fine search ran inside the corridors' region and found a route.
    assert (run_dir / "clip_region.tif").exists() and rec["routes"][0]["cost_true"] > 0
    # The graph's estimate is the same order as the fine search's cost.
    assert 0.3 < co["corridors"][0]["cost"] / rec["routes"][0]["cost_true"] < 3.0


def test_node_and_cost_sampling(tmp_path):
    """Both samplings are optional; with them, every sample is counted once (a corridor's share,
    or a new corridor), and a new corridor joins the fine search's region."""
    from roadsearch.params import load_params
    from roadsearch.tin import tin_corridor

    path, s, g = dem_file(tmp_path)
    p = load_params(overrides={"tin_res": 5.0, "tin_buffer": 100.0, "tin_block": 25.0, "tin_edge_max": 60.0,
                               "tin_node_samples": 3, "tin_node_density": 4000.0, "cost_samples": 8})
    grid, region, info, gdf = tin_corridor(path, p, s, g, [], log=lambda m: None)
    kept = gdf[gdf.found_by != "sampled"]
    new = gdf[gdf.found_by == "sampled"]
    assert info["node_samples"] == 3 and info["node_sample_nodes"] > 0
    node_total = kept.node_sample_share.sum() + new.node_sample_share.sum()
    assert abs(node_total - 1.0) < 1e-9
    assert abs(kept.cost_sample_share.sum() + info["cost_sample_share_none"] - 1.0) < 1e-9
    # Without sampling, no shares.
    p0 = load_params(overrides={"tin_res": 5.0, "tin_buffer": 100.0, "tin_block": 25.0, "tin_edge_max": 60.0})
    _, _, info0, gdf0 = tin_corridor(path, p0, s, g, [], log=lambda m: None)
    assert gdf0.node_sample_share.isna().all() and info0["node_sample_share_new"] is None


def test_graph_window_on_a_geographic_dem(tmp_path):
    """A NAD83 latitude/longitude DEM larger than the endpoints' box plus tin_margin: the graph
    covers only that window (reprojected to UTM), not the whole DEM."""
    import numpy as np
    from affine import Affine
    from pyproj import CRS, Transformer

    from roadsearch.params import load_params
    from roadsearch.tin import graph_window, tin_corridor

    from .conftest import write_dem
    from .test_search import random_terrain

    n = 240
    z = random_terrain(5, n=n, relief=20.0) + 1500.0
    d = 1.0 / 10800.0                                   # 1/3 arc-second
    lon0, lat0 = -115.5, 44.95
    path = write_dem(tmp_path / "geo.tif", z, Affine(d, 0, lon0, 0, -d, lat0), CRS.from_epsg(4269))
    # Endpoints 60 cells in from opposite corners of the middle third, in UTM 11N.
    to_utm = Transformer.from_crs(4269, 32611, always_xy=True)
    s = to_utm.transform(lon0 + 90 * d, lat0 - 90 * d)
    g = to_utm.transform(lon0 + 150 * d, lat0 - 150 * d)
    p = load_params(overrides={"tin_res": 5.0, "tin_margin": 150.0, "tin_buffer": 100.0, "tin_block": 25.0,
                               "tin_edge_max": 60.0})
    bounds, res = graph_window(path, p, s, g)
    dem_w = n * d * 111_320 * np.cos(np.radians(lat0))   # m
    assert res == 5.0 and bounds[2] - bounds[0] < 0.75 * dem_w
    grid, region, info, gdf = tin_corridor(path, p, s, g, [], log=lambda m: None)
    assert grid.crs.to_epsg() == 32611 and info["window_bounds"] == [round(v, 1) for v in bounds]
    assert len(gdf) >= 1 and gdf.crs.to_epsg() == 32611
