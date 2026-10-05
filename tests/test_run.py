"""End-to-end run on a synthetic DEM: outputs (section 10) and test 10 (determinism)."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine

from roadsearch.cli import main
from roadsearch.outputs import ROUTE_COLUMNS

from .conftest import UTM, X0, Y0, make_grid, write_dem
from .test_search import random_terrain

RES = 5.0


def dem_file(tmp_path):
    z = random_terrain(6, n=90, relief=5.0)
    path = write_dem(tmp_path / "dem.tif", z, Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    g = make_grid(z, RES)
    return path, g.xy(12, 12), g.xy(78, 70)


def args(path, s, g, out, *extra):
    return ["run", "--dem", str(path), "--start", f"{s[0]},{s[1]}", "--goal", f"{g[0]},{g[1]}",
            "--out", str(out), "--clip", "none", "--K_alt", "3", "--D_alt", "60", *extra]


def test_run_writes_all_outputs(tmp_path):
    path, s, g = dem_file(tmp_path)
    out = tmp_path / "out"
    assert main(args(path, s, g, out)) == 0
    routes = gpd.read_parquet(out / "routes.parquet")
    summary = gpd.read_parquet(out / "route_summary.parquet")
    feats = gpd.read_parquet(out / "features.parquet")
    for col in ROUTE_COLUMNS:
        assert col in routes.columns, col
    assert routes.geometry.has_z.all() and routes.crs.equals(UTM)
    run_id = routes["run_id"].iloc[0]
    assert sorted(summary["rank"]) == list(range(1, len(summary) + 1))
    # Ranked by true cost.
    assert list(summary.sort_values("rank")["cost_true"]) == sorted(summary["cost_true"])
    # Segments add up to the route summary.
    for _, r in summary.iterrows():
        segs = routes[routes["route_id"] == r["route_id"]]
        assert segs["length"].sum() == pytest.approx(r["length"])
        assert segs["cost_total"].sum() == pytest.approx(r["cost_true"], rel=1e-9)
        assert np.all(np.diff(segs["station_start"].to_numpy()) > 0)
    assert "hairpin_candidate" in set(feats["feature_type"])
    run_dir = out / "runs" / run_id
    for name in ["dem_smoothed", "slope", "soil_depth", "exclusion_code", "reachable", "contributing_area_ha",
                 "corridor_field", "cost_to_come_forward", "cost_to_come_backward", "explored_astar"]:
        assert (run_dir / f"{name}.tif").exists(), name
    assert (run_dir / "quicklook.png").exists()
    rec = json.loads((run_dir / "params.json").read_text())
    assert rec["params"]["g_max"] == 0.1 and rec["dem"]["crs_epsg"] == 32611
    assert rec["routes"][0]["rank"] == 1
    with rasterio.open(run_dir / "corridor_field.tif") as ds:
        F = ds.read(1)
    assert np.nanmin(F) == pytest.approx(rec["corridor_F_min"], rel=1e-6)


def test_runs_accumulate_by_run_id(tmp_path):
    path, s, g = dem_file(tmp_path)
    out = tmp_path / "out"
    assert main(args(path, s, g, out)) == 0
    assert main(args(path, s, g, out, "--C_base", "20")) == 0
    summary = gpd.read_parquet(out / "route_summary.parquet")
    assert summary["run_id"].nunique() == 2


def test_determinism(tmp_path):
    """Test 10: two identical runs produce byte-identical route output."""
    path, s, g = dem_file(tmp_path)
    blobs = []
    for k in range(2):
        out = tmp_path / f"out{k}"
        assert main(args(path, s, g, out)) == 0
        blobs.append([(out / f).read_bytes() for f in ("routes.parquet", "route_summary.parquet",
                                                       "features.parquet")])
    assert blobs[0] == blobs[1]


def test_run_reports_connectivity_failure(tmp_path):
    z = random_terrain(6, n=90, relief=5.0)
    z[30:60, 30:33] += 100
    z[30:60, 57:60] += 100
    z[30:33, 30:60] += 100
    z[57:60, 30:60] += 100
    path = write_dem(tmp_path / "dem.tif", z, Affine(RES, 0, X0, 0, -RES, Y0), UTM)
    gr = make_grid(z, RES)
    s, g = gr.xy(45, 45), gr.xy(80, 80)
    assert main(args(path, s, g, tmp_path / "out")) == 2
    rec = json.loads(next((tmp_path / "out" / "runs").iterdir()).joinpath("params.json").read_text())
    assert "ConnectivityError" in rec["error"]
