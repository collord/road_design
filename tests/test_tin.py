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
