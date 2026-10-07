"""The sweep command: one run per operating-cost weight, finished runs reused, one viewer."""

import json

import pytest

from affine import Affine

from roadsearch.cli import main

from .conftest import UTM, X0, Y0, make_grid, write_dem
from .test_search import random_terrain


def test_sweep_runs_each_k_once_and_writes_viewer(tmp_path, capsys):
    z = random_terrain(6, n=60, relief=5.0)
    path = write_dem(tmp_path / "dem.tif", z, Affine(5.0, 0, X0, 0, -5.0, Y0), UTM)
    g = make_grid(z, 5.0)
    (sx, sy), (gx, gy) = g.xy(8, 8), g.xy(50, 48)
    out = tmp_path / "out"
    argv = ["sweep", "--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}", "--out", str(out),
            "--clip", "none", "--K_alt", "1", "--C_travel", "1", "--C_vertical", "20", "--k", "0,0.1,10", "--tree"]
    assert main(argv) == 0
    runs = sorted((out / "runs").iterdir())
    assert len(runs) == 3
    ops = sorted((json.loads((r / "params.json").read_text())["params"]["C_travel"],
                  json.loads((r / "params.json").read_text())["params"]["C_vertical"]) for r in runs)
    assert ops == [(0.0, 0.0), (0.1, 2.0), (10.0, 200.0)]
    page = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    assert all(f'"run_id":"{r.name}"' in page for r in runs)
    data = out / "viewer" / "data"
    assert sorted(f.name for f in data.iterdir()) == sorted([f"{r.name}.json" for r in runs] + [f"{r.name}_tree.json" for r in runs])
    d = json.loads((data / f"{runs[0].name}.json").read_text(encoding="utf-8"))
    assert {"zs", "P", "tree_file"} <= d.keys() and d["P"]["sp"]["w"] > 0
    tree = json.loads((data / f"{runs[0].name}_tree.json").read_text(encoding="utf-8"))
    assert tree["fwd"]["n"] > 0 and tree["bwd"]["n"] > 0
    assert 'id="v-draw"' in page and "root.RoadEval = api" in page and "/*__EVALUATOR__*/" not in page
    assert 'id="v-method"' in page and "<h2>3. Finding the cheapest chain: Dijkstra</h2>" in page and page.count('<svg class="fig') == 4
    assert "__METHOD__" not in page and "[[figure:" not in page

    capsys.readouterr()
    assert main(argv + ["--no-viewer"]) == 0                      # second time: nothing is re-run
    assert capsys.readouterr().out.count("already finished, reusing it") == 3


def test_sweep_needs_unit_operating_costs(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        main(["sweep", "--dem", "x.tif", "--start", "0,0", "--goal", "1,1", "--out", str(tmp_path)])


def test_merge_combines_separate_output_roots(tmp_path):
    """Two sweeps into separate roots (as CI jobs), merged, give the same site as one root."""
    import geopandas as gpd

    z = random_terrain(6, n=60, relief=5.0)
    path = write_dem(tmp_path / "dem.tif", z, Affine(5.0, 0, X0, 0, -5.0, Y0), UTM)
    g = make_grid(z, 5.0)
    (sx, sy), (gx, gy) = g.xy(8, 8), g.xy(50, 48)
    common = ["--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}", "--clip", "none", "--K_alt", "1",
              "--C_travel", "1", "--C_vertical", "20", "--no-viewer"]
    assert main(["sweep", *common, "--k", "0", "--out", str(tmp_path / "a")]) == 0
    assert main(["sweep", *common, "--k", "10", "--out", str(tmp_path / "b")]) == 0
    assert main(["merge", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "m")]) == 0
    s = gpd.read_parquet(tmp_path / "m" / "route_summary.parquet")
    assert s["run_id"].nunique() == 2 and len(list((tmp_path / "m" / "runs").iterdir())) == 2
    assert main(["sweep", *common[:-1], "--k", "0,10", "--out", str(tmp_path / "m")]) == 0     # reuses both
    # ...and the merged runs export to a GeoPackage.
    import pyogrio
    gp = tmp_path / "r.gpkg"
    assert main(["export", str(gp), "--out-root", str(tmp_path / "m"), "--prefix", "t"]) == 0
    assert {n for n, _ in pyogrio.list_layers(gp)} == {"t_routes", "t_segments", "t_features"}
    r = gpd.read_file(gp, layer="t_routes")
    assert r["run_id"].nunique() == 2 and r.geometry.has_z.all() and (r["self_overlaps"] == 0).all()
    assert set(r["label"]) == {"C_travel 0, C_vertical 0", "C_travel 10, C_vertical 200"}
    assert (tmp_path / "m" / "viewer" / "index.html").exists()


def test_sweep_over_a_parameter(tmp_path):
    z = random_terrain(6, n=60, relief=5.0)
    path = write_dem(tmp_path / "dem.tif", z, Affine(5.0, 0, X0, 0, -5.0, Y0), UTM)
    g = make_grid(z, 5.0)
    (sx, sy), (gx, gy) = g.xy(8, 8), g.xy(50, 48)
    out = tmp_path / "out"
    assert main(["sweep", "--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}", "--out", str(out),
                 "--clip", "none", "--K_alt", "1", "--vary", "R_min=10,20", "--vary-label", "Minimum turn radius",
                 "--vary-unit", "m", "--title", "Turn radii"]) == 0
    radii = sorted(json.loads((r / "params.json").read_text())["params"]["R_min"] for r in (out / "runs").iterdir())
    assert radii == [10.0, 20.0]
    page = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    m = json.loads(page.split('<script id="manifest" type="application/json">')[1].split("</script>")[0])
    assert [r["label"] for r in m["runs"]] == ["10 m", "20 m"] and m["column"] == "Minimum turn radius"
    assert not m["sweep"] and "<title>Turn radii</title>" in page


def test_examples_registry():
    from roadsearch.examples import load_examples, run_matrix, sweep_argv

    ex = load_examples()
    assert {"haul-road", "red-mountain"} <= ex.keys()
    for name, e in ex.items():
        argv = sweep_argv(name, e)
        assert argv[0] == "sweep" and ("--vary" in argv) != ("--k" in argv)
    jobs = json.loads(run_matrix())["include"]
    assert {"example": "red-mountain", "value": "30"} in jobs and {"example": "haul-road", "value": "0.1"} in jobs


def test_extra_runs_and_a_reference_line(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString

    z = random_terrain(6, n=60, relief=5.0)
    path = write_dem(tmp_path / "dem.tif", z, Affine(5.0, 0, X0, 0, -5.0, Y0), UTM)
    g = make_grid(z, 5.0)
    (sx, sy), (gx, gy) = g.xy(8, 8), g.xy(50, 48)
    ref = tmp_path / "ref.geojson"
    gpd.GeoDataFrame(geometry=[LineString([(gx, gy), ((sx + gx) / 2 + 30, (sy + gy) / 2), (sx, sy)])], crs=UTM).to_file(ref)
    from shapely.geometry import LineString as LS, Polygon, box
    zones = tmp_path / "zones.gpkg"           # one zone to avoid, one to ignore (filtered out by type)
    cx, cy = (sx + gx) / 2, (sy + gy) / 2
    gpd.GeoDataFrame({"type": ["avoid", "keep"]}, geometry=[box(cx - 20, cy - 20, cx + 20, cy + 20),
                     box(sx + 30, sy - 60, sx + 60, sy - 30)], crs=UTM).to_file(zones)
    out = tmp_path / "out"
    assert main(["sweep", "--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}", "--out", str(out),
                 "--clip", "none", "--K_alt", "1", "--vary", "R_min=20", "--vary-unit", "m",
                 "--extra", "20 m, 12% grade|R_min=20,g_max=0.12",
                 "--extra", f"avoid|R_min=20|exclude={zones}|type='avoid'", "--reference", str(ref),
                 "--reference-label", "As built"]) == 0
    page = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    m = json.loads(page.split('<script id="manifest" type="application/json">')[1].split("</script>")[0])
    assert [r["label"] for r in m["runs"]] == ["20 m", "20 m, 12% grade", "avoid"]
    assert len(m["zones"]) == 1 and m["zones"][0]["runs"] == ["avoid"] and m["zones"][0]["where"] == "type='avoid'"
    av = [x for x in m["runs"] if x["label"] == "avoid"][0]["routes"][0]["xy"]
    assert not LS(av).intersects(Polygon(m["zones"][0]["rings"][0]))        # kept out of the zone
    # Just the extra run (as a CI job runs it): an empty --vary list with --extra.
    assert main(["sweep", "--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}", "--out", str(out),
                 "--clip", "none", "--K_alt", "1", "--vary", "R_min=", "--no-viewer",
                 "--extra", "20 m, 12% grade|R_min=20,g_max=0.12"]) == 0
    assert [r["params"]["g_max"] for r in m["runs"]] == [0.1, 0.12, 0.1]
    r = m["reference"]
    assert r["label"] == "As built" and set(r["runs"]) == {x["run_id"] for x in m["runs"]}
    assert r["xy"][0] == [round(sx, 1), round(sy, 1)]                 # turned to run from the start
    t = r["runs"][m["runs"][0]["run_id"]]
    assert t["construction"] > 0 and t["length"] == pytest.approx(r["profile"]["length"], rel=1e-6)
    assert set(r["profile"]["grades"]) == {"100", "200"} and r["g_max"] == [0.1, 0.12]
