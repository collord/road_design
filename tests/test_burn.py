"""Burning a route into the DEM: design surface and cut/fill against the section model."""

import json

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString

from roadsearch.burn import burn_surface, centerline_samples
from roadsearch.cli import main
from roadsearch.params import load_params
from roadsearch.section import section, soil_depth
from roadsearch.terrain import Grid

from .conftest import UTM, X0, Y0, make_grid, write_dem
from .test_search import random_terrain

RES = 0.5
NR, NC = 240, 400


def plane(S):
    rows, cols = np.mgrid[0:NR, 0:NC]
    tr = Affine(RES, 0, 0, 0, -RES, NR * RES)
    y = tr.f + (rows + 0.5) * tr.e
    return Grid(1000 + S * y, tr, UTM)                          # rising north


def straight(grid, kind, a, b, cut_soil=0.0, cut_rock=0.0, z=None, y=None):
    y = NR * RES / 2 if y is None else y
    z0 = float(grid.z[0, 0]) if z is None else z
    line = LineString([(20, y, z0), (180, y, z0)])
    return gpd.GeoDataFrame({"seq": [0], "section_type": [kind], "cut_width": [a], "fill_width": [b],
                             "cut_soil_m3": [cut_soil], "cut_rock_m3": [cut_rock]}, geometry=[line])


def per_meter(b, grid):
    cols = np.arange(NC)
    x = grid.transform.c + (cols + 0.5) * RES
    mid = (x > 60) & (x < 140)                                  # 80 m clear of the ends
    dep = b["depth"][:, mid]
    return (np.maximum(-dep, 0).sum() * RES * RES / 80, np.maximum(dep, 0).sum() * RES * RES / 80)


@pytest.mark.parametrize("S", [0.2, 0.4, 0.6])
def test_half_bench_matches_section_model(S):
    p = load_params()
    g = plane(S)
    d = float(soil_depth(S, p))
    sec = section(S, d, p)
    assert sec.kind == "half_bench"
    segs = straight(g, "half_bench", sec.cut_width, sec.fill_width, sec.A_soil * 160, sec.A_rock * 160)
    cl = centerline_samples(segs, g, g.z, p, RES / 2)
    b = burn_surface(g, g.z, np.full(g.shape, d), cl, p, 60.0)
    cut, fill = per_meter(b, g)
    assert cut == pytest.approx(sec.A_soil + sec.A_rock, rel=0.04)
    assert fill == pytest.approx(sec.A_fill, rel=0.04)
    # The road surface is level across its width and as wide as the road.
    assert b["volumes"]["road_area_m2"] == pytest.approx(160 * p.w, rel=0.05)


def test_full_bench_has_no_fill():
    p = load_params()
    S = 0.9
    g = plane(S)
    sec = section(S, 0.0, p)
    assert sec.kind == "full_bench" and sec.fill_width == 0
    segs = straight(g, "full_bench", sec.cut_width, 0.0, sec.A_soil * 160, sec.A_rock * 160)
    cl = centerline_samples(segs, g, g.z, p, RES / 2)
    b = burn_surface(g, g.z, np.zeros(g.shape), cl, p, 60.0)
    cut, fill = per_meter(b, g)
    assert fill < 0.02
    assert cut == pytest.approx(sec.A_soil + sec.A_rock, rel=0.04)


def test_flat_ground_is_unchanged():
    p = load_params()
    g = plane(0.0)
    segs = straight(g, "flat", p.w / 2, p.w / 2)
    cl = centerline_samples(segs, g, g.z, p, RES / 2)
    b = burn_surface(g, g.z, np.full(g.shape, p.d0), cl, p, 60.0)
    assert np.abs(b["depth"]).max() < 1e-9
    assert b["volumes"]["road_area_m2"] > 0


def test_fill_crossing_embankment():
    """A level embankment over a V gully: w*h + h^2/f_rock integrated along it."""
    p = load_params()
    rows, cols = np.mgrid[0:NR, 0:NC]
    tr = Affine(RES, 0, 0, 0, -RES, NR * RES)
    x = tr.c + (cols + 0.5) * RES
    z = 1000 - 6.0 * np.clip(1 - np.abs(x - 100) / 25, 0, None)   # 6 m deep, 50 m wide
    g = Grid(z, tr, UTM)
    segs = straight(g, "fill_crossing", p.w / 2, p.w / 2, z=1000.0)
    segs.geometry = [LineString([(70, NR * RES / 2, 1000.0), (130, NR * RES / 2, 1000.0)])]
    cl = centerline_samples(segs, g, g.z, p, RES / 2)
    b = burn_surface(g, g.z, np.zeros(g.shape), cl, p, 60.0)
    analytic = p.w * 6 * 25 + (2 * 36 * 25 / 3) / p.f_rock
    fill = np.maximum(b["depth"], 0).sum() * RES * RES
    assert fill == pytest.approx(analytic, rel=0.08)     # plus the slopes round the embankment ends


def test_burn_command_end_to_end(tmp_path):
    z = random_terrain(6, n=90, relief=5.0)
    path = write_dem(tmp_path / "dem.tif", z, Affine(5.0, 0, X0, 0, -5.0, Y0), UTM)
    g = make_grid(z, 5.0)
    (sx, sy), (gx, gy) = g.xy(12, 12), g.xy(78, 70)
    out = tmp_path / "out"
    assert main(["run", "--dem", str(path), "--start", f"{sx},{sy}", "--goal", f"{gx},{gy}", "--out", str(out),
                 "--clip", "none", "--K_alt", "1"]) == 0
    run_id = next((out / "runs").iterdir()).name
    assert main(["burn", run_id, "--out", str(out), "--res", "2", "--buffer", "60"]) == 0
    d = out / "runs" / run_id / "burn" / "rank1"
    for f in ["dem_existing.tif", "dem_design.tif", "cut_fill_depth.tif", "cut_material.tif", "road_surface.tif",
              "wall_cells.tif", "burn_summary.json", "burn_quicklook.png"]:
        assert (d / f).exists(), f
    s = json.loads((d / "burn_summary.json").read_text())
    v, m = s["burned"], s["section_model"]
    assert v["cut_soil_m3"] + v["cut_rock_m3"] > 0
    cut_m = m["cut_soil_m3"] + m["cut_rock_m3"]
    assert v["cut_soil_m3"] + v["cut_rock_m3"] == pytest.approx(cut_m, rel=0.5)
