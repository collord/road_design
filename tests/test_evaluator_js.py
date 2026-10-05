"""The browser evaluator (roadsearch/evaluator.js) against the Python model.

Runs the JavaScript under Node; skipped when Node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from roadsearch.lattice import ACTION_PRIM
from roadsearch.params import load_params
from roadsearch.route import evaluate_route, vertical_travel
from roadsearch.section import section, section_params

from roadsearch.preprocess import preprocess
from roadsearch.search import make_problem, reconstruct, run_search

from .conftest import make_grid
from .test_search import random_terrain

NODE = shutil.which("node")
EVALUATOR = Path(__file__).resolve().parent.parent / "roadsearch" / "evaluator.js"
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


def run_js(tmp_path, body: str, data) -> dict:
    """Run `body` (JS, with E = the evaluator and D = data) and return what it prints as JSON."""
    (tmp_path / "in.json").write_text(json.dumps(data))
    script = tmp_path / "t.js"
    script.write_text(f"const E = require({json.dumps(str(EVALUATOR))});\n"
                      f"const D = JSON.parse(require('fs').readFileSync({json.dumps(str(tmp_path / 'in.json'))}, 'utf8'));\n"
                      + body)
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def js_params(p) -> dict:
    return {"sp": section_params(p)._asdict(), "d0": p.d0, "S0": p.S0, "g_max": p.g_max, "R_min": p.R_min,
            "C_travel": p.C_travel, "C_vertical": p.C_vertical,
            "culvert_area_breaks": list(p.culvert_area_breaks), "culvert_costs": list(p.culvert_costs)}


def test_section_matches_python(tmp_path):
    p = load_params()
    sp = section_params(p)
    cases = [(s, d) for s in np.linspace(0.0, p.s_mask * 1.02, 47) for d in (0.0, 0.3, 1.2, 2.0)]
    got = run_js(tmp_path, "console.log(JSON.stringify(D.cases.map(([s, d]) => E.sectionCore(s, d, D.sp))));",
                 {"cases": cases, "sp": sp._asdict()})
    for (s, d), g in zip(cases, got):
        want = section(s, d, p)
        assert E_KIND[g["kind"]] == want.kind, (s, d)
        if want.kind == "infeasible":
            continue
        for k_js, k_py in (("cost", "cost"), ("As", "A_soil"), ("Ar", "A_rock"), ("Af", "A_fill"), ("a", "cut_width"),
                           ("catch", "cut_catch")):
            assert g[k_js] == pytest.approx(getattr(want, k_py), rel=1e-9, abs=1e-9), (s, d, k_js)


E_KIND = {-1: "infeasible", 0: "flat", 1: "half_bench", 2: "full_bench"}


def test_route_edges_match_python(tmp_path):
    """Every primitive edge of a best route, costed in JS from the same samples."""
    p = load_params(overrides={"C_travel": 3.0, "C_vertical": 40.0, "A_min": 0.3, "culvert_costs": [1.0, 2.0, 3.0]})
    # Gentle hills with a north-south gully across the route, so it must cross a channel
    # (a straight 4-connected channel: a diagonal D8 channel can be crossed between its cells).
    rr, cc = np.mgrid[0:80, 0:80]
    z = random_terrain(11, n=80, relief=2.0) + 0.3 * np.abs(cc - 40) + 0.05 * rr
    # Primitives only, so the whole route can be compared as one line.
    g = make_grid(z, 5.0)
    prob = make_problem(preprocess(g, g.xy(6, 6), g.xy(72, 70), p), p, hairpins=False, crossings=False, bends=False)
    res = run_search(prob, (6, 6), (72, 70), astar=True)
    edges = reconstruct(res, prob.lat, res.goal_state)
    segs = evaluate_route(prob, edges)
    L = prob.lat
    cases = []
    for (r, c, a), sg in zip(edges, segs):
        if L.kind[a] != ACTION_PRIM:
            continue
        s0, n = int(L.s0[a]), int(L.ns[a])
        length = float(L.length[a] * prob.res)
        w = length / n
        samples = [{"fr": float(r + L.sdr[k]), "fc": float(c + L.sdc[k]), "tx": float(L.tx[k]), "ty": float(L.ty[k]),
                    "w": w, "station": (k - s0 + 0.5) * w} for k in range(s0, s0 + n)]
        cases.append({"samples": samples, "rc0": [r, c], "rc1": [r + int(L.dr[a]), c + int(L.dc[a])],
                      "length": length, "construction": sg.cost_base + sg.cost_earthwork,
                      "culvert": sg.cost_culvert, "vertical": vertical_travel(prob, r, c, a)})
    assert len(cases) > 10
    nr, nc = prob.zs.shape
    flags = (prob.open_ == 0).astype(int) | (prob.channel.astype(int) << 1)
    data = {"T": {"w": nc, "h": nr, "res": prob.res, "x0": 0.0, "y0": 0.0, "zs": prob.zs.ravel().tolist(),
                  "flags": flags.ravel().tolist(), "area": prob.area.ravel().tolist()},
            "P": js_params(p), "cases": cases}
    got = run_js(tmp_path, """
const T = E.makeTerrain(D.T);
console.log(JSON.stringify({
  gx: Array.from(T.gx), gy: Array.from(T.gy),
  edges: D.cases.map(q => E.costSamples(T, D.P, q.samples, q.rc0, q.rc1, q.length).tot),
}));""", data)
    assert np.allclose(np.array(got["gx"]).reshape(nr, nc), prob.gx, rtol=1e-12, atol=1e-12)
    assert np.allclose(np.array(got["gy"]).reshape(nr, nc), prob.gy, rtol=1e-12, atol=1e-12)
    for q, t in zip(cases, got["edges"]):
        assert t["construction"] - t["culvert_cost"] == pytest.approx(q["construction"], rel=1e-9)
        assert t["vertical"] == pytest.approx(q["vertical"], rel=1e-9, abs=1e-9)
        assert t["operating"] == pytest.approx(3.0 * q["length"] + 40.0 * q["vertical"], rel=1e-9)
        assert t["over_grade_m"] == 0 and t["excluded_m"] == 0 and t["infeasible_m"] == 0
    # The whole route as one sample list: one culvert per run of channel cells, as the search
    # charges them (half from each edge at a shared node).
    whole = run_js(tmp_path, """
const T = E.makeTerrain(D.T);
let st = 0; const S = [];
for (const q of D.cases) { for (const s of q.samples) S.push(Object.assign({}, s, { station: st + s.station })); st += q.length; }
console.log(JSON.stringify(E.costSamples(T, D.P, S, D.cases[0].rc0, D.cases[D.cases.length - 1].rc1, st).tot));""", data)
    assert whole["culvert_cost"] == pytest.approx(sum(sg.cost_culvert for sg in segs), rel=1e-9, abs=1e-9)
    assert whole["construction"] == pytest.approx(sum(sg.cost_construction for sg in segs), rel=1e-9)
    assert (prob.channel[tuple(np.floor(np.vstack([sg.pts_rc for sg in segs]) + 0.5).astype(int).T)] != 0).any()


def test_fillet_rounds_corners(tmp_path):
    got = run_js(tmp_path, """
const f1 = E.fillet([[0, 0], [200, 0], [200, 200]], 40, 1);
const f2 = E.fillet([[0, 0], [30, 0], [30, 30]], 40, 1);
const rad = pts => { // radius from three points mid-arc
  const k = Math.floor(pts.length / 2), [a, b, c] = [pts[k - 1], pts[k], pts[k + 1]];
  const A = Math.hypot(b[0] - c[0], b[1] - c[1]), B = Math.hypot(a[0] - c[0], a[1] - c[1]), C = Math.hypot(a[0] - b[0], a[1] - b[1]);
  const area = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
  return A * B * C / (4 * area); };
console.log(JSON.stringify({ c1: f1.corners, c2: f2.corners, r1: rad(f1.pts), start: f1.pts[0], end: f1.pts[f1.pts.length - 1],
  len1: f1.pts.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - f1.pts[i][0], p[1] - f1.pts[i][1]), 0) }));""", {})
    assert got["c1"][0]["tight"] is False and got["c1"][0]["angle"] == pytest.approx(90)
    assert got["r1"] == pytest.approx(40, rel=1e-3)
    assert got["start"] == [0, 0] and got["end"] == [200, 200]
    assert got["len1"] == pytest.approx(400 - 2 * 40 + np.pi * 40 / 2, rel=1e-3)
    assert got["c2"][0]["tight"] is True and got["c2"][0]["radius"] == pytest.approx(30, rel=1e-9)


def test_python_polyline_costing_matches_the_browser(tmp_path):
    """polyline.evaluate_polyline (reference lines) and evaluator.js (drawn routes) agree."""
    from affine import Affine

    from roadsearch.polyline import RunTerrain, evaluate_polyline, sample_polyline

    p = load_params(overrides={"A_min": 0.3, "C_travel": 2.0, "C_vertical": 30.0})
    rr, cc = np.mgrid[0:80, 0:80]
    z = random_terrain(3, n=80, relief=8.0) + 0.3 * np.abs(cc - 40) + 0.05 * rr
    g = make_grid(z, 5.0)
    t = preprocess(g, g.xy(6, 6), g.xy(72, 70), p)
    tr = Affine(5.0, 0, 0.0, 0, -5.0, 400.0)
    T = RunTerrain(t.z_smooth, g.z, t.code, t.area_ha, tr, p)
    xy = np.array([[20.0, 380.0], [120.0, 300.0], [210.0, 330.0], [330.0, 120.0], [380.0, 30.0]])
    py = evaluate_polyline(T, xy)
    pts, tan, w, st, length = sample_polyline(xy, 2.5)
    fr, fc = T.rc(pts)
    samples = [{"fr": float(a), "fc": float(b), "tx": float(u), "ty": float(v), "w": float(ww), "station": float(s)}
               for a, b, (u, v), ww, s in zip(fr, fc, tan, w, st)]
    r0, c0 = T.rc(xy[:1]); r1, c1 = T.rc(xy[-1:])
    flags = (t.code != 0).astype(int) | ((np.nan_to_num(t.area_ha) >= p.A_min).astype(int) << 1)
    data = {"T": {"w": 80, "h": 80, "res": 5.0, "x0": 0.0, "y0": 400.0, "zs": t.z_smooth.ravel().tolist(),
                  "flags": flags.ravel().tolist(), "area": np.nan_to_num(t.area_ha).ravel().tolist()},
            "P": js_params(p), "S": samples, "a": [float(r0[0]), float(c0[0])], "b": [float(r1[0]), float(c1[0])],
            "L": length}
    js = run_js(tmp_path, """
const T = E.makeTerrain(D.T);
console.log(JSON.stringify(E.costSamples(T, D.P, D.S, D.a, D.b, D.L).tot));""", data)
    assert py["culverts"] > 0 and py["over_grade_m"] > 0
    for k in ("length", "base", "earthwork", "culverts", "culvert_cost", "soil_m3", "rock_m3", "fill_m3", "vertical",
              "over_grade_m", "excluded_m", "infeasible_m", "construction", "operating", "total"):
        assert py[k] == pytest.approx(js[k], rel=1e-9, abs=1e-6), k
