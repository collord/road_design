"""The corridor tree: routes through a cell rebuilt from the saved tree match route_through."""

import numpy as np
import pytest

from roadsearch.explore import corridor_tree
from roadsearch.params import load_params
from roadsearch.pipeline import corridor_fields, route_through
from roadsearch.preprocess import preprocess
from roadsearch.search import make_problem

from .conftest import make_grid
from .test_search import random_terrain


def chain_cells(t, i, nc):
    """Cells from node i up to the root, as the viewer walks them."""
    out = [divmod(int(t["cell"][i]), nc)]
    while t["parent"][i] >= 0:
        i = int(t["parent"][i])
        out.append(divmod(int(t["cell"][i]), nc))
    return out


def test_tree_routes_match_route_through():
    p = load_params()
    z = random_terrain(4, n=70, relief=5.0)
    g = make_grid(z, 5.0)
    s, e = (5, 5), (64, 60)
    prob = make_problem(preprocess(g, g.xy(*s), g.xy(*e), p), p, hairpins=False)
    fields = corridor_fields(prob, s, e)
    tree = corridor_tree(prob, fields, rel_max=0.5, spacing_m=None)
    nh, nc = prob.n_theta, prob.nc
    keys = {d: tree[d]["cell"].astype(np.int64) * nh + tree[d]["h"] for d in ("fwd", "bwd")}
    ok = np.argwhere(np.isfinite(fields.F) & (fields.F <= 1.5 * fields.F_min))
    rng = np.random.default_rng(0)
    for r, c in ok[rng.choice(len(ok), 12, replace=False)]:
        edges, via = route_through(prob, fields, (r, c))
        L = prob.lat
        nodes = [s]
        for er, ec, a in edges:
            nodes.append((er + int(L.dr[a]), ec + int(L.dc[a])))
        hb = int(tree["best_h"][r, c])
        fi = int(np.searchsorted(keys["fwd"], (r * nc + c) * nh + hb))
        bi = int(np.searchsorted(keys["bwd"], (r * nc + c) * nh + (hb + nh // 2) % nh))
        assert keys["fwd"][fi] == (r * nc + c) * nh + hb
        got = chain_cells(tree["fwd"], fi, nc)[::-1] + chain_cells(tree["bwd"], bi, nc)[1:]
        assert got == [tuple(map(int, n)) for n in nodes]
        length = float(tree["fwd"]["length"][fi] + tree["bwd"]["length"][bi])
        assert length == pytest.approx(sum(L.length[a] for _, _, a in edges) * prob.res, rel=1e-4)       # stored as float32
    # Cells beyond the threshold have no entry.
    assert (tree["best_h"][fields.F > 1.5 * fields.F_min] == -1).all()


def test_viewer_tree_keeps_routes_through_via_points():
    from roadsearch.explore import viewer_tree

    p = load_params()
    z = random_terrain(4, n=70, relief=5.0)
    g = make_grid(z, 5.0)
    s, e = (5, 5), (64, 60)
    prob = make_problem(preprocess(g, g.xy(*s), g.xy(*e), p), p, hairpins=False)
    fields = corridor_fields(prob, s, e)
    v = viewer_tree(corridor_tree(prob, fields, spacing_m=15.0), fields.F, prob.res, spacing_m=15.0)
    nh, nc, step = prob.n_theta, prob.nc, v["step"]
    assert step == 3
    vias = np.argwhere(v["best_h"] >= 0)
    assert len(vias) and (vias % step == 0).all()
    keys = {d: v[d]["cell"].astype(np.int64) * nh + v[d]["h"] for d in ("fwd", "bwd")}
    for r, c in vias[:: max(1, len(vias) // 10)]:
        edges, _ = route_through(prob, fields, (r, c))
        L = prob.lat
        nodes = [s] + [(er + int(L.dr[a]), ec + int(L.dc[a])) for er, ec, a in edges]
        hb = int(v["best_h"][r, c])
        fi = int(np.searchsorted(keys["fwd"], (r * nc + c) * nh + hb))
        bi = int(np.searchsorted(keys["bwd"], (r * nc + c) * nh + (hb + nh // 2) % nh))
        got = chain_cells(v["fwd"], fi, nc)[::-1] + chain_cells(v["bwd"], bi, nc)[1:]
        assert got == [tuple(map(int, n)) for n in nodes]
