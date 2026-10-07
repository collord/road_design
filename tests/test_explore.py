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


def _dist_to_polyline(pts, line):
    """Distance from each point to a polyline (cells)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.einsum("ijk,jk->ij", pts[:, None, :] - a[None], ab) / np.maximum((ab * ab).sum(1), 1e-12)
    proj = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(-1)).min(1)


def test_tree_macro_shapes_match_edge_points():
    """The batched hairpin and bend centerlines are the shapes edge_points draws one at a time."""
    from roadsearch.explore import macro_shapes
    from roadsearch.lattice import ACTION_FILL, ACTION_PRIM
    from roadsearch.search import edge_points, reconstruct

    p = load_params()
    z = random_terrain(7, n=90, relief=15.0)
    g = make_grid(z, 5.0)
    s, e = (5, 5), (84, 80)
    prob = make_problem(preprocess(g, g.xy(*s), g.xy(*e), p), p)
    fields = corridor_fields(prob, s, e)
    tree = corridor_tree(prob, fields, rel_max=0.5, spacing_m=None)
    L = prob.lat
    edges = set()
    for d, res in (("fwd", fields.fwd), ("bwd", fields.bwd)):
        t = tree[d]
        for i in np.flatnonzero(t["geo_off"] >= 0):
            cell = int(t["cell"][i])
            edges.add(reconstruct(res, L, (cell // prob.nc, cell % prob.nc, int(t["h"][i])))[-1])
    kinds = {int(L.kind[a]) for _, _, a in edges}
    assert len(edges) > 50 and kinds - {ACTION_PRIM, ACTION_FILL} == kinds and len(kinds) >= 3, kinds
    rows, cols, acts = map(np.array, zip(*sorted(edges)))
    lens, off, geo = macro_shapes(prob, rows, cols, acts)
    assert (off >= 0).all()
    for i, (r, c, a) in enumerate(zip(rows, cols, acts)):
        ref = edge_points(prob, int(r), int(c), int(a))
        n = int(geo[off[i], 0])
        got = geo[off[i] + 1: off[i] + 1 + n].astype(float)
        assert np.allclose(got[0], ref[0], atol=1e-3) and np.allclose(got[-1], ref[-1], atol=1e-3)
        assert _dist_to_polyline(got, ref).max() < 0.02             # on the same curve (cells)
        assert n >= np.hypot(*np.diff(ref, axis=0).T).sum() - 1      # about one point per cell
        ref_len = np.hypot(*np.diff(ref, axis=0).T).sum() * prob.res
        assert lens[i] == pytest.approx(ref_len, rel=5e-3)
