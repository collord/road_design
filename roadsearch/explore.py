"""Search trees for "the best route through here", and rebuilding a finished run.

The corridor field gives, for every cell, the cost of the cheapest route
through it. Drawing that route needs the two search trees, which are too big
to keep whole (one parent per state). Only the states on the cheapest route
through some corridor cell matter, and those form two much smaller trees:
the union of the forward paths start -> (cell, best heading), and of the
backward paths goal -> (cell, best heading reversed). corridor_tree() keeps
just those, with each node's parent, cumulative length and, for hairpins and
bends, the edge's centerline (other edges are short and drawn as chords).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numba
import numpy as np
import rasterio

from .lattice import ACTION_FILL, ACTION_PRIM
from .params import load_params
from .search import SearchProblem, edge_points, make_problem
from .terrain import load_grid

TREE_REL_MAX = 1.0      # cells up to twice the best cost, the range the viewer colors
TREE_SPACING_M = 50.0   # via cells kept for drawing routes: a grid this far apart


@numba.njit(cache=True)
def _mark_ancestors(parent, dr, dc, h0, nc, nh, seeds, mark):
    """Mark every state on the tree path from a source to each seed state."""
    for s in seeds:
        while s >= 0 and not mark[s]:
            mark[s] = True
            a = parent[s]
            if a < 0:
                break
            cell, h = divmod(s, nh)
            r, c = divmod(cell, nc)
            s = ((r - dr[a]) * nc + (c - dc[a])) * nh + h0[a]


def _one_tree(prob: SearchProblem, res, seeds: np.ndarray) -> dict:
    L = prob.lat
    nh, nc = prob.n_theta, prob.nc
    parent = res.parent.reshape(-1)
    mark = np.zeros(parent.shape[0], dtype=np.bool_)
    _mark_ancestors(parent, L.dr.astype(np.int64), L.dc.astype(np.int64), L.h0.astype(np.int64), nc, nh,
                    seeds.astype(np.int64), mark)
    ids = np.flatnonzero(mark)
    act = parent[ids]
    cell, h = np.divmod(ids, nh)
    r, c = np.divmod(cell, nc)
    par = np.full(ids.shape, -1, dtype=np.int64)
    has = act >= 0
    a = act[has]
    pid = ((r[has] - L.dr[a]) * nc + (c[has] - L.dc[a])) * nh + L.h0[a]
    par[has] = np.searchsorted(ids, pid)
    assert np.array_equal(ids[par[has]], pid)
    # Edge lengths: lattice length for primitives and crossings, the centerline for macro-actions.
    elen = np.zeros(ids.shape)
    elen[has] = L.length[a] * prob.res
    geo_off = np.full(ids.shape, -1, dtype=np.int64)
    pts_all = []
    n_pts = 0
    kind = np.full(ids.shape, ACTION_PRIM)
    kind[has] = L.kind[a]
    for i in np.flatnonzero(has & (kind != ACTION_PRIM) & (kind != ACTION_FILL)):
        pr, pc = int(r[par[i]]), int(c[par[i]])
        pts = edge_points(prob, pr, pc, int(act[i]))
        seg = np.hypot(*np.diff(pts, axis=0).T)
        elen[i] = float(seg.sum() * prob.res)
        # About one point per cell is plenty to draw it.
        keep = np.flatnonzero(np.diff(np.floor(np.concatenate([[0.0], np.cumsum(seg)])), prepend=-1) > 0)
        pts = pts[np.union1d(keep, [len(pts) - 1])]
        geo_off[i] = n_pts
        pts_all.append(np.vstack([[len(pts), 0.0], pts]))
        n_pts += len(pts) + 1
    # Cumulative length from the source, parents first (in order of cost-to-come).
    order = np.argsort(res.G.reshape(-1)[ids], kind="stable")
    cum = np.zeros(ids.shape)
    for i in order:
        if par[i] >= 0:
            cum[i] = cum[par[i]] + elen[i]
    geo = np.vstack(pts_all).astype(np.float32) if pts_all else np.zeros((0, 2), np.float32)
    return {"cell": cell.astype(np.int32), "h": h.astype(np.uint8), "parent": par.astype(np.int32),
            "length": cum.astype(np.float32), "geo_off": geo_off.astype(np.int32), "geo": geo}


def corridor_tree(prob: SearchProblem, fields, rel_max: float = TREE_REL_MAX,
                  spacing_m: float | None = TREE_SPACING_M) -> dict:
    """The forward and backward trees restricted to cheapest routes through
    cells within (1 + rel_max) x the best cost, on a grid of via cells about
    spacing_m apart (None: every cell). Routes through every cell would keep
    about two nodes per cell, and drawing every hairpin and bend among them
    takes hours on a fine grid."""
    nh = prob.n_theta
    ok = np.isfinite(fields.F) & (fields.F <= (1.0 + rel_max) * fields.F_min) & (fields.best_h >= 0)
    if spacing_m:
        step = max(1, int(round(spacing_m / prob.res)))
        rr, cc = np.mgrid[0:prob.nr, 0:prob.nc]
        ok &= (rr % step == 0) & (cc % step == 0)
    cells = np.flatnonzero(ok)
    hb = fields.best_h.reshape(-1)[cells].astype(np.int64)
    fwd = _one_tree(prob, fields.fwd, cells * nh + hb)
    bwd = _one_tree(prob, fields.bwd, cells * nh + (hb + nh // 2) % nh)
    best_h = np.where(ok, fields.best_h, -1).astype(np.int16)
    return {"fwd": fwd, "bwd": bwd, "best_h": best_h, "n_theta": nh, "rel_max": rel_max}


def write_corridor_tree(run_dir: str | Path, prob: SearchProblem, fields, rel_max: float = TREE_REL_MAX,
                        spacing_m: float | None = TREE_SPACING_M) -> None:
    t = corridor_tree(prob, fields, rel_max, spacing_m)
    flat = {f"{d}_{k}": v for d in ("fwd", "bwd") for k, v in t[d].items()}
    np.savez_compressed(Path(run_dir) / "corridor_tree.npz", best_h=t["best_h"], n_theta=t["n_theta"],
                        rel_max=t["rel_max"], **flat)


def read_corridor_tree(run_dir: str | Path) -> dict | None:
    path = Path(run_dir) / "corridor_tree.npz"
    if not path.exists():
        return None
    z = np.load(path)
    out = {"best_h": z["best_h"], "n_theta": int(z["n_theta"]), "rel_max": float(z["rel_max"])}
    for d in ("fwd", "bwd"):
        out[d] = {k[len(d) + 1:]: z[k] for k in z.files if k.startswith(d + "_")}
    return out


def rebuild_problem(run_dir: str | Path) -> tuple[SearchProblem, dict]:
    """The search problem of a finished run, as it stood for the corridor
    fields: same grid, clip region, blocked cells and final fill price."""
    from .preprocess import preprocess

    run_dir = Path(run_dir)
    rec = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    p = load_params(overrides=rec["params"])
    gi = rec["grid"]
    bounds = tuple(gi["clip_bounds"]) if gi.get("clip_bounds") else None
    grid, _ = load_grid(rec["inputs"]["dem"], p, gi["working_res_m"], bounds)
    assert list(grid.shape) == gi["working_shape"], (grid.shape, gi["working_shape"])
    with tempfile.TemporaryDirectory() as tmp:
        terrain = preprocess(grid, tuple(rec["inputs"]["start"]), tuple(rec["inputs"]["goal"]), p,
                             rec["inputs"]["exclude"], tmp)
    clip = run_dir / "clip_region.tif"
    if clip.exists():
        with rasterio.open(clip) as ds:
            terrain.clip = ds.read(1) == 1
    prob = make_problem(terrain, p)
    blocked = run_dir / "overlap_blocked.tif"
    if blocked.exists():
        with rasterio.open(blocked) as ds:
            prob.open_[ds.read(1) == 1] = 0
    prob.lam_fill = float(rec["lam_fill_final"])
    return prob, rec


def backfill_tree(run_dir: str | Path, log=print) -> dict:
    """Recompute the corridor fields of a finished run and save its corridor tree.
    Checks the rebuilt best cost against the recorded one."""
    from .pipeline import corridor_fields

    prob, rec = rebuild_problem(run_dir)
    pre = rec["preprocess"]
    fields = corridor_fields(prob, tuple(pre["start_rc"]), tuple(pre["goal_rc"]))
    want = float(rec["corridor_F_min"])
    rel = abs(fields.F_min - want) / want
    if rel > 1e-9:
        raise RuntimeError(f"rebuilt best cost {fields.F_min:,.2f} differs from the recorded {want:,.2f}")
    write_corridor_tree(run_dir, prob, fields)
    t = read_corridor_tree(run_dir)
    info = {"F_min": fields.F_min, "fwd_nodes": int(t["fwd"]["cell"].size), "bwd_nodes": int(t["bwd"]["cell"].size)}
    log(f"{Path(run_dir).name}: corridor tree {info['fwd_nodes']:,} + {info['bwd_nodes']:,} nodes "
        f"(best cost {fields.F_min:,.0f} matches)")
    return info


@numba.njit(cache=True)
def _mark_from(parent, seeds, mark):
    for i in seeds:
        while i >= 0 and not mark[i]:
            mark[i] = True
            i = parent[i]


def viewer_tree(t: dict, F: np.ndarray, res: float, rel_max: float = TREE_REL_MAX,
                spacing_m: float = TREE_SPACING_M) -> dict:
    """A compact copy of a corridor tree for the viewer: routes through via
    cells on a grid about spacing_m apart (every cell would be ~2 nodes per
    cell), hairpin and bend shapes as int16 offsets from the parent cell in
    1/8 cells, no stored lengths (the viewer measures its polyline)."""
    nh = t["n_theta"]
    nr, nc = F.shape
    step = max(1, int(round(spacing_m / res)))
    F_min = float(np.nanmin(F))
    rr, cc = np.mgrid[0:nr, 0:nc]
    ok = (np.isfinite(F) & (F <= (1.0 + rel_max) * F_min) & (t["best_h"] >= 0)
          & (rr % step == 0) & (cc % step == 0))
    cells = np.flatnonzero(ok)
    hb = t["best_h"].reshape(-1)[cells].astype(np.int64)
    out = {"n_theta": nh, "step": step, "best_h": np.where(ok, t["best_h"], -1).astype(np.int8)}
    for d, seeds in (("fwd", cells * nh + hb), ("bwd", cells * nh + (hb + nh // 2) % nh)):
        a = t[d]
        key = a["cell"].astype(np.int64) * nh + a["h"]
        idx = np.searchsorted(key, seeds)
        assert np.array_equal(key[idx], seeds)
        mark = np.zeros(key.size, dtype=np.bool_)
        _mark_from(a["parent"].astype(np.int64), idx.astype(np.int64), mark)
        keep = np.flatnonzero(mark)
        new = np.full(key.size, -1, dtype=np.int64)
        new[keep] = np.arange(keep.size)
        par = a["parent"][keep]
        par = np.where(par >= 0, new[np.maximum(par, 0)], -1)
        ng = np.zeros(keep.size, dtype=np.uint8)
        geo = []
        for j, i in enumerate(keep):
            o = a["geo_off"][i]
            if o < 0:
                continue
            n = int(a["geo"][o, 0])
            pts = a["geo"][o + 1:o + 1 + n].astype(np.float64)
            if n > 255:
                pts = pts[np.unique(np.linspace(0, n - 1, 255).round().astype(int))]
            pc = divmod(int(a["cell"][a["parent"][i]]), nc)
            off = np.clip(np.round((pts - np.array(pc, dtype=float)) * 8), -32768, 32767).astype(np.int16)
            ng[j] = len(off)
            geo.append(off)
        out[d] = {"cell": a["cell"][keep].astype(np.int32), "h": a["h"][keep].astype(np.uint8),
                  "parent": par.astype(np.int32), "ng": ng,
                  "geo": np.vstack(geo) if geo else np.zeros((0, 2), np.int16)}
    return out
