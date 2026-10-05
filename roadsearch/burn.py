"""Burn a route into the DEM: design surface, cut/fill depth and cut material.

Every cell near the route takes its nearest point on the centerline (the
section hinge) and the design elevation there, z_road. With d its horizontal
distance outside the road edge (0 on the road):

    cut side  (ground above road): new = min(ground, z_road + backslope(d))
    fill side (ground below road): new = max(ground, z_road - f * d)

so the road surface is level across its width, cut faces rise from the
uphill edge and fill slopes fall from the downhill edge until they meet the
ground, the geometry the section model costs. The backslope rises at c_rock
up to the rock surface (ground minus soil depth) and at c_soil above it, and
is clipped at catch_max.

Ordinary sections (flat, half_bench, full_bench) are a half bench about the
hinge: the segment's cut width lies uphill of the centerline and its fill
width downhill (a full bench has no fill). The hinge sits on the ground, so
z_road is the fine-grid ground at the centerline. Hairpins, bends and fill
crossings are symmetric about the centerline and keep their own design
profile, adjusted linearly so their ends meet the ground of the neighboring
sections. Half-bench fill slopes blend f_soil and f_rock by each segment's
spoil composition, as the section model does; macros use f_soil, crossings
f_rock.

Spoil sidecast from full-bench cuts is not placed (its volume is reported).
Where two non-adjacent parts of the route claim neighboring cells with a
step of more than 1 m, the cells are flagged as a wall.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .params import Params, load_params
from .preprocess import gradient, smooth_dem
from .section import soil_depth
from .terrain import Grid, load_grid, write_raster

ORDINARY = ("flat", "half_bench", "full_bench")
MATERIAL = {"none": 0, "soil": 1, "rock": 2, "fill": 3}


def _bil(z: np.ndarray, grid: Grid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = grid.transform
    fc = (x - t.c) / t.a - 0.5
    fr = (y - t.f) / t.e - 0.5
    nr, nc = z.shape
    r0 = np.clip(np.floor(fr).astype(int), 0, nr - 2)
    c0 = np.clip(np.floor(fc).astype(int), 0, nc - 2)
    wr, wc = np.clip(fr - r0, 0, 1), np.clip(fc - c0, 0, 1)
    return ((1 - wr) * ((1 - wc) * z[r0, c0] + wc * z[r0, c0 + 1])
            + wr * ((1 - wc) * z[r0 + 1, c0] + wc * z[r0 + 1, c0 + 1]))


def spoil_fill_slope(s, p: Params) -> float:
    """Fill slope of a segment's sidecast spoil, blended by its loose soil and
    rock volumes as the section model does. The fill area is very sensitive to
    it near the limit (f / (f - s)), so the soil slope alone would overstate fill."""
    vs = p.k_soil * float(s.get("cut_soil_m3", 0.0) or 0.0)
    vr = p.k_rock * float(s.get("cut_rock_m3", 0.0) or 0.0)
    return (vs * p.f_soil + vr * p.f_rock) / (vs + vr) if vs + vr > 0 else p.f_soil


def centerline_samples(segs: gpd.GeoDataFrame, grid: Grid, ground: np.ndarray, p: Params, step: float) -> dict:
    """Points every <= step meters along the route, with what each needs:
    design z, uphill and downhill road extent, fill slope, segment number."""
    xs, ys, zs, up, down, fslope, seq, macro = [], [], [], [], [], [], [], []
    for _, s in segs.sort_values("seq").iterrows():
        xyz = np.asarray(s.geometry.coords, dtype=float)
        d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xyz[:, :2], axis=0).T))])
        n = max(2, int(math.ceil(d[-1] / step)) + 1)
        u = np.linspace(0.0, d[-1], n)
        x, y = np.interp(u, d, xyz[:, 0]), np.interp(u, d, xyz[:, 1])
        if s.section_type in ORDINARY:
            z = _bil(ground, grid, x, y)                       # the hinge sits on the ground
            a = float(s.cut_width) if s.cut_width == s.cut_width else p.w / 2
            b = float(s.fill_width) if s.fill_width == s.fill_width else p.w / 2
            f = spoil_fill_slope(s, p)
            is_macro = False
        else:
            z = np.interp(u, d, xyz[:, 2])
            # Shift the design profile so its ends meet the fine-grid ground.
            g0 = float(_bil(ground, grid, x[:1], y[:1])[0])
            g1 = float(_bil(ground, grid, x[-1:], y[-1:])[0])
            z = z + (g0 - z[0]) + ((g1 - z[-1]) - (g0 - z[0])) * u / max(d[-1], 1e-9)
            width = p.w_hp if s.section_type == "hairpin" else p.w
            a = b = width / 2
            f = p.f_rock if s.section_type == "fill_crossing" else p.f_soil
            is_macro = True
        xs.append(x), ys.append(y), zs.append(z)
        up.append(np.full(n, a)), down.append(np.full(n, b)), fslope.append(np.full(n, f))
        seq.append(np.full(n, int(s.seq))), macro.append(np.full(n, is_macro))
    cat = np.concatenate
    return {"x": cat(xs), "y": cat(ys), "z": cat(zs), "up": cat(up), "down": cat(down),
            "f": cat(fslope), "seq": cat(seq), "macro": cat(macro)}


def burn_surface(grid: Grid, ground: np.ndarray, soil: np.ndarray, cl: dict, p: Params,
                 buffer: float) -> dict:
    """Design surface and its by-products on `grid` (cells beyond `buffer` m of the
    centerline keep the ground)."""
    nr, nc = ground.shape
    t = grid.transform
    cols, rows = np.meshgrid(np.arange(nc), np.arange(nr))
    X = t.c + (cols + 0.5) * t.a
    Y = t.f + (rows + 0.5) * t.e
    tree = cKDTree(np.column_stack([cl["x"], cl["y"]]))
    D, idx = tree.query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=buffer)
    near = np.isfinite(D) & np.isfinite(ground.ravel())
    k = idx[near]
    D = D[near]
    zg = ground.ravel()[near]
    sd = np.nan_to_num(soil.ravel()[near])
    zr = cl["z"][k]
    h = zg - zr
    cut = h > 0
    half = np.where(cut, cl["up"][k], cl["down"][k])
    d = np.maximum(D - half, 0.0)

    # Cut: rock backslope up to the rock surface, soil backslope above it.
    rock_top = h - sd                                     # rock surface above road level
    d_rock = np.where(rock_top > 0, rock_top / p.c_rock, 0.0)
    rise = np.where(d <= d_rock, p.c_rock * d, np.maximum(rock_top, 0.0) + p.c_soil * (d - d_rock))
    new = zg.copy()
    in_cut = cut & (d <= p.catch_max)
    new[in_cut] = np.minimum(zg[in_cut], zr[in_cut] + rise[in_cut])
    # Fill: down from the road edge at the fill slope (none beyond a full bench's hinge).
    fill_side = ~cut & (d <= p.catch_max) & ((d == 0) | (cl["down"][k] > 0) | cl["macro"][k])
    new[fill_side] = np.maximum(zg[fill_side], zr[fill_side] - cl["f"][k][fill_side] * d[fill_side])

    out = np.array(ground, dtype=float, copy=True).ravel()
    out[near] = new
    design = out.reshape(nr, nc)
    road = np.zeros(nr * nc, dtype=bool)
    road[np.flatnonzero(near)[D <= half + 1e-9]] = True
    seg = np.full(nr * nc, -1, dtype=np.int64)
    seg[near] = cl["seq"][k]
    seg = seg.reshape(nr, nc)

    depth = np.nan_to_num(design - ground)                # + fill, - cut; 0 where there is no ground
    cut_d = np.maximum(-depth, 0.0)
    soil_part = np.minimum(cut_d, np.nan_to_num(soil))
    rock_part = cut_d - soil_part
    material = np.zeros((nr, nc), dtype=np.uint8)
    material[cut_d > 0.01] = MATERIAL["soil"]
    material[rock_part > 0.01] = MATERIAL["rock"]
    material[depth > 0.01] = MATERIAL["fill"]

    # Walls: neighboring cells claimed by parts of the route far apart along it,
    # with a step between them.
    wall = np.zeros((nr, nc), dtype=bool)
    for dr, dc in ((0, 1), (1, 0)):
        a_ = seg[:nr - dr, :nc - dc]
        b_ = seg[dr:, dc:]
        step = np.abs(design[:nr - dr, :nc - dc] - design[dr:, dc:])
        clash = (a_ >= 0) & (b_ >= 0) & (np.abs(a_ - b_) > 2) & (step > 1.0)
        wall[:nr - dr, :nc - dc] |= clash
        wall[dr:, dc:] |= clash

    cell = grid.cell_area
    return {
        "design": design, "depth": depth, "material": material, "wall": wall, "seg": seg,
        "road": road.reshape(nr, nc),
        "volumes": {
            "cut_soil_m3": float(soil_part.sum() * cell), "cut_rock_m3": float(rock_part.sum() * cell),
            "fill_m3": float(np.maximum(depth, 0).sum() * cell),
            "road_area_m2": float(road.sum() * cell), "wall_cells": int(wall.sum()),
        },
    }


def burn_route(run_dir: str | Path, out_root: str | Path, rank: int = 1, res: float = 2.0,
               buffer: float = 150.0, dest: str | Path | None = None) -> dict:
    """Burn the route of `rank` from a finished run; writes GeoTIFFs and a summary."""
    run_dir, out_root = Path(run_dir), Path(out_root)
    rec = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    run_id = rec["run_id"]
    p = load_params(overrides=rec["params"])
    segs = gpd.read_parquet(out_root / "routes.parquet")
    segs = segs[(segs["run_id"] == run_id) & (segs["rank"] == rank)]
    if segs.empty:
        raise ValueError(f"run {run_id} has no route of rank {rank}")
    minx, miny, maxx, maxy = segs.total_bounds
    pad = buffer + 2 * res
    grid, _ = load_grid(rec["inputs"]["dem"], p, res, (minx - pad, miny - pad, maxx + pad, maxy + pad))
    ground = grid.z
    zs = smooth_dem(ground, p.sigma_m / res)
    gx, gy = gradient(zs, grid)
    S = np.hypot(gx, gy)
    soil = np.where(np.isfinite(S), soil_depth(np.nan_to_num(S, nan=np.inf), p), np.nan)
    cl = centerline_samples(segs, grid, ground, p, step=res / 2)
    b = burn_surface(grid, ground, soil, cl, p, buffer)

    model = {
        "cut_soil_m3": float(segs["cut_soil_m3"].sum()), "cut_rock_m3": float(segs["cut_rock_m3"].sum()),
        "fill_m3": float(segs["fill_m3"].sum()), "spoil_sidecast_m3": float(segs["spoil_m3"].sum()),
    }
    by_type = (segs.groupby("section_type")[["length", "cut_soil_m3", "cut_rock_m3", "fill_m3"]].sum()
               .round(1).to_dict(orient="index"))
    kind = dict(zip(segs["seq"].astype(int), segs["section_type"]))
    cell = grid.cell_area
    burned_by_type = {}
    for st in sorted(set(kind.values())):
        m = np.isin(b["seg"], [q for q, v in kind.items() if v == st])
        dep = b["depth"][m]
        burned_by_type[st] = {"cut_m3": round(float(np.maximum(-dep, 0).sum() * cell), 1),
                              "fill_m3": round(float(np.maximum(dep, 0).sum() * cell), 1)}
    dest = Path(dest) if dest else run_dir / "burn" / f"rank{rank}"
    dest.mkdir(parents=True, exist_ok=True)
    write_raster(dest / "dem_existing.tif", ground, grid)
    write_raster(dest / "dem_design.tif", b["design"], grid)
    write_raster(dest / "cut_fill_depth.tif", np.where(np.isfinite(ground), b["depth"], np.nan), grid)
    write_raster(dest / "cut_material.tif", b["material"], grid, nodata=255)
    write_raster(dest / "road_surface.tif", b["road"], grid, nodata=255)
    write_raster(dest / "wall_cells.tif", b["wall"], grid, nodata=255)
    summary = {"run_id": run_id, "rank": rank, "res_m": res, "buffer_m": buffer,
               "burned": b["volumes"], "section_model": model, "section_model_by_type": by_type,
               "burned_by_type": burned_by_type,
               "material_codes": MATERIAL,
               "notes": ["fill slopes blended by each segment's spoil (crossings f_rock); full-bench spoil not placed",
                         "ordinary sections: hinge on the fine-grid ground; macros: design profile shifted to meet it"]}
    (dest / "burn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    quicklook(dest / "burn_quicklook.png", grid, ground, b, segs)
    return summary


def quicklook(path: Path, grid: Grid, ground: np.ndarray, b: dict, segs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, TwoSlopeNorm

    ls = LightSource(315, 45)
    fill = lambda z: np.where(np.isfinite(z), z, np.nanmin(z))  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(ls.hillshade(fill(b["design"]), dx=grid.res, dy=grid.res), cmap="gray")
    ax[0].set_title("Design surface (route burned in), hillshade")
    ax[1].imshow(ls.hillshade(fill(ground), dx=grid.res, dy=grid.res), cmap="gray")
    dep = np.where(np.abs(b["depth"]) > 0.05, b["depth"], np.nan)
    lim = max(1.0, float(np.nanpercentile(np.abs(dep), 99))) if np.isfinite(dep).any() else 1.0
    im = ax[1].imshow(dep, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), alpha=0.85)
    ax[1].imshow(np.where(b["wall"], 1.0, np.nan), cmap="autumn", alpha=1)
    fig.colorbar(im, ax=ax[1], shrink=0.6, label="design - existing (m): red cut, blue fill")
    ax[1].set_title("Cut and fill depth; walls in yellow")
    for a in ax:
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
