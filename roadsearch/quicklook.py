"""Quick-look PNG for sanity checking (brief section 10)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap  # noqa: E402

from .exclusion import MANUAL, SLOPE  # noqa: E402

ROUTE_COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3", "#4daf4a", "#a65628", "#f781bf"]


# Rasters are drawn at most this many cells across: each panel is about 675 pixels wide, so
# finer detail would not show, and full-size layers on a big grid cost gigabytes.
MAX_CELLS_ACROSS = 1000


def quicklook(path, terrain, fields, routes, prob, p, title: str) -> None:
    g = terrain.grid
    h, w = terrain.z_smooth.shape
    k = max(1, -(-max(h, w) // MAX_CELLS_ACROSS))        # every k-th cell
    sub = lambda a: np.asarray(a)[::k, ::k]  # noqa: E731
    ext = dict(extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="nearest")   # full-grid (col, row) units
    rr, cc = np.arange(0, h, k), np.arange(0, w, k)
    zs = sub(terrain.z_smooth)
    z = np.where(np.isnan(zs), np.nanmin(zs), zs)
    hs = LightSource(315, 45).hillshade(z, dx=g.res * k, dy=g.res * k)
    figw = 15
    fig, axes = plt.subplots(1, 2, figsize=(figw, figw / 2 * h / w + 1.2))
    ax0, ax1 = axes
    for ax in axes:
        ax.imshow(hs, cmap="gray", **ext)
    code = sub(terrain.code)
    ax0.imshow(np.where((code & SLOPE) > 0, 1.0, np.nan), cmap=ListedColormap(["#d7301f"]), alpha=0.45, **ext)
    ax0.imshow(np.where((code & MANUAL) > 0, 1.0, np.nan), cmap=ListedColormap(["#6a51a3"]), alpha=0.6, **ext)
    if not np.all(terrain.clip):
        ax0.contour(cc, rr, sub(terrain.clip).astype(float), levels=[0.5], colors="k", linewidths=0.8,
                    linestyles="--")
    ax0.contour(cc, rr, sub(terrain.channel).astype(float), levels=[0.5], colors="#41b6c4", linewidths=0.4)
    F = sub(fields.F)
    F = np.where(np.isfinite(F), F, np.nan)
    vmax = (1 + p.eps_alt) * fields.F_min * 1.5
    im = ax1.imshow(F, cmap="viridis_r", vmin=fields.F_min, vmax=vmax, alpha=0.8, **ext)
    cb = fig.colorbar(im, ax=ax1, shrink=0.7)
    cb.set_label("corridor field F (cheapest route through cell)")
    ax1.contour(cc, rr, F, levels=[(1 + p.eps_alt) * fields.F_min], colors="w", linewidths=0.8)
    for rank, r in reversed(list(enumerate(routes, start=1))):
        from .search import path_xy

        pts = path_xy(prob, r.edges, terrain.start_rc)
        col = ROUTE_COLORS[(rank - 1) % len(ROUTE_COLORS)]
        lw = 2.2 if rank == 1 else 1.3
        for ax in axes:
            ax.plot(pts[:, 1], pts[:, 0], color=col, lw=lw,
                    label=f"{rank}: {r.cost.total:,.0f}" if ax is ax0 else None)
        for s in r.segments:
            if s.section_type == "hairpin":
                m = s.pts_rc[len(s.pts_rc) // 2]
                ax0.plot(m[1], m[0], "o", ms=3, mfc="none", mec=col)
    for ax in axes:
        ax.plot(terrain.start_rc[1], terrain.start_rc[0], "*", ms=14, mfc="w", mec="k")
        ax.plot(terrain.goal_rc[1], terrain.goal_rc[0], "o", ms=9, mfc="w", mec="k")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    ax0.set_title("slope mask (red), manual zones (purple), channels (cyan), clip (dashed)\n"
                  "routes by rank with true cost; circles = hairpins", fontsize=9)
    ax1.set_title(f"corridor field; white contour = (1 + eps_alt) x best = {(1 + p.eps_alt) * fields.F_min:,.0f}",
                  fontsize=9)
    ax0.legend(loc="lower left", fontsize=8, title="rank: true cost", title_fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
