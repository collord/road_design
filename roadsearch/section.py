"""Half-bench sidecast cross-section model (brief section 4).

Section plane: x is horizontal distance from the hinge, y is vertical. The
hinge sits at ground level on the centerline. Uphill (x > 0 on the cut side)
the ground is y = s*x and the rock surface is y = s*x - d. Downhill (x > 0 on
the fill side) the ground is y = -s*x.

The core is a Numba function of scalars so the search kernel can call it
directly. `section()` wraps it for Python callers.
"""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass

import numba
import numpy as np

from .params import Params

# Section type codes, shared with the route outputs.
INFEASIBLE = -1
FLAT = 0
HALF_BENCH = 1
FULL_BENCH = 2
HAIRPIN = 3
FILL_CROSSING = 4
SECTION_NAMES = {
    INFEASIBLE: "infeasible",
    FLAT: "flat",
    HALF_BENCH: "half_bench",
    FULL_BENCH: "full_bench",
    HAIRPIN: "hairpin",
    FILL_CROSSING: "fill_crossing",
}

# The subset of Params the section needs, as a Numba-friendly namedtuple.
SectionParams = namedtuple(
    "SectionParams",
    "w c_soil c_rock f_soil f_rock k_soil k_rock b_min s_mask s_flat C_base C_soil C_rock catch_max",
)


def section_params(p: Params, w: float | None = None) -> SectionParams:
    return SectionParams(
        float(p.w if w is None else w), p.c_soil, p.c_rock, p.f_soil, p.f_rock,
        p.k_soil, p.k_rock, p.b_min, p.s_mask, p.s_flat, p.C_base, p.C_soil, p.C_rock,
        p.catch_max,
    )


_BISECT_ITERS = 60
_S_ZERO = 1e-12


@numba.njit(cache=True)
def _shoelace4(x0, y0, x1, y1, x2, y2, x3, y3):
    """Unsigned area of the polygon (x0,y0)-(x1,y1)-(x2,y2)-(x3,y3)."""
    acc = (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3)
    return 0.5 * abs(acc)


@numba.njit(cache=True)
def _shoelace3(x0, y0, x1, y1, x2, y2):
    acc = (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y0 - x0 * y2)
    return 0.5 * abs(acc)


@numba.njit(cache=True)
def _clipped_area(xs, ys, n, xmax):
    """Shoelace area of the polygon (xs[:n], ys[:n]) clipped to x <= xmax
    (one Sutherland-Hodgman pass)."""
    cx = np.empty(2 * n)
    cy = np.empty(2 * n)
    m = 0
    for i in range(n):
        j = (i + 1) % n
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        in_i = xi <= xmax
        in_j = xj <= xmax
        if in_i:
            cx[m] = xi
            cy[m] = yi
            m += 1
        if in_i != in_j:
            t = (xmax - xi) / (xj - xi)
            cx[m] = xmax
            cy[m] = yi + t * (yj - yi)
            m += 1
    acc = 0.0
    for i in range(m):
        j = (i + 1) % m
        acc += cx[i] * cy[j] - cx[j] * cy[i]
    return 0.5 * abs(acc)


@numba.njit(cache=True)
def cut_geometry(a, s, d, c_soil, c_rock, catch_max=math.inf):
    """Cut of bench width `a` on sidehill `s` with vertical soil depth `d`.

    Returns (A_soil, A_rock, catch) where catch is the horizontal distance
    from the hinge to where the backslope meets the ground (x2 in the brief).
    If x2 exceeds catch_max, the cut and its rock part are clipped at
    x = catch_max (as if ended by a vertical face) and catch = catch_max.
    Requires s < c_rock, and s < c_soil wherever d > 0.
    """
    if a <= 0.0:
        return 0.0, 0.0, 0.0
    if s <= _S_ZERO:
        return 0.0, 0.0, min(a, catch_max)
    if a * s <= d:
        # Bench floor stays in soil: one triangle, backslope at c_soil.
        x2 = c_soil * a / (c_soil - s)
        if x2 <= catch_max:
            return _shoelace3(0.0, 0.0, a, 0.0, x2, s * x2), 0.0, x2
        xs = np.array([0.0, a, x2])
        ys = np.array([0.0, 0.0, s * x2])
        return _clipped_area(xs, ys, 3, catch_max), 0.0, catch_max
    # Floor enters rock at xr = d/s. Backslope rises at c_rock to the rock
    # surface at x1, then at c_soil through the soil to ground at x2.
    xr = d / s
    x1 = (c_rock * a - d) / (c_rock - s)
    y1 = c_rock * (x1 - a)
    x2 = x1 + d / (c_soil - s) if d > 0.0 else x1
    y2 = s * x2
    if x2 <= catch_max:
        total = _shoelace4(0.0, 0.0, a, 0.0, x1, y1, x2, y2)
        rock = _shoelace3(xr, 0.0, a, 0.0, x1, y1)
        return max(total - rock, 0.0), rock, x2
    xs = np.array([0.0, a, x1, x2])
    ys = np.array([0.0, 0.0, y1, y2])
    total = _clipped_area(xs, ys, 4, catch_max)
    rock = _clipped_area(np.array([xr, a, x1]), np.array([0.0, 0.0, y1]), 3, catch_max)
    return max(total - rock, 0.0), rock, catch_max


@numba.njit(cache=True)
def _spoil_fill_slope(vs, vr, d, s, f_soil, f_rock, c_soil):
    """Volume-weighted fill slope of the sidecast spoil."""
    tot = vs + vr
    if tot > 0.0:
        return (vs * f_soil + vr * f_rock) / tot
    # No cut yet: the first spoil is soil if there is soil at the hinge.
    return f_soil if (d > 0.0 and s < c_soil) else f_rock


@numba.njit(cache=True)
def _fill_area(b, s, f):
    if b <= 0.0:
        return 0.0
    if s >= f:
        return math.inf
    return 0.5 * b * b * s * f / (f - s)


@numba.njit(cache=True)
def section_core(s, d, sp):
    """Cross-section on sidehill slope s with soil depth d.

    Returns a tuple:
      (kind, a, b, A_soil, A_rock, A_fill, spoil, cut_catch, fill_toe, f, cost)
    Areas are per meter of road (m2 = m3/m). `a` is cut width, `b` fill width,
    `spoil` the loose volume per meter left over (full bench only), `f` the
    spoil fill slope, `cost` the cost per meter. Infeasible sections have
    kind == INFEASIBLE and cost == inf.
    """
    s = abs(s)
    w = sp.w
    if s >= sp.s_mask:
        return (INFEASIBLE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf)
    # Soil cannot stand steeper than its own cut slope.
    if d < 0.0 or s >= sp.c_soil:
        d = 0.0

    if s <= _S_ZERO:
        k = sp.k_soil if d > 0.0 else sp.k_rock
        f = sp.f_soil if d > 0.0 else sp.f_rock
        a = w / (1.0 + math.sqrt(k))  # limit of the balance as s -> 0
        return (FLAT, a, w - a, 0.0, 0.0, 0.0, 0.0, a, w - a, f, sp.C_base)

    full = s >= max(sp.f_soil, sp.f_rock)
    a = w
    if not full:
        # Bisection on r(a) = loose cut - fill(w - a). r(0) < 0 < r(w).
        lo = 0.0
        hi = w
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            As, Ar, _c = cut_geometry(mid, s, d, sp.c_soil, sp.c_rock, sp.catch_max)
            vs = sp.k_soil * As
            vr = sp.k_rock * Ar
            f = _spoil_fill_slope(vs, vr, d, s, sp.f_soil, sp.f_rock, sp.c_soil)
            r = vs + vr - _fill_area(w - mid, s, f)
            if r < 0.0:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)

    As, Ar, catch = cut_geometry(a, s, d, sp.c_soil, sp.c_rock, sp.catch_max)
    vs = sp.k_soil * As
    vr = sp.k_rock * Ar
    f = _spoil_fill_slope(vs, vr, d, s, sp.f_soil, sp.f_rock, sp.c_soil)
    b = w - a
    if not full and (s >= f or b < sp.b_min):
        full = True

    if full:
        a = w
        As, Ar, catch = cut_geometry(a, s, d, sp.c_soil, sp.c_rock, sp.catch_max)
        vs = sp.k_soil * As
        vr = sp.k_rock * Ar
        f = _spoil_fill_slope(vs, vr, d, s, sp.f_soil, sp.f_rock, sp.c_soil)
        cost = sp.C_base + sp.C_soil * As + sp.C_rock * Ar
        return (FULL_BENCH, w, 0.0, As, Ar, 0.0, vs + vr, catch, 0.0, f, cost)

    A_fill = _fill_area(b, s, f)
    toe = f * b / (f - s)
    kind = FLAT if s < sp.s_flat else HALF_BENCH
    cost = sp.C_base + sp.C_soil * As + sp.C_rock * Ar
    return (kind, a, b, As, Ar, A_fill, 0.0, catch, toe, f, cost)


@numba.njit(cache=True)
def section_arrays(s, d, sp):
    """Vectorized section_core. Returns an (n, 11) array of its outputs."""
    n = s.shape[0]
    out = np.empty((n, 11))
    for i in range(n):
        r = section_core(s[i], d[i], sp)
        out[i, 0] = r[0]
        out[i, 1] = r[1]
        out[i, 2] = r[2]
        out[i, 3] = r[3]
        out[i, 4] = r[4]
        out[i, 5] = r[5]
        out[i, 6] = r[6]
        out[i, 7] = r[7]
        out[i, 8] = r[8]
        out[i, 9] = r[9]
        out[i, 10] = r[10]
    return out


@dataclass(frozen=True)
class Section:
    kind: str
    cut_width: float
    fill_width: float
    A_soil: float
    A_rock: float
    A_fill: float
    spoil: float
    cut_catch: float
    fill_toe: float
    fill_slope: float
    cost: float

    @property
    def feasible(self) -> bool:
        return self.kind != "infeasible"


def section(s: float, d: float, params: Params | SectionParams, w: float | None = None) -> Section:
    """Pure Python entry point: section on sidehill slope s with soil depth d."""
    sp = params if isinstance(params, SectionParams) else section_params(params, w)
    r = section_core(float(s), float(d), sp)
    return Section(SECTION_NAMES[int(r[0])], *(float(v) for v in r[1:]))


def soil_depth(S, p: Params):
    """Vertical soil depth as a function of terrain slope magnitude S."""
    S = np.asarray(S, dtype=float)
    d = p.d0 * np.exp(-S / p.S0)
    return np.where(S >= p.c_soil, 0.0, d)


def max_catch_at_mask(p: Params) -> float:
    """Largest cut catch distance that can occur at the mask threshold slope.

    At the threshold the sidehill slope is s_mask and the terrain slope is at
    least that, so soil depth is at most soil_depth(s_mask). Catch grows with
    cut width, so the worst case is a full bench. Never more than catch_max.
    """
    s = p.s_mask
    worst = 0.0
    for d in (0.0, float(soil_depth(s, p))):
        _, _, catch = cut_geometry(p.w, s, d if s < p.c_soil else 0.0, p.c_soil, p.c_rock, p.catch_max)
        worst = max(worst, catch)
    return worst
