"""State-lattice motion primitives (brief section 5.1).

Headings are the lattice directions of small integer vectors (for 16 bins:
the axis, diagonal and knight-move directions), so straight motions land
exactly on nodes. A turning primitive joins two lattice poses with a
straight segment and a circular arc (or arc then straight): the arc is the
largest circle tangent to both heading lines, so the curve is determined
by its end poses alone, and the reverse of a primitive is the same curve.
Every arc radius is at least R_min.

Coordinates in this module are in cells with x east and y north. Grid rows
increase southward, so a displacement (X, Y) is (drow, dcol) = (-Y, X).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HEADINGS = {
    8: [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
    16: [(1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
         (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1)],
}
# 32 bins: per octant 0, atan(1/4), atan(1/2), atan(3/4), rotated by quarter turns.
_Q32 = [(1, 0), (4, 1), (2, 1), (4, 3), (1, 1), (3, 4), (1, 2), (1, 4)]
HEADINGS[32] = [v for k in range(4) for v in
                [(x, y) if k == 0 else (-y, x) if k == 1 else (-x, -y) if k == 2 else (y, -x)
                 for x, y in _Q32]]
SAMPLE_SPACING = 0.5  # cells; the brief requires no more than half a cell


@dataclass(frozen=True)
class Primitive:
    h0: int
    h1: int
    X: int                 # end displacement, cells east
    Y: int                 # end displacement, cells north
    length: float          # arc length in cells
    radius: float          # arc radius in cells (inf for straight)
    pts: np.ndarray        # (n, 2) sample positions (x, y), cells, midpoints of n equal pieces
    tan: np.ndarray        # (n, 2) unit tangents at the samples

    @property
    def key(self):
        return (self.h0, self.X, self.Y, self.h1)


def heading_vectors(n_theta: int) -> np.ndarray:
    if n_theta not in HEADINGS:
        raise ValueError(f"N_theta must be one of {sorted(HEADINGS)}")
    return np.array(HEADINGS[n_theta], dtype=np.int64)


def contour_slope_limit(n_theta: int, g_max: float) -> float:
    """Steepest terrain slope on which some lattice heading has grade <= g_max
    for every aspect: g_max / sin(half the widest gap between headings)."""
    ang = np.sort(heading_angles(n_theta) % (2 * np.pi))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(g_max / np.sin(gaps.max() / 2))


def heading_angles(n_theta: int) -> np.ndarray:
    v = heading_vectors(n_theta)
    return np.arctan2(v[:, 1], v[:, 0])


def _unit(a):
    return np.array([math.cos(a), math.sin(a)])


def _turn_geometry(th0, th1, X, Y):
    """Tangent-arc construction between poses (0,0,th0) and (X,Y,th1).

    Returns (a, b, R, dth) where the heading lines meet at P = a*u0 and
    E = P + b*u1, and R = min(a, b) / tan(|dth|/2); or None if invalid.
    """
    dth = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
    if abs(dth) < 1e-12 or abs(abs(dth) - math.pi) < 1e-12:
        return None
    u0, u1 = _unit(th0), _unit(th1)
    det = u0[0] * u1[1] - u0[1] * u1[0]
    a = (X * u1[1] - Y * u1[0]) / det
    b = (u0[0] * Y - u0[1] * X) / det
    if a <= 1e-12 or b <= 1e-12:
        return None
    R = min(a, b) / math.tan(abs(dth) / 2)
    return a, b, R, dth


def _sample_turn(th0, a, b, R, dth, n):
    """Points and tangents at the midpoints of n equal arc-length pieces."""
    T = R * math.tan(abs(dth) / 2)
    L1, La, L2 = a - T, R * abs(dth), b - T
    L = L1 + La + L2
    u0 = _unit(th0)
    sgn = 1.0 if dth > 0 else -1.0
    left = np.array([-u0[1], u0[0]]) * sgn
    p1 = L1 * u0
    center = p1 + R * left
    pts, tans = np.empty((n, 2)), np.empty((n, 2))
    for k in range(n):
        s = (k + 0.5) * L / n
        if s <= L1:
            pts[k], tans[k] = s * u0, u0
        elif s <= L1 + La:
            # Rotate (p1 - center) by phi about the center.
            phi = sgn * (s - L1) / R
            th = th0 + phi
            rel = p1 - center
            c, s_ = math.cos(phi), math.sin(phi)
            pts[k] = center + np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
            tans[k] = _unit(th)
        else:
            th1 = th0 + dth
            e2 = center + np.array([math.cos(dth) * (p1 - center)[0] - math.sin(dth) * (p1 - center)[1],
                                    math.sin(dth) * (p1 - center)[0] + math.cos(dth) * (p1 - center)[1]])
            pts[k], tans[k] = e2 + (s - L1 - La) * _unit(th1), _unit(th1)
    return pts, tans, L


def _straight(h, v, th, h1=None):
    """Straight move along v (direction th) from heading h to heading h1 (default h)."""
    L = math.hypot(*v)
    n = max(1, math.ceil(L / SAMPLE_SPACING - 1e-9))
    s = (np.arange(n) + 0.5) * L / n
    u = _unit(th)
    return Primitive(h, h if h1 is None else h1, int(v[0]), int(v[1]), L, math.inf, s[:, None] * u,
                     np.tile(u, (n, 1)))


def _turn(h0, h1, th0, th1, X, Y):
    geo = _turn_geometry(th0, th1, X, Y)
    if geo is None:
        return None
    a, b, R, dth = geo
    T = R * math.tan(abs(dth) / 2)
    L = (a - T) + R * abs(dth) + (b - T)
    n = max(1, math.ceil(L / SAMPLE_SPACING - 1e-9))
    pts, tans, L = _sample_turn(th0, a, b, R, dth, n)
    return Primitive(h0, h1, X, Y, L, R, pts, tans)


def heading_bins(n_theta: int) -> np.ndarray:
    """(n_theta, 2) angular bounds [lo, hi) of each heading's bin: halfway to
    the neighboring lattice headings."""
    ths = heading_angles(n_theta)
    out = np.empty((n_theta, 2))
    for h in range(n_theta):
        prev = (ths[h] - ths[(h - 1) % n_theta]) % (2 * math.pi)
        nxt = (ths[(h + 1) % n_theta] - ths[h]) % (2 * math.pi)
        out[h] = (ths[h] - prev / 2, ths[h] + nxt / 2)
    return out


def binned_moves(n_theta: int, max_cells: int, kink_max_deg: float = 10.0) -> list[tuple[int, int, int, int]]:
    """Straight moves (h0, X, Y, h1) to lattice points with coprime coordinates
    and max(|X|, |Y|) <= max_cells, binned to lattice headings.

    A move in direction a may start in heading h0 and end in heading h1 when
    both are within kink_max_deg of a and at least one of them is the heading
    nearest to a. Moves along an exact lattice heading are included only when
    they change heading (the plain straight already exists). So a road can
    hold directions between the lattice headings, e.g. climb at close to
    g_max on a uniform slope, and pass from one heading class to the next,
    at the price of a kink of at most kink_max_deg where consecutive moves
    differ in direction (Stueckelberger et al. 2007 round links to 16
    classes with up to 18 degrees). The set is closed under reversal.
    """
    if max_cells <= 0:
        return []
    ths = heading_angles(n_theta)
    kmax = math.radians(kink_max_deg)
    out = []
    for X in range(-max_cells, max_cells + 1):
        for Y in range(-max_cells, max_cells + 1):
            if (X, Y) == (0, 0) or math.gcd(X, Y) != 1:
                continue
            a = math.atan2(Y, X)
            d = (a - ths + math.pi) % (2 * math.pi) - math.pi
            near = int(np.argmin(np.abs(d)))
            ok = [h for h in range(n_theta) if abs(d[h]) <= kmax + 1e-12]
            exact = abs(d[near]) < 1e-12
            for h0 in ok:
                for h1 in ok:
                    if h0 != near and h1 != near:
                        continue
                    if exact and h0 == h1:
                        continue
                    out.append((h0, X, Y, h1))
    return out


def build_primitives(n_theta: int, R_min_cells: float, turns=(1, 2), move_max_cells: int = 0,
                     kink_max_deg: float = 10.0) -> list[Primitive]:
    """Straight plus the shortest valid left/right turn for each heading change
    in `turns`, plus binned straight moves (see binned_moves), closed under
    reversal."""
    vecs = heading_vectors(n_theta)
    ths = heading_angles(n_theta)
    M = int(math.ceil(3 * R_min_cells)) + 4
    prims: dict[tuple, Primitive] = {}
    for h0 in range(n_theta):
        p = _straight(h0, vecs[h0], ths[h0])
        prims[p.key] = p
        for k in turns:
            for sgn in (1, -1):
                h1 = (h0 + sgn * k) % n_theta
                best, best_key = None, None
                for X in range(-M, M + 1):
                    for Y in range(-M, M + 1):
                        geo = _turn_geometry(ths[h0], ths[h1], X, Y)
                        if geo is None or geo[2] < R_min_cells - 1e-9:
                            continue
                        a, b, R, dth = geo
                        T = R * math.tan(abs(dth) / 2)
                        L = (a - T) + R * abs(dth) + (b - T)
                        u0 = _unit(ths[h0])
                        fwd = X * u0[0] + Y * u0[1]
                        lat = abs(X * u0[1] - Y * u0[0])
                        key = (round(L, 9), round(abs(a - b), 9), round(fwd, 9), round(lat, 9))
                        if best_key is None or key < best_key:
                            best, best_key = (X, Y), key
                if best is None:
                    raise ValueError(f"no turn primitive for heading {h0} -> {h1} within {M} cells")
                p = _turn(h0, h1, ths[h0], ths[h1], *best)
                prims[p.key] = p
    for h0, X, Y, h1 in binned_moves(n_theta, move_max_cells, kink_max_deg):
        p = _straight(h0, (X, Y), math.atan2(Y, X), h1)
        prims[p.key] = p
    # Close under reversal: (h0, X, Y, h1) reversed is (h1+pi, -X, -Y, h0+pi).
    half = n_theta // 2
    for p in list(prims.values()):
        rk = ((p.h1 + half) % n_theta, -p.X, -p.Y, (p.h0 + half) % n_theta)
        if rk not in prims:
            if math.isinf(p.radius):
                prims[rk] = _straight(rk[0], (rk[1], rk[2]), math.atan2(rk[2], rk[1]), rk[3])
            else:
                prims[rk] = _turn(rk[0], rk[3], ths[rk[0]], ths[rk[3]], rk[1], rk[2])
    return [prims[k] for k in sorted(prims)]


BEND_ANGLES = (90.0, 135.0)   # bend turns in degrees, rounded to whole heading bins


def bend_steps(n_theta: int, angles=BEND_ANGLES) -> tuple[int, ...]:
    """Bend turns in heading bins (8 and 12 for 32 headings, 4 and 6 for 16)."""
    return tuple(int(round(a / 360.0 * n_theta)) for a in angles)


@dataclass
class BendGeometry:
    """Left bends: straight L1, arc of radius R through dth, straight L2 (cells).

    Arrays are indexed (heading, slot); slot j turns bend_steps(n_theta)[j] bins left.
    """

    X: np.ndarray        # end displacement east (cells)
    Y: np.ndarray        # end displacement north (cells)
    h1: np.ndarray       # end heading
    L1: np.ndarray
    L2: np.ndarray
    dth: np.ndarray      # signed turn (radians, > 0 left)
    R: float             # arc radius (cells)


def build_bends(n_theta: int, R_cells: float, steps=None) -> BendGeometry:
    """Shortest left bend for each heading and turn, ending on a lattice node
    with the exact end heading: straight, arc of radius R_cells, straight."""
    steps = bend_steps(n_theta) if steps is None else steps
    if any(not 0 < k < n_theta // 2 for k in steps):
        raise ValueError(f"bend turns must be between 0 and 180 degrees, got {steps} of {n_theta} bins")
    vecs = heading_vectors(n_theta)
    ths = heading_angles(n_theta)
    ns = len(steps)
    X = np.zeros((n_theta, ns), dtype=np.int64)
    Y = np.zeros((n_theta, ns), dtype=np.int64)
    H1 = np.zeros((n_theta, ns), dtype=np.int64)
    L1 = np.zeros((n_theta, ns))
    L2 = np.zeros((n_theta, ns))
    DT = np.zeros((n_theta, ns))
    for h0 in range(n_theta):
        for j, k in enumerate(steps):
            h1 = (h0 + k) % n_theta
            dth = (ths[h1] - ths[h0] + math.pi) % (2 * math.pi) - math.pi
            T = R_cells * math.tan(abs(dth) / 2)
            M = min(int(math.ceil(3 * (R_cells + T))) + 4, 200)
            u0 = _unit(ths[h0])
            best, best_key = None, None
            for x in range(-M, M + 1):
                for y in range(-M, M + 1):
                    geo = _turn_geometry(ths[h0], ths[h1], x, y)
                    if geo is None:
                        continue
                    a, b, _, d = geo
                    if a < T - 1e-9 or b < T - 1e-9 or d * dth <= 0:
                        continue
                    length = (a - T) + R_cells * abs(dth) + (b - T)
                    fwd = x * u0[0] + y * u0[1]
                    lat = abs(x * u0[1] - y * u0[0])
                    key = (round(length, 9), round(abs(a - b), 9), round(fwd, 9), round(lat, 9))
                    if best_key is None or key < best_key:
                        best, best_key = (x, y, a - T, b - T), key
            if best is None:
                raise ValueError(f"no bend for heading {h0} turning {k} bins within {M} cells")
            X[h0, j], Y[h0, j], L1[h0, j], L2[h0, j] = best
            L1[h0, j] = 0.0 if L1[h0, j] < 1e-9 else L1[h0, j]
            L2[h0, j] = 0.0 if L2[h0, j] < 1e-9 else L2[h0, j]
            H1[h0, j] = h1
            DT[h0, j] = dth
    return BendGeometry(X, Y, H1, L1, L2, DT, float(R_cells))


@dataclass
class LatticeArrays:
    """Flat arrays for the Numba kernel."""

    n_theta: int
    prims: list[Primitive]
    h0: np.ndarray
    h1: np.ndarray
    dr: np.ndarray
    dc: np.ndarray
    length: np.ndarray       # cells
    rev: np.ndarray
    s0: np.ndarray           # first sample index
    ns: np.ndarray           # sample count
    sdr: np.ndarray          # sample row offset (cells)
    sdc: np.ndarray          # sample col offset (cells)
    tx: np.ndarray           # sample unit tangent, east
    ty: np.ndarray           # sample unit tangent, north
    by_heading: np.ndarray   # (n_theta, max_out) action indices, -1 padded
    kind: np.ndarray         # ACTION_PRIM, ACTION_HP_*, ACTION_FILL or ACTION_BEND_*
    slot: np.ndarray         # hairpin leg-spacing slot or bend slot (0 for other actions)
    turn: np.ndarray | None = None   # change in heading, radians (|h1 - h0|; pi for a hairpin)

    @property
    def n_prims(self) -> int:
        return len(self.prims)


# Action kinds. Hairpins are macro-actions appended after the primitives: a
# left/right pair per heading and leg-spacing slot. Fill crossings follow:
# straight runs of m >= 2 lattice steps. Then bends: left bends per heading
# and slot, followed by right bends, each defined as the reverse of a left
# bend (so the canonical, lower-numbered direction of every bend is a left
# bend and one table of left bends prices both).
ACTION_PRIM, ACTION_HP_LEFT, ACTION_HP_RIGHT, ACTION_FILL, ACTION_BEND_L, ACTION_BEND_R = 0, 1, 2, 3, 4, 5


def lattice_arrays(n_theta: int, R_min_cells: float, hp_slots: np.ndarray | None = None,
                   cross_max_cells: float = 0.0, bends: BendGeometry | None = None,
                   move_max_cells: int = 0, kink_max_deg: float = 10.0) -> LatticeArrays:
    """Primitive arrays, plus left/right hairpin actions for each leg-spacing
    slot if hp_slots ((n_theta, n_slots) lateral lattice steps, 0 = unused)
    is given, plus fill-crossing actions (straight runs of m >= 2 heading
    steps, at most cross_max_cells long)."""
    prims = build_primitives(n_theta, R_min_cells, move_max_cells=move_max_cells, kink_max_deg=kink_max_deg)
    idx = {p.key: i for i, p in enumerate(prims)}
    half = n_theta // 2
    rev = np.array([idx[((p.h1 + half) % n_theta, -p.X, -p.Y, (p.h0 + half) % n_theta)] for p in prims])
    ns = np.array([len(p.pts) for p in prims], dtype=np.int64)
    s0 = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64)
    pts = np.concatenate([p.pts for p in prims])
    tans = np.concatenate([p.tan for p in prims])
    out = [[i for i, p in enumerate(prims) if p.h0 == h] for h in range(n_theta)]
    h0 = [p.h0 for p in prims]
    h1 = [p.h1 for p in prims]
    dr = [-p.Y for p in prims]
    dc = [p.X for p in prims]
    length = [p.length for p in prims]
    rev = rev.tolist()
    kind = [ACTION_PRIM] * len(prims)
    s0 = s0.tolist()
    ns = ns.tolist()
    slot = [0] * len(prims)
    if hp_slots is not None:
        vecs = heading_vectors(n_theta)
        for h in range(n_theta):
            x, y = vecs[h]
            for j, k in enumerate(int(v) for v in hp_slots[h]):
                if k == 0:
                    continue
                # Left normal of (x, y) is (-y, x); as (drow, dcol) that is (-x, -y).
                for side, sgn in ((ACTION_HP_LEFT, 1), (ACTION_HP_RIGHT, -1)):
                    i = len(h0)
                    h0.append(h)
                    h1.append((h + half) % n_theta)
                    dr.append(-sgn * k * x)
                    dc.append(-sgn * k * y)
                    length.append(math.pi * 0.5 * k * math.hypot(x, y))  # nominal arc, cells
                    rev.append(i + 1 if side == ACTION_HP_LEFT else i - 1)
                    kind.append(side)
                    slot.append(j)
                    s0.append(0)
                    ns.append(0)
                    out[h].append(i)
    vecs = heading_vectors(n_theta)
    fill_index = {}
    for h in range(n_theta):
        x, y = vecs[h]
        vlen = math.hypot(x, y)
        for m in range(2, int(math.floor(cross_max_cells / vlen + 1e-9)) + 1):
            fill_index[(h, m)] = len(h0)
            h0.append(h)
            h1.append(h)
            dr.append(-m * y)
            dc.append(m * x)
            length.append(m * vlen)
            rev.append(-1)
            kind.append(ACTION_FILL)
            slot.append(0)
            s0.append(0)
            ns.append(0)
            out[h].append(fill_index[(h, m)])
    for (h, m), i in fill_index.items():
        rev[i] = fill_index[((h + half) % n_theta, m)]
    if bends is not None:
        left = {}
        for h in range(n_theta):
            for j in range(bends.X.shape[1]):
                i = len(h0)
                left[(h, j)] = i
                h0.append(h)
                h1.append(int(bends.h1[h, j]))
                dr.append(-int(bends.Y[h, j]))
                dc.append(int(bends.X[h, j]))
                length.append(bends.L1[h, j] + bends.R * abs(bends.dth[h, j]) + bends.L2[h, j])
                rev.append(-1)
                kind.append(ACTION_BEND_L)
                slot.append(j)
                s0.append(0)
                ns.append(0)
                out[h].append(i)
        for (h, j), il in left.items():
            i = len(h0)
            hr = (int(bends.h1[h, j]) + half) % n_theta
            h0.append(hr)
            h1.append((h + half) % n_theta)
            dr.append(int(bends.Y[h, j]))
            dc.append(-int(bends.X[h, j]))
            length.append(length[il])
            rev.append(il)
            rev[il] = i
            kind.append(ACTION_BEND_R)
            slot.append(j)
            s0.append(0)
            ns.append(0)
            out[hr].append(i)
    by_heading = np.full((n_theta, max(map(len, out))), -1, dtype=np.int64)
    for h, lst in enumerate(out):
        by_heading[h, :len(lst)] = lst
    i64 = lambda a: np.array(a, dtype=np.int64)  # noqa: E731
    ang = heading_angles(n_theta)
    d = ang[i64(h1)] - ang[i64(h0)]
    turn = np.abs((d + np.pi) % (2 * np.pi) - np.pi)
    turn[np.isin(i64(kind), [ACTION_HP_LEFT, ACTION_HP_RIGHT])] = np.pi
    turn = np.minimum(turn, turn[i64(rev)])     # bit-identical both ways (the search relies on it)
    return LatticeArrays(
        n_theta, prims,
        h0=i64(h0), h1=i64(h1), dr=i64(dr), dc=i64(dc),
        length=np.array(length, dtype=float),
        rev=i64(rev), s0=i64(s0), ns=i64(ns),
        sdr=-pts[:, 1].copy(), sdc=pts[:, 0].copy(),
        tx=tans[:, 0].copy(), ty=tans[:, 1].copy(),
        by_heading=by_heading, kind=i64(kind), slot=i64(slot), turn=turn,
    )
