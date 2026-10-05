"""Search orchestration: the fill-price iteration (brief section 8), corridor
fields and alternatives (section 9)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy import ndimage

from .masshaul import TrueCost, true_cost
from .route import Segment, evaluate_route
from .search import SearchProblem, edge_cost, edge_points, reconstruct, reverse_edges, run_search


@dataclass
class RouteResult:
    edges: list
    segments: list[Segment]
    cost: TrueCost
    lam_fill: float


@dataclass
class Iteration:
    iteration: int
    lam_fill: float
    surrogate: float
    true: float
    n_crossings: int
    shared_with_previous: float


@dataclass
class IterationResult:
    best: RouteResult | None
    history: list[Iteration] = field(default_factory=list)
    final_lam: float = math.nan
    last_search: object = None       # SearchResult of the last A* run (explored set)


def shared_length_fraction(a: list[Segment], b: list[Segment]) -> float:
    """Fraction of route a's length made of edges that also appear in b."""
    keys = {(s.r, s.c, s.action) for s in b}
    tot = sum(s.length for s in a)
    return sum(s.length for s in a if (s.r, s.c, s.action) in keys) / tot if tot else 1.0


def fill_price_iteration(prob: SearchProblem, start_rc, goal_rc, max_iter: int = 5,
                         same: float = 0.98) -> IterationResult:
    """Search with lam_fill = C_borrow, cost the route truly, set lam_fill to the
    realized cost per m3 of crossing fill, and search again. Stops when the
    route keeps at least `same` of its length, when it has no fill
    crossings, or after max_iter searches. Returns the route with the lowest
    true cost."""
    p = prob.params
    lam = p.C_borrow
    out = IterationResult(None)
    prev = None
    for it in range(max_iter):
        prob.lam_fill = lam
        res = run_search(prob, start_rc, goal_rc, astar=True)
        out.last_search = res
        if res.goal_state is None:
            break
        edges = reconstruct(res, prob.lat, res.goal_state)
        segs = evaluate_route(prob, edges)
        tc = true_cost(segs, p)
        n_cross = sum(s.section_type == "fill_crossing" for s in segs)
        shared = shared_length_fraction(segs, prev) if prev is not None else math.nan
        out.history.append(Iteration(it, lam, res.goal_cost, tc.total, n_cross, shared))
        if out.best is None or tc.total < out.best.cost.total:
            out.best = RouteResult(edges, segs, tc, lam)
        if prev is not None and shared >= same:
            break
        if n_cross == 0 or tc.mass.fill_m3 <= 0:
            break
        prev = segs
        lam = tc.mass.cost_per_m3
    out.final_lam = out.best.lam_fill if out.best else lam
    prob.lam_fill = out.final_lam
    return out


def avoid_self_overlap(prob: SearchProblem, start_rc, goal_rc, it: IterationResult, max_rounds: int = 5,
                       log=None) -> tuple[IterationResult, list[dict], np.ndarray]:
    """Re-search while the best route overlaps itself (loops back over its own
    road, e.g. a balloon turn-around). Each round blocks the cells of the
    excursion between the overlapping pieces, buffered by half the road
    width, and repeats the fill-price search. If blocking leaves no route,
    the previous route is kept (and its overlap stays reported).

    Returns the final iteration result, a record per round, and the mask of
    blocked cells. prob.open_ is left with the blocks applied.
    """
    from .route import self_overlaps

    p = prob.params
    blocked = np.zeros((prob.nr, prob.nc), dtype=bool)
    rounds = []
    for k in range(max_rounds):
        if it.best is None:
            break
        segs = it.best.segments
        ov = self_overlaps([s.pts_rc for s in segs], prob.res, p.w)
        if not ov:
            break
        lo = min(i for i, _, _ in ov) + 1
        hi = max(j for _, j, _ in ov) - 1
        geo = _Geometry(prob)
        m = geo.mask(it.best.edges[lo:hi + 1])
        m = ndimage.binary_dilation(m, iterations=max(1, int(math.ceil(0.5 * p.w / prob.res))))
        for rc in (start_rc, goal_rc):
            m[max(rc[0] - 1, 0):rc[0] + 2, max(rc[1] - 1, 0):rc[1] + 2] = False
        saved = prob.open_.copy()
        prob.open_[m] = 0
        new = fill_price_iteration(prob, start_rc, goal_rc)
        rec = {"round": k, "overlapping_pairs": len(ov), "seq_range": [lo, hi], "blocked_cells": int(m.sum()),
               "cost_before": it.best.cost.total}
        if new.best is None:
            prob.open_ = saved
            rec["result"] = "no route with the excursion blocked; kept the previous route"
            rounds.append(rec)
            if log:
                log(f"self-overlap round {k}: {rec['result']}")
            break
        blocked |= m
        rec["cost_after"] = new.best.cost.total
        rec["result"] = "re-searched"
        rounds.append(rec)
        if log:
            log(f"self-overlap round {k}: route overlapped itself over segments {lo}-{hi}; blocked "
                f"{int(m.sum())} cells, true cost {it.best.cost.total:,.0f} -> {new.best.cost.total:,.0f}")
        it = new
    return it, rounds, blocked


# --- bidirectional fields and alternatives (brief section 9) ------------------------

@dataclass
class Fields:
    fwd: object                 # SearchResult from the start
    bwd: object                 # SearchResult from the goal (same graph: it is reversal-symmetric)
    F: np.ndarray               # corridor field: cheapest route cost through each cell
    best_h: np.ndarray          # heading (forward sense) achieving F, -1 where F is inf
    F_min: float


def corridor_fields(prob: SearchProblem, start_rc, goal_rc, max_cost: float = math.inf) -> Fields:
    """Full Dijkstra from both ends; F(i,j) = min_h Gf(i,j,h) + Gb(i,j,h+pi).

    The backward search is a forward search from the goal: every action's
    reverse is an action with identical cost, so cost-to-come from the goal
    to (i, j, h+pi) equals cost-to-go from (i, j, h) to the goal. F is built
    one heading at a time to keep memory to two (rows, cols) arrays.
    """
    fwd = run_search(prob, start_rc, max_cost=max_cost)
    bwd = run_search(prob, goal_rc, max_cost=max_cost)
    nh = prob.n_theta
    F = np.full((prob.nr, prob.nc), np.inf)
    best_h = np.full((prob.nr, prob.nc), -1, dtype=np.int16)
    for h in range(nh):
        tot = fwd.G[:, :, h] + bwd.G[:, :, (h + nh // 2) % nh]
        better = tot < F
        F[better] = tot[better]
        best_h[better] = h
    return Fields(fwd, bwd, F, best_h, float(F.min()))


def route_through(prob: SearchProblem, fields: Fields, rc) -> tuple[list, int]:
    """Edges of the cheapest route through cell rc (start -> rc -> goal), and
    the index of the via node in the route's node sequence."""
    r, c = rc
    h = int(fields.best_h[r, c])
    nh = prob.n_theta
    head = reconstruct(fields.fwd, prob.lat, (r, c, h))
    tail = reconstruct(fields.bwd, prob.lat, (r, c, (h + nh // 2) % nh))
    return head + reverse_edges(tail, prob.lat), len(head)


class _Geometry:
    """Cached edge polylines (hairpin and bend geometry is costly to rebuild)."""

    def __init__(self, prob: SearchProblem):
        self.prob = prob
        self.cache: dict = {}

    def points(self, e) -> np.ndarray:
        pts = self.cache.get(e)
        if pts is None:
            pts = edge_points(self.prob, *e)
            self.cache[e] = pts
        return pts

    def cells(self, edges) -> np.ndarray:
        out = [np.floor(self.points(e) + 0.5).astype(np.int64) for e in edges]
        if not out:
            return np.empty((0, 2), np.int64)
        # A point within half a cell of the grid edge rounds onto the cell past it.
        cells = np.clip(np.vstack(out), 0, [self.prob.nr - 1, self.prob.nc - 1])
        return np.unique(cells, axis=0)

    def overlaps(self, edges) -> list:
        from .route import self_overlaps
        return self_overlaps([self.points(e) for e in edges], self.prob.res, self.prob.params.w)

    def mask(self, edges) -> np.ndarray:
        m = np.zeros((self.prob.nr, self.prob.nc), dtype=bool)
        c = self.cells(edges)
        m[c[:, 0], c[:, 1]] = True
        return m


def _shared_edge_fraction(prob: SearchProblem, a, b) -> float:
    keys = set(b)
    la = [prob.lat.length[e[2]] for e in a]
    return sum(l for e, l in zip(a, la) if e in keys) / sum(la) if la else 1.0


def crop_problem(prob: SearchProblem, r0: int, r1: int, c0: int, c1: int) -> SearchProblem:
    """The search problem restricted to rows r0:r1, cols c0:c1 (copies)."""
    import dataclasses

    sl = (slice(r0, r1), slice(c0, c1))
    cp = lambda a: np.ascontiguousarray(a[sl])  # noqa: E731
    hp = prob.hp if prob.hp.shape[:2] != (prob.nr, prob.nc) else cp(prob.hp)
    bend = prob.bend if prob.bend is None or prob.bend.shape[:2] != (prob.nr, prob.nc) else cp(prob.bend)
    return dataclasses.replace(
        prob, nr=r1 - r0, nc=c1 - c0, open_=cp(prob.open_), gx=cp(prob.gx), gy=cp(prob.gy), zs=cp(prob.zs),
        soil=cp(prob.soil), z_raw=cp(prob.z_raw), channel=cp(prob.channel), area=cp(prob.area),
        chdist=cp(prob.chdist), hp=hp, bend=bend)


def locally_optimal(prob: SearchProblem, edges, via_idx: int, window: float, tol: float,
                    geo: "_Geometry | None" = None, pad_m: float = 200.0) -> tuple[bool, dict]:
    """T-test for a via route: the part within `window` cost either side of
    the via node must be (within tol) a cheapest path between its ends.

    Out-and-back detours to a via point fail: going straight between the
    window ends is much cheaper than visiting the via point. The cheapest
    path is searched in a box around that part of the route, padded by half
    the distance between its ends plus pad_m. The box only ever makes the
    test more lenient (a shortcut outside it is missed), never stricter.
    """
    lt, tt = prob.lat_tuple(), prob.terr_tuple()
    costs = np.array([edge_cost(r, c, a, lt, tt) for r, c, a in edges])
    cum = np.concatenate([[0.0], np.cumsum(costs)])          # cost at node k (k = 0..len(edges))
    gv = cum[via_idx]
    ia = int(np.searchsorted(cum, gv - window, side="right")) - 1
    ib = int(np.searchsorted(cum, gv + window, side="left"))
    ia = max(ia, 0)
    ib = min(ib, len(edges))
    if ib - ia < 2:
        return True, {}
    L = prob.lat
    ra, ca, a0 = edges[ia]
    ha = int(L.h0[a0])
    rb, cb, b0 = edges[ib - 1]
    rb, cb = rb + int(L.dr[b0]), cb + int(L.dc[b0])
    sub = float(cum[ib] - cum[ia])
    cells = (geo or _Geometry(prob)).cells(edges[ia:ib])
    pad = int(math.ceil((0.5 * math.hypot(ra - rb, ca - cb) * prob.res + pad_m) / prob.res))
    r0 = max(int(cells[:, 0].min()) - pad, 0)
    r1 = min(int(cells[:, 0].max()) + pad + 1, prob.nr)
    c0 = max(int(cells[:, 1].min()) - pad, 0)
    c1 = min(int(cells[:, 1].max()) + pad + 1, prob.nc)
    sub_prob = crop_problem(prob, r0, r1, c0, c1)
    sub_prob.lam_fill = prob.lam_fill
    res = run_search(sub_prob, (ra - r0, ca - c0), (rb - r0, cb - c0), astar=True,
                     max_cost=sub * (1 + tol) + 1e-9, source_heading=ha)
    best = res.goal_cost
    return sub <= (1 + tol) * best, {"window_cost": sub, "shortest": best}


@numba.njit(cache=True)
def _branch_costs(G, parent, order, on_route, dr, dc, h0, nc, nh):
    """For every state, the cost-to-come at which its tree path last runs on
    the route (states visited in order of increasing cost, so each parent is
    done before its children)."""
    out = np.zeros(G.shape[0], dtype=np.float32)
    for s in order:
        if on_route[s]:
            out[s] = G[s]
            continue
        p = parent[s]
        if p < 0:
            continue
        cell = s // nh
        r = cell // nc
        c = cell - r * nc
        prev = ((r - dr[p]) * nc + (c - dc[p])) * nh + h0[p]
        out[s] = out[prev]
    return out


def _tree_branch(res, prob: SearchProblem, states) -> np.ndarray:
    G = res.G.ravel()
    idx = np.flatnonzero(np.isfinite(G))
    order = idx[np.argsort(G[idx], kind="stable")]
    del idx
    on = np.zeros(G.shape[0], dtype=np.bool_)
    nh = prob.n_theta
    for r, c, h in states:
        on[(r * prob.nc + c) * nh + h] = True
    L = prob.lat
    return _branch_costs(G, res.parent.ravel(), order, on, L.dr, L.dc, L.h0, prob.nc, nh).reshape(res.G.shape)


def shared_cost_fraction(prob: SearchProblem, fields: Fields, edges) -> np.ndarray:
    """Per cell, the fraction of the cost of the route through that cell that
    runs along `edges` (a start-to-goal route), from both search trees."""
    L = prob.lat
    nh = prob.n_theta
    states = [(r, c, int(L.h0[a])) for r, c, a in edges]
    r, c, a = edges[-1]
    states.append((r + int(L.dr[a]), c + int(L.dc[a]), int(L.h1[a])))
    af = _tree_branch(fields.fwd, prob, states)
    hb = np.where(fields.best_h >= 0, fields.best_h, 0).astype(np.int64)
    rr, cc = np.indices(hb.shape)
    share = af[rr, cc, hb].astype(np.float64)
    del af
    back = [(r, c, (h + nh // 2) % nh) for r, c, h in states]
    ab = _tree_branch(fields.bwd, prob, back)
    share += ab[rr, cc, (hb + nh // 2) % nh]
    del ab
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(fields.F) & (fields.F > 0), share / fields.F, 1.0)


# Skip via cells whose route would follow the best route for more than this
# share of its cost (a cost proxy for the brief's half-the-length rule, which
# is still checked exactly on every candidate that gets through).
PREFILTER_SHARE = 0.7


@dataclass
class Alternative:
    edges: list
    via: tuple[int, int]
    F: float                     # surrogate cost of the route through `via`
    local_check: dict = field(default_factory=dict)


def alternatives(prob: SearchProblem, fields: Fields, start_rc, log=None) -> list[Alternative]:
    """Best route plus up to K_alt - 1 alternatives through distinct via cells.

    Each further via cell has the lowest F among cells at least D_alt from
    every chosen route and within (1 + eps_alt) * F_min. A candidate is
    discarded if it shares more than half its length with a chosen route, or
    if it is not locally optimal around its via point (see locally_optimal).
    Cells whose route would run more than PREFILTER_SHARE of its cost along
    the best route are skipped without reconstruction (computed for all
    cells at once from the search trees). Cells near a discarded candidate's
    route are not tried as via cells again, and at most alt_max_tries
    candidates are examined.
    """
    p = prob.params
    F = fields.F
    if not np.isfinite(fields.F_min):
        return []
    geo = _Geometry(prob)
    flat = np.argmin(F)
    via0 = (int(flat // prob.nc), int(flat % prob.nc))
    edges0, _ = route_through(prob, fields, via0)
    chosen = [Alternative(edges0, via0, float(F[via0]))]
    near = geo.mask(edges0)
    dist = ndimage.distance_transform_edt(~near) * prob.res
    tried = np.zeros_like(near)
    thr = (1 + p.eps_alt) * fields.F_min
    share = shared_cost_fraction(prob, fields, edges0)
    cand = np.flatnonzero((F.ravel() <= thr) & (share.ravel() <= PREFILTER_SHARE))
    del share
    order = cand[np.lexsort((cand, F.ravel()[cand]))]
    window = p.alt_window * fields.F_min
    blur = max(1, int(round(0.25 * p.D_alt / prob.res)))
    blur_overlap = max(1, int(round(p.D_alt / prob.res)))
    n_tries = 0
    stats = {"overlap": 0, "not_local": 0, "self_overlap": 0}
    for idx in order:
        if len(chosen) >= p.K_alt or n_tries >= p.alt_max_tries:
            break
        rc = (int(idx // prob.nc), int(idx % prob.nc))
        if dist[rc] < p.D_alt or tried[rc]:
            continue
        n_tries += 1
        edges, via_idx = route_through(prob, fields, rc)
        reject = None
        if any(_shared_edge_fraction(prob, edges, ch.edges) > 0.5 for ch in chosen):
            reject = "overlap"
        elif geo.overlaps(edges):
            reject = "self_overlap"
        else:
            ok, info = locally_optimal(prob, edges, via_idx, window, p.alt_local_tol, geo)
            if not ok:
                reject = "not_local"
        if reject:
            stats[reject] += 1
            m = geo.mask(edges) & ~near
            tried |= ndimage.binary_dilation(m, iterations=blur if reject == "not_local" else blur_overlap)
            continue
        chosen.append(Alternative(edges, rc, float(F[rc]), info))
        near |= geo.mask(edges)
        dist = ndimage.distance_transform_edt(~near) * prob.res
    if log:
        log(f"alternatives: {len(chosen)} kept after {n_tries} candidates "
            f"({stats['overlap']} overlapping, {stats['not_local']} not locally optimal, "
            f"{stats['self_overlap']} overlapping themselves)")
    return chosen
