"""Search orchestration: the fill-price iteration (brief section 8), corridor
fields and alternatives (section 9)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy import ndimage

from .masshaul import TrueCost, true_cost
from .route import Segment, evaluate_route
from .search import SearchProblem, edge_cost, edge_points, reconstruct, reverse_edges, run_search
from .timing import add, stage


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
        with stage(f"A* search {it + 1}", fill_price=round(lam, 3)) as row:
            res = run_search(prob, start_rc, goal_rc, astar=True)
            row["settled"] = int(res.n_expanded)
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
    one heading at a time to keep memory to two per-cell arrays. Each search is given the
    other end as its goal (only recorded, not stopped at), so both cover the same cells and
    their state indices agree.
    """
    with stage("Dijkstra from the start") as row:
        fwd = run_search(prob, start_rc, goal_rc, max_cost=max_cost)
        row["settled"] = int(fwd.n_expanded)
    with stage("Dijkstra from the goal") as row:
        bwd = run_search(prob, goal_rc, start_rc, max_cost=max_cost)
        row["settled"] = int(bwd.n_expanded)
    assert np.array_equal(fwd.cells, bwd.cells)
    fwd.closed = bwd.closed = None          # not needed downstream: a byte per state each
    nh = prob.n_theta
    F = np.full(fwd.cells.size, np.inf)
    best_h = np.full(fwd.cells.size, -1, dtype=np.int16)
    for h in range(nh):
        tot = fwd.G[:, h] + bwd.G[:, (h + nh // 2) % nh]
        better = tot < F
        F[better] = tot[better]
        best_h[better] = h
    F = fwd.to_grid(F, np.inf)
    return Fields(fwd, bwd, F, fwd.to_grid(best_h, np.int16(-1)), float(F.min()))


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
    """The search problem restricted to rows r0:r1, cols c0:c1 (copies of the rasters; the
    hairpin and bend tables are shared, through a cropped lookup)."""
    import dataclasses

    sl = (slice(r0, r1), slice(c0, c1))
    cp = lambda a: np.ascontiguousarray(a[sl])  # noqa: E731
    look = cp(prob.macro_lookup) if prob.macro_lookup is not None else None
    return dataclasses.replace(
        prob, nr=r1 - r0, nc=c1 - c0, open_=cp(prob.open_), gx=cp(prob.gx), gy=cp(prob.gy), zs=cp(prob.zs),
        soil=cp(prob.soil), z_raw=cp(prob.z_raw), channel=cp(prob.channel), area=cp(prob.area),
        chdist=cp(prob.chdist), macro_lookup=look,
        open_tables=cp(prob.open_tables) if prob.open_tables is not None else None)


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
    return sub <= (1 + tol) * best, {"window_cost": sub, "shortest": best, "settled": int(res.n_expanded)}


@numba.njit(cache=True, inline="always")
def _pred(s, a, cells, lookup, dr, dc, h0, nc, nh):
    """State index of the state that action a, arriving at state index s, left from."""
    flat = cells[s // nh]
    r = flat // nc
    c = flat - r * nc
    return lookup[(r - dr[a]) * nc + (c - dc[a])] * nh + h0[a]


@numba.njit(cache=True)
def _branch_costs(G, parent, order, on_route, cells, lookup, dr, dc, h0, nc, nh):
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
        out[s] = out[_pred(s, p, cells, lookup, dr, dc, h0, nc, nh)]
    return out


def _tree_branch(res, prob: SearchProblem, states) -> np.ndarray:
    G = res.G.ravel()
    idx = np.flatnonzero(np.isfinite(G))
    order = idx[np.argsort(G[idx], kind="stable")]
    del idx
    on = np.zeros(G.shape[0], dtype=np.bool_)
    for r, c, h in states:
        on[res.state(r, c, h)] = True
    L = prob.lat
    return _branch_costs(G, res.parent.ravel(), order, on, res.cells, res.lookup, L.dr, L.dc, L.h0, prob.nc,
                         prob.n_theta).reshape(res.G.shape)


def shared_cost_fraction(prob: SearchProblem, fields: Fields, edges) -> np.ndarray:
    """Per cell, the fraction of the cost of the route through that cell that
    runs along `edges` (a start-to-goal route), from both search trees."""
    L = prob.lat
    nh = prob.n_theta
    states = [(r, c, int(L.h0[a])) for r, c, a in edges]
    r, c, a = edges[-1]
    states.append((r + int(L.dr[a]), c + int(L.dc[a]), int(L.h1[a])))
    af = _tree_branch(fields.fwd, prob, states)
    k = np.arange(fields.fwd.cells.size)
    hb = np.maximum(fields.best_h.reshape(-1)[fields.fwd.cells], 0).astype(np.int64)
    share = af[k, hb].astype(np.float64)
    del af
    back = [(r, c, (h + nh // 2) % nh) for r, c, h in states]
    ab = _tree_branch(fields.bwd, prob, back)
    share += ab[k, (hb + nh // 2) % nh]
    del ab
    share = fields.fwd.to_grid(share, 0.0)
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


@numba.njit(cache=True)
def _plateau_chains(fpar, bpar, cells, lookup, dr, dc, h0, rev, nc, nh, Gf, Gb, length, res, F_max, min_cost):
    """Plateaus: chains of edges that are in both search trees. Edge u -> v (action a) is in
    the forward tree when v's parent is a, and in the backward tree when u-reversed's parent
    is rev[a]. Each state has at most one plateau edge in and one out, so plateaus are chains.
    States are state indices of the two searches (which cover the same cells).
    Returns, for chains whose route costs at most F_max and whose own cost is at least min_cost:
    (head state, tail state, number of edges, length m, plateau cost, route cost)."""
    n = fpar.shape[0]
    nxt = np.full(n, -1, dtype=np.int32)          # state indices fit (plateaus() checks)
    has_in = np.zeros(n, dtype=np.bool_)
    half = nh // 2
    for v in range(n):
        a = fpar[v]
        if a < 0:
            continue
        u = _pred(v, a, cells, lookup, dr, dc, h0, nc, nh)
        uh = h0[a]
        urev = u - uh + (uh + half) % nh
        if bpar[urev] == rev[a]:
            nxt[u] = v
            has_in[v] = True
    n_heads = 0
    for u in range(n):
        if nxt[u] >= 0 and not has_in[u]:
            n_heads += 1
    heads = np.empty(n_heads, dtype=np.int64)
    tails = np.empty(n_heads, dtype=np.int64)
    edges = np.empty(n_heads, dtype=np.int64)
    lens = np.empty(n_heads)
    pcost = np.empty(n_heads)
    rcost = np.empty(n_heads)
    m = 0
    for u in range(n):
        if nxt[u] < 0 or has_in[u]:
            continue
        cell = u // nh
        hrev = (u - cell * nh + half) % nh
        F = Gf[u] + Gb[cell * nh + hrev]
        if not F <= F_max:
            continue
        x = u
        k = 0
        L = 0.0
        while nxt[x] >= 0:
            x = nxt[x]
            k += 1
            L += length[fpar[x]] * res
        cost = Gf[x] - Gf[u]
        if cost < min_cost:
            continue
        heads[m] = u
        tails[m] = x
        edges[m] = k
        lens[m] = L
        pcost[m] = cost
        rcost[m] = F
        m += 1
    return heads[:m], tails[:m], edges[:m], lens[:m], pcost[:m], rcost[:m]


def plateaus(prob: SearchProblem, fields: Fields, F_max: float, min_cost: float) -> dict:
    """Plateaus of the two search trees (see _plateau_chains), as arrays, cheapest route first."""
    L = prob.lat
    assert fields.fwd.parent.size < 2**31, "too many states for 32-bit plateau links"
    h, t, k, ln, pc, rc = _plateau_chains(
        fields.fwd.parent.reshape(-1), fields.bwd.parent.reshape(-1), fields.fwd.cells,
        fields.fwd.lookup, L.dr.astype(np.int64),
        L.dc.astype(np.int64), L.h0.astype(np.int64), L.rev.astype(np.int64), prob.nc, prob.n_theta,
        fields.fwd.G.reshape(-1), fields.bwd.G.reshape(-1), L.length.astype(np.float64), float(prob.res),
        float(F_max), float(min_cost))
    order = np.lexsort((-pc, rc))
    return {"head": h[order], "tail": t[order], "edges": k[order], "length": ln[order], "cost": pc[order],
            "route_cost": rc[order]}


def route_through_state(prob: SearchProblem, fields: Fields, state: int) -> list:
    """Edges of the cheapest route through a given state (a state index of the searches):
    start -> state on the forward tree, state -> goal on the backward tree."""
    nh = prob.n_theta
    r, c, h = fields.fwd.rc(state)
    head = reconstruct(fields.fwd, prob.lat, (r, c, h))
    tail = reconstruct(fields.bwd, prob.lat, (r, c, (h + nh // 2) % nh))
    return head + reverse_edges(tail, prob.lat)


def alternatives(prob: SearchProblem, fields: Fields, start_rc, log=None) -> list[Alternative]:
    """Best route plus up to K_alt - 1 alternatives, by the method alt_method names."""
    if prob.params.alt_method == "plateau":
        return _alternatives_plateau(prob, fields, log)
    return _alternatives_via(prob, fields, start_rc, log)


def _alternatives_plateau(prob: SearchProblem, fields: Fields, log=None) -> list[Alternative]:
    """Alternatives along plateaus: stretches that are the cheapest path from both ends.

    A route made of the forward tree up to a plateau, the plateau, and the backward tree
    beyond it is optimal over any window costing less than its plateau, so no extra search
    is needed to show it is not a detour. Plateaus whose route costs at most (1 + eps_alt)
    times the best and whose own cost is at least alt_plateau times the best are taken
    cheapest route first; a candidate is kept if its via point (the plateau's middle) is at
    least D_alt from every route already chosen, it shares at most half its length with
    each, and it does not run over itself.
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
    with stage("find plateaus") as row:
        pl = plateaus(prob, fields, (1 + p.eps_alt) * fields.F_min * (1 + 1e-12), p.alt_plateau * fields.F_min)
        row["plateaus"] = int(len(pl["head"]))
    nh = prob.n_theta
    stats = {"near": 0, "overlap": 0, "self_overlap": 0}
    n_tries = 0
    t_route = t_geom = 0.0
    for i, (tail, k) in enumerate(zip(pl["tail"], pl["edges"])):
        if len(chosen) >= p.K_alt or n_tries >= p.alt_max_tries:
            break
        # The plateau's middle state: walk half its edges back from the tail along the forward tree.
        x = int(tail)
        fpar = fields.fwd.parent.reshape(-1)
        for _ in range(int(k) // 2):
            r, c, _h = fields.fwd.rc(x)
            a = int(fpar[x])
            x = fields.fwd.state(r - int(prob.lat.dr[a]), c - int(prob.lat.dc[a]), int(prob.lat.h0[a]))
        rc = fields.fwd.rc(x)[:2]
        if dist[rc] < p.D_alt:
            stats["near"] += 1
            continue
        n_tries += 1
        t0 = time.perf_counter()
        edges = route_through_state(prob, fields, x)
        t1 = time.perf_counter()
        t_route += t1 - t0
        reject = None
        if any(_shared_edge_fraction(prob, edges, ch.edges) > 0.5 for ch in chosen):
            reject = "overlap"
        elif geo.overlaps(edges):
            reject = "self_overlap"
        t_geom += time.perf_counter() - t1
        if reject:
            stats[reject] += 1
            continue
        chosen.append(Alternative(edges, (int(rc[0]), int(rc[1])), float(pl["route_cost"][i]),
                                  {"plateau_cost": float(pl["cost"][i]), "plateau_length_m": float(pl["length"][i])}))
        near |= geo.mask(edges)
        dist = ndimage.distance_transform_edt(~near) * prob.res
    add("rebuild candidate routes from the trees", t_route, candidates=n_tries)
    add("overlap checks", t_geom)
    if log:
        log(f"alternatives (plateaus): {len(chosen)} kept from {len(pl['head'])} plateaus; {n_tries} candidates "
            f"({stats['overlap']} overlapping, {stats['self_overlap']} overlapping themselves, "
            f"{stats['near']} too near a chosen route)")
    return chosen


def _alternatives_via(prob: SearchProblem, fields: Fields, start_rc, log=None) -> list[Alternative]:
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
    with stage("prefilter (shared cost, all cells)"):
        share = shared_cost_fraction(prob, fields, edges0)
    cand = np.flatnonzero((F.ravel() <= thr) & (share.ravel() <= PREFILTER_SHARE))
    del share
    order = cand[np.lexsort((cand, F.ravel()[cand]))]
    window = p.alt_window * fields.F_min
    blur = max(1, int(round(0.25 * p.D_alt / prob.res)))
    blur_overlap = max(1, int(round(p.D_alt / prob.res)))
    n_tries = 0
    stats = {"overlap": 0, "not_local": 0, "self_overlap": 0}
    t_route = t_geom = t_local = 0.0
    n_local = settled_local = 0
    for idx in order:
        if len(chosen) >= p.K_alt or n_tries >= p.alt_max_tries:
            break
        rc = (int(idx // prob.nc), int(idx % prob.nc))
        if dist[rc] < p.D_alt or tried[rc]:
            continue
        n_tries += 1
        t0 = time.perf_counter()
        edges, via_idx = route_through(prob, fields, rc)
        t1 = time.perf_counter()
        t_route += t1 - t0
        reject = None
        if any(_shared_edge_fraction(prob, edges, ch.edges) > 0.5 for ch in chosen):
            reject = "overlap"
        elif geo.overlaps(edges):
            reject = "self_overlap"
        t_geom += time.perf_counter() - t1
        if reject is None:
            t2 = time.perf_counter()
            ok, info = locally_optimal(prob, edges, via_idx, window, p.alt_local_tol, geo)
            t_local += time.perf_counter() - t2
            n_local += 1
            settled_local += int(info.get("settled", 0))
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
    add("rebuild candidate routes from the trees", t_route, candidates=n_tries)
    add("overlap checks", t_geom)
    add("local-optimality A* searches", t_local, searches=n_local, settled=settled_local)
    if log:
        log(f"alternatives: {len(chosen)} kept after {n_tries} candidates "
            f"({stats['overlap']} overlapping, {stats['not_local']} not locally optimal, "
            f"{stats['self_overlap']} overlapping themselves)")
    return chosen
