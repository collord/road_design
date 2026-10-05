"""True cost evaluator with spoil reuse (brief section 8).

Spoil sources are full-bench segments and hairpin surpluses; fill sinks are
fill crossings. A small transport LP assigns spoil to sinks along the route
(haul distance = difference in station), with haul priced per m3*m by
whether the loaded haul runs downhill or uphill and no haul beyond haul_max.
Unmet fill is bought at C_borrow; unused spoil is sidecast for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .params import Params
from .route import Segment


@dataclass
class MassHaul:
    haul_cost: float
    borrow_m3: float
    borrow_cost: float
    reused_m3: float
    fill_m3: float
    assignments: list = field(default_factory=list)   # (source seq, sink seq, m3, distance m, cost per m3)
    sink_source: dict = field(default_factory=dict)   # sink seq -> {"spoil_m3", "borrow_m3"}

    @property
    def total(self) -> float:
        return self.haul_cost + self.borrow_cost

    @property
    def cost_per_m3(self) -> float:
        return self.total / self.fill_m3 if self.fill_m3 > 0 else float("nan")


def solve_mass_haul(segs: list[Segment], p: Params) -> MassHaul:
    stations = np.concatenate([[0.0], np.cumsum([s.length for s in segs])])
    mid = 0.5 * (stations[:-1] + stations[1:])
    zmid = np.array([float(np.mean(s.z)) for s in segs])
    src = [i for i, s in enumerate(segs) if s.spoil_m3 > 0 and s.section_type != "fill_crossing"]
    snk = [i for i, s in enumerate(segs) if s.section_type == "fill_crossing" and s.fill_m3 > 0]
    fill = float(sum(segs[j].fill_m3 for j in snk))
    if not snk:
        return MassHaul(0.0, 0.0, 0.0, 0.0, 0.0)
    pairs = []
    for i in src:
        for j in snk:
            dist = abs(mid[i] - mid[j])
            if dist <= p.haul_max:
                rate = p.C_haul_down if zmid[i] >= zmid[j] else p.C_haul_up
                pairs.append((i, j, dist, rate * dist))
    nx = len(pairs)
    nb = len(snk)
    c = np.array([q[3] for q in pairs] + [p.C_borrow] * nb, dtype=float)
    # Demand: sum_i x_ij + b_j = fill_j
    A_eq = np.zeros((nb, nx + nb))
    b_eq = np.array([segs[j].fill_m3 for j in snk])
    jpos = {j: k for k, j in enumerate(snk)}
    for v, (i, j, _, _) in enumerate(pairs):
        A_eq[jpos[j], v] = 1.0
    A_eq[:, nx:] = np.eye(nb)
    # Supply: sum_j x_ij <= spoil_i
    ipos = {i: k for k, i in enumerate(src)}
    A_ub = np.zeros((len(src), nx + nb)) if src else None
    b_ub = np.array([segs[i].spoil_m3 for i in src]) if src else None
    for v, (i, j, _, _) in enumerate(pairs):
        A_ub[ipos[i], v] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"mass-haul LP failed: {res.message}")
    x = res.x
    assign = [(segs[i].seq, segs[j].seq, float(x[v]), d, cpm)
              for v, (i, j, d, cpm) in enumerate(pairs) if x[v] > 1e-9]
    haul = float(sum(x[v] * pairs[v][3] for v in range(nx)))
    borrow = x[nx:]
    sink_source = {}
    for k, j in enumerate(snk):
        reused = float(sum(x[v] for v, q in enumerate(pairs) if q[1] == j))
        sink_source[segs[j].seq] = {"spoil_m3": reused, "borrow_m3": float(borrow[k])}
    return MassHaul(haul, float(borrow.sum()), float(borrow.sum() * p.C_borrow),
                    float(fill - borrow.sum()), fill, assign, sink_source)


@dataclass
class TrueCost:
    total: float
    surrogate: float
    base: float
    excavation: float        # section and hairpin cut, hairpin borrow
    culverts: float
    crossing_fill: float     # haul + borrow for crossing fill
    mass: MassHaul
    operating: float = 0.0   # lifetime truck costs (C_travel, C_vertical)

    @property
    def construction(self) -> float:
        return self.total - self.operating


def true_cost(segs: list[Segment], p: Params) -> TrueCost:
    mass = solve_mass_haul(segs, p)
    base = sum(s.cost_base for s in segs)
    excav = sum(s.cost_earthwork for s in segs if s.section_type != "fill_crossing")
    culv = sum(s.cost_culvert for s in segs)
    op = sum(s.cost_operating for s in segs)
    return TrueCost(base + excav + culv + mass.total + op, sum(s.cost_surrogate for s in segs),
                    base, excav, culv, mass.total, mass, op)
