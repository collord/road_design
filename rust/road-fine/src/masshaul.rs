//! The true cost of a route with spoil reuse (a port of roadsearch/masshaul.py).
//!
//! Spoil sources are full-bench segments and hairpin surpluses; fill sinks are fill crossings.
//! Spoil goes to sinks along the route (haul distance = difference in station), priced per
//! m3 m by whether the loaded haul runs down or up, never beyond haul_max; unmet fill is bought
//! at C_borrow, unused spoil is sidecast for free. The Python solves this transport problem as
//! a linear program; here it is a min-cost flow (successive shortest paths), which reaches the
//! same optimum.

use crate::params::Params;
use crate::route::{Segment, FILL_CROSSING};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MassHaul {
    pub haul_cost: f64,
    pub borrow_m3: f64,
    pub borrow_cost: f64,
    pub reused_m3: f64,
    pub fill_m3: f64,
}

impl MassHaul {
    pub fn total(&self) -> f64 {
        self.haul_cost + self.borrow_cost
    }

    pub fn cost_per_m3(&self) -> f64 {
        if self.fill_m3 > 0.0 { self.total() / self.fill_m3 } else { f64::NAN }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrueCost {
    pub total: f64,
    pub surrogate: f64,
    pub base: f64,
    /// Section and hairpin cut, hairpin borrow.
    pub excavation: f64,
    pub culverts: f64,
    /// Haul and borrow for crossing fill.
    pub crossing_fill: f64,
    /// Lifetime truck costs (C_travel, C_vertical, C_turn).
    pub operating: f64,
    pub mass: MassHaul,
}

impl TrueCost {
    pub fn construction(&self) -> f64 {
        self.total - self.operating
    }
}

const EPS: f64 = 1e-12;

struct Arc {
    to: usize,
    cap: f64,
    cost: f64,
}

/// A small min-cost flow network: arcs in pairs (forward, residual).
struct Network {
    arcs: Vec<Arc>,
    out: Vec<Vec<usize>>,
}

impl Network {
    fn new(n: usize) -> Self {
        Network { arcs: Vec::new(), out: vec![Vec::new(); n] }
    }

    fn add(&mut self, from: usize, to: usize, cap: f64, cost: f64) -> usize {
        let k = self.arcs.len();
        self.arcs.push(Arc { to, cap, cost });
        self.arcs.push(Arc { to: from, cap: 0.0, cost: -cost });
        self.out[from].push(k);
        self.out[to].push(k + 1);
        k
    }

    fn flow(&self, k: usize) -> f64 {
        self.arcs[k + 1].cap
    }

    /// Successive shortest paths (Bellman-Ford on the residual graph) from s to t until t is
    /// unreachable.
    fn run(&mut self, s: usize, t: usize) {
        let n = self.out.len();
        loop {
            let mut dist = vec![f64::INFINITY; n];
            let mut via = vec![usize::MAX; n];
            dist[s] = 0.0;
            for _ in 0..n {
                let mut changed = false;
                for u in 0..n {
                    if !dist[u].is_finite() {
                        continue;
                    }
                    for &k in &self.out[u] {
                        let a = &self.arcs[k];
                        if a.cap > EPS && dist[u] + a.cost < dist[a.to] - EPS {
                            dist[a.to] = dist[u] + a.cost;
                            via[a.to] = k;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            if !dist[t].is_finite() {
                return;
            }
            let mut push = f64::INFINITY;
            let mut v = t;
            while v != s {
                let k = via[v];
                push = push.min(self.arcs[k].cap);
                v = self.arcs[k ^ 1].to;
            }
            let mut v = t;
            while v != s {
                let k = via[v];
                self.arcs[k].cap -= push;
                self.arcs[k ^ 1].cap += push;
                v = self.arcs[k ^ 1].to;
            }
        }
    }
}

pub fn solve_mass_haul(segs: &[Segment], p: &Params) -> MassHaul {
    let mut stations = vec![0.0];
    for s in segs {
        stations.push(stations.last().unwrap() + s.length);
    }
    let mid: Vec<f64> = (0..segs.len()).map(|i| 0.5 * (stations[i] + stations[i + 1])).collect();
    let zmid: Vec<f64> = segs.iter().map(|s| s.z.iter().sum::<f64>() / s.z.len() as f64).collect();
    let src: Vec<usize> = (0..segs.len()).filter(|&i| segs[i].spoil_m3 > 0.0 && segs[i].kind != FILL_CROSSING).collect();
    let snk: Vec<usize> = (0..segs.len()).filter(|&j| segs[j].kind == FILL_CROSSING && segs[j].fill_m3 > 0.0).collect();
    if snk.is_empty() {
        return MassHaul::default();
    }
    let fill: f64 = snk.iter().map(|&j| segs[j].fill_m3).sum();
    // Nodes: 0 source, 1 sink, 2 borrow pit, then spoil sources, then fill sinks.
    let (s, t, b) = (0, 1, 2);
    let mut net = Network::new(3 + src.len() + snk.len());
    net.add(s, b, f64::INFINITY, 0.0);
    let mut borrow_arcs = Vec::new();
    for (k, &j) in snk.iter().enumerate() {
        let node = 3 + src.len() + k;
        borrow_arcs.push(net.add(b, node, f64::INFINITY, p.c_borrow));
        net.add(node, t, segs[j].fill_m3, 0.0);
    }
    let mut haul_arcs = Vec::new();
    for (a, &i) in src.iter().enumerate() {
        net.add(s, 3 + a, segs[i].spoil_m3, 0.0);
        for (k, &j) in snk.iter().enumerate() {
            let dist = (mid[i] - mid[j]).abs();
            if dist <= p.haul_max {
                let rate = if zmid[i] >= zmid[j] { p.c_haul_down } else { p.c_haul_up };
                haul_arcs.push((net.add(3 + a, 3 + src.len() + k, f64::INFINITY, rate * dist), rate * dist));
            }
        }
    }
    net.run(s, t);
    let haul: f64 = haul_arcs.iter().map(|&(k, c)| net.flow(k) * c).sum();
    let borrow: f64 = borrow_arcs.iter().map(|&k| net.flow(k)).sum();
    MassHaul { haul_cost: haul, borrow_m3: borrow, borrow_cost: borrow * p.c_borrow, reused_m3: fill - borrow, fill_m3: fill }
}

pub fn true_cost(segs: &[Segment], p: &Params) -> TrueCost {
    let mass = solve_mass_haul(segs, p);
    let base: f64 = segs.iter().map(|s| s.cost_base).sum();
    let excavation: f64 = segs.iter().filter(|s| s.kind != FILL_CROSSING).map(|s| s.cost_earthwork).sum();
    let culverts: f64 = segs.iter().map(|s| s.cost_culvert).sum();
    let operating: f64 = segs.iter().map(|s| s.cost_operating).sum();
    TrueCost {
        total: base + excavation + culverts + mass.total() + operating,
        surrogate: segs.iter().map(|s| s.cost_surrogate).sum(),
        base,
        excavation,
        culverts,
        crossing_fill: mass.total(),
        operating,
        mass,
    }
}
