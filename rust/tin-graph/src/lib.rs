//! Coarse road search on a sparse graph over a DEM.
//!
//! Nodes are where roads plausibly go or turn: the vertices of a heightfield triangulation of the
//! smoothed DEM (Delatin: dense on ridges, valley floors and slope breaks, sparse on uniform
//! slopes), points along the channel network, saddles, and a sparse background grid for coverage.
//! Each node is joined to its nearest neighbors within a maximum length, and every edge is costed
//! on the full-resolution rasters with the road cost model:
//!
//! * construction: earthwork per meter from the section table at each sample's sidehill slope,
//!   culverts for each run of channel cells crossed;
//! * operating: truck travel on the length, climb on every rise and fall of the profile;
//! * a switchback proxy for edges steeper than g_max: the road is stretched to |dz| / g_max, with
//!   a hairpin allowance for each extra pass along the edge;
//! * infeasible across excluded cells (slope mask, avoid zones) or nodata.
//!
//! Two exact Dijkstra searches (from the start and from the goal) give the corridor field, the
//! cost of the best route through each node. Distinct corridors come from saddle candidates
//! (routes forced through a saddle, ranked by the field) and from a penalty loop.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use road_cost::{culvert_cost, Grid, SectionTable};

/// Why a node is in the graph.
pub const KIND_TIN: u8 = 0;
pub const KIND_CHANNEL: u8 = 1;
pub const KIND_SADDLE: u8 = 2;
pub const KIND_BACKGROUND: u8 = 3;
pub const KIND_ENDPOINT: u8 = 4;
/// Added at random by node sampling.
pub const KIND_SAMPLED: u8 = 5;

/// Settings of the graph and its corridors.
#[derive(Clone, Debug)]
pub struct TinParams {
    /// Maximum vertical error of the triangulation (m).
    pub tolerance: f64,
    /// Edges per node (nearest neighbors).
    pub k: usize,
    /// Longest edge (cells).
    pub edge_max: f64,
    /// Block size (cells) for thinning channel and saddle nodes and for the background grid.
    pub block: usize,
    /// Smallest contributing area (ha) for a channel node.
    pub channel_node_area: f64,
    /// Alternatives sought besides the best route.
    pub n_alt: usize,
    /// Factor applied to an edge's weight each time a kept or tried route uses it.
    pub penalty: f64,
    /// Keep a route only if it costs at most (1 + stretch) times the best.
    pub stretch: f64,
    /// Keep a route only if at most this share of its length runs on edges of a kept route.
    pub max_shared: f64,
}

/// Road cost settings for edges.
#[derive(Clone, Debug)]
pub struct EdgeParams {
    pub g_max: f64,
    pub c_travel: f64,
    pub c_vertical: f64,
    /// Contributing area (ha) above which a cell is a channel (culverts).
    pub a_min: f64,
    pub culvert_breaks: Vec<f64>,
    pub culvert_costs: Vec<f64>,
    /// Allowance per implied hairpin on a steep edge.
    pub hairpin_cost: f64,
}

/// The rasters of the coarse grid, all nr x nc.
#[derive(Clone, Copy)]
pub struct Rasters<'a> {
    /// Working DEM (profiles).
    pub z: Grid<'a, f64>,
    /// Smoothed DEM (triangulation, saddles) and its gradient (sidehill).
    pub zs: Grid<'a, f64>,
    pub gx: Grid<'a, f64>,
    pub gy: Grid<'a, f64>,
    /// 1 where a road may go.
    pub open: Grid<'a, u8>,
    /// Contributing area, ha (0 at nodata).
    pub area: Grid<'a, f64>,
    /// Earthwork cost per meter over (sidehill slope, |grade|), including the base cost.
    pub section: SectionTable<'a>,
    /// Cell size (m).
    pub res: f64,
}

impl Rasters<'_> {
    fn nr(&self) -> usize {
        self.z.nr
    }

    fn nc(&self) -> usize {
        self.z.nc
    }

    fn inside(&self, r: i64, c: i64) -> bool {
        r >= 0 && c >= 0 && (r as usize) < self.nr() && (c as usize) < self.nc()
    }

    fn is_open(&self, r: i64, c: i64) -> bool {
        self.inside(r, c) && self.open.at(r, c) != 0
    }
}

fn bilinear(a: &Grid<f64>, fr: f64, fc: f64) -> f64 {
    let (nr, nc) = (a.nr as i64, a.nc as i64);
    if fr < 0.0 || fc < 0.0 || fr > (nr - 1) as f64 || fc > (nc - 1) as f64 {
        return f64::NAN;
    }
    let r0 = (fr as i64).min(nr - 2).max(0);
    let c0 = (fc as i64).min(nc - 2).max(0);
    let (wr, wc) = (fr - r0 as f64, fc - c0 as f64);
    let (r1, c1) = ((r0 + 1).min(nr - 1), (c0 + 1).min(nc - 1));
    (1.0 - wr) * ((1.0 - wc) * a.at(r0, c0) + wc * a.at(r0, c1)) + wr * ((1.0 - wc) * a.at(r1, c0) + wc * a.at(r1, c1))
}

/// Saddles of the smoothed DEM: interior cells whose 8 neighbors, in circular order, switch
/// between higher and lower at least four times.
pub fn saddles(zs: &Grid<f64>) -> Vec<(usize, usize)> {
    const RING: [(i64, i64); 8] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];
    let mut out = Vec::new();
    for r in 1..zs.nr as i64 - 1 {
        for c in 1..zs.nc as i64 - 1 {
            let z0 = zs.at(r, c);
            if z0.is_nan() {
                continue;
            }
            let mut signs = [0i8; 8];
            let mut ok = true;
            for (i, &(dr, dc)) in RING.iter().enumerate() {
                let v = zs.at(r + dr, c + dc);
                if v.is_nan() {
                    ok = false;
                    break;
                }
                signs[i] = if v > z0 { 1 } else { -1 };
            }
            if ok && (0..8).filter(|&i| signs[i] != signs[(i + 1) % 8]).count() >= 4 {
                out.push((r as usize, c as usize));
            }
        }
    }
    out
}

/// A graph node: grid cell and kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub r: usize,
    pub c: usize,
    pub kind: u8,
}

/// Nodes: triangulation vertices, channel and saddle nodes (one per block), background (one per
/// block), and the two endpoints (always, and first: indices 0 and 1). Only open cells.
pub fn build_nodes(t: &Rasters, p: &TinParams, start: (usize, usize), goal: (usize, usize)) -> Vec<Node> {
    let (nr, nc) = (t.nr(), t.nc());
    let mut nodes = vec![
        Node { r: start.0, c: start.1, kind: KIND_ENDPOINT },
        Node { r: goal.0, c: goal.1, kind: KIND_ENDPOINT },
    ];
    let mut seen: HashSet<usize> = [start.0 * nc + start.1, goal.0 * nc + goal.1].into_iter().collect();
    let mut add = |nodes: &mut Vec<Node>, r: usize, c: usize, kind: u8| {
        if t.is_open(r as i64, c as i64) && seen.insert(r * nc + c) {
            nodes.push(Node { r, c, kind });
        }
    };
    // Triangulation of the smoothed DEM, nodata filled from the nearest data: a cliff at the
    // data's edge would draw vertices all along it.
    let heights = terrain::fill_nearest(&t.zs.data, nr, nc);
    if let Ok((pts, _tris)) = delatin::triangulate(&heights, (nc, nr), delatin::Error(p.tolerance)) {
        for (x, y) in pts {
            add(&mut nodes, y, x, KIND_TIN);
        }
    }
    let b = p.block.max(1);
    let (br, bc) = (nr.div_ceil(b), nc.div_ceil(b));
    // Channel nodes: the largest open channel cell per block.
    let mut best: HashMap<usize, (f64, usize, usize)> = HashMap::new();
    for r in 0..nr {
        for c in 0..nc {
            let a = t.area.at(r as i64, c as i64);
            if a >= p.channel_node_area && t.is_open(r as i64, c as i64) {
                let key = (r / b) * bc + c / b;
                if best.get(&key).is_none_or(|&(a0, _, _)| a > a0) {
                    best.insert(key, (a, r, c));
                }
            }
        }
    }
    let mut chan: Vec<_> = best.into_values().collect();
    chan.sort_by(|a, b| (a.1, a.2).cmp(&(b.1, b.2)));
    for (_, r, c) in chan {
        add(&mut nodes, r, c, KIND_CHANNEL);
    }
    // Saddles: the lowest open saddle per block.
    let mut low: HashMap<usize, (f64, usize, usize)> = HashMap::new();
    for (r, c) in saddles(&t.zs) {
        if !t.is_open(r as i64, c as i64) {
            continue;
        }
        let z = t.zs.at(r as i64, c as i64);
        let key = (r / b) * bc + c / b;
        if low.get(&key).is_none_or(|&(z0, _, _)| z < z0) {
            low.insert(key, (z, r, c));
        }
    }
    let mut sad: Vec<_> = low.into_values().collect();
    sad.sort_by(|a, b| (a.1, a.2).cmp(&(b.1, b.2)));
    for (_, r, c) in sad {
        add(&mut nodes, r, c, KIND_SADDLE);
    }
    // Background: block centers.
    for i in 0..br {
        for j in 0..bc {
            let (r, c) = ((i * b + b / 2).min(nr - 1), (j * b + b / 2).min(nc - 1));
            add(&mut nodes, r, c, KIND_BACKGROUND);
        }
    }
    nodes
}

/// What an edge costs, split up.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeCost {
    pub total: f64,
    pub construction: f64,
    pub operating: f64,
    /// Road length (m), stretched on steep edges.
    pub length: f64,
    /// Implied hairpins.
    pub hairpins: u32,
}

/// Cost of a straight edge between two cells (None if infeasible). Symmetric.
pub fn edge_cost(t: &Rasters, e: &EdgeParams, a: (usize, usize), b: (usize, usize)) -> Option<EdgeCost> {
    let (r0, c0) = (a.0 as f64, a.1 as f64);
    let (dr, dc) = (b.0 as f64 - r0, b.1 as f64 - c0);
    let lc = dr.hypot(dc);
    if lc == 0.0 {
        return None;
    }
    let n = ((2.0 * lc).ceil() as usize).max(1);
    // Unit tangent, east and north.
    let (tx, ty) = (dc / lc, -dr / lc);
    let z_a = t.z.at(a.0 as i64, a.1 as i64);
    let z_b = t.z.at(b.0 as i64, b.1 as i64);
    if z_a.is_nan() || z_b.is_nan() {
        return None;
    }
    let mut zprev = z_a;
    let mut vert = 0.0;
    let mut sides = Vec::with_capacity(n);
    let mut culv = 0.0;
    let (mut in_run, mut run_max) = (false, 0.0f64);
    for i in 0..n {
        let f = (i as f64 + 0.5) / n as f64;
        let (fr, fc) = (r0 + f * dr, c0 + f * dc);
        let (cr, cc) = ((fr + 0.5).floor() as i64, (fc + 0.5).floor() as i64);
        if !t.is_open(cr, cc) {
            return None;
        }
        let zk = bilinear(&t.z, fr, fc);
        let ax = bilinear(&t.gx, fr, fc);
        let ay = bilinear(&t.gy, fr, fc);
        if !(zk.is_finite() && ax.is_finite() && ay.is_finite()) {
            return None;
        }
        vert += (zk - zprev).abs();
        zprev = zk;
        sides.push((-ax * ty + ay * tx).abs());
        let area = t.area.at(cr, cc);
        if area >= e.a_min {
            run_max = if in_run { run_max.max(area) } else { area };
            in_run = true;
        } else if in_run {
            culv += culvert_cost(run_max, &e.culvert_breaks, &e.culvert_costs);
            in_run = false;
        }
    }
    if in_run {
        culv += culvert_cost(run_max, &e.culvert_breaks, &e.culvert_costs);
    }
    vert += (z_b - zprev).abs();
    let lm = lc * t.res;
    let dz = (z_b - z_a).abs();
    let (le, hairpins) = if dz <= e.g_max * lm {
        (lm, 0u32)
    } else {
        let le = dz / e.g_max;
        (le, ((le / lm).ceil() as u32).saturating_sub(1))
    };
    let g = (dz / le).min(e.g_max);
    let mut earth = 0.0;
    for s in sides {
        let cpm = t.section.cost(s, g);
        if !cpm.is_finite() {
            return None;
        }
        earth += cpm * le / n as f64;
    }
    let construction = earth + culv + hairpins as f64 * e.hairpin_cost;
    let operating = e.c_travel * le + e.c_vertical * vert.max(dz);
    Some(EdgeCost { total: construction + operating, construction, operating, length: le, hairpins })
}

/// An undirected graph in compressed sparse row form.
pub struct Graph {
    pub nodes: Vec<Node>,
    /// Edges (u, v) with u < v, and their costs.
    pub edges: Vec<(u32, u32)>,
    pub costs: Vec<EdgeCost>,
    /// Adjacency: for node u, neighbors start[u]..start[u + 1] in (node, edge index).
    pub start: Vec<usize>,
    pub adj: Vec<(u32, u32)>,
}

/// Join each node to its `k` nearest neighbors within `edge_max` cells; keep feasible edges.
pub fn build_graph(t: &Rasters, e: &EdgeParams, p: &TinParams, nodes: Vec<Node>) -> Graph {
    let bucket = p.edge_max.max(1.0);
    let mut buckets: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for (i, nd) in nodes.iter().enumerate() {
        let key = ((nd.r as f64 / bucket) as i64, (nd.c as f64 / bucket) as i64);
        buckets.entry(key).or_default().push(i as u32);
    }
    let mut pairs: HashSet<(u32, u32)> = HashSet::new();
    for (i, nd) in nodes.iter().enumerate() {
        let (br, bc) = ((nd.r as f64 / bucket) as i64, (nd.c as f64 / bucket) as i64);
        let mut cand: Vec<(f64, u32)> = Vec::new();
        for dr in -1..=1 {
            for dc in -1..=1 {
                if let Some(list) = buckets.get(&(br + dr, bc + dc)) {
                    for &j in list {
                        if j as usize == i {
                            continue;
                        }
                        let o = &nodes[j as usize];
                        let d = (o.r as f64 - nd.r as f64).hypot(o.c as f64 - nd.c as f64);
                        if d <= p.edge_max {
                            cand.push((d, j));
                        }
                    }
                }
            }
        }
        cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(_, j) in cand.iter().take(p.k) {
            let (u, v) = if (i as u32) < j { (i as u32, j) } else { (j, i as u32) };
            pairs.insert((u, v));
        }
    }
    let mut pairs: Vec<(u32, u32)> = pairs.into_iter().collect();
    pairs.sort_unstable();
    let costed: Vec<Option<EdgeCost>> = map_edges(&pairs, |&(u, v)| {
        let (a, b) = (&nodes[u as usize], &nodes[v as usize]);
        edge_cost(t, e, (a.r, a.c), (b.r, b.c))
    });
    let mut edges = Vec::new();
    let mut costs = Vec::new();
    for (pr, c) in pairs.into_iter().zip(costed) {
        if let Some(c) = c {
            edges.push(pr);
            costs.push(c);
        }
    }
    let mut deg = vec![0usize; nodes.len() + 1];
    for &(u, v) in &edges {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut start = vec![0usize; nodes.len() + 1];
    for i in 0..nodes.len() {
        start[i + 1] = start[i] + deg[i];
    }
    let mut fill = start.clone();
    let mut adj = vec![(0u32, 0u32); start[nodes.len()]];
    for (k, &(u, v)) in edges.iter().enumerate() {
        adj[fill[u as usize]] = (v, k as u32);
        fill[u as usize] += 1;
        adj[fill[v as usize]] = (u, k as u32);
        fill[v as usize] += 1;
    }
    Graph { nodes, edges, costs, start, adj }
}

#[cfg(feature = "parallel")]
fn map_edges<F: Fn(&(u32, u32)) -> Option<EdgeCost> + Sync + Send>(pairs: &[(u32, u32)], f: F) -> Vec<Option<EdgeCost>> {
    use rayon::prelude::*;
    pairs.par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
fn map_edges<F: Fn(&(u32, u32)) -> Option<EdgeCost>>(pairs: &[(u32, u32)], f: F) -> Vec<Option<EdgeCost>> {
    pairs.iter().map(f).collect()
}

#[derive(Clone, Copy, PartialEq)]
struct Item(f64, u32);

impl Eq for Item {}

impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then_with(|| o.1.cmp(&self.1))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Dijkstra over nodes from `source` with edge weights `w`: (distance, parent edge or u32::MAX).
/// No turning costs; see [`search_directed`] for those.
pub fn dijkstra(g: &Graph, w: &[f64], source: usize) -> (Vec<f64>, Vec<u32>) {
    let n = g.nodes.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut par = vec![u32::MAX; n];
    let mut heap = BinaryHeap::new();
    dist[source] = 0.0;
    heap.push(Item(0.0, source as u32));
    while let Some(Item(d, u)) = heap.pop() {
        let u = u as usize;
        if d > dist[u] {
            continue;
        }
        for &(v, k) in &g.adj[g.start[u]..g.start[u + 1]] {
            let nd = d + w[k as usize];
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                par[v as usize] = k;
                heap.push(Item(nd, v));
            }
        }
    }
    (dist, par)
}

// --- directed edges: states that know where they came from, so turns can be charged -------------
//
// Undirected edge k = (u, v), u < v, gives directed edges 2k (u -> v) and 2k + 1 (v -> u).

#[inline(always)]
fn tail(g: &Graph, d: u32) -> usize {
    let (a, b) = g.edges[(d / 2) as usize];
    (if d % 2 == 0 { a } else { b }) as usize
}

#[inline(always)]
fn head(g: &Graph, d: u32) -> usize {
    let (a, b) = g.edges[(d / 2) as usize];
    (if d % 2 == 0 { b } else { a }) as usize
}

/// The directed edge leaving u along undirected edge k.
#[inline(always)]
fn out_of(g: &Graph, u: usize, k: u32) -> u32 {
    if g.edges[k as usize].0 as usize == u { 2 * k } else { 2 * k + 1 }
}

/// Heading change (radians, 0..pi) from directed edge a to directed edge b (b leaves a's head).
fn turn_angle(g: &Graph, a: u32, b: u32) -> f64 {
    let (p, q, r) = (&g.nodes[tail(g, a)], &g.nodes[head(g, a)], &g.nodes[head(g, b)]);
    let (ax, ay) = (q.c as f64 - p.c as f64, q.r as f64 - p.r as f64);
    let (bx, by) = (r.c as f64 - q.c as f64, r.r as f64 - q.r as f64);
    let na = ax.hypot(ay) * bx.hypot(by);
    if na == 0.0 {
        return 0.0;
    }
    ((ax * bx + ay * by) / na).clamp(-1.0, 1.0).acos()
}

/// A search over directed edges: dist[d] is the cheapest cost of arriving at head(d) along d,
/// counting edge weights and c_turn per radian of turn between consecutive edges.
pub struct DirectedSearch {
    pub dist: Vec<f64>,
    /// The directed edge before d, or u32::MAX if d leaves the source.
    pub prev: Vec<u32>,
}

/// Dijkstra over directed edges from `source`. A route may not double straight back along the
/// edge it arrived on.
pub fn search_directed(g: &Graph, w: &[f64], c_turn: f64, source: usize) -> DirectedSearch {
    let nd = 2 * g.edges.len();
    let mut dist = vec![f64::INFINITY; nd];
    let mut prev = vec![u32::MAX; nd];
    let mut heap = BinaryHeap::new();
    for &(_, k) in &g.adj[g.start[source]..g.start[source + 1]] {
        let d = out_of(g, source, k);
        if w[k as usize] < dist[d as usize] {
            dist[d as usize] = w[k as usize];
            heap.push(Item(w[k as usize], d));
        }
    }
    while let Some(Item(cost, d)) = heap.pop() {
        if cost > dist[d as usize] {
            continue;
        }
        let v = head(g, d);
        for &(_, k2) in &g.adj[g.start[v]..g.start[v + 1]] {
            if k2 == d / 2 {
                continue;
            }
            let d2 = out_of(g, v, k2);
            let nc = cost + c_turn * turn_angle(g, d, d2) + w[k2 as usize];
            if nc < dist[d2 as usize] {
                dist[d2 as usize] = nc;
                prev[d2 as usize] = d;
                heap.push(Item(nc, d2));
            }
        }
    }
    DirectedSearch { dist, prev }
}

/// Directed edges from the source to d, in order.
fn directed_path(s: &DirectedSearch, d: u32) -> Vec<u32> {
    let mut out = vec![d];
    let mut x = d;
    while s.prev[x as usize] != u32::MAX {
        x = s.prev[x as usize];
        out.push(x);
    }
    out.reverse();
    out
}

/// The cheapest arrival at node v: (cost, directed edge), or (inf, MAX).
fn best_arrival(g: &Graph, s: &DirectedSearch, v: usize) -> (f64, u32) {
    let mut best = (f64::INFINITY, u32::MAX);
    for &(_, k) in &g.adj[g.start[v]..g.start[v + 1]] {
        let d = out_of(g, v, k) ^ 1; // arriving at v along k
        if s.dist[d as usize] < best.0 {
            best = (s.dist[d as usize], d);
        }
    }
    best
}

/// The best route through node v from the two searches: (cost, arriving edge, leaving edge).
/// fwd is from the start; bwd from the goal, so bwd.dist[b ^ 1] is the cost from v's next node
/// back to the goal including edge b.
fn through(g: &Graph, fwd: &DirectedSearch, bwd: &DirectedSearch, c_turn: f64, v: usize, start: usize, goal: usize) -> (f64, u32, u32) {
    if v == start {
        let (c, d) = best_arrival(g, bwd, v);
        return (c, u32::MAX, if d == u32::MAX { u32::MAX } else { d ^ 1 });
    }
    if v == goal {
        let (c, d) = best_arrival(g, fwd, v);
        return (c, d, u32::MAX);
    }
    let mut best = (f64::INFINITY, u32::MAX, u32::MAX);
    let nbrs = &g.adj[g.start[v]..g.start[v + 1]];
    for &(_, ka) in nbrs {
        let a = out_of(g, v, ka) ^ 1;
        let ga = fwd.dist[a as usize];
        if !ga.is_finite() || ga >= best.0 {
            continue;
        }
        for &(_, kb) in nbrs {
            if kb == ka {
                continue;
            }
            let b = out_of(g, v, kb);
            let c = ga + c_turn * turn_angle(g, a, b) + bwd.dist[(b ^ 1) as usize];
            if c < best.0 {
                best = (c, a, b);
            }
        }
    }
    best
}

/// A route on the graph.
#[derive(Clone, Debug)]
pub struct Route {
    /// Nodes from the start to the goal.
    pub nodes: Vec<usize>,
    /// Undirected edges, in order.
    pub edges: Vec<u32>,
    /// Edge costs plus turning (in the operating part).
    pub cost: f64,
    pub construction: f64,
    pub operating: f64,
    pub length: f64,
    pub hairpins: u32,
    /// Total heading change at the nodes, radians.
    pub turning: f64,
    /// How it was found: "best", "saddle", "penalty" or "sampled".
    pub how: &'static str,
    /// The nodes' cells (row, col): routes from different graphs compare by these.
    pub coords: Vec<(usize, usize)>,
}

fn make_route(g: &Graph, path: &[u32], c_turn: f64, how: &'static str) -> Route {
    let mut nodes = vec![tail(g, path[0])];
    let (mut cons, mut op, mut len, mut hp, mut turn) = (0.0, 0.0, 0.0, 0u32, 0.0);
    for (i, &d) in path.iter().enumerate() {
        nodes.push(head(g, d));
        let c = &g.costs[(d / 2) as usize];
        cons += c.construction;
        op += c.operating;
        len += c.length;
        hp += c.hairpins;
        if i > 0 {
            turn += turn_angle(g, path[i - 1], d);
        }
    }
    op += c_turn * turn;
    let coords = route_coords(g, &nodes);
    Route {
        coords,
        nodes,
        edges: path.iter().map(|d| d / 2).collect(),
        cost: cons + op,
        construction: cons,
        operating: op,
        length: len,
        hairpins: hp,
        turning: turn,
        how,
    }
}

/// Share of route `a`'s length that runs on edges in `kept`.
fn shared(g: &Graph, a: &[u32], kept: &HashSet<u32>) -> f64 {
    let tot: f64 = a.iter().map(|&k| g.costs[k as usize].length).sum();
    let sh: f64 = a.iter().filter(|k| kept.contains(k)).map(|&k| g.costs[k as usize].length).sum();
    if tot > 0.0 { sh / tot } else { 1.0 }
}

/// The corridor search's result.
pub struct Corridors {
    pub routes: Vec<Route>,
    /// Cost of the best route through each node.
    pub field: Vec<f64>,
}

/// How arbitrary a route's position at node v is: near 1 where a neighbor off the route is about
/// as good (the route could have gone there), near 0 where every neighbor is much worse (a
/// constriction: a saddle, a narrow bench, a crossing).
fn arbitrariness(g: &Graph, field: &[f64], on_route: &HashSet<usize>, v: usize, best: f64, delta: f64) -> f64 {
    let lateral = g.adj[g.start[v]..g.start[v + 1]]
        .iter()
        .map(|&(u, _)| u as usize)
        .filter(|u| !on_route.contains(u))
        .map(|u| field[u])
        .fold(f64::INFINITY, f64::min);
    if !lateral.is_finite() {
        return 0.0;
    }
    (-((lateral / best - 1.0).max(0.0)) / delta).exp()
}

/// The best route, the corridor field, and up to n_alt distinct alternatives: routes forced
/// through saddles (ranked by the field), then a penalty loop whose penalties are graded by how
/// arbitrary each kept route's position was (constrictions are left open).
pub fn corridors(g: &Graph, p: &TinParams, c_turn: f64, start: usize, goal: usize) -> Corridors {
    let w: Vec<f64> = g.costs.iter().map(|c| c.total).collect();
    let fwd = search_directed(g, &w, c_turn, start);
    let bwd = search_directed(g, &w, c_turn, goal);
    let n = g.nodes.len();
    let thr: Vec<(f64, u32, u32)> = (0..n).map(|v| through(g, &fwd, &bwd, c_turn, v, start, goal)).collect();
    let field: Vec<f64> = thr.iter().map(|t| t.0).collect();
    let mut routes = Vec::new();
    let (best_cost, arrive) = best_arrival(g, &fwd, goal);
    if !best_cost.is_finite() {
        return Corridors { routes, field };
    }
    let best = make_route(g, &directed_path(&fwd, arrive), c_turn, "best");
    let limit = (1.0 + p.stretch) * best.cost;
    let mut kept: Vec<HashSet<u32>> = vec![best.edges.iter().copied().collect()];
    routes.push(best);
    let distinct = |edges: &[u32], kept: &[HashSet<u32>]| kept.iter().all(|k| shared(g, edges, k) <= p.max_shared);
    // Saddle candidates, cheapest forced route first.
    let mut sad: Vec<usize> = (0..n).filter(|&i| g.nodes[i].kind == KIND_SADDLE && field[i] <= limit).collect();
    sad.sort_by(|&a, &b| field[a].total_cmp(&field[b]).then(a.cmp(&b)));
    for s in sad {
        if routes.len() > p.n_alt {
            break;
        }
        let (_, a, b) = thr[s];
        if a == u32::MAX || b == u32::MAX {
            continue;
        }
        let mut path = directed_path(&fwd, a);
        let back: Vec<u32> = directed_path(&bwd, b ^ 1).into_iter().rev().map(|d| d ^ 1).collect();
        path.extend(back);
        let r = make_route(g, &path, c_turn, "saddle");
        if r.cost <= limit && distinct(&r.edges, &kept) {
            kept.push(r.edges.iter().copied().collect());
            routes.push(r);
        }
    }
    // Penalty loop, graded by the field: discourage a kept route where it could as well have
    // been elsewhere, not where it passes a constriction.
    let delta = 0.02;
    let best_cost = routes[0].cost;
    let mut wp = w.clone();
    let penalize = |wp: &mut Vec<f64>, r: &Route| {
        let on: HashSet<usize> = r.nodes.iter().copied().collect();
        let arb: Vec<f64> = r.nodes.iter().map(|&v| arbitrariness(g, &field, &on, v, best_cost, delta)).collect();
        for (i, &k) in r.edges.iter().enumerate() {
            let a = arb[i].min(arb[i + 1]);
            wp[k as usize] *= 1.0 + (p.penalty - 1.0) * a;
        }
    };
    for r in routes.clone().iter() {
        penalize(&mut wp, r);
    }
    for _ in 0..4 * p.n_alt {
        if routes.len() > p.n_alt {
            break;
        }
        let s = search_directed(g, &wp, c_turn, start);
        let (c, a) = best_arrival(g, &s, goal);
        if !c.is_finite() {
            break;
        }
        let r = make_route(g, &directed_path(&s, a), c_turn, "penalty");
        penalize(&mut wp, &r);
        if r.cost <= limit && distinct(&r.edges, &kept) {
            kept.push(r.edges.iter().copied().collect());
            routes.push(r);
        }
    }
    Corridors { routes, field }
}

// --- sampling: cost uncertainty, and random node placements ------------------------------------

/// Points along a route, bucketed, for "how much of this other route lies near it".
pub struct Near {
    cell: f64,
    near: f64,
    buckets: HashMap<(i64, i64), Vec<(f64, f64)>>,
}

impl Near {
    /// `coords`: the route's nodes (row, col); `near`: the distance (cells) that counts as near.
    pub fn new(coords: &[(usize, usize)], near: f64) -> Self {
        let cell = near.max(1.0);
        let mut buckets: HashMap<(i64, i64), Vec<(f64, f64)>> = HashMap::new();
        let step = (0.5 * near).max(0.5);
        for w in coords.windows(2) {
            let (a, b) = ((w[0].0 as f64, w[0].1 as f64), (w[1].0 as f64, w[1].1 as f64));
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            let n = ((len / step).ceil() as usize).max(1);
            for i in 0..=n {
                let f = i as f64 / n as f64;
                let pt = (a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1));
                buckets.entry(((pt.0 / cell) as i64, (pt.1 / cell) as i64)).or_default().push(pt);
            }
        }
        Near { cell, near, buckets }
    }

    pub fn is_near(&self, r: f64, c: f64) -> bool {
        let (br, bc) = ((r / self.cell) as i64, (c / self.cell) as i64);
        (-1..=1).any(|dr| {
            (-1..=1).any(|dc| {
                self.buckets
                    .get(&(br + dr, bc + dc))
                    .is_some_and(|l| l.iter().any(|&(pr, pc)| (pr - r).hypot(pc - c) <= self.near))
            })
        })
    }

    /// Share of a route's length (its nodes, row and col) whose segments lie near this route.
    pub fn share(&self, coords: &[(usize, usize)]) -> f64 {
        let (mut tot, mut within) = (0.0, 0.0);
        for w in coords.windows(2) {
            let (a, b) = ((w[0].0 as f64, w[0].1 as f64), (w[1].0 as f64, w[1].1 as f64));
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            tot += len;
            let mid = (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
            if self.is_near(a.0, a.1) && self.is_near(b.0, b.1) && self.is_near(mid.0, mid.1) {
                within += len;
            }
        }
        if tot > 0.0 { within / tot } else { 0.0 }
    }
}

/// Which of the kept routes a route is: the one with the largest share of its length near it, if at
/// least half (ties go to the earlier, cheaper one); routes.len() for none of them.
fn classify(near: &[Near], coords: &[(usize, usize)]) -> usize {
    let (mut who, mut most) = (near.len(), 0.5);
    for (j, nr) in near.iter().enumerate() {
        let sh = nr.share(coords);
        if sh > most || (who == near.len() && sh >= most) {
            who = j;
            most = sh;
        }
    }
    who
}

/// Uncertainty of the costs, as lognormal factors with mean 1 (a sensitivity check: the spreads
/// are themselves judgments).
#[derive(Clone, Debug)]
pub struct McParams {
    pub samples: usize,
    /// Spread (sigma of the log) of one factor on all construction costs (the cost unit, rock...).
    pub sigma_construction: f64,
    /// ... of one factor on all operating costs (fuel, fleet, haul life).
    pub sigma_operating: f64,
    /// ... of an independent factor on each edge (local soil, rock, drainage).
    pub sigma_edge: f64,
    pub seed: u64,
    /// A sample's route counts as a kept route when most of its length lies within this many
    /// cells of it (the corridor buffer).
    pub near: f64,
}

/// What the samples found: the share won by each kept route (index len() is "none of them").
pub struct McResult {
    pub wins: Vec<f64>,
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(seed: u64) -> f64 {
    ((splitmix(seed) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A standard normal value from a seed (Box-Muller on two hashed uniforms).
fn normal(seed: u64) -> f64 {
    let (u1, u2) = (uniform(seed), uniform(seed ^ 0xD1B5_4A32_D192_ED03));
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lognormal(sigma: f64, seed: u64) -> f64 {
    if sigma == 0.0 { 1.0 } else { (sigma * normal(seed) - 0.5 * sigma * sigma).exp() }
}

fn route_coords(g: &Graph, nodes: &[usize]) -> Vec<(usize, usize)> {
    nodes.iter().map(|&v| (g.nodes[v].r, g.nodes[v].c)).collect()
}

/// Cost sensitivity: re-cost the graph for sampled cost factors and count which kept route each
/// sample's best route is.
pub fn monte_carlo(g: &Graph, routes: &[Route], c_turn: f64, start: usize, goal: usize, mc: &McParams) -> McResult {
    let near: Vec<Near> = routes.iter().map(|r| Near::new(&r.coords, mc.near)).collect();
    let one = |i: usize| -> Option<usize> {
        let base = splitmix(mc.seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let fc = lognormal(mc.sigma_construction, base ^ 1);
        let fo = lognormal(mc.sigma_operating, base ^ 2);
        let w: Vec<f64> = g
            .costs
            .iter()
            .enumerate()
            .map(|(k, c)| (c.construction * fc + c.operating * fo) * lognormal(mc.sigma_edge, base ^ (k as u64 + 3) << 1))
            .collect();
        let s = search_directed(g, &w, c_turn * fo, start);
        let (cost, a) = best_arrival(g, &s, goal);
        if !cost.is_finite() {
            return None;
        }
        let path = directed_path(&s, a);
        let mut nodes = vec![tail(g, path[0])];
        nodes.extend(path.iter().map(|&d| head(g, d)));
        Some(classify(&near, &route_coords(g, &nodes)))
    };
    let results: Vec<Option<usize>> = map_samples(mc.samples, one);
    let mut wins = vec![0.0; routes.len() + 1];
    let n = results.iter().filter(|r| r.is_some()).count().max(1) as f64;
    for who in results.into_iter().flatten() {
        wins[who] += 1.0 / n;
    }
    McResult { wins }
}

/// Random node placements: each sample adds `nodes` random open cells to the terrain-informed
/// nodes, rebuilds the graph, and finds its best route.
#[derive(Clone, Debug)]
pub struct NodeSampling {
    pub samples: usize,
    pub nodes: usize,
    pub seed: u64,
    /// As McParams::near.
    pub near: f64,
}

/// What the node samples found: the share each kept route won (index len(): none of them), and
/// the routes that were none of them, distinct from each other (new corridors), with the share of
/// samples each stands for.
pub struct NodeSampleResult {
    pub wins: Vec<f64>,
    pub found: Vec<(Route, f64)>,
}

/// Run the node samples. Each builds its own graph, so samples run one after another (the graph
/// build is itself parallel); `progress` is told after each (sample index, which route it was).
#[allow(clippy::too_many_arguments)]
pub fn node_samples(
    t: &Rasters,
    e: &EdgeParams,
    p: &TinParams,
    c_turn: f64,
    start: (usize, usize),
    goal: (usize, usize),
    routes: &[Route],
    ns: &NodeSampling,
    progress: &mut dyn FnMut(usize, usize),
) -> NodeSampleResult {
    let mut near: Vec<Near> = routes.iter().map(|r| Near::new(&r.coords, ns.near)).collect();
    let k_kept = routes.len();
    let seed_nodes = build_nodes(t, p, start, goal);
    let (nr, nc) = (t.nr(), t.nc());
    let mut counts = vec![0usize; k_kept + 1];
    let mut found: Vec<(Route, usize)> = Vec::new();
    let mut done = 0usize;
    for i in 0..ns.samples {
        let mut nodes = seed_nodes.clone();
        let mut seen: HashSet<usize> = nodes.iter().map(|n| n.r * nc + n.c).collect();
        let base = splitmix(ns.seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (mut added, mut tries) = (0usize, 0u64);
        while added < ns.nodes && tries < 50 * ns.nodes as u64 + 100 {
            let r = (uniform(base ^ (2 * tries)) * nr as f64) as usize;
            let c = (uniform(base ^ (2 * tries + 1)) * nc as f64) as usize;
            tries += 1;
            if r < nr && c < nc && t.is_open(r as i64, c as i64) && seen.insert(r * nc + c) {
                nodes.push(Node { r, c, kind: KIND_SAMPLED });
                added += 1;
            }
        }
        let g = build_graph(t, e, p, nodes);
        let w: Vec<f64> = g.costs.iter().map(|c| c.total).collect();
        let s = search_directed(&g, &w, c_turn, 0);
        let (cost, a) = best_arrival(&g, &s, 1);
        if !cost.is_finite() {
            continue;
        }
        let r = make_route(&g, &directed_path(&s, a), c_turn, "sampled");
        done += 1;
        let who = classify(&near, &r.coords);
        if who < k_kept {
            counts[who] += 1;
        } else if who < near.len() {
            found[who - k_kept].1 += 1;
            counts[k_kept] += 1;
        } else {
            near.push(Near::new(&r.coords, ns.near));
            found.push((r, 1));
            counts[k_kept] += 1;
        }
        progress(i, who);
    }
    let n = done.max(1) as f64;
    NodeSampleResult {
        wins: counts.iter().map(|&c| c as f64 / n).collect(),
        found: found.into_iter().map(|(r, c)| (r, c as f64 / n)).collect(),
    }
}

#[cfg(feature = "parallel")]
fn map_samples<T: Send, F: Fn(usize) -> T + Sync + Send>(n: usize, f: F) -> Vec<T> {
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
fn map_samples<T, F: Fn(usize) -> T>(n: usize, f: F) -> Vec<T> {
    (0..n).map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Vec<f64> {
        // Flat cost per meter (10) up to sidehill 0.9, infeasible beyond.
        let (ns, ng) = (11, 3);
        (0..ns * ng).map(|k| if k / ng == ns - 1 { f64::INFINITY } else { 10.0 }).collect()
    }

    #[test]
    fn a_ridge_is_crossed_at_its_saddle() {
        // A ridge along the middle row with a low saddle near the east end; endpoints either side.
        let (nr, nc) = (41usize, 61usize);
        let z: Vec<f64> = (0..nr * nc)
            .map(|k| {
                let (r, c) = ((k / nc) as f64, (k % nc) as f64);
                let ridge = 60.0 * (-((r - 20.0) / 4.0).powi(2)).exp();
                let notch = 1.0 - 0.9 * (-((c - 48.0) / 4.0).powi(2)).exp();
                ridge * notch
            })
            .collect();
        let (mut gx, mut gy) = (vec![0.0; nr * nc], vec![0.0; nr * nc]);
        for r in 1..nr - 1 {
            for c in 1..nc - 1 {
                let k = r * nc + c;
                gx[k] = (z[k + 1] - z[k - 1]) / 20.0;
                gy[k] = -(z[k + nc] - z[k - nc]) / 20.0;
            }
        }
        let open = vec![1u8; nr * nc];
        let area = vec![0.0; nr * nc];
        let tab = table();
        let t = Rasters {
            z: Grid::new(&z, nr, nc),
            zs: Grid::new(&z, nr, nc),
            gx: Grid::new(&gx, nr, nc),
            gy: Grid::new(&gy, nr, nc),
            open: Grid::new(&open, nr, nc),
            area: Grid::new(&area, nr, nc),
            section: SectionTable { data: &tab, ns: 11, ng: 3, ds: 0.1, dg: 0.05 },
            res: 10.0,
        };
        let e = EdgeParams { g_max: 0.1, c_travel: 1.0, c_vertical: 200.0, a_min: 2.0, culvert_breaks: vec![10.0],
                             culvert_costs: vec![50.0, 100.0], hairpin_cost: 500.0 };
        let p = TinParams { tolerance: 1.0, k: 12, edge_max: 8.0, block: 5, channel_node_area: 1e9, n_alt: 2,
                            penalty: 1.5, stretch: 0.5, max_shared: 0.5 };
        let nodes = build_nodes(&t, &p, (5, 10), (35, 10));
        // The notch in the ridge is found as a saddle.
        assert!(saddles(&t.zs).iter().any(|&(r, c)| r == 20 && (c as i64 - 48).abs() <= 1));
        let g = build_graph(&t, &e, &p, nodes);
        let c = corridors(&g, &p, 0.0, 0, 1);
        let best = &c.routes[0];
        // The best route crosses the ridge row near the notch, not straight over the top.
        let crossing = best.nodes.iter().map(|&i| g.nodes[i]).min_by_key(|n| (n.r as i64 - 20).abs()).unwrap();
        assert!((crossing.c as i64 - 48).abs() <= 6, "crossed at column {}", crossing.c);
        // No node's best route is cheaper than the best route.
        assert!(c.field.iter().all(|&f| !f.is_finite() || f >= best.cost - 1e-6));
        // Charging turns straightens the route.
        let ct = corridors(&g, &p, 1e4, 0, 1);
        assert!(ct.routes[0].turning <= best.turning + 1e-9);
        assert!(ct.routes[0].cost >= best.cost - 1e-6);
        // Monte Carlo: shares sum to one, and every sample's route starts at the start.
        let mc = McParams { samples: 24, sigma_construction: 0.3, sigma_operating: 0.3, sigma_edge: 0.2, seed: 7, near: 4.0 };
        let r = monte_carlo(&g, &c.routes, 0.0, 0, 1, &mc);
        assert!((r.wins.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let again = monte_carlo(&g, &c.routes, 0.0, 0, 1, &mc);
        assert_eq!(r.wins, again.wins); // deterministic for a seed
        // Node sampling: every sample is one of the kept routes or a new one, and the best route
        // (through the notch) keeps winning when nodes move.
        let ns = NodeSampling { samples: 6, nodes: 150, seed: 3, near: 4.0 };
        let mut seen = 0;
        let res = node_samples(&t, &e, &p, 0.0, (5, 10), (35, 10), &c.routes, &ns, &mut |_, _| seen += 1);
        assert_eq!(seen, 6);
        let total: f64 = res.wins.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(res.wins[0] >= 0.5, "best corridor share {}", res.wins[0]);
        assert!((res.found.iter().map(|f| f.1).sum::<f64>() - res.wins[c.routes.len()]).abs() < 1e-9);
    }
}
