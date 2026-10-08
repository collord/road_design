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
    // Triangulation of the smoothed DEM (nodata filled with the lowest value).
    let zmin = t.zs.data.iter().copied().filter(|v| !v.is_nan()).fold(f64::INFINITY, f64::min);
    let heights: Vec<f64> = t.zs.data.iter().map(|&v| if v.is_nan() { zmin } else { v }).collect();
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

/// Dijkstra from `source` with edge weights `w`: (distance, parent edge index or u32::MAX).
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

fn other(g: &Graph, k: u32, u: usize) -> usize {
    let (a, b) = g.edges[k as usize];
    if a as usize == u { b as usize } else { a as usize }
}

/// Edges of the tree path from the source to `node` (in order from the source).
fn tree_path(g: &Graph, par: &[u32], node: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut u = node;
    while par[u] != u32::MAX {
        out.push(par[u]);
        u = other(g, par[u], u);
    }
    out.reverse();
    out
}

/// A route on the graph.
#[derive(Clone, Debug)]
pub struct Route {
    /// Nodes from the start to the goal.
    pub nodes: Vec<usize>,
    pub edges: Vec<u32>,
    pub cost: f64,
    pub construction: f64,
    pub operating: f64,
    pub length: f64,
    pub hairpins: u32,
    /// How it was found: "best", "saddle" or "penalty".
    pub how: &'static str,
}

fn make_route(g: &Graph, edges: Vec<u32>, start: usize, how: &'static str) -> Route {
    let mut nodes = vec![start];
    let (mut cost, mut cons, mut op, mut len, mut hp) = (0.0, 0.0, 0.0, 0.0, 0u32);
    for &k in &edges {
        let u = *nodes.last().unwrap();
        nodes.push(other(g, k, u));
        let c = &g.costs[k as usize];
        cost += c.total;
        cons += c.construction;
        op += c.operating;
        len += c.length;
        hp += c.hairpins;
    }
    Route { nodes, edges, cost, construction: cons, operating: op, length: len, hairpins: hp, how }
}

/// Share of route `a`'s length that runs on edges in `kept`.
fn shared(g: &Graph, a: &Route, kept: &HashSet<u32>) -> f64 {
    let tot: f64 = a.edges.iter().map(|&k| g.costs[k as usize].length).sum();
    let sh: f64 = a.edges.iter().filter(|k| kept.contains(k)).map(|&k| g.costs[k as usize].length).sum();
    if tot > 0.0 { sh / tot } else { 1.0 }
}

/// The corridor search: the best route, the corridor field (best route cost through each node),
/// and up to n_alt distinct alternatives (saddle candidates first, then a penalty loop).
pub struct Corridors {
    pub routes: Vec<Route>,
    pub field: Vec<f64>,
}

pub fn corridors(g: &Graph, p: &TinParams, start: usize, goal: usize) -> Corridors {
    let w: Vec<f64> = g.costs.iter().map(|c| c.total).collect();
    let (gf, pf) = dijkstra(g, &w, start);
    let (gb, pb) = dijkstra(g, &w, goal);
    let field: Vec<f64> = gf.iter().zip(&gb).map(|(a, b)| a + b).collect();
    let mut routes = Vec::new();
    if !gf[goal].is_finite() {
        return Corridors { routes, field };
    }
    let best = make_route(g, tree_path(g, &pf, goal), start, "best");
    let limit = (1.0 + p.stretch) * best.cost;
    let mut kept_edges: Vec<HashSet<u32>> = vec![best.edges.iter().copied().collect()];
    routes.push(best);
    let distinct = |r: &Route, kept: &[HashSet<u32>]| kept.iter().all(|k| shared(g, r, k) <= p.max_shared);
    // Saddle candidates, cheapest forced route first.
    let mut sad: Vec<usize> = (0..g.nodes.len())
        .filter(|&i| g.nodes[i].kind == KIND_SADDLE && field[i] <= limit)
        .collect();
    sad.sort_by(|&a, &b| field[a].total_cmp(&field[b]).then(a.cmp(&b)));
    for s in sad {
        if routes.len() > p.n_alt {
            break;
        }
        let mut edges = tree_path(g, &pf, s);
        let mut back = tree_path(g, &pb, s);
        back.reverse();
        edges.extend(back);
        let r = make_route(g, edges, start, "saddle");
        if r.cost <= limit && distinct(&r, &kept_edges) {
            kept_edges.push(r.edges.iter().copied().collect());
            routes.push(r);
        }
    }
    // Penalty loop: discourage the edges routes have used, search again.
    let mut wp = w.clone();
    for r in &routes {
        for &k in &r.edges {
            wp[k as usize] *= p.penalty;
        }
    }
    for _ in 0..4 * p.n_alt {
        if routes.len() > p.n_alt {
            break;
        }
        let (d, par) = dijkstra(g, &wp, start);
        if !d[goal].is_finite() {
            break;
        }
        let r = make_route(g, tree_path(g, &par, goal), start, "penalty");
        for &k in &r.edges {
            wp[k as usize] *= p.penalty;
        }
        if r.cost <= limit && distinct(&r, &kept_edges) {
            kept_edges.push(r.edges.iter().copied().collect());
            routes.push(r);
        }
    }
    Corridors { routes, field }
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
        let c = corridors(&g, &p, 0, 1);
        let best = &c.routes[0];
        // The best route crosses the ridge row near the notch, not straight over the top.
        let crossing = best.nodes.iter().map(|&i| g.nodes[i]).min_by_key(|n| (n.r as i64 - 20).abs()).unwrap();
        assert!((crossing.c as i64 - 48).abs() <= 6, "crossed at column {}", crossing.c);
        assert!(c.field.iter().zip(0..).all(|(&f, i)| !f.is_finite() || f >= best.cost - 1e-6 || i > g.nodes.len()));
    }
}
