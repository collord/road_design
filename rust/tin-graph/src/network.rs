//! A road network joining an access point to many pads (drill pads, say), on the graph.
//!
//! The cheapest such network is a Steiner tree, which is NP-hard to find; this builds a good one.
//! It grows the network from the access point, each time connecting the pad cheapest to reach from
//! the network so far (a search starting from every node already on it), then improves it by
//! taking each pad's own end branch off and reconnecting the pad at least cost, until nothing
//! gets cheaper.
//!
//! Costs: construction once per edge; operating for every pad's traffic over its whole way to the
//! access point (one unit per pad), so a trunk pays the operating cost of all the pads beyond it.
//!
//! Pads: a rig working on a pad, or stuck there, blocks the road through it. A new branch never
//! passes through another pad, and a pad may lie on the way to at most `b` other pads (its
//! "pads beyond"); with b = 0 every pad ends its own branch.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{head, out_of, tail, turn_angle, Graph};

const NONE: u32 = u32::MAX;

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

/// The network so far: a tree rooted at the access point, as parent pointers on the graph.
pub struct Network {
    /// The access point's node, and each pad's.
    pub root: usize,
    pub pads: Vec<usize>,
    /// Pads beyond which no more than this many pads may lie.
    pub b: usize,
    c_turn: f64,
    pub connected: Vec<bool>,
    /// Pads no search could reach (closed ground, or the pad rule).
    pub unreachable: Vec<bool>,
    /// Per node: on the network; its parent node and the edge to it (NONE at the root, off it).
    pub in_tree: Vec<bool>,
    pub parent: Vec<u32>,
    pub parent_edge: Vec<u32>,
    children: Vec<u32>,
    /// Per node: the pad there, or -1.
    pad_of: Vec<i32>,
    /// Pads in the order they were connected.
    pub order: Vec<usize>,
}

/// A pad connected: which, the directed edges of its new branch from the network to it, and
/// what connecting it added (construction of the branch plus the pad's operating cost to the
/// access point).
pub struct Step {
    pub pad: usize,
    pub path: Vec<u32>,
    pub cost: f64,
}

impl Network {
    pub fn new(g: &Graph, root: usize, pads: &[usize], b: usize, c_turn: f64) -> Self {
        let n = g.nodes.len();
        let mut pad_of = vec![-1i32; n];
        for (i, &v) in pads.iter().enumerate() {
            if pad_of[v] < 0 && v != root {
                pad_of[v] = i as i32;
            }
        }
        let mut in_tree = vec![false; n];
        in_tree[root] = true;
        // A pad on the access point's node is connected already. (Pads sharing a node: give each
        // its own cell; the second would never be reached.)
        let connected: Vec<bool> = pads.iter().map(|&v| v == root).collect();
        Network {
            root,
            pads: pads.to_vec(),
            b,
            c_turn,
            unreachable: vec![false; pads.len()],
            connected,
            in_tree,
            parent: vec![NONE; n],
            parent_edge: vec![NONE; n],
            children: vec![0; n],
            pad_of,
            order: Vec::new(),
        }
    }

    /// Nodes from v up to the access point, v first.
    fn up(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        let mut cur = Some(v);
        std::iter::from_fn(move || {
            let x = cur?;
            cur = if self.parent[x] == NONE { None } else { Some(self.parent[x] as usize) };
            Some(x)
        })
    }

    /// Per pad, how many connected pads' way to the access point passes through it.
    pub fn beyond(&self) -> Vec<usize> {
        let mut out = vec![0; self.pads.len()];
        for (i, &v) in self.pads.iter().enumerate() {
            if !self.connected[i] || v == self.root {
                continue;
            }
            for u in self.up(v).skip(1) {
                if self.pad_of[u] >= 0 {
                    out[self.pad_of[u] as usize] += 1;
                }
            }
        }
        out
    }

    /// Per node on the network: the operating cost of the way from it to the access point.
    fn op_to_root(&self, g: &Graph) -> Vec<f64> {
        let mut op = vec![f64::NAN; g.nodes.len()];
        op[self.root] = 0.0;
        for v in 0..g.nodes.len() {
            if !self.in_tree[v] || !op[v].is_nan() {
                continue;
            }
            let chain: Vec<usize> = self.up(v).take_while(|&u| op[u].is_nan()).collect();
            for &u in chain.iter().rev() {
                op[u] = op[self.parent[u] as usize] + g.costs[self.parent_edge[u] as usize].operating;
            }
        }
        op
    }

    /// Whether a new pad may join the network at v: every pad on v's way to the access point
    /// (v included) must stay within b pads beyond.
    fn junction_ok(&self, v: usize, beyond: &[usize]) -> bool {
        self.up(v).all(|u| self.pad_of[u] < 0 || beyond[self.pad_of[u] as usize] < self.b)
    }

    /// The cheapest new branch from the network to any of the target pads.
    fn search(&self, g: &Graph, target: &[bool]) -> Option<Step> {
        let w: Vec<f64> = g.costs.iter().map(|c| c.total).collect();
        let op = self.op_to_root(g);
        let beyond = self.beyond();
        let nd = 2 * g.edges.len();
        let mut dist = vec![f64::INFINITY; nd];
        let mut prev = vec![NONE; nd];
        let mut heap = BinaryHeap::new();
        for v in 0..g.nodes.len() {
            if !self.in_tree[v] || !self.junction_ok(v, &beyond) {
                continue;
            }
            for &(u, k) in &g.adj[g.start[v]..g.start[v + 1]] {
                if self.in_tree[u as usize] {
                    continue;
                }
                let d = out_of(g, v, k);
                let c = op[v] + w[k as usize];
                if c < dist[d as usize] {
                    dist[d as usize] = c;
                    heap.push(Item(c, d));
                }
            }
        }
        while let Some(Item(cost, d)) = heap.pop() {
            if cost > dist[d as usize] {
                continue;
            }
            let h = head(g, d);
            let p = self.pad_of[h];
            if p >= 0 {
                if target[p as usize] {
                    let mut path = vec![d];
                    while prev[*path.last().unwrap() as usize] != NONE {
                        path.push(prev[*path.last().unwrap() as usize]);
                    }
                    path.reverse();
                    return Some(Step { pad: p as usize, path, cost });
                }
                continue; // a new branch never passes through another pad
            }
            for &(u, k2) in &g.adj[g.start[h]..g.start[h + 1]] {
                if k2 == d / 2 || self.in_tree[u as usize] {
                    continue;
                }
                let d2 = out_of(g, h, k2);
                let nc = cost + self.c_turn * turn_angle(g, d, d2) + w[k2 as usize];
                if nc < dist[d2 as usize] {
                    dist[d2 as usize] = nc;
                    prev[d2 as usize] = d;
                    heap.push(Item(nc, d2));
                }
            }
        }
        None
    }

    fn attach(&mut self, g: &Graph, path: &[u32], pad: usize) {
        for &d in path {
            let (a, b) = (tail(g, d), head(g, d));
            self.parent[b] = a as u32;
            self.parent_edge[b] = d / 2;
            self.in_tree[b] = true;
            self.children[a] += 1;
        }
        self.connected[pad] = true;
    }

    /// Connect the pad cheapest to reach. None when every pad is connected or unreachable.
    pub fn grow(&mut self, g: &Graph) -> Option<Step> {
        let target: Vec<bool> = (0..self.pads.len()).map(|i| !self.connected[i] && !self.unreachable[i]).collect();
        if !target.iter().any(|&t| t) {
            return None;
        }
        match self.search(g, &target) {
            Some(step) => {
                self.attach(g, &step.path, step.pad);
                self.order.push(step.pad);
                Some(step)
            }
            None => {
                for (i, &t) in target.iter().enumerate() {
                    if t {
                        self.unreachable[i] = true;
                    }
                }
                None
            }
        }
    }

    /// Construction (each edge once) and operating (each pad's way to the access point) cost.
    pub fn cost(&self, g: &Graph) -> (f64, f64) {
        let construction = (0..g.nodes.len())
            .filter(|&v| self.in_tree[v] && v != self.root)
            .map(|v| g.costs[self.parent_edge[v] as usize].construction)
            .sum();
        let op = self.op_to_root(g);
        let operating = (0..self.pads.len())
            .filter(|&i| self.connected[i] && self.pads[i] != self.root)
            .map(|i| op[self.pads[i]])
            .sum();
        (construction, operating)
    }

    /// Take off a pad's own end branch (back to a junction, another pad or the access point).
    /// None if other pads lie beyond it. Returns the removed (node, parent, edge), pad first.
    fn detach(&mut self, pad: usize) -> Option<Vec<(usize, u32, u32)>> {
        let v0 = self.pads[pad];
        if !self.connected[pad] || v0 == self.root || self.children[v0] > 0 || self.pad_of[v0] != pad as i32 {
            return None;
        }
        let mut removed = Vec::new();
        let mut v = v0;
        loop {
            let p = self.parent[v] as usize;
            removed.push((v, self.parent[v], self.parent_edge[v]));
            self.in_tree[v] = false;
            self.parent[v] = NONE;
            self.parent_edge[v] = NONE;
            self.children[p] -= 1;
            if p == self.root || self.pad_of[p] >= 0 || self.children[p] > 0 {
                break;
            }
            v = p;
        }
        self.connected[pad] = false;
        Some(removed)
    }

    fn restore(&mut self, removed: &[(usize, u32, u32)], pad: usize) {
        for &(v, p, e) in removed.iter().rev() {
            self.parent[v] = p;
            self.parent_edge[v] = e;
            self.in_tree[v] = true;
            self.children[p as usize] += 1;
        }
        self.connected[pad] = true;
    }

    /// One improvement pass: each pad at the end of a branch is taken off and reconnected at
    /// least cost; kept if the whole network got cheaper. Returns the pads that moved.
    pub fn improve(&mut self, g: &Graph) -> Vec<usize> {
        let mut moved = Vec::new();
        for pad in 0..self.pads.len() {
            let (c0, o0) = self.cost(g);
            let Some(removed) = self.detach(pad) else { continue };
            let mut target = vec![false; self.pads.len()];
            target[pad] = true;
            match self.search(g, &target) {
                Some(step) => {
                    self.attach(g, &step.path, pad);
                    let (c1, o1) = self.cost(g);
                    if c1 + o1 < c0 + o0 - 1e-6 * (c0 + o0).abs().max(1.0) {
                        moved.push(pad);
                    } else {
                        let new = self.detach(pad).expect("a branch just attached is an end branch");
                        debug_assert!(!new.is_empty());
                        self.restore(&removed, pad);
                    }
                }
                None => self.restore(&removed, pad),
            }
        }
        moved
    }

    /// The network's edges, each with the number of pads whose way to the access point uses it.
    pub fn edges(&self, g: &Graph) -> Vec<(u32, u32)> {
        let mut uses = vec![0u32; g.nodes.len()];
        for (i, &v) in self.pads.iter().enumerate() {
            if self.connected[i] && v != self.root {
                for u in self.up(v) {
                    if u != self.root {
                        uses[u] += 1;
                    }
                }
            }
        }
        (0..g.nodes.len())
            .filter(|&v| self.in_tree[v] && v != self.root)
            .map(|v| (self.parent_edge[v], uses[v]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assemble, EdgeCost, Node, KIND_TIN};

    /// A graph on the given cells and undirected edges, each costing its length to build.
    fn graph(cells: &[(usize, usize)], pairs: &[(u32, u32)]) -> Graph {
        let nodes: Vec<Node> = cells.iter().map(|&(r, c)| Node { r, c, kind: KIND_TIN }).collect();
        let costed = pairs
            .iter()
            .map(|&(u, v)| {
                let (a, b) = (cells[u as usize], cells[v as usize]);
                let len = (a.0 as f64 - b.0 as f64).hypot(a.1 as f64 - b.1 as f64);
                Some(EdgeCost { total: len, construction: len, operating: 0.0, length: len, hairpins: 0 })
            })
            .collect();
        assemble(nodes, pairs.to_vec(), costed)
    }

    #[test]
    fn pads_beyond_other_pads_follow_the_limit() {
        // A line 0-1-2-3-4 from the access point (0), pads at 2 and 4: pad 4 lies beyond pad 2.
        let cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)];
        let g = graph(&cells, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let mut strict = Network::new(&g, 0, &[2, 4], 0, 0.0);
        while strict.grow(&g).is_some() {}
        assert!(strict.connected[0] && !strict.connected[1] && strict.unreachable[1]);
        let mut loose = Network::new(&g, 0, &[2, 4], 1, 0.0);
        while loose.grow(&g).is_some() {}
        assert!(loose.connected.iter().all(|&c| c));
        assert_eq!(loose.beyond(), vec![1, 0]);
        assert_eq!(loose.cost(&g).0, 4.0);
    }

    #[test]
    fn branches_share_a_trunk_and_improvement_never_costs_more() {
        // A 5 x 5 grid, access point at a corner, pads at the other three corners and the middle.
        let mut cells = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                cells.push((r, c));
            }
        }
        let id = |r: u32, c: u32| r * 5 + c;
        let mut pairs = Vec::new();
        for r in 0..5 {
            for c in 0..5 {
                if c + 1 < 5 {
                    pairs.push((id(r, c), id(r, c + 1)));
                }
                if r + 1 < 5 {
                    pairs.push((id(r, c), id(r + 1, c)));
                }
            }
        }
        let g = graph(&cells, &pairs);
        let pads = [id(0, 4) as usize, id(4, 0) as usize, id(4, 4) as usize, id(2, 2) as usize];
        let mut net = Network::new(&g, 0, &pads, 2, 0.0);
        while net.grow(&g).is_some() {}
        assert!(net.connected.iter().all(|&c| c));
        let (c0, _) = net.cost(&g);
        // A tree joining four pads on this grid needs at least 12 edges; separate roads would need 4+4+8+4.
        assert!(c0 < 20.0, "{c0}");
        net.improve(&g);
        let (c1, _) = net.cost(&g);
        assert!(c1 <= c0);
        assert!(net.beyond().iter().all(|&b| b <= 2));
        // Every pad's way to the access point is on the network.
        for &p in &pads {
            assert!(net.up(p).last() == Some(0));
        }
    }
}
