use crate::{CellIndex, Lattice, MinHeap};

/// Parent action of a state that has none (a source, or not reached).
pub const NO_PARENT: i16 = -1;

/// What an action costs from a cell.
pub trait EdgeCost: Sync {
    /// Cost of `action` from cell (r, c): non-negative, `f64::INFINITY` if infeasible.
    fn cost(&self, r: i32, c: i32, action: usize) -> f64;
}

/// A lower bound on the cost from a cell to the goal. Consistent bounds make A* exact.
pub trait Heuristic: Sync {
    fn estimate(&self, r: i32, c: i32) -> f64;
}

/// No estimate: A* becomes Dijkstra.
pub struct ZeroHeuristic;

impl Heuristic for ZeroHeuristic {
    #[inline(always)]
    fn estimate(&self, _r: i32, _c: i32) -> f64 {
        0.0
    }
}

/// Where a search starts and when it stops.
#[derive(Clone, Debug)]
pub struct SearchSpec<'a> {
    /// State ids starting at cost 0.
    pub sources: &'a [u64],
    /// A goal cell: its cheapest settled state is reported as `goal_state`.
    pub goal: Option<(i32, i32)>,
    /// Stop as soon as the goal cell is settled (A*); otherwise settle everything reachable.
    pub stop_at_goal: bool,
    /// Stop once the smallest key (cost, plus the estimate for A*) exceeds this.
    pub max_cost: f64,
}

impl<'a> SearchSpec<'a> {
    pub fn dijkstra(sources: &'a [u64]) -> Self {
        SearchSpec { sources, goal: None, stop_at_goal: false, max_cost: f64::INFINITY }
    }

    pub fn astar(sources: &'a [u64], goal: (i32, i32)) -> Self {
        SearchSpec { sources, goal: Some(goal), stop_at_goal: true, max_cost: f64::INFINITY }
    }
}

/// Per-state results, indexed by state id (`cell_id * n_theta + h`).
pub struct SearchResult {
    /// Cost to come; infinite where not reached.
    pub g: Vec<f64>,
    /// Action used to arrive; [`NO_PARENT`] at sources and unreached states.
    pub parent: Vec<i16>,
    /// 1 where settled.
    pub closed: Vec<u8>,
    pub n_expanded: u64,
    /// The cheapest settled state at the goal cell, if any.
    pub goal_state: Option<u64>,
}

impl SearchResult {
    /// The state an action arrived from, if any.
    pub fn predecessor(&self, lat: &Lattice, index: &CellIndex, state: u64) -> Option<u64> {
        let a = self.parent[state as usize];
        if a < 0 {
            return None;
        }
        let nh = lat.n_theta() as u64;
        let (r, c) = index.cell((state / nh) as u32);
        let act = lat.action(a as usize);
        index.state((r - act.dr) as i64, (c - act.dc) as i64, act.h0, nh as usize)
    }

    /// Edges (row, col, action) from a source to `state`, in travel order.
    pub fn edges_to(&self, lat: &Lattice, index: &CellIndex, state: u64) -> Vec<(i32, i32, usize)> {
        let nh = lat.n_theta() as u64;
        let mut out = Vec::new();
        let mut s = state;
        while let Some(prev) = self.predecessor(lat, index, s) {
            let (r, c) = index.cell((prev / nh) as u32);
            out.push((r, c, self.parent[s as usize] as usize));
            s = prev;
        }
        out.reverse();
        out
    }
}

/// Dijkstra or A* over the states of `index`.
///
/// States are settled in order of (key, state id) with key = cost to come
/// (plus the estimate for A*). A state's parent is the first relaxation that
/// reached its final cost, in settle order and then in the lattice's
/// expansion order. Edges into cells outside the index are skipped.
pub fn search<C, H>(lat: &Lattice, index: &CellIndex, cost: &C, heur: &H, spec: &SearchSpec) -> SearchResult
where
    C: EdgeCost + ?Sized,
    H: Heuristic + ?Sized,
{
    let nh = lat.n_theta();
    let n = index.len() * nh;
    let mut g = vec![f64::INFINITY; n];
    let mut parent = vec![NO_PARENT; n];
    let mut closed = vec![0u8; n];
    let mut heap = MinHeap::with_capacity(1 << 16);
    let (goal_r, goal_c) = spec.goal.unwrap_or((-1, -1));
    for &s in spec.sources {
        assert!((s as usize) < n, "source state {s} is not in the index");
        g[s as usize] = 0.0;
        let key = if spec.stop_at_goal {
            let (r, c) = index.cell((s / nh as u64) as u32);
            heur.estimate(r, c)
        } else {
            0.0
        };
        heap.push(key, s);
    }
    let mut n_exp = 0u64;
    let mut goal_state: Option<u64> = None;
    while let Some((key, u)) = heap.pop() {
        let ui = u as usize;
        if closed[ui] != 0 {
            continue;
        }
        if key > spec.max_cost {
            break;
        }
        closed[ui] = 1;
        n_exp += 1;
        let cell = (u / nh as u64) as u32;
        let h = (u % nh as u64) as usize;
        let (r, c) = index.cell(cell);
        if r == goal_r && c == goal_c && goal_state.map_or(true, |gs| g[ui] < g[gs as usize]) {
            goal_state = Some(u);
            if spec.stop_at_goal {
                break;
            }
        }
        let gu = g[ui];
        for &a in lat.leaving(h) {
            let act = lat.action(a as usize);
            let (r2, c2) = (r + act.dr, c + act.dc);
            let v = match index.state(r2 as i64, c2 as i64, act.h1, nh) {
                Some(v) => v as usize,
                None => continue,
            };
            if closed[v] != 0 {
                continue;
            }
            let w = cost.cost(r, c, a as usize);
            if w == f64::INFINITY {
                continue;
            }
            let ng = gu + w;
            if ng < g[v] {
                g[v] = ng;
                parent[v] = a as i16;
                let f = if spec.stop_at_goal { ng + heur.estimate(r2, c2) } else { ng };
                heap.push(f, v as u64);
            }
        }
    }
    SearchResult { g, parent, closed, n_expanded: n_exp, goal_state }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Action;

    /// Eight headings, a step forward or a 45 degree turn; diagonals cost sqrt 2.
    fn grid_lattice() -> Lattice {
        let d = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)];
        let mut acts = Vec::new();
        for h in 0..8u16 {
            for dh in [0i32, 1, 7] {
                let h1 = ((h as i32 + dh) % 8) as u16;
                let (dr, dc) = d[h1 as usize];
                acts.push(Action { h0: h, h1, dr, dc });
            }
        }
        Lattice::new(8, acts)
    }

    struct Len<'a>(&'a Lattice, Vec<bool>, usize);
    impl EdgeCost for Len<'_> {
        fn cost(&self, r: i32, c: i32, a: usize) -> f64 {
            let act = self.0.action(a);
            let (r2, c2) = (r + act.dr, c + act.dc);
            if !self.1[r2 as usize * self.2 + c2 as usize] {
                return f64::INFINITY;
            }
            ((act.dr * act.dr + act.dc * act.dc) as f64).sqrt()
        }
    }

    struct Octile(i32, i32);
    impl Heuristic for Octile {
        fn estimate(&self, r: i32, c: i32) -> f64 {
            let (a, b) = ((r - self.0).abs() as f64, (c - self.1).abs() as f64);
            a.max(b) + (2f64.sqrt() - 1.0) * a.min(b)
        }
    }

    #[test]
    fn astar_matches_dijkstra_and_sparse_matches_dense() {
        let (nr, nc) = (30usize, 40usize);
        let mut open = vec![true; nr * nc];
        for r in 3..27 {
            open[r * nc + 20] = false; // a wall with gaps at the ends
        }
        let lat = grid_lattice();
        let cost = Len(&lat, open.clone(), nc);
        let dense = CellIndex::full(nr, nc);
        let sparse = CellIndex::from_mask(nr, nc, &open);
        let (s, goal) = ((15i64, 2i64), (15i32, 37i32));
        let src_d: Vec<u64> = (0..8).map(|h| dense.state(s.0, s.1, h, 8).unwrap()).collect();
        let src_s: Vec<u64> = (0..8).map(|h| sparse.state(s.0, s.1, h, 8).unwrap()).collect();
        let dj = search(&lat, &dense, &cost, &ZeroHeuristic, &SearchSpec::dijkstra(&src_d));
        let sp = search(&lat, &sparse, &cost, &ZeroHeuristic, &SearchSpec::dijkstra(&src_s));
        // Same costs and parents state for state, and the same number settled.
        for (k, &cell) in sparse.cells().iter().enumerate() {
            for h in 0..8 {
                let (i, j) = (cell as usize * 8 + h, k * 8 + h);
                assert_eq!(dj.g[i].to_bits(), sp.g[j].to_bits());
                assert_eq!(dj.parent[i], sp.parent[j]);
            }
        }
        assert_eq!(dj.n_expanded, sp.n_expanded);
        let a = search(&lat, &sparse, &cost, &Octile(goal.0, goal.1), &SearchSpec::astar(&src_s, goal));
        let gs = a.goal_state.unwrap();
        let best = (0..8).map(|h| sp.g[sparse.state(15, 37, h, 8).unwrap() as usize]).fold(f64::INFINITY, f64::min);
        assert!((a.g[gs as usize] - best).abs() < 1e-9);
        assert!(a.n_expanded < sp.n_expanded);
        // The path goes round the wall.
        let edges = a.edges_to(&lat, &sparse, gs);
        assert!(edges.iter().all(|&(r, c, _)| open[r as usize * nc + c as usize]));
        assert!(edges.iter().any(|&(r, _, _)| r < 3 || r >= 27));
    }

    #[test]
    fn max_cost_stops_early() {
        let lat = grid_lattice();
        let open = vec![true; 400];
        let cost = Len(&lat, open, 20);
        let ix = CellIndex::full(20, 20);
        let src = [ix.state(10, 10, 0, 8).unwrap()];
        let spec = SearchSpec { max_cost: 3.0, ..SearchSpec::dijkstra(&src) };
        let res = search(&lat, &ix, &cost, &ZeroHeuristic, &spec);
        assert!(res.g.iter().zip(&res.closed).all(|(&g, &cl)| cl == 0 || g <= 3.0));
        assert!(res.n_expanded < 400 * 8);
    }
}
