//! A road network from an access point to many pads on the fine lattice: the refinement of the
//! coarse network (tin_graph::network).
//!
//! Pads are level platforms with their long axis along the contour; the road arrives along that
//! axis at one of the two short ends, heading into the pad. Each pad's footprint is closed to roads.
//! Each pad gets a branch: one A* search from the network so far (the access point, any node along
//! the branches already built, or out of the far end of a connected pad, where the pad rule
//! allows) to the pad's entries. A junction's start costs the operating cost of the way from it to
//! the access point, so trunks carry every pad's traffic. Pads are connected in a given order (the
//! coarse network's), then each pad's end branch is taken off and searched again while the whole
//! network gets cheaper.
//!
//! The pad rule: a rig on a pad blocks the road through it, so a pad may lie on the way to at most
//! `b` other pads.

use std::collections::HashMap;

use lattice_search::{search_with_progress, CellIndex, Heuristic, Progress, SearchSpec};
use road_cost::{ClimbHeuristic, Grid};

use crate::lattice::heading_angles;
use crate::pipeline::{FineProgress, Report, PROGRESS_EVERY};
use crate::problem::Problem;
use crate::route::{Edge, Segment};

/// One end of a pad: the cell a road arrives at and its heading into the pad.
#[derive(Clone, Copy, Debug)]
pub struct End {
    pub cell: (i64, i64),
    pub heading: usize,
}

#[derive(Clone, Debug)]
pub struct Pad {
    /// Center (row, col, cell coordinates) and long axis (radians, east = 0, counterclockwise).
    pub center: (f64, f64),
    pub axis: f64,
    /// The two short ends (None where the ground there is closed).
    pub ends: [Option<End>; 2],
    /// Cells of the footprint.
    pub cells: Vec<(i64, i64)>,
}

/// Where a branch leaves the network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum From {
    Root,
    /// At node `node` of pad `pad`'s branch (node i: the start of its edge i).
    Branch { pad: usize, node: usize },
    /// Out of the far end of a connected pad.
    Through { pad: usize },
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub from: From,
    /// The end of the pad it arrives at.
    pub end: usize,
    pub edges: Vec<Edge>,
    pub segments: Vec<Segment>,
    /// Operating cost from the branch's start to each of its nodes (len edges + 1).
    pub op: Vec<f64>,
    pub construction: f64,
}

pub struct FineNetwork {
    pub root: (i64, i64),
    pub pads: Vec<Pad>,
    pub b: usize,
    /// Per pad: its branch, once connected.
    pub branches: Vec<Option<Branch>>,
    pub unreachable: Vec<bool>,
}

/// The smallest of several climb heuristics (one per goal cell): still a lower bound.
struct MinClimb<'a>(Vec<ClimbHeuristic<'a>>);

impl Heuristic for MinClimb<'_> {
    fn estimate(&self, r: i32, c: i32) -> f64 {
        self.0.iter().map(|h| h.estimate(r, c)).fold(f64::INFINITY, f64::min)
    }
}

/// The lattice heading nearest a direction (radians, east = 0, counterclockwise).
fn nearest_heading(angles: &[f64], a: f64) -> usize {
    let d = |b: f64| ((a - b + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI).abs();
    (0..angles.len()).min_by(|&i, &j| d(angles[i]).total_cmp(&d(angles[j]))).unwrap()
}

impl FineNetwork {
    /// Pads at (row, col) centers, `length` x `width` m, long axis along the contour of the
    /// smoothed DEM; their footprints closed in prob.open.
    pub fn new(prob: &mut Problem, root: (i64, i64), centers: &[(f64, f64)], length: f64, width: f64, b: usize) -> Self {
        let (nr, nc, res) = (prob.nr as i64, prob.nc as i64, prob.res);
        let angles = heading_angles(prob.n_theta());
        let mut pads = Vec::new();
        for &(r, c) in centers {
            let k = (r.round().clamp(0.0, (nr - 1) as f64) as usize) * prob.nc + c.round().clamp(0.0, (nc - 1) as f64) as usize;
            let (gx, gy) = (prob.gx[k], prob.gy[k]);
            let axis = if gx.hypot(gy) > 1e-6 { (-gx).atan2(gy) } else { 0.0 };
            let (ux, uy) = (axis.cos(), axis.sin());
            // Footprint: cells whose center is within the rectangle (x east, y north from the center).
            let mut cells = Vec::new();
            let reach = (0.5 * length.hypot(width) / res).ceil() as i64 + 1;
            let (r0, c0) = (r.round() as i64, c.round() as i64);
            for rr in (r0 - reach).max(0)..=(r0 + reach).min(nr - 1) {
                for cc in (c0 - reach).max(0)..=(c0 + reach).min(nc - 1) {
                    let (x, y) = ((cc as f64 - c) * res, (r - rr as f64) * res);
                    if (x * ux + y * uy).abs() <= 0.5 * length && (-x * uy + y * ux).abs() <= 0.5 * width {
                        cells.push((rr, cc));
                    }
                }
            }
            // The ends: just past each short side, arriving along the axis toward the center.
            let mut ends = [None, None];
            for (e, s) in [1.0f64, -1.0].into_iter().enumerate() {
                let d = 0.5 * length + 0.75 * res;
                let (er, ec) = ((r - s * uy * d / res).round() as i64, (c + s * ux * d / res).round() as i64);
                if er < 0 || ec < 0 || er >= nr || ec >= nc {
                    continue;
                }
                let heading = nearest_heading(&angles, (-s * uy).atan2(-s * ux));
                ends[e] = Some(End { cell: (er, ec), heading });
            }
            pads.push(Pad { center: (r, c), axis, ends, cells });
        }
        for p in &pads {
            for &(r, c) in &p.cells {
                prob.open[(r * nc + c) as usize] = 0;
            }
        }
        // An end on closed ground cannot be reached.
        for p in &mut pads {
            for e in p.ends.iter_mut() {
                if let Some(end) = e {
                    if prob.open[(end.cell.0 * nc + end.cell.1) as usize] == 0 {
                        *e = None;
                    }
                }
            }
        }
        let n = pads.len();
        FineNetwork { root, pads, b, branches: vec![None; n], unreachable: vec![false; n] }
    }

    /// Operating cost from the start of pad p's branch to the access point.
    fn op_at_start(&self, p: usize) -> f64 {
        match self.branches[p].as_ref().map(|b| b.from) {
            None | Some(From::Root) => 0.0,
            Some(From::Branch { pad, node }) => self.op_at_start(pad) + self.branches[pad].as_ref().unwrap().op[node],
            Some(From::Through { pad }) => self.op_to_root(pad),
        }
    }

    /// Operating cost from pad p (its entry) to the access point.
    pub fn op_to_root(&self, p: usize) -> f64 {
        let b = self.branches[p].as_ref().expect("connected pad");
        self.op_at_start(p) + b.op.last().copied().unwrap_or(0.0)
    }

    /// Pads on the way from pad p's branch start to the access point.
    fn pads_before(&self, p: usize) -> Vec<usize> {
        match self.branches[p].as_ref().map(|b| b.from) {
            None | Some(From::Root) => Vec::new(),
            Some(From::Branch { pad, .. }) => self.pads_before(pad),
            Some(From::Through { pad }) => {
                let mut v = self.pads_before(pad);
                v.push(pad);
                v
            }
        }
    }

    /// Per pad, the connected pads whose way to the access point passes through it.
    pub fn beyond(&self) -> Vec<usize> {
        let mut out = vec![0; self.pads.len()];
        for p in 0..self.pads.len() {
            if self.branches[p].is_some() {
                for q in self.pads_before(p) {
                    out[q] += 1;
                }
            }
        }
        out
    }

    /// (construction, operating) of the network: each branch built once, each pad's traffic
    /// over its whole way to the access point.
    pub fn cost(&self) -> (f64, f64) {
        let c = self.branches.iter().flatten().map(|b| b.construction).sum();
        let o = (0..self.pads.len()).filter(|&p| self.branches[p].is_some()).map(|p| self.op_to_root(p)).sum();
        (c, o)
    }

    /// Pads whose branch starts on pad p's branch or out of pad p.
    fn dependents(&self, p: usize) -> usize {
        self.branches
            .iter()
            .flatten()
            .filter(|b| matches!(b.from, From::Branch { pad, .. } | From::Through { pad } if pad == p))
            .count()
    }

    /// The cheapest branch from the network to pad p, or None.
    fn search(&self, prob: &Problem, p: usize, stage: &str, report: &mut Report) -> Option<Branch> {
        let goals: Vec<End> = self.pads[p].ends.iter().flatten().copied().collect();
        if goals.is_empty() {
            return None;
        }
        let (nr, nc, nh) = (prob.nr, prob.nc, prob.n_theta());
        let beyond = self.beyond();
        let ok = |q: usize| beyond[q] < self.b;
        // Sources: (cell, heading or all, cost, where).
        let mut src: Vec<((i64, i64), Option<usize>, f64, From)> = vec![(self.root, None, 0.0, From::Root)];
        for q in 0..self.pads.len() {
            let Some(br) = self.branches[q].as_ref() else { continue };
            if !self.pads_before(q).into_iter().all(ok) {
                continue;
            }
            let start = self.op_at_start(q);
            // Interior nodes (not the junction itself, not the pad's door).
            for node in 1..br.edges.len() {
                let (r, c, _) = br.edges[node];
                src.push(((r, c), None, start + br.op[node], From::Branch { pad: q, node }));
            }
            // Through the pad, out of its far end, continuing the way it arrived.
            if ok(q) {
                if let Some(far) = self.pads[q].ends[1 - br.end] {
                    let h = self.pads[q].ends[br.end].unwrap().heading;
                    src.push((far.cell, Some(h), self.op_to_root(q), From::Through { pad: q }));
                }
                // With reversing switchbacks the pad is a landing: pull in, reverse out of the door
                // onto a new branch, any way. A rig on the pad blocks it, as for the far end.
                if prob.params.switchback_landing > 0.0 {
                    let door = self.pads[q].ends[br.end].unwrap();
                    src.push((door.cell, None, self.op_to_root(q), From::Through { pad: q }));
                }
            }
        }
        let mut mask: Vec<bool> = prob.open.iter().map(|&o| o != 0).collect();
        for s in &src {
            mask[(s.0 .0 as usize) * nc + s.0 .1 as usize] = true;
        }
        for g in &goals {
            mask[(g.cell.0 as usize) * nc + g.cell.1 as usize] = true;
        }
        let index = CellIndex::from_mask(nr, nc, &mask);
        let mut sources = Vec::new();
        let mut costs = Vec::new();
        let mut from_of: HashMap<u64, From> = HashMap::new();
        for &((r, c), h, cost, from) in &src {
            let hs: Vec<usize> = h.map_or((0..nh).collect(), |h| vec![h]);
            for h in hs {
                if let Some(s) = index.state(r, c, h as u16, nh) {
                    sources.push(s);
                    costs.push(cost);
                    from_of.entry(s).or_insert(from);
                }
            }
        }
        // The goal: arriving at an end heading into the pad (or one heading bin either side).
        let mut goal_states = Vec::new();
        for g in &goals {
            for dh in [0, 1, nh - 1] {
                if let Some(s) = index.state(g.cell.0, g.cell.1, ((g.heading + dh) % nh) as u16, nh) {
                    goal_states.push(s);
                }
            }
        }
        let pr = &prob.params;
        let heur = MinClimb(
            goals
                .iter()
                .map(|g| ClimbHeuristic {
                    goal: (g.cell.0 as i32, g.cell.1 as i32),
                    goal_z: prob.zs[g.cell.0 as usize * nc + g.cell.1 as usize],
                    zs: Grid::new(&prob.zs, nr, nc),
                    res: prob.res,
                    g_max: pr.g_max,
                    c_len: pr.c_base + pr.c_travel,
                    c_vertical: pr.c_vertical,
                })
                .collect(),
        );
        let spec = SearchSpec {
            sources: &sources, goal: None, stop_at_goal: true, max_cost: f64::INFINITY, goal_states: Some(&goal_states),
            source_costs: Some(&costs),
        };
        let start = self.root;
        let mut rep = |pr: &Progress| {
            let cells = pr.new_cells.iter().map(|&k| index.cells()[k as usize]).collect();
            let edges: Vec<Edge> = lattice_search::edges_to(&prob.lattice, &index, pr.parent, pr.state)
                .into_iter()
                .map(|(r, c, a)| (r as i64, c as i64, a))
                .collect();
            let from = edges.first().map_or(start, |e| (e.0, e.1));
            let path = prob.path_points(&edges, from).unwrap_or_default();
            report(stage, &FineProgress { settled: pr.n_expanded, key: pr.key, cells, g: pr.new_g, path })
        };
        let (res, completed) =
            prob.with_model(|model| search_with_progress(&prob.lattice, &index, model, &heur, &spec, PROGRESS_EVERY, &mut rep));
        let gs = res.goal_state.filter(|_| completed)?;
        // The path back to its source state, and where that is on the network.
        let mut s = gs;
        while let Some(prev) = res.predecessor(&prob.lattice, &index, s) {
            s = prev;
        }
        let from = *from_of.get(&s)?;
        let edges: Vec<Edge> =
            res.edges_to(&prob.lattice, &index, gs).into_iter().map(|(r, c, a)| (r as i64, c as i64, a)).collect();
        if edges.is_empty() {
            return None;
        }
        let cell = index.cell((gs / nh as u64) as u32);
        let cell = (cell.0 as i64, cell.1 as i64);
        let end = self.pads[p].ends.iter().position(|e| e.map(|e| e.cell) == Some(cell)).unwrap_or(0);
        let segments = prob.evaluate_route(&edges).ok()?;
        let mut op = vec![0.0];
        for sg in &segments {
            op.push(op.last().unwrap() + sg.cost_operating);
        }
        let construction = segments.iter().map(|s| s.cost_construction()).sum();
        Some(Branch { from, end, edges, segments, op, construction })
    }

    /// Connect pad p (the stage names the progress reports). False if no branch reaches it.
    pub fn connect(&mut self, prob: &Problem, p: usize, report: &mut Report) -> bool {
        match self.search(prob, p, &format!("pad {}", p + 1), report) {
            Some(b) => {
                self.branches[p] = Some(b);
                self.unreachable[p] = false;
                true
            }
            None => {
                self.unreachable[p] = true;
                false
            }
        }
    }

    /// One improvement pass: each pad at the end of the network (nothing branching from its
    /// branch or through it) is searched again with its branch off; kept if the whole network
    /// got cheaper. Returns the pads that moved.
    pub fn improve(&mut self, prob: &Problem, report: &mut Report) -> Vec<usize> {
        let mut moved = Vec::new();
        for p in 0..self.pads.len() {
            if self.branches[p].is_none() || self.dependents(p) > 0 {
                continue;
            }
            let (c0, o0) = self.cost();
            let old = self.branches[p].take();
            match self.search(prob, p, &format!("improve: pad {}", p + 1), report) {
                Some(b) => {
                    self.branches[p] = Some(b);
                    let (c1, o1) = self.cost();
                    if c1 + o1 < c0 + o0 - 1e-6 * (c0 + o0).abs().max(1.0) {
                        moved.push(p);
                    } else {
                        self.branches[p] = old;
                    }
                }
                None => self.branches[p] = old,
            }
        }
        moved
    }

    /// Per pad's branch, per segment: the pads whose way to the access point runs over it.
    pub fn segment_uses(&self) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> =
            self.branches.iter().map(|b| vec![0; b.as_ref().map_or(0, |b| b.segments.len())]).collect();
        for p in 0..self.pads.len() {
            if self.branches[p].is_none() {
                continue;
            }
            // The whole of its own branch, then the part of each branch it continues along.
            let mut q = p;
            let mut upto = out[p].len();
            loop {
                for u in &mut out[q][..upto] {
                    *u += 1;
                }
                match self.branches[q].as_ref().unwrap().from {
                    From::Root => break,
                    From::Branch { pad, node } => {
                        q = pad;
                        upto = node;
                    }
                    From::Through { pad } => {
                        q = pad;
                        upto = out[pad].len();
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Params;
    use crate::problem::{FineInputs, Macros};

    const PARAMS: &str = r#"{"w": 5.0, "g_max": 0.1, "R_min": 20.0, "R_hp": 10.0, "g_hp": 0.04, "w_hp": 7.0, "hp_leg_max": 60.0, "hp_gap_min": 8.0, "C_wall": null, "d0": 2.0, "S0": 0.3, "c_soil": 1.0, "c_rock": 4.0, "f_soil": 0.67, "f_rock": 0.8, "k_soil": 1.0, "k_rock": 1.3, "b_min": 0.5, "mask_margin": 0.9, "catch_max": 50.0, "C_base": 10.0, "C_soil": 1.0, "C_rock": 6.0, "C_borrow": 4.0, "C_haul_down": 0.01, "C_haul_up": 0.02, "haul_max": 500.0, "C_travel": 1.0, "C_vertical": 10.0, "C_turn": 0.0, "A_min": 2.0, "culvert_area_breaks": [10.0, 50.0], "culvert_costs": [50.0, 150.0, 400.0], "H_fill_max": 8.0, "L_cross_max": 60.0, "fill_tol": 0.25, "N_theta": 32, "move_max_cells": 7, "kink_max_deg": 10.0, "smooth_sigma": 0.0, "s_flat": 0.02}"#;

    /// A uniform 5% slope rising east, 120 x 120 cells of 5 m.
    fn slope() -> Problem {
        let (nr, nc, res) = (120usize, 120usize, 5.0);
        let z: Vec<f64> = (0..nr * nc).map(|k| 1000.0 + 0.05 * (k % nc) as f64 * res).collect();
        let inputs = FineInputs {
            nr, nc, res, zs: z.clone(), z_raw: z, gx: vec![0.05; nr * nc], gy: vec![0.0; nr * nc], area: vec![0.0; nr * nc],
            open: vec![1; nr * nc], rough: Vec::new(),
        };
        Problem::new(inputs, Params::from_json(PARAMS).unwrap(), Macros::Lazy).unwrap()
    }

    /// A uniform 40% slope rising east: too steep to climb straight at 10%.
    fn steep(switchback_landing: f64) -> Problem {
        let (nr, nc, res) = (100usize, 60usize, 5.0);
        let z: Vec<f64> = (0..nr * nc).map(|k| 1000.0 + 0.4 * (k % nc) as f64 * res).collect();
        let inputs = FineInputs {
            nr, nc, res, zs: z.clone(), z_raw: z, gx: vec![0.4; nr * nc], gy: vec![0.0; nr * nc], area: vec![0.0; nr * nc],
            open: vec![1; nr * nc], rough: Vec::new(),
        };
        let json = PARAMS.replacen('{', &format!("{{\"switchback_landing\": {switchback_landing}, "), 1);
        Problem::new(inputs, Params::from_json(&json).unwrap(), Macros::Lazy).unwrap()
    }

    #[test]
    fn steep_ground_is_climbed_by_reversing_switchbacks() {
        use crate::pipeline::find_route;
        use crate::route::{HAIRPIN, SWITCHBACK};
        let mut prob = steep(20.0);
        // No loop hairpins among the moves; switchbacks instead.
        assert!(!prob.lat.kind.iter().any(|&k| k == road_cost::ACTION_HP_LEFT as i64));
        assert!(prob.lat.kind.iter().any(|&k| k == road_cost::ACTION_SWITCH as i64));
        for (i, &k) in prob.lat.kind.iter().enumerate() {
            if k == road_cost::ACTION_SWITCH as i64 {
                let j = prob.lat.rev[i] as usize;
                assert_eq!(prob.lat.rev[j] as usize, i);
                assert_eq!(prob.lat.kind[j], k);
            }
        }
        // 80 m up from (50, 5) to (50, 45): at most 10% means 800 m of road on 200 m of ground.
        let (it, _, _) = find_route(&mut prob, (50, 5), (50, 45), &mut |_, _| true, &mut |_, _| {}).unwrap();
        let best = it.best.expect("a route up the slope");
        let n_switch = best.segments.iter().filter(|s| s.kind == SWITCHBACK).count();
        assert!(n_switch >= 2, "{n_switch} switchbacks");
        assert!(best.segments.iter().all(|s| s.kind != HAIRPIN));
        let length: f64 = best.segments.iter().filter(|s| s.kind != SWITCHBACK).map(|s| s.length).sum();
        assert!(length >= 780.0, "{length} m");
        // Each landing lies along the contour (north-south here), level, about 20 m.
        for s in best.segments.iter().filter(|s| s.kind == SWITCHBACK) {
            let (a, b) = (s.pts[0], s.pts[1]);
            assert!((b.1 - a.1).abs() < 0.25 * (b.0 - a.0).abs(), "landing {a:?} -> {b:?}");
            assert!((s.length - 20.0).abs() < 1e-9);
            assert!(s.cost_earthwork.is_finite() && s.cost_surrogate.is_finite());
        }
        // The search's cost and the evaluation agree on the landings.
        for s in best.segments.iter().filter(|s| s.kind == SWITCHBACK) {
            let t = s.cost_construction();
            assert!((t - s.cost_surrogate).abs() <= 0.05 * t + 1.0, "evaluated {t}, searched {}", s.cost_surrogate);
        }
    }

    #[test]
    fn pads_are_reached_along_their_long_axis_and_footprints_stay_clear() {
        let mut prob = slope();
        let centers = [(30.0, 90.0), (90.0, 90.0), (90.0, 30.0)];
        let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 15.0, 10.0, 2);
        // Contours run north-south: the long axis is north (or south).
        for p in &net.pads {
            assert!((p.axis.sin().abs() - 1.0).abs() < 1e-9, "{}", p.axis);
            assert!(p.cells.len() >= 4);
        }
        let mut report = |_: &str, _: &FineProgress| true;
        for p in 0..centers.len() {
            assert!(net.connect(&prob, p, &mut report), "pad {p}");
        }
        let angles = heading_angles(prob.n_theta());
        for (p, b) in net.branches.iter().enumerate() {
            let b = b.as_ref().unwrap();
            let end = net.pads[p].ends[b.end].unwrap();
            let &(r, c, a) = b.edges.last().unwrap();
            // The last edge ends at the pad's door, heading along the axis into the pad.
            assert_eq!((r + prob.lat.dr[a], c + prob.lat.dc[a]), end.cell);
            let h1 = prob.lat.h1[a] as usize;
            assert!((angles[h1] - angles[end.heading]).sin().abs() < 0.25 && (angles[h1] - angles[end.heading]).cos() > 0.9);
            // No road on any footprint.
            for s in &b.segments {
                for &(fr, fc) in &s.pts {
                    let cell = ((fr + 0.5).floor() as i64, (fc + 0.5).floor() as i64);
                    assert!(net.pads.iter().all(|pd| !pd.cells.contains(&cell)), "pad {p} road on a footprint at {cell:?}");
                }
            }
        }
        assert!(net.beyond().iter().all(|&b| b <= 2));
        let (c0, o0) = net.cost();
        let mut report = |_: &str, _: &FineProgress| true;
        net.improve(&prob, &mut report);
        let (c1, o1) = net.cost();
        assert!(c1 + o1 <= c0 + o0 + 1e-6);
        // Sharing: the network costs less to build than three separate roads from the access point.
        let separate: f64 = (0..centers.len())
            .map(|p| {
                let mut pr = slope();
                let mut lone = FineNetwork::new(&mut pr, (10, 10), &centers[p..p + 1], 15.0, 10.0, 2);
                let mut report = |_: &str, _: &FineProgress| true;
                assert!(lone.connect(&pr, 0, &mut report));
                lone.cost().0
            })
            .sum();
        assert!(c1 < separate, "{c1} vs {separate}");
    }
}
