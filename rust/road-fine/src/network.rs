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
//!
//! Pads are level benches, priced with the road's section model (earthwork only): a pad at the end
//! of its branch is a bench of its own. With `alongside`, a pad may instead sit beside the road,
//! which runs along its long side at its level: the road's bench widened by the pad's width, priced
//! as the widening (on steep or rocky ground a wider bench cuts much deeper, so it can cost more
//! than a separate pad). A road beside a pad is not blocked by its rig, and branches may go on from
//! there. Each pad's branch is searched both ways and the cheaper kept.
//!
//! With `rigs` = N > 0 the pad rule gives way to N branches: roads may run through any number of
//! pads, and the network has at most N dead ends, so N rigs can work at once, each from the far
//! end of its branch back toward the access point. Once there are N, a pad joins only by carrying
//! a branch on past its last pad, not by forking a new one. For a grid of holes, rig_pieces lays
//! the branches out as a person would: rows along the contour, linked end to end as a serpentine
//! from the bottom up, cut into N runs; each run's first pad joins the network where cheapest and
//! the rest carry its branch on, pad after pad (each rig then drills its run from the top down).
//!
//! With orient_by_road, pads lie the way the road arrives rather than along the contour, as drill
//! pads at the ends of spurs are built: a level bench centered on the hole, its long axis along
//! the arriving heading, reached at its near end from any of the lattice's headings. Each heading
//! has its own bench (across the slope, plus the cut or fill that keeps it level along its
//! length), and the search finishes at the cheapest road plus bench.

use std::collections::HashMap;

use lattice_search::{search_with_progress, CellIndex, Heuristic, Progress, SearchSpec};
use road_cost::{ClimbHeuristic, Grid};

use crate::lattice::heading_angles;
use crate::pipeline::{FineProgress, Report, PROGRESS_EVERY};
use crate::problem::Problem;
use crate::route::{bil_clamped, Edge, Segment};
use crate::section::{section, soil_depth, SectionParams};

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
    /// Where a road passes beside it: cells off each long side, heading along the axis either way.
    pub sides: Vec<End>,
    /// Earthwork of its bench alone, and of widening a road's bench by it (infinite where the
    /// ground is too steep for the bench).
    pub cost_alone: f64,
    pub cost_beside: f64,
    /// With orient_by_road: each way in (the near end of a pad lying along that heading) and its
    /// bench's cost (infinite where it cannot be built).
    pub by_road: Vec<(End, f64)>,
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
    /// The end of the pad it arrives at, or (beside) its entry in the pad's sides.
    pub end: usize,
    /// Arriving beside the pad rather than at an end.
    pub beside: bool,
    /// The pad's bench (alone, or the widening), included in construction.
    pub pad_cost: f64,
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
    /// Pads may sit beside the road (see the module docs).
    pub alongside: bool,
    /// Rigs to design for (see the module docs); 0: the pad rule (`b`).
    pub rigs: usize,
    /// While set, a branch may only carry this pad's branch on (past it, or reversing out of it).
    pub only_from: Option<usize>,
    /// Pads lie the way the road arrives (see the module docs).
    pub by_road: bool,
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
    /// smoothed DEM; their footprints closed in prob.open. `alongside`: pads may sit beside the road.
    pub fn new(prob: &mut Problem, root: (i64, i64), centers: &[(f64, f64)], length: f64, width: f64, b: usize,
               alongside: bool) -> Self {
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
            // The bench: across the slope at the center, with the soil there.
            let (s, d) = (gx.hypot(gy), prob.soil[k]);
            let bench = |w: f64| {
                let mut sp = SectionParams::new(&prob.params);
                sp.w = w;
                section(s, d, &sp).cost
            };
            let road_w = prob.params.w;
            // Its own bench along the axis, as pads by road are priced (bench_along).
            let cost_alone = bench_along(prob, (r, c), axis, length, width);
            let cost_beside = length * (bench(road_w + width) - bench(road_w));
            pads.push(Pad { center: (r, c), axis, ends, cells, sides: Vec::new(), cost_alone, cost_beside,
                            by_road: Vec::new() });
        }
        // Beside each long side: the road's centerline half its width and half the pad's out.
        for p in &mut pads {
            let (ux, uy) = (p.axis.cos(), p.axis.sin());
            let (h_fwd, h_back) = (nearest_heading(&angles, p.axis), nearest_heading(&angles, p.axis + std::f64::consts::PI));
            for s in [1.0f64, -1.0] {
                let mut d = 0.5 * (width + prob.params.w);
                for _ in 0..4 {
                    let (sr, sc) = ((p.center.0 - s * ux * d / res).round() as i64, (p.center.1 - s * uy * d / res).round() as i64);
                    if sr < 0 || sc < 0 || sr >= nr || sc >= nc {
                        break;
                    }
                    if p.cells.contains(&(sr, sc)) {
                        d += res;
                        continue;
                    }
                    for h in [h_fwd, h_back] {
                        p.sides.push(End { cell: (sr, sc), heading: h });
                    }
                    break;
                }
            }
        }
        for p in &pads {
            for &(r, c) in &p.cells {
                prob.open[(r * nc + c) as usize] = 0;
            }
        }
        // An end or side on closed ground cannot be reached.
        for p in &mut pads {
            p.sides.retain(|e| prob.open[(e.cell.0 * nc + e.cell.1) as usize] != 0);
            for e in p.ends.iter_mut() {
                if let Some(end) = e {
                    if prob.open[(end.cell.0 * nc + end.cell.1) as usize] == 0 {
                        *e = None;
                    }
                }
            }
        }
        let n = pads.len();
        FineNetwork {
            root, pads, b, alongside, rigs: 0, only_from: None, by_road: false, branches: vec![None; n],
            unreachable: vec![false; n],
        }
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

    /// The pads in rows along the contour, linked end to end as a serpentine from the bottom row up
    /// (starting at the end nearer the access point), cut into `rigs` runs of about equal size.
    /// Rows run the way each pad's nearest neighbor lies (the commonest such direction); a row
    /// ends where the next pad across that direction is more than half a spacing further on.
    pub fn rig_pieces(&self, prob: &Problem) -> Vec<Vec<usize>> {
        let n = self.pads.len();
        if n == 0 || self.rigs == 0 {
            return Vec::new();
        }
        // East, north in cells.
        let xy: Vec<(f64, f64)> = self.pads.iter().map(|p| (p.center.1, -p.center.0)).collect();
        let (mut sx, mut sy, mut gx, mut gy) = (0.0, 0.0, 0.0, 0.0);
        let mut nn = Vec::with_capacity(n);
        for i in 0..n {
            let j = (0..n).filter(|&j| j != i)
                .min_by(|&a, &b| {
                    let d = |j: usize| (xy[j].0 - xy[i].0).hypot(xy[j].1 - xy[i].1);
                    d(a).total_cmp(&d(b))
                });
            if let Some(j) = j {
                let (dx, dy) = (xy[j].0 - xy[i].0, xy[j].1 - xy[i].1);
                let a = 2.0 * dy.atan2(dx);
                sx += a.cos();
                sy += a.sin();
                nn.push(dx.hypot(dy));
            }
            let (r, c) = (self.pads[i].center.0.round() as usize, self.pads[i].center.1.round() as usize);
            let k = r.min(prob.nr - 1) * prob.nc + c.min(prob.nc - 1);
            gx += prob.gx[k];
            gy += prob.gy[k];
        }
        let along = 0.5 * sy.atan2(sx);
        let (ax, ay) = (along.cos(), along.sin());
        // Across the rows, uphill.
        let (mut ux, mut uy) = (-ay, ax);
        if ux * gx + uy * gy < 0.0 {
            ux = -ux;
            uy = -uy;
        }
        nn.sort_by(f64::total_cmp);
        let spacing = if nn.is_empty() { 1.0 } else { nn[nn.len() / 2] };
        let mut order: Vec<usize> = (0..n).collect();
        let v = |i: usize| xy[i].0 * ux + xy[i].1 * uy;
        let u = |i: usize| xy[i].0 * ax + xy[i].1 * ay;
        order.sort_by(|&a, &b| v(a).total_cmp(&v(b)));
        let mut rows: Vec<Vec<usize>> = vec![vec![order[0]]];
        for w in order.windows(2) {
            if v(w[1]) - v(w[0]) > 0.5 * spacing {
                rows.push(Vec::new());
            }
            rows.last_mut().unwrap().push(w[1]);
        }
        for r in &mut rows {
            r.sort_by(|&a, &b| u(a).total_cmp(&u(b)));
        }
        // The bottom row starts at the end nearer the access point; rows then alternate.
        let root = (self.root.1 as f64, -(self.root.0 as f64));
        let dist = |i: usize| (xy[i].0 - root.0).hypot(xy[i].1 - root.1);
        let mut flip = dist(*rows[0].last().unwrap()) < dist(rows[0][0]);
        let mut snake = Vec::with_capacity(n);
        for r in rows {
            if flip {
                snake.extend(r.iter().rev());
            } else {
                snake.extend(r.iter());
            }
            flip = !flip;
        }
        let k = self.rigs.min(n);
        (0..k).map(|i| snake[i * n / k..(i + 1) * n / k].to_vec()).collect()
    }

    /// Connect pad p by carrying pad `prev`'s branch on (past it, or reversing out of it); if no
    /// such branch reaches it, by the cheapest branch from anywhere. False if none reaches it.
    pub fn connect_after(&mut self, prob: &Problem, p: usize, prev: usize, report: &mut Report) -> bool {
        self.only_from = Some(prev);
        let b = self.search(prob, p, &format!("pad {} after {}", p + 1, prev + 1), report);
        self.only_from = None;
        match b {
            Some(b) => {
                self.branches[p] = Some(b);
                self.unreachable[p] = false;
                true
            }
            None => self.connect(prob, p, report),
        }
    }

    /// Pads lie the way the road arrives: their footprints reopen (their direction is not known
    /// yet), and each heading gets its way in and its bench.
    pub fn orient_by_road(&mut self, prob: &mut Problem, length: f64, width: f64) {
        self.by_road = true;
        let (nr, nc, res) = (prob.nr as i64, prob.nc as i64, prob.res);
        let angles = heading_angles(prob.n_theta());
        for pad in &mut self.pads {
            for &(r, c) in &pad.cells {
                let k = (r * nc + c) as usize;
                prob.open[k] = prob.open_tables[k];
            }
            pad.by_road.clear();
            for (h, &a) in angles.iter().enumerate() {
                let (ux, uy) = (a.cos(), a.sin());
                let d = 0.5 * length + 0.75 * res;
                let (er, ec) = ((pad.center.0 + uy * d / res).round() as i64, (pad.center.1 - ux * d / res).round() as i64);
                if er < 0 || ec < 0 || er >= nr || ec >= nc || prob.open[(er * nc + ec) as usize] == 0 {
                    continue;
                }
                let cost = bench_along(prob, pad.center, a, length, width);
                pad.by_road.push((End { cell: (er, ec), heading: h }, cost));
            }
        }
    }

    /// Connected pads no other branch continues from: the network's dead ends.
    pub fn dead_ends(&self) -> usize {
        (0..self.pads.len()).filter(|&q| self.branches[q].is_some() && self.dependents(q) == 0).count()
    }

    /// Pads whose branch starts on pad p's branch or out of pad p.
    fn dependents(&self, p: usize) -> usize {
        self.branches
            .iter()
            .flatten()
            .filter(|b| matches!(b.from, From::Branch { pad, .. } | From::Through { pad } if pad == p))
            .count()
    }

    /// Operating cost from where a branch would start to the access point.
    fn op_from(&self, from: From) -> f64 {
        match from {
            From::Root => 0.0,
            From::Branch { pad, node } => self.op_at_start(pad) + self.branches[pad].as_ref().unwrap().op[node],
            From::Through { pad } => self.op_to_root(pad),
        }
    }

    /// The cheapest branch from the network to pad p (its bench included), or None: to an end,
    /// or with `alongside` also beside it, whichever costs the network less.
    fn search(&self, prob: &Problem, p: usize, stage: &str, report: &mut Report) -> Option<Branch> {
        let pad = &self.pads[p];
        if self.by_road {
            let (goals, costs): (Vec<End>, Vec<f64>) = pad.by_road.iter().filter(|g| g.1.is_finite()).copied().unzip();
            return self.search_to(prob, p, &goals, Some(&costs), false, stage, report);
        }
        let ends: Vec<End> = pad.ends.iter().flatten().copied().collect();
        let mut best: Option<(f64, Branch)> = None;
        let mut consider = |b: Option<Branch>, me: &Self| {
            if let Some(b) = b {
                let total = b.construction + me.op_from(b.from) + b.op.last().copied().unwrap_or(0.0);
                if total.is_finite() && best.as_ref().is_none_or(|(t, _)| total < *t) {
                    best = Some((total, b));
                }
            }
        };
        if pad.cost_alone.is_finite() {
            consider(self.search_to(prob, p, &ends, None, false, stage, report), self);
        }
        if self.alongside && pad.cost_beside.is_finite() && !pad.sides.is_empty() {
            consider(self.search_to(prob, p, &pad.sides, None, true, &format!("{stage}, beside"), report), self);
        }
        best.map(|(_, b)| b)
    }

    /// The cheapest branch from the network to one of `goals` of pad p (beside it, at an end, or
    /// by road with `goal_costs`, each goal's bench).
    #[allow(clippy::too_many_arguments)]
    fn search_to(&self, prob: &Problem, p: usize, goals: &[End], goal_costs: Option<&[f64]>, beside: bool, stage: &str,
                 report: &mut Report) -> Option<Branch> {
        if goals.is_empty() {
            return None;
        }
        let (nr, nc, nh) = (prob.nr, prob.nc, prob.n_theta());
        let beyond = self.beyond();
        let ok = |q: usize| self.rigs > 0 || beyond[q] < self.b;
        // With N rigs and N dead ends already, only carrying a branch on past its last pad.
        let leaf = |q: usize| self.branches[q].is_some() && self.dependents(q) == 0;
        let full = self.rigs > 0 && (0..self.pads.len()).filter(|&q| leaf(q)).count() >= self.rigs;
        // Sources: (cell, headings, cost, where).
        let forced = self.only_from;
        let all: Vec<usize> = (0..nh).collect();
        // Leaving a road at a junction: at most a right angle from its direction of travel (a
        // sharper turn means reversing, which only a switchback or a pad's landing allows).
        let angles = heading_angles(nh);
        let turn_off = |h: usize| -> Vec<usize> {
            (0..nh)
                .filter(|&k| ((angles[k] - angles[h] + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
                    - std::f64::consts::PI).abs() <= 0.5 * std::f64::consts::PI + 1e-9)
                .collect()
        };
        let mut src: Vec<((i64, i64), Vec<usize>, f64, From)> =
            if full || forced.is_some() { Vec::new() } else { vec![(self.root, all.clone(), 0.0, From::Root)] };
        for q in 0..self.pads.len() {
            let Some(br) = self.branches[q].as_ref() else { continue };
            if forced.is_some_and(|f| f != q) {
                continue;
            }
            if !self.pads_before(q).into_iter().all(ok) || (full && !leaf(q)) {
                continue;
            }
            let start = self.op_at_start(q);
            // Interior nodes (not the junction itself, not the pad's door): forks.
            for node in (1..br.edges.len()).filter(|_| !full && forced.is_none()) {
                let (r, c, a) = br.edges[node];
                src.push(((r, c), turn_off(prob.lat.h0[a] as usize), start + br.op[node], From::Branch { pad: q, node }));
            }
            // Beside a pad the road goes on: its last node is a junction like any other.
            if br.beside {
                // (A branch of no edges arrived on the network itself: nothing new to start from.)
                if let Some(&(r, c, a)) = br.edges.last() {
                    let node = br.edges.len();
                    let cell = (r + prob.lat.dr[a], c + prob.lat.dc[a]);
                    src.push((cell, turn_off(prob.lat.h1[a] as usize), start + br.op[node], From::Branch { pad: q, node }));
                }
                continue;
            }
            // Through the pad, out of its far end, continuing the way it arrived (a pad by road
            // ends its spur).
            if ok(q) {
                let door = if self.by_road { self.pads[q].by_road[br.end].0 } else { self.pads[q].ends[br.end].unwrap() };
                if !self.by_road {
                    if let Some(far) = self.pads[q].ends[1 - br.end] {
                        src.push((far.cell, vec![door.heading], self.op_to_root(q), From::Through { pad: q }));
                    }
                }
                // With reversing switchbacks the pad is a landing: pull in, reverse out of the door
                // onto a new branch, any way. A rig on the pad blocks it, as for the far end.
                if prob.params.switchback_landing > 0.0 {
                    src.push((door.cell, all.clone(), self.op_to_root(q), From::Through { pad: q }));
                }
            }
        }
        let mut mask: Vec<bool> = prob.open.iter().map(|&o| o != 0).collect();
        for s in &src {
            mask[(s.0 .0 as usize) * nc + s.0 .1 as usize] = true;
        }
        for g in goals {
            mask[(g.cell.0 as usize) * nc + g.cell.1 as usize] = true;
        }
        let index = CellIndex::from_mask(nr, nc, &mask);
        let mut sources = Vec::new();
        let mut costs = Vec::new();
        let mut from_of: HashMap<u64, From> = HashMap::new();
        for ((r, c), hs, cost, from) in &src {
            let (r, c, cost, from) = (*r, *c, *cost, *from);
            for &h in hs {
                if let Some(s) = index.state(r, c, h as u16, nh) {
                    sources.push(s);
                    costs.push(cost);
                    from_of.entry(s).or_insert(from);
                }
            }
        }
        // The goal: arriving at an end heading into the pad (or one heading bin either side), each
        // with its goal's finishing cost if any.
        let (mut goal_states, mut goal_of, mut finish) = (Vec::new(), Vec::new(), Vec::new());
        for (i, g) in goals.iter().enumerate() {
            for dh in [0, 1, nh - 1] {
                if let Some(s) = index.state(g.cell.0, g.cell.1, ((g.heading + dh) % nh) as u16, nh) {
                    // A state two goals share keeps the cheaper.
                    let c = goal_costs.map_or(0.0, |gc| gc[i]);
                    match goal_states.iter().position(|&x| x == s) {
                        Some(j) if finish[j] <= c => {}
                        Some(j) => {
                            goal_of[j] = i;
                            finish[j] = c;
                        }
                        None => {
                            goal_states.push(s);
                            goal_of.push(i);
                            finish.push(c);
                        }
                    }
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
            source_costs: Some(&costs), goal_costs: goal_costs.map(|_| &finish[..]),
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
        // No edges: the network already arrives at a way in (the pad sits on it), so only its bench.
        let edges: Vec<Edge> =
            res.edges_to(&prob.lattice, &index, gs).into_iter().map(|(r, c, a)| (r as i64, c as i64, a)).collect();
        let cell = index.cell((gs / nh as u64) as u32);
        let cell = (cell.0 as i64, cell.1 as i64);
        let reached = goal_states.iter().position(|&x| x == gs).map(|j| goal_of[j]);
        let end = if self.by_road {
            // Its entry in the pad's ways in.
            let g = goals[reached?];
            self.pads[p].by_road.iter().position(|w| w.0.cell == g.cell && w.0.heading == g.heading)?
        } else if beside {
            goals.iter().position(|e| e.cell == cell).unwrap_or(0)
        } else {
            self.pads[p].ends.iter().position(|e| e.map(|e| e.cell) == Some(cell)).unwrap_or(0)
        };
        let segments = prob.evaluate_route(&edges).ok()?;
        let mut op = vec![0.0];
        for sg in &segments {
            op.push(op.last().unwrap() + sg.cost_operating);
        }
        let pad_cost = if self.by_road {
            self.pads[p].by_road[end].1
        } else if beside {
            self.pads[p].cost_beside
        } else {
            self.pads[p].cost_alone
        };
        let construction = segments.iter().map(|s| s.cost_construction()).sum::<f64>() + pad_cost;
        Some(Branch { from, end, beside, pad_cost, edges, segments, op, construction })
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

    /// Pad p and every pad whose branch starts on its branch or out of it, recursively: the
    /// branches that move with p's. In the order to connect them again (p first).
    fn subtree(&self, p: usize) -> Vec<usize> {
        let mut out = vec![p];
        let mut i = 0;
        while i < out.len() {
            let q = out[i];
            for (r, b) in self.branches.iter().enumerate() {
                if let Some(b) = b {
                    if matches!(b.from, From::Branch { pad, .. } | From::Through { pad } if pad == q) && !out.contains(&r) {
                        out.push(r);
                    }
                }
            }
            i += 1;
        }
        out
    }

    /// One improvement pass: each pad's branch, with the branches hanging off it (at most
    /// MAX_SUBTREE pads), is taken up and the pads connected again in turn, each to anywhere on
    /// the network (a neighbor's road it ran beside, say); kept if every pad connects and the
    /// whole network got cheaper. Returns the pads whose branch moved.
    pub fn improve(&mut self, prob: &Problem, report: &mut Report) -> Vec<usize> {
        const MAX_SUBTREE: usize = 12;
        let mut moved = Vec::new();
        for p in 0..self.pads.len() {
            if self.branches[p].is_none() {
                continue;
            }
            let sub = self.subtree(p);
            if sub.len() > MAX_SUBTREE {
                continue;
            }
            let (c0, o0) = self.cost();
            let old: Vec<Option<Branch>> = sub.iter().map(|&q| self.branches[q].take()).collect();
            let mut ok = true;
            for &q in &sub {
                match self.search(prob, q, &format!("improve: pad {}", q + 1), report) {
                    Some(b) => self.branches[q] = Some(b),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            let (c1, o1) = self.cost();
            if ok && c1 + o1 < c0 + o0 - 1e-6 * (c0 + o0).abs().max(1.0) {
                moved.push(p);
            } else {
                for (&q, b) in sub.iter().zip(old) {
                    self.branches[q] = b;
                }
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

/// A level pad bench centered at (row, col), `length` along `angle` (radians, east = 0,
/// counterclockwise) by `width`, at the smoothed DEM's elevation at its center: the road section
/// model's earthwork for a bench of that width across the slope (as the switchbacks' landings),
/// plus the prism that keeps it level along its length at the fill price. Infinite off the grid,
/// on ground too steep for the section, or past H_fill_max of cut or fill.
pub fn bench_along(prob: &Problem, center: (f64, f64), angle: f64, length: f64, width: f64) -> f64 {
    let (nr, nc, res) = (prob.nr, prob.nc, prob.res);
    let p = &prob.params;
    let mut sp = SectionParams::new(p);
    sp.w = width;
    let (ux, uy) = (angle.cos(), angle.sin());
    let n = ((2.0 * length / res).ceil() as usize).max(2);
    let ds = length / n as f64;
    let z0 = bil_clamped(&prob.zs, nr, nc, center.0, center.1);
    let (mut earth, mut prism) = (0.0, 0.0);
    for k in 0..n {
        let t = (k as f64 + 0.5) * ds - 0.5 * length;
        let (fr, fc) = (center.0 - t * uy / res, center.1 + t * ux / res);
        if fr < 0.0 || fc < 0.0 || fr > (nr - 1) as f64 || fc > (nc - 1) as f64 {
            return f64::INFINITY;
        }
        let ax = bil_clamped(&prob.gx, nr, nc, fr, fc);
        let ay = bil_clamped(&prob.gy, nr, nc, fr, fc);
        let s = (-ax * uy + ay * ux).abs();
        let g = ax * ux + ay * uy;
        let sec = section(s, soil_depth(s.hypot(g), p), &sp);
        let h = z0 - bil_clamped(&prob.zs, nr, nc, fr, fc);
        if !sec.cost.is_finite() || h.abs() > p.h_fill_max {
            return f64::INFINITY;
        }
        earth += (sec.cost - p.c_base) * ds;
        prism += (width * h.abs() + h * h / p.f_rock) * ds;
    }
    earth + prob.lam_fill * prism
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
    fn pads_in_a_row_share_a_road_beside_them() {
        // Three pads along the contour (north-south on this slope), no pad may block another.
        let centers = [(30.0, 60.0), (60.0, 60.0), (90.0, 60.0)];
        let run = |alongside: bool| {
            let mut prob = slope();
            let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 15.0, 10.0, 0, alongside);
            let mut report = |_: &str, _: &FineProgress| true;
            for p in 0..centers.len() {
                assert!(net.connect(&prob, p, &mut report), "pad {p}");
            }
            net.improve(&prob, &mut report);
            assert!(net.beyond().iter().all(|&b| b == 0));
            net
        };
        let ends = run(false);
        let beside = run(true);
        let pad = &beside.pads[0];
        assert!(pad.cost_alone > 0.0 && pad.cost_beside.is_finite());
        assert_eq!(pad.sides.len(), 4);
        // Every pad's bench is paid for either way.
        for net in [&ends, &beside] {
            for b in net.branches.iter().flatten() {
                assert!(b.pad_cost > 0.0);
            }
        }
        // The row is served by a road beside a pad that others branch from (no pad blocks another).
        assert!(beside.branches.iter().flatten().any(|b| b.beside));
        let (c_ends, o_ends) = ends.cost();
        let (c_beside, o_beside) = beside.cost();
        assert!(c_beside + o_beside < c_ends + o_ends, "beside {c_beside} + {o_beside} vs ends {c_ends} + {o_ends}");
        // On 40% ground a bench costs far more than on 5%, and widening the road's more than a
        // pad of its own.
        let mut steep = steep(20.0);
        let net = FineNetwork::new(&mut steep, (50, 5), &[(50.0, 30.0)], 15.0, 10.0, 0, true);
        let sp = &net.pads[0];
        assert!(sp.cost_alone > 20.0 * pad.cost_alone && sp.cost_beside > sp.cost_alone,
                "40%: alone {} widening {}; 5%: alone {} widening {}", sp.cost_alone, sp.cost_beside, pad.cost_alone,
                pad.cost_beside);
    }

    #[test]
    fn one_rig_makes_one_branch_through_the_pads() {
        let centers = [(30.0, 60.0), (60.0, 60.0), (90.0, 60.0), (60.0, 100.0)];
        let mut prob = slope();
        let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 15.0, 10.0, 0, false);
        net.rigs = 1;
        let mut report = |_: &str, _: &FineProgress| true;
        for p in 0..centers.len() {
            assert!(net.connect(&prob, p, &mut report), "pad {p}");
            assert_eq!(net.dead_ends(), 1);
        }
        net.improve(&prob, &mut report);
        assert_eq!(net.dead_ends(), 1);
        // One chain: the pads lie on the way to one another, three of them to the last.
        let mut beyond = net.beyond();
        beyond.sort();
        assert_eq!(beyond, vec![0, 1, 2, 3]);
        // Two rigs: two dead ends at most.
        let mut prob = slope();
        let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 15.0, 10.0, 0, false);
        net.rigs = 2;
        for p in 0..centers.len() {
            assert!(net.connect(&prob, p, &mut report), "pad {p}");
            assert!(net.dead_ends() <= 2);
        }
    }

    #[test]
    fn a_grid_is_drilled_by_rigs_along_its_rows() {
        // Rows along the contour (north-south here), 12 cells apart up the slope (east), 4 holes a
        // row 12 cells apart, alternate rows offset by half.
        let mut centers = Vec::new();
        for (j, c) in [40.0, 52.0, 64.0].into_iter().enumerate() {
            for i in 0..4 {
                centers.push((30.0 + 12.0 * i as f64 + if j % 2 == 1 { 6.0 } else { 0.0 }, c));
            }
        }
        let mut prob = slope();
        let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 15.0, 10.0, 0, false);
        net.rigs = 2;
        let pieces = net.rig_pieces(&prob);
        assert_eq!(pieces.iter().map(Vec::len).collect::<Vec<_>>(), vec![6, 6]);
        let snake: Vec<usize> = pieces.concat();
        // Bottom row first, from its end nearer the access point (north-west), then back along the next.
        assert_eq!(snake[..4], [0, 1, 2, 3]);
        assert_eq!(snake[4..8], [7, 6, 5, 4]);
        assert_eq!(snake[8..], [8, 9, 10, 11]);
        let mut report = |_: &str, _: &FineProgress| true;
        for piece in &pieces {
            assert!(net.connect(&prob, piece[0], &mut report));
            for w in piece.windows(2) {
                assert!(net.connect_after(&prob, w[1], w[0], &mut report), "pad {}", w[1]);
            }
        }
        assert!(net.branches.iter().all(Option::is_some));
        assert!(net.dead_ends() <= 2, "{} dead ends", net.dead_ends());
        // Each run is a chain: every pad after the first carries the previous one's branch on.
        for piece in &pieces {
            for w in piece.windows(2) {
                let from = net.branches[w[1]].as_ref().unwrap().from;
                assert_eq!(from, From::Through { pad: w[0] }, "pad {} from {from:?}", w[1]);
            }
        }
    }

    #[test]
    fn pads_lie_the_way_the_road_arrives() {
        let centers = [(30.0, 90.0), (90.0, 90.0), (90.0, 30.0)];
        let build = |by_road: bool| {
            let mut prob = slope();
            let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 20.0, 15.0, 0, false);
            if by_road {
                net.orient_by_road(&mut prob, 20.0, 15.0);
            }
            let mut report = |_: &str, _: &FineProgress| true;
            for p in 0..centers.len() {
                assert!(net.connect(&prob, p, &mut report), "pad {p}");
            }
            (prob, net)
        };
        let (prob, net) = build(true);
        let angles = heading_angles(prob.n_theta());
        for (p, b) in net.branches.iter().enumerate() {
            let b = b.as_ref().unwrap();
            let (way, cost) = net.pads[p].by_road[b.end];
            assert!(cost.is_finite() && (b.pad_cost - cost).abs() < 1e-9);
            // The last move ends at the way in, heading into the pad (within a heading bin).
            let &(r, c, a) = b.edges.last().unwrap();
            assert_eq!((r + prob.lat.dr[a], c + prob.lat.dc[a]), way.cell);
            let d = angles[prob.lat.h1[a] as usize] - angles[way.heading];
            assert!(d.cos() > 0.95, "pad {p}");
        }
        // Every heading has a way in on open ground; the footprint is open again.
        assert!(net.pads[0].by_road.len() >= 30);
        assert!(net.pads[0].cells.iter().all(|&(r, c)| prob.open[(r * prob.nc as i64 + c) as usize] != 0));
        let (_, contour) = build(false);
        let total = |n: &FineNetwork| { let (c, o) = n.cost(); c + o };
        assert!(total(&net) <= 1.05 * total(&contour), "by road {} vs contour {}", total(&net), total(&contour));
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
        let mut net = FineNetwork::new(&mut prob, (10, 10), &centers, 15.0, 10.0, 2, false);
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
                let mut lone = FineNetwork::new(&mut pr, (10, 10), &centers[p..p + 1], 15.0, 10.0, 2, false);
                let mut report = |_: &str, _: &FineProgress| true;
                assert!(lone.connect(&pr, 0, &mut report));
                lone.cost().0
            })
            .sum();
        assert!(c1 < separate, "{c1} vs {separate}");
    }
}
