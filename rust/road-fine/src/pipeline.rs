//! The fine search's orchestration (pipeline.py): A* with fill-price repeats, then re-searching
//! while the route overlaps itself.

use lattice_search::{search_with_progress, CellIndex, Progress, SearchSpec};
use road_cost::{ClimbHeuristic, Grid};

use crate::masshaul::{true_cost, TrueCost};
use crate::problem::Problem;
use crate::route::{self_overlaps, Edge, Segment, FILL_CROSSING};

/// Settled states between progress reports.
pub const PROGRESS_EVERY: u64 = 1 << 16;

/// Called with the stage and the search's progress; returning false stops the search.
pub type Report<'a> = dyn FnMut(&str, &Progress) -> bool + 'a;

/// One A* search: the route's edges (None if the goal is unreachable), its cost, and counts.
pub struct Found {
    pub edges: Option<Vec<Edge>>,
    pub cost: f64,
    pub n_expanded: u64,
    pub completed: bool,
}

/// A* from every heading at `start` to the cheapest state at `goal` (search.run_search with
/// astar=True), over the open cells (and the two endpoints).
pub fn astar(prob: &Problem, start: (i64, i64), goal: (i64, i64), stage: &str, report: &mut Report) -> Found {
    let (nr, nc, nh) = (prob.nr, prob.nc, prob.n_theta());
    let mut mask: Vec<bool> = prob.open.iter().map(|&o| o != 0).collect();
    mask[start.0 as usize * nc + start.1 as usize] = true;
    mask[goal.0 as usize * nc + goal.1 as usize] = true;
    let index = CellIndex::from_mask(nr, nc, &mask);
    let sources: Vec<u64> = (0..nh).map(|h| index.state(start.0, start.1, h as u16, nh).unwrap()).collect();
    let cp = prob.cost_params();
    let model = prob.model(&cp);
    let p = &prob.params;
    let heur = ClimbHeuristic {
        goal: (goal.0 as i32, goal.1 as i32),
        goal_z: prob.zs[goal.0 as usize * nc + goal.1 as usize],
        zs: Grid::new(&prob.zs, nr, nc),
        res: prob.res,
        g_max: p.g_max,
        c_len: p.c_base + p.c_travel,
        c_vertical: p.c_vertical,
    };
    let spec = SearchSpec::astar(&sources, (goal.0 as i32, goal.1 as i32));
    let (res, completed) =
        search_with_progress(&prob.lattice, &index, &model, &heur, &spec, PROGRESS_EVERY, &mut |pr: &Progress| report(stage, pr));
    let edges = res.goal_state.map(|s| {
        res.edges_to(&prob.lattice, &index, s).into_iter().map(|(r, c, a)| (r as i64, c as i64, a)).collect()
    });
    let cost = res.goal_state.map_or(f64::INFINITY, |s| res.g[s as usize]);
    Found { edges, cost, n_expanded: res.n_expanded, completed }
}

#[derive(Clone, Debug)]
pub struct RouteResult {
    pub edges: Vec<Edge>,
    pub segments: Vec<Segment>,
    pub cost: TrueCost,
    pub lam_fill: f64,
}

#[derive(Clone, Debug)]
pub struct Iteration {
    pub iteration: usize,
    pub lam_fill: f64,
    pub surrogate: f64,
    pub true_cost: f64,
    pub n_crossings: usize,
    pub shared_with_previous: f64,
    pub settled: u64,
}

#[derive(Clone, Debug, Default)]
pub struct IterationResult {
    pub best: Option<RouteResult>,
    pub history: Vec<Iteration>,
    pub final_lam: f64,
    /// False if a report stopped a search.
    pub completed: bool,
}

/// Fraction of route a's length made of edges that also appear in b.
pub fn shared_length_fraction(a: &[Segment], b: &[Segment]) -> f64 {
    let keys: std::collections::HashSet<(i64, i64, usize)> = b.iter().map(|s| (s.r, s.c, s.action)).collect();
    let tot: f64 = a.iter().map(|s| s.length).sum();
    let shared: f64 = a.iter().filter(|s| keys.contains(&(s.r, s.c, s.action))).map(|s| s.length).sum();
    if tot > 0.0 { shared / tot } else { 1.0 }
}

/// Search with lam_fill = C_borrow, cost the route truly, set lam_fill to the realized cost per
/// m3 of crossing fill and search again; stop when the route keeps at least `same` of its
/// length, has no fill crossings, or after max_iter searches. Keeps the cheapest true cost.
pub fn fill_price_iteration(prob: &mut Problem, start: (i64, i64), goal: (i64, i64), max_iter: usize, same: f64,
                            report: &mut Report) -> Result<IterationResult, String> {
    let mut lam = prob.params.c_borrow;
    let mut out = IterationResult { completed: true, ..Default::default() };
    let mut prev: Option<Vec<Segment>> = None;
    for it in 0..max_iter {
        prob.lam_fill = lam;
        let found = astar(prob, start, goal, &format!("A* search {}", it + 1), report);
        if !found.completed {
            out.completed = false;
            break;
        }
        let Some(edges) = found.edges else { break };
        let segs = prob.evaluate_route(&edges)?;
        let tc = true_cost(&segs, &prob.params);
        let n_cross = segs.iter().filter(|s| s.kind == FILL_CROSSING).count();
        let shared = prev.as_ref().map_or(f64::NAN, |pv| shared_length_fraction(&segs, pv));
        out.history.push(Iteration {
            iteration: it, lam_fill: lam, surrogate: found.cost, true_cost: tc.total, n_crossings: n_cross,
            shared_with_previous: shared, settled: found.n_expanded,
        });
        let better = out.best.as_ref().is_none_or(|b| tc.total < b.cost.total);
        let (fill, cpm) = (tc.mass.fill_m3, tc.mass.cost_per_m3());
        if better {
            out.best = Some(RouteResult { edges, segments: segs.clone(), cost: tc, lam_fill: lam });
        }
        if prev.is_some() && shared >= same {
            break;
        }
        if n_cross == 0 || fill <= 0.0 {
            break;
        }
        prev = Some(segs);
        lam = cpm;
    }
    out.final_lam = out.best.as_ref().map_or(lam, |b| b.lam_fill);
    prob.lam_fill = out.final_lam;
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct OverlapRound {
    pub round: usize,
    pub overlapping_pairs: usize,
    pub seq_range: (usize, usize),
    pub blocked_cells: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub result: &'static str,
}

/// 4-connected dilation, `iterations` times (scipy's binary_dilation with its default cross).
fn dilate(m: &[bool], nr: usize, nc: usize, iterations: usize) -> Vec<bool> {
    let mut cur = m.to_vec();
    for _ in 0..iterations {
        let prev = cur.clone();
        for r in 0..nr {
            for c in 0..nc {
                let k = r * nc + c;
                if prev[k] {
                    continue;
                }
                cur[k] = (r > 0 && prev[k - nc]) || (r + 1 < nr && prev[k + nc]) || (c > 0 && prev[k - 1])
                    || (c + 1 < nc && prev[k + 1]);
            }
        }
    }
    cur
}

/// Re-search while the best route overlaps itself (loops back over its own road): each round
/// blocks the cells of the excursion between the overlapping pieces, buffered by half the road
/// width, and repeats the fill-price search; if that leaves no route the previous one is kept.
/// Returns the final result, a record per round, and the blocked cells (prob.open keeps them).
pub fn avoid_self_overlap(prob: &mut Problem, start: (i64, i64), goal: (i64, i64), mut it: IterationResult,
                          max_rounds: usize, report: &mut Report)
                          -> Result<(IterationResult, Vec<OverlapRound>, Vec<bool>), String> {
    let (nr, nc) = (prob.nr, prob.nc);
    let mut blocked = vec![false; nr * nc];
    let mut rounds = Vec::new();
    for k in 0..max_rounds {
        let Some(best) = it.best.as_ref() else { break };
        if !it.completed {
            break;
        }
        let pieces: Vec<Vec<(f64, f64)>> = best.segments.iter().map(|s| s.pts.clone()).collect();
        let ov = self_overlaps(&pieces, prob.res, prob.params.w, 1.3);
        if ov.is_empty() {
            break;
        }
        let lo = ov.iter().map(|o| o.0).min().unwrap() + 1;
        let hi = ov.iter().map(|o| o.1).max().unwrap() - 1;
        let excursion: Vec<Edge> = if lo <= hi { best.edges[lo..=hi].to_vec() } else { Vec::new() };
        let m = prob.edges_mask(&excursion)?;
        let mut m = dilate(&m, nr, nc, ((0.5 * prob.params.w / prob.res).ceil() as usize).max(1));
        for (r, c) in [start, goal] {
            for rr in (r - 1).max(0)..(r + 2).min(nr as i64) {
                for cc in (c - 1).max(0)..(c + 2).min(nc as i64) {
                    m[rr as usize * nc + cc as usize] = false;
                }
            }
        }
        let saved = prob.open.clone();
        for (o, &b) in prob.open.iter_mut().zip(&m) {
            if b {
                *o = 0;
            }
        }
        let cost_before = best.cost.total;
        let n_blocked = m.iter().filter(|&&b| b).count();
        let new = fill_price_iteration(prob, start, goal, 5, 0.98, report)?;
        let Some(new_best) = new.best.as_ref() else {
            prob.open = saved;
            rounds.push(OverlapRound {
                round: k, overlapping_pairs: ov.len(), seq_range: (lo, hi), blocked_cells: n_blocked, cost_before,
                cost_after: f64::NAN, result: "no route with the excursion blocked; kept the previous route",
            });
            break;
        };
        for (b, &x) in blocked.iter_mut().zip(&m) {
            *b |= x;
        }
        rounds.push(OverlapRound {
            round: k, overlapping_pairs: ov.len(), seq_range: (lo, hi), blocked_cells: n_blocked, cost_before,
            cost_after: new_best.cost.total, result: "re-searched",
        });
        it = new;
    }
    Ok((it, rounds, blocked))
}

/// The fine stage for one problem: the fill-price iteration, then self-overlap avoidance.
pub fn find_route(prob: &mut Problem, start: (i64, i64), goal: (i64, i64), report: &mut Report)
                  -> Result<(IterationResult, Vec<OverlapRound>, Vec<bool>), String> {
    let it = fill_price_iteration(prob, start, goal, 5, 0.98, report)?;
    avoid_self_overlap(prob, start, goal, it, 5, report)
}
