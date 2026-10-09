//! The fine search for one corridor: `new` (preprocessing of the corridor's grid, clipped to its
//! buffer), then `search` (A* with fill-price repeats and self-overlap avoidance, reporting
//! progress and each route found through JS callbacks). Hairpin and bend costs are computed as
//! the search reaches them; other workers can compute them ahead, in parallel: a session made
//! `for_rows` computes `macro_rows` for slices of the cells, and the searching session stores
//! them with `set_macro_rows` (the same values it would compute).

use serde::Serialize;
use wasm_bindgen::prelude::*;

use road_fine::params::Params;
use road_fine::masshaul::TrueCost;
use road_fine::network::FineNetwork;
use road_fine::pipeline::{find_route, FineProgress};
use road_fine::problem::{own_ground, preprocess, Macros, Problem};
use road_fine::route::Segment;

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[derive(Serialize)]
struct CostOut {
    total: f64,
    construction: f64,
    operating: f64,
    base: f64,
    excavation: f64,
    culverts: f64,
    crossing_fill: f64,
    fill_m3: f64,
    borrow_m3: f64,
    surrogate: f64,
}

#[derive(Serialize)]
struct SegmentOut {
    kind: &'static str,
    /// Index of the segment's first point in the route's points.
    first: usize,
    length: f64,
    grade: f64,
    cost: f64,
    construction: f64,
    operating: f64,
}

#[derive(Serialize)]
struct IterationOut {
    iteration: usize,
    lam_fill: f64,
    surrogate: f64,
    true_cost: f64,
    n_crossings: usize,
    settled: u64,
}

#[derive(Serialize)]
struct FineRoute {
    cost: CostOut,
    length: f64,
    hairpins: usize,
    bends: usize,
    fill_crossings: usize,
    lam_fill: f64,
    /// Centerline points (row, col on this grid) and the road's design elevation at each.
    rows: Vec<f64>,
    cols: Vec<f64>,
    z: Vec<f64>,
    /// Ground elevation at each point (the final route only).
    ground: Vec<f64>,
    segments: Vec<SegmentOut>,
    history: Vec<IterationOut>,
    overlap_rounds: usize,
    completed: bool,
    /// Hairpin and bend designs computed (lazily) for the search.
    macros_computed: usize,
}

/// A route's centerline, costs and segments, for the page.
fn fine_route(segs: &[Segment], c: &TrueCost, lam_fill: f64) -> FineRoute {
    let (mut rows, mut cols, mut z, mut segments) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, s) in segs.iter().enumerate() {
        let skip = usize::from(i > 0);
        segments.push(SegmentOut {
            kind: s.kind, first: rows.len().saturating_sub(skip), length: s.length, grade: s.grade,
            cost: s.cost_total(), construction: s.cost_construction(), operating: s.cost_operating,
        });
        for (&(r, c), &zz) in s.pts.iter().zip(&s.z).skip(skip) {
            rows.push(r);
            cols.push(c);
            z.push(zz);
        }
    }
    let count = |k: &str| segs.iter().filter(|s| s.kind == k).count();
    FineRoute {
        cost: CostOut {
            total: c.total, construction: c.construction(), operating: c.operating, base: c.base,
            excavation: c.excavation, culverts: c.culverts, crossing_fill: c.crossing_fill, fill_m3: c.mass.fill_m3,
            borrow_m3: c.mass.borrow_m3, surrogate: c.surrogate,
        },
        length: segs.iter().map(|s| s.length).sum(),
        hairpins: count(road_fine::route::HAIRPIN),
        bends: count(road_fine::route::BEND),
        fill_crossings: count(road_fine::route::FILL_CROSSING),
        lam_fill,
        rows, cols, z, segments,
        ground: Vec::new(),
        history: Vec::new(),
        overlap_rounds: 0,
        completed: true,
        macros_computed: 0,
    }
}

/// One corridor's fine grid and search problem.
#[wasm_bindgen]
pub struct FineSession {
    prob: Problem,
    excluded: Vec<bool>,
    network: Option<FineNetwork>,
}

/// Polylines from points with NaN between lines.
fn polylines(rows: &[f64], cols: &[f64]) -> Vec<Vec<(f64, f64)>> {
    let mut out: Vec<Vec<(f64, f64)>> = vec![Vec::new()];
    for (&r, &c) in rows.iter().zip(cols) {
        if r.is_nan() || c.is_nan() {
            out.push(Vec::new());
        } else {
            out.last_mut().unwrap().push((r, c));
        }
    }
    out.retain(|l| !l.is_empty());
    out
}

/// A progress callback for the page: {stage, settled, key, cells, g, rows, cols} (see search).
fn js_report(progress: &js_sys::Function) -> impl FnMut(&str, &FineProgress) -> bool + '_ {
    move |stage: &str, pr: &FineProgress| -> bool {
        let o = js_sys::Object::new();
        let set = |k: &str, v: &JsValue| {
            let _ = js_sys::Reflect::set(&o, &k.into(), v);
        };
        set("stage", &JsValue::from_str(stage));
        set("settled", &JsValue::from_f64(pr.settled as f64));
        set("key", &JsValue::from_f64(pr.key));
        set("cells", &js_sys::Uint32Array::from(&pr.cells[..]));
        set("g", &js_sys::Float32Array::from(pr.g));
        let rows: Vec<f64> = pr.path.iter().map(|p| p.0).collect();
        let cols: Vec<f64> = pr.path.iter().map(|p| p.1).collect();
        set("rows", &js_sys::Float64Array::from(&rows[..]));
        set("cols", &js_sys::Float64Array::from(&cols[..]));
        let _ = progress.call1(&JsValue::NULL, &o);
        true
    }
}

#[wasm_bindgen]
impl FineSession {
    /// The DEM on the corridor's grid (row-major w x h, meters, NaN for nodata) at `res` m;
    /// parameters as JSON (Params.to_dict(), infinities as null); the coarse corridor as
    /// (row, col) points on this grid, and the buffer (m) the search may use around it. The
    /// other corridors' lines (points, NaN between lines) mark the ground nearer to them than to
    /// this corridor's line (within two cells): `apart` is the search's cost factor there (1: the
    /// ground is shared freely; infinite: it is closed).
    #[wasm_bindgen(constructor)]
    pub fn new(z: &[f32], w: usize, h: usize, res: f64, params_json: &str, line_rows: &[f64], line_cols: &[f64],
               buffer: f64, other_rows: &[f64], other_cols: &[f64], apart: f64, for_rows: bool)
               -> Result<FineSession, JsValue> {
        if z.len() != w * h || w < 2 || h < 2 {
            return Err(js_err("the grid needs w x h values, at least 2 x 2"));
        }
        let p = Params::from_json(params_json).map_err(js_err)?;
        // The corridor's line, or a network's edges (NaN between them).
        let lines = polylines(line_rows, line_cols);
        let mut others: Vec<Vec<(f64, f64)>> = vec![Vec::new()];
        let apart = if apart.is_nan() { 1.0 } else { apart.max(1.0) };
        if apart > 1.0 {
            for (&r, &c) in other_rows.iter().zip(other_cols) {
                if r.is_nan() || c.is_nan() {
                    others.push(Vec::new());
                } else {
                    others.last_mut().unwrap().push((r, c));
                }
            }
        }
        others.retain(|l| !l.is_empty());
        let (d_own, nearer) = own_ground(h, w, res, &lines, &others, 2.0 * res);
        let closed = apart.is_infinite();
        let clip: Vec<u8> = d_own.iter().zip(&nearer).map(|(&d, &n)| u8::from(d <= buffer && (n || !closed))).collect();
        let z: Vec<f64> = z.iter().map(|&v| v as f64).collect();
        let (inputs, excluded) = preprocess(z, h, w, res, &p, &clip);
        let mut prob = Problem::new(inputs, p, if for_rows { Macros::Rows } else { Macros::Lazy }).map_err(js_err)?;
        if apart > 1.0 && !closed && !others.is_empty() {
            prob.penalty = nearer.iter().map(|&n| if n { 1.0 } else { apart as f32 }).collect();
        }
        Ok(FineSession { prob, excluded, network: None })
    }

    /// Cells the search may use.
    pub fn cells(&self) -> usize {
        self.prob.table_cells.len()
    }

    /// The grid cell (row * w + col) of each of the search's cells, in table order (the order of
    /// macro_rows).
    pub fn table_cells(&self) -> Vec<u32> {
        self.prob.table_cells.iter().map(|&k| k as u32).collect()
    }

    /// Hairpin and bend costs of cells a..b: {hp, bend} Float32Arrays for set_macro_rows.
    pub fn macro_rows(&self, a: usize, b: usize) -> js_sys::Object {
        let (hp, bend) = self.prob.macro_rows(a, b);
        let o = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&o, &"hp".into(), &js_sys::Float32Array::from(&hp[..]));
        let _ = js_sys::Reflect::set(&o, &"bend".into(), &js_sys::Float32Array::from(&bend[..]));
        o
    }

    /// Store another session's macro_rows(a, ..) (the same grid and parameters).
    pub fn set_macro_rows(&mut self, a: usize, hp: &[f32], bend: &[f32]) -> Result<(), JsValue> {
        self.prob.set_macro_rows(a, hp, bend).map_err(js_err)
    }

    /// 1 where the search may go (the corridor's buffer, minus ground too steep for a road).
    pub fn open_mask(&self) -> Vec<u8> {
        self.prob.open.clone()
    }

    /// Start a pad network on this grid: the access point (row, col) and the pads' centers (row,
    /// col, cell coordinates), `length` x `width` m each with the long axis along the contour, at
    /// most `b` pads beyond any pad. Returns each pad's long axis (radians, east = 0).
    #[allow(clippy::too_many_arguments)]
    pub fn network_begin(&mut self, ar: i32, ac: i32, pad_rows: &[f64], pad_cols: &[f64], length: f64, width: f64,
                         b: usize) -> Result<Vec<f64>, JsValue> {
        let (nr, nc) = (self.prob.nr as i32, self.prob.nc as i32);
        if ar < 0 || ac < 0 || ar >= nr || ac >= nc {
            return Err(js_err("the access point is outside the network's grid"));
        }
        let centers: Vec<(f64, f64)> = pad_rows.iter().zip(pad_cols).map(|(&r, &c)| (r, c)).collect();
        let net = FineNetwork::new(&mut self.prob, (ar as i64, ac as i64), &centers, length, width, b);
        let axes = net.pads.iter().map(|p| p.axis).collect();
        self.network = Some(net);
        Ok(axes)
    }

    /// Connect pad `p` to the network so far; false if no branch reaches it. Progress as for search.
    pub fn network_connect(&mut self, p: usize, progress: &js_sys::Function) -> Result<bool, JsValue> {
        let net = self.network.as_mut().ok_or_else(|| js_err("network_begin() first"))?;
        if p >= net.pads.len() {
            return Err(js_err("no such pad"));
        }
        let mut report = js_report(progress);
        Ok(net.connect(&self.prob, p, &mut report))
    }

    /// One improvement pass; returns the pads whose branch moved.
    pub fn network_improve(&mut self, progress: &js_sys::Function) -> Result<Vec<u32>, JsValue> {
        let net = self.network.as_mut().ok_or_else(|| js_err("network_begin() first"))?;
        let mut report = js_report(progress);
        Ok(net.improve(&self.prob, &mut report).into_iter().map(|p| p as u32).collect())
    }

    /// The network: {runs: [{pad, uses, rows, cols, z}] (stretches of a branch used by the same
    /// number of pads; grid coordinates and design elevations), connected, unreachable, beyond,
    /// construction, operating}.
    pub fn network_state(&self) -> Result<JsValue, JsValue> {
        let net = self.network.as_ref().ok_or_else(|| js_err("network_begin() first"))?;
        #[derive(Serialize)]
        struct Run {
            pad: usize,
            uses: u32,
            rows: Vec<f64>,
            cols: Vec<f64>,
            z: Vec<f64>,
        }
        #[derive(Serialize)]
        struct State {
            runs: Vec<Run>,
            connected: Vec<bool>,
            unreachable: Vec<bool>,
            beyond: Vec<usize>,
            /// Per pad, its ends on open ground (0: no road can arrive).
            ends: Vec<usize>,
            construction: f64,
            operating: f64,
        }
        let uses = net.segment_uses();
        let mut runs: Vec<Run> = Vec::new();
        for (p, b) in net.branches.iter().enumerate() {
            let Some(b) = b else { continue };
            for (i, s) in b.segments.iter().enumerate() {
                let u = uses[p][i];
                let same = runs.last().is_some_and(|r| r.pad == p && r.uses == u) && i > 0;
                if !same {
                    runs.push(Run { pad: p, uses: u, rows: Vec::new(), cols: Vec::new(), z: Vec::new() });
                }
                let run = runs.last_mut().unwrap();
                let skip = usize::from(!run.rows.is_empty());
                for (&(r, c), &z) in s.pts.iter().zip(&s.z).skip(skip) {
                    run.rows.push(r);
                    run.cols.push(c);
                    run.z.push(z);
                }
            }
        }
        let (construction, operating) = net.cost();
        let st = State {
            runs,
            connected: net.branches.iter().map(|b| b.is_some()).collect(),
            unreachable: net.unreachable.clone(),
            beyond: net.beyond(),
            ends: net.pads.iter().map(|p| p.ends.iter().flatten().count()).collect(),
            construction,
            operating,
        };
        serde_wasm_bindgen::to_value(&st).map_err(js_err)
    }

    fn check_endpoints(&self, sr: i64, sc: i64, gr: i64, gc: i64) -> Result<(), JsValue> {
        let (nr, nc) = (self.prob.nr as i64, self.prob.nc as i64);
        for (name, r, c) in [("start", sr, sc), ("finish", gr, gc)] {
            if r < 0 || c < 0 || r >= nr || c >= nc {
                return Err(js_err(format!("the {name} is outside the corridor's grid")));
            }
            let k = (r * nc + c) as usize;
            if self.prob.z_raw[k].is_nan() {
                return Err(js_err(format!("the {name} is on a nodata cell")));
            }
            if self.excluded[k] {
                return Err(js_err(format!("the {name} is on ground too steep for a road (slope mask)")));
            }
        }
        Ok(())
    }

    /// The route from (sr, sc) to (gr, gc). `progress({stage, settled, key, cells, g, rows, cols})`
    /// is called every 65,536 settled states: the grid cells first settled since the last call
    /// and their cost to come (the wavefront), and the centerline of the route to the state just
    /// settled (the leading candidate). `on_route(route)` is called with each route a search
    /// finds (the same fields as the result, without the history).
    pub fn search(&mut self, sr: i32, sc: i32, gr: i32, gc: i32, progress: &js_sys::Function,
                  on_route: &js_sys::Function) -> Result<JsValue, JsValue> {
        let (sr, sc, gr, gc) = (sr as i64, sc as i64, gr as i64, gc as i64);
        self.check_endpoints(sr, sc, gr, gc)?;
        let this = JsValue::NULL;
        let mut report = |stage: &str, pr: &FineProgress| -> bool {
            let o = js_sys::Object::new();
            let set = |k: &str, v: &JsValue| {
                let _ = js_sys::Reflect::set(&o, &k.into(), v);
            };
            set("stage", &JsValue::from_str(stage));
            set("settled", &JsValue::from_f64(pr.settled as f64));
            set("key", &JsValue::from_f64(pr.key));
            set("cells", &js_sys::Uint32Array::from(&pr.cells[..]));
            set("g", &js_sys::Float32Array::from(pr.g));
            let rows: Vec<f64> = pr.path.iter().map(|p| p.0).collect();
            let cols: Vec<f64> = pr.path.iter().map(|p| p.1).collect();
            set("rows", &js_sys::Float64Array::from(&rows[..]));
            set("cols", &js_sys::Float64Array::from(&cols[..]));
            let _ = progress.call1(&this, &o);
            true
        };
        let mut found = |segs: &[Segment], tc: &TrueCost| {
            if let Ok(v) = serde_wasm_bindgen::to_value(&fine_route(segs, tc, f64::NAN)) {
                let _ = on_route.call1(&this, &v);
            }
        };
        let (it, rounds, _blocked) =
            find_route(&mut self.prob, (sr, sc), (gr, gc), &mut report, &mut found).map_err(js_err)?;
        let Some(best) = it.best else { return Ok(JsValue::NULL) };
        let mut out = fine_route(&best.segments, &best.cost, best.lam_fill);
        out.history = it.history.iter().map(|h| IterationOut {
            iteration: h.iteration, lam_fill: h.lam_fill, surrogate: h.surrogate, true_cost: h.true_cost,
            n_crossings: h.n_crossings, settled: h.settled,
        }).collect();
        out.ground = out.rows.iter().zip(&out.cols).map(|(&r, &c)| self.prob.ground_at(r, c)).collect();
        out.overlap_rounds = rounds.len();
        out.completed = it.completed;
        let (a, b) = self.prob.lazy_computed();
        out.macros_computed = a + b;
        serde_wasm_bindgen::to_value(&out).map_err(js_err)
    }
}
