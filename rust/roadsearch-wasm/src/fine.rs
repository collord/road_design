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
use road_fine::pipeline::find_route;
use road_fine::problem::{corridor_mask, preprocess, Macros, Problem};
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
}

#[wasm_bindgen]
impl FineSession {
    /// The DEM on the corridor's grid (row-major w x h, meters, NaN for nodata) at `res` m;
    /// parameters as JSON (Params.to_dict(), infinities as null); the coarse corridor as
    /// (row, col) points on this grid, and the buffer (m) the search may use around it.
    #[wasm_bindgen(constructor)]
    pub fn new(z: &[f32], w: usize, h: usize, res: f64, params_json: &str, line_rows: &[f64], line_cols: &[f64],
               buffer: f64, for_rows: bool) -> Result<FineSession, JsValue> {
        if z.len() != w * h || w < 2 || h < 2 {
            return Err(js_err("the grid needs w x h values, at least 2 x 2"));
        }
        let p = Params::from_json(params_json).map_err(js_err)?;
        let line: Vec<(f64, f64)> = line_rows.iter().zip(line_cols).map(|(&r, &c)| (r, c)).collect();
        let clip = corridor_mask(h, w, res, &line, buffer);
        let z: Vec<f64> = z.iter().map(|&v| v as f64).collect();
        let (inputs, excluded) = preprocess(z, h, w, res, &p, &clip);
        let prob = Problem::new(inputs, p, if for_rows { Macros::Rows } else { Macros::Lazy }).map_err(js_err)?;
        Ok(FineSession { prob, excluded })
    }

    /// Cells the search may use.
    pub fn cells(&self) -> usize {
        self.prob.table_cells.len()
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

    /// The route from (sr, sc) to (gr, gc). `progress(stage, settled, key)` is called every
    /// 65,536 settled states, and `on_route(route)` with each route a search finds (the same
    /// fields as the result, without the history).
    pub fn search(&mut self, sr: i32, sc: i32, gr: i32, gc: i32, progress: &js_sys::Function,
                  on_route: &js_sys::Function) -> Result<JsValue, JsValue> {
        let (sr, sc, gr, gc) = (sr as i64, sc as i64, gr as i64, gc as i64);
        self.check_endpoints(sr, sc, gr, gc)?;
        let this = JsValue::NULL;
        let mut report = |stage: &str, pr: &lattice_search::Progress| -> bool {
            let _ = progress.call3(&this, &JsValue::from_str(stage), &JsValue::from_f64(pr.n_expanded as f64),
                                   &JsValue::from_f64(pr.key));
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
