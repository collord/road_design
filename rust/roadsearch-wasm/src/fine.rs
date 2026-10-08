//! The fine search for one corridor: `new` (preprocessing of the corridor's grid, clipped to its
//! buffer), then `search` (A* with fill-price repeats and self-overlap avoidance, reporting
//! progress through a JS callback). Hairpin and bend costs are computed as the search reaches
//! them, not tabulated for every cell first.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use road_fine::params::Params;
use road_fine::pipeline::find_route;
use road_fine::problem::{corridor_mask, preprocess, Problem};

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
    segments: Vec<SegmentOut>,
    history: Vec<IterationOut>,
    overlap_rounds: usize,
    completed: bool,
    /// Hairpin and bend designs computed (lazily) for the search.
    macros_computed: usize,
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
               buffer: f64) -> Result<FineSession, JsValue> {
        if z.len() != w * h || w < 2 || h < 2 {
            return Err(js_err("the grid needs w x h values, at least 2 x 2"));
        }
        let p = Params::from_json(params_json).map_err(js_err)?;
        let line: Vec<(f64, f64)> = line_rows.iter().zip(line_cols).map(|(&r, &c)| (r, c)).collect();
        let clip = corridor_mask(h, w, res, &line, buffer);
        let z: Vec<f64> = z.iter().map(|&v| v as f64).collect();
        let (inputs, excluded) = preprocess(z, h, w, res, &p, &clip);
        let prob = Problem::new(inputs, p, true).map_err(js_err)?;
        Ok(FineSession { prob, excluded })
    }

    /// Cells the search may use.
    pub fn cells(&self) -> usize {
        self.prob.table_cells.len()
    }

    /// 1 where the search may go (the corridor's buffer, minus ground too steep for a road).
    pub fn open_mask(&self) -> Vec<u8> {
        self.prob.open.clone()
    }

    /// The route from (sr, sc) to (gr, gc). `progress(stage, settled, key)` is called every
    /// 65,536 settled states.
    pub fn search(&mut self, sr: i32, sc: i32, gr: i32, gc: i32, progress: &js_sys::Function) -> Result<JsValue, JsValue> {
        let (sr, sc, gr, gc) = (sr as i64, sc as i64, gr as i64, gc as i64);
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
        let this = JsValue::NULL;
        let mut report = |stage: &str, pr: &lattice_search::Progress| -> bool {
            let _ = progress.call3(&this, &JsValue::from_str(stage), &JsValue::from_f64(pr.n_expanded as f64),
                                   &JsValue::from_f64(pr.key));
            true
        };
        let (it, rounds, _blocked) = find_route(&mut self.prob, (sr, sc), (gr, gc), &mut report).map_err(js_err)?;
        let Some(best) = it.best else { return Ok(JsValue::NULL) };
        let (mut rows, mut cols, mut z, mut segments) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, s) in best.segments.iter().enumerate() {
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
        let count = |k: &str| best.segments.iter().filter(|s| s.kind == k).count();
        let c = &best.cost;
        let out = FineRoute {
            cost: CostOut {
                total: c.total, construction: c.construction(), operating: c.operating, base: c.base,
                excavation: c.excavation, culverts: c.culverts, crossing_fill: c.crossing_fill,
                fill_m3: c.mass.fill_m3, borrow_m3: c.mass.borrow_m3, surrogate: c.surrogate,
            },
            length: best.segments.iter().map(|s| s.length).sum(),
            hairpins: count(road_fine::route::HAIRPIN),
            bends: count(road_fine::route::BEND),
            fill_crossings: count(road_fine::route::FILL_CROSSING),
            lam_fill: best.lam_fill,
            rows, cols, z, segments,
            history: it.history.iter().map(|h| IterationOut {
                iteration: h.iteration, lam_fill: h.lam_fill, surrogate: h.surrogate, true_cost: h.true_cost,
                n_crossings: h.n_crossings, settled: h.settled,
            }).collect(),
            overlap_rounds: rounds.len(),
            completed: it.completed,
            macros_computed: {
                let (a, b) = self.prob.lazy_computed();
                a + b
            },
        };
        serde_wasm_bindgen::to_value(&out).map_err(js_err)
    }
}
