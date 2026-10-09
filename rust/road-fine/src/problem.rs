//! The search problem on a fine grid (search.py's make_problem and SearchProblem): the masks
//! and rasters the cost model reads, the lattice, the section cost table and the macro tables.

use std::sync::atomic::AtomicU32;

use lattice_search::Lattice;
use road_cost::macros::{
    self, BendParams, BendShapes, HairpinParams, Headings, LazyMacro, MacroShape, MacroTerrain, N_ACROSS, N_SLOTS,
};
use road_cost::{Actions, CostParams, Grid, MacroTable, RoadCost, SectionTable, Terrain};

use crate::lattice::{build_bends, heading_vectors, lattice_arrays, BendGeometry, LatticeArrays};
use crate::params::Params;
use crate::section::{cost_table, max_catch_at_mask, soil_depth, TABLE_NS};

/// The rasters of a fine grid, as preprocessing leaves them (row-major nr x nc).
pub struct FineInputs {
    pub nr: usize,
    pub nc: usize,
    pub res: f64,
    /// Working DEM (NaN at nodata).
    pub z_raw: Vec<f64>,
    /// Smoothed DEM and its gradient (x east, y north).
    pub zs: Vec<f64>,
    pub gx: Vec<f64>,
    pub gy: Vec<f64>,
    /// Contributing area, ha (NaN or 0 at nodata).
    pub area: Vec<f64>,
    /// 1 where the search may go: not excluded (slope mask, zones) and inside the clip region.
    pub open: Vec<u8>,
    /// Roughness (m) from the native DEM (roughness.py); empty: none.
    pub rough: Vec<f64>,
}

/// Preprocessing of a fine grid (preprocess.py without exclusion zones): smoothing, gradients,
/// the dilated slope mask, flow accumulation. `clip` (1 = the search may go there, e.g. a
/// corridor's buffer) is combined into the open mask. Also returns the slope mask, for checking
/// the endpoints.
pub fn preprocess(z_raw: Vec<f64>, nr: usize, nc: usize, res: f64, p: &Params, clip: &[u8], rough: Vec<f64>)
                  -> (FineInputs, Vec<bool>) {
    let zs = terrain::smooth(&z_raw, nr, nc, p.sigma_m() / res);
    let (gx, gy) = terrain::gradient(&zs, nr, nc, res, -res);
    let slope: Vec<f64> = gx.iter().zip(&gy).map(|(a, b)| a.hypot(*b)).collect();
    let (_raw, excluded) = terrain::slope_mask(&slope, nr, nc, p.g_max, p.s_mask(), max_catch_at_mask(p), res, res);
    let area = terrain::contributing_area(&z_raw, nr, nc, res, res, res * res / 1e4);
    let open = (0..nr * nc).map(|k| u8::from(!excluded[k] && clip[k] != 0)).collect();
    (FineInputs { nr, nc, res, z_raw, zs, gx, gy, area, open, rough }, excluded)
}

/// 1 within `buffer` meters of a polyline of (row, col) points (cell coordinates, centers at
/// whole numbers): the region a corridor's fine search may use.
pub fn corridor_mask(nr: usize, nc: usize, res: f64, line: &[(f64, f64)], buffer: f64) -> Vec<u8> {
    line_distance(nr, nc, res, line).iter().map(|&d| u8::from(d <= buffer)).collect()
}

/// A corridor's own region: within `buffer` meters of its line and no farther from it than from
/// any other corridor's line, give or take `slack` meters. Where corridors share a stretch both
/// may use it; where they part, the ground between them is split at the midline, so each
/// corridor's fine search stays its own and they cannot converge on one route.
pub fn corridor_region(nr: usize, nc: usize, res: f64, own: &[(f64, f64)], others: &[Vec<(f64, f64)>], buffer: f64,
                       slack: f64) -> Vec<u8> {
    let (d_own, nearer) = own_ground(nr, nc, res, std::slice::from_ref(&own.to_vec()), others, slack);
    d_own.iter().zip(&nearer).map(|(&a, &n)| u8::from(a <= buffer && n)).collect()
}

/// Each cell's distance (m) from the own lines, and whether it is no farther from them than from
/// every other line, give or take `slack` m.
pub fn own_ground(nr: usize, nc: usize, res: f64, own: &[Vec<(f64, f64)>], others: &[Vec<(f64, f64)>], slack: f64)
                  -> (Vec<f64>, Vec<bool>) {
    let d_own = lines_distance(nr, nc, res, own);
    if others.is_empty() {
        let n = d_own.len();
        return (d_own, vec![true; n]);
    }
    let d_other = lines_distance(nr, nc, res, others);
    let nearer = d_own.iter().zip(&d_other).map(|(&a, &b)| a <= b + slack).collect();
    (d_own, nearer)
}

/// Distance (m) from every cell to the nearest cell a polyline of (row, col) points crosses.
pub fn line_distance(nr: usize, nc: usize, res: f64, line: &[(f64, f64)]) -> Vec<f64> {
    lines_distance(nr, nc, res, std::slice::from_ref(&line.to_vec()))
}

/// Distance (m) from every cell to the nearest cell any of the polylines crosses.
pub fn lines_distance(nr: usize, nc: usize, res: f64, lines: &[Vec<(f64, f64)>]) -> Vec<f64> {
    let mut on = vec![false; nr * nc];
    let mut mark = |fr: f64, fc: f64| {
        let (r, c) = ((fr + 0.5).floor(), (fc + 0.5).floor());
        if r >= 0.0 && c >= 0.0 && (r as usize) < nr && (c as usize) < nc {
            on[r as usize * nc + c as usize] = true;
        }
    };
    for line in lines {
        if let Some(&(r, c)) = line.first() {
            mark(r, c);
        }
        for w in line.windows(2) {
            let ((r0, c0), (r1, c1)) = (w[0], w[1]);
            let n = (4.0 * (r1 - r0).hypot(c1 - c0)).ceil().max(1.0) as usize;
            for k in 1..=n {
                let f = k as f64 / n as f64;
                mark(r0 + f * (r1 - r0), c0 + f * (c1 - c0));
            }
        }
    }
    terrain::distance_to(&on, nr, nc, res, res)
}

/// How a problem holds its hairpin and bend costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Macros {
    /// Whole tables, filled by build_tables (as the Python does).
    Tables,
    /// Computed when the search first asks for an entry and remembered; entries computed
    /// elsewhere (macro_rows, in other threads or workers) can be stored with set_macro_rows.
    Lazy,
    /// Neither: only macro_rows, to compute rows for another problem's cache.
    Rows,
}

pub struct Problem {
    pub nr: usize,
    pub nc: usize,
    pub res: f64,
    pub params: Params,
    /// The current mask (self-overlap avoidance closes cells), and the one the macro tables
    /// were built on (their costs and shapes are evaluated on it).
    pub open: Vec<u8>,
    pub open_tables: Vec<u8>,
    pub gx: Vec<f64>,
    pub gy: Vec<f64>,
    pub zs: Vec<f64>,
    pub soil: Vec<f64>,
    pub z_raw: Vec<f64>,
    pub channel: Vec<u8>,
    pub area: Vec<f64>,
    /// Distance to the nearest channel cell, cells.
    pub chdist: Vec<f64>,
    /// Roughness (m; zeros if none) and the soil slope it adds per meter (0: none).
    pub rough: Vec<f64>,
    pub k_rough: f64,
    pub lat: LatticeArrays,
    pub lattice: Lattice,
    pub actions: Actions,
    /// Cost per meter over (sidehill, grade), TABLE_NS x table_ng, and its steps.
    pub table: Vec<f64>,
    pub table_ng: usize,
    pub ds: f64,
    pub dg: f64,
    pub hp_slots: Vec<i64>,
    pub bend_geo: BendGeometry,
    pub hp_params: HairpinParams,
    pub bend_params: BendParams,
    /// Flat indices of the cells the macro tables cover (the open ones), each grid cell's row
    /// among them (-1: none), and the tables (cells, headings, slots), filled by build_tables;
    /// or, for a lazy problem, caches filled as the search asks for entries.
    pub table_cells: Vec<i64>,
    pub macro_lookup: Vec<i32>,
    pub hp: Vec<f32>,
    pub bend: Vec<f32>,
    pub lazy: bool,
    pub hp_cache: Vec<AtomicU32>,
    pub bend_cache: Vec<AtomicU32>,
    pub n_bend_slots: usize,
    /// Cells of the tables built so far.
    pub tables_done: usize,
    /// Heading vectors (x east, y north), in lattice cells.
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    /// Current price of crossing fill per m3.
    pub lam_fill: f64,
    /// Per-cell cost factors for the search (empty: none); see RoadCost::penalty.
    pub penalty: Vec<f32>,
}

pub fn hairpin_params(p: &Params, res: f64) -> HairpinParams {
    HairpinParams {
        w_hp: p.w_hp, g_hp: p.g_hp, g_max: p.g_max, leg_max: p.hp_leg_max, c_soil: p.c_soil, c_rock: p.c_rock,
        f_soil: p.f_soil, f_rock: p.f_rock, k_soil: p.k_soil, k_rock: p.k_rock, cost_base: p.c_base,
        cost_soil: p.c_soil_cost, cost_rock: p.c_rock_cost, cost_borrow: p.c_borrow, catch_max: p.catch_max,
        delta: (0.5 * res).min(p.w_hp / N_ACROSS as f64), r_hp: p.r_hp, rho: p.r_min, cost_wall: p.c_wall,
        c_travel: p.c_travel, c_vertical: p.c_vertical,
    }
}

pub fn bend_params(p: &Params, res: f64) -> BendParams {
    BendParams {
        w: p.w, g_max: p.g_max, c_soil: p.c_soil, c_rock: p.c_rock, f_soil: p.f_soil, f_rock: p.f_rock,
        k_soil: p.k_soil, k_rock: p.k_rock, cost_base: p.c_base, cost_soil: p.c_soil_cost, cost_rock: p.c_rock_cost,
        cost_borrow: p.c_borrow, catch_max: p.catch_max, delta: (0.5 * res).min(p.w / N_ACROSS as f64),
        c_travel: p.c_travel, c_vertical: p.c_vertical,
    }
}

/// (N_theta, N_SLOTS) row-major lattice steps along the left normal for each hairpin leg
/// spacing; 0 marks an unused slot (hairpin.py's hairpin_slots).
pub fn hairpin_slots(p: &Params, res: f64) -> Vec<i64> {
    let mut out = Vec::with_capacity(p.n_theta * N_SLOTS);
    for (x, y) in heading_vectors(p.n_theta) {
        let step = (x as f64).hypot(y as f64) * res;
        let k_gap = ((p.w_hp + p.hp_gap_min) / step - 1e-9).ceil().max(1.0) as i64;
        let k_semi = k_gap.max((2.0 * p.r_hp / step - 1e-9).ceil() as i64);
        out.push(k_gap);
        out.push(if k_semi > k_gap { k_semi } else { 0 });
    }
    out
}

impl Problem {
    /// Everything but the macro tables, which build_tables fills (in slices, for progress); or,
    /// if `lazy`, with macro costs computed as the search asks for them (the same values; a
    /// search reaches a fraction of the states, and each entry costs a whole macro design).
    pub fn new(inputs: FineInputs, params: Params, macros_mode: Macros) -> Result<Problem, String> {
        let lazy = macros_mode == Macros::Lazy;
        let tables = macros_mode == Macros::Tables;
        let FineInputs { nr, nc, res, z_raw, zs, gx, gy, area, open, rough } = inputs;
        let n = nr * nc;
        for (name, len) in [("z_raw", z_raw.len()), ("zs", zs.len()), ("gx", gx.len()), ("gy", gy.len()),
                            ("area", area.len()), ("open", open.len())] {
            if len != n {
                return Err(format!("{name} has {len} cells, the grid {n}"));
            }
        }
        let p = &params;
        let has_rough = rough.len() == n;
        let k_rough = if has_rough { p.rough_k() } else { 0.0 };
        let rough: Vec<f64> = if has_rough { rough.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect() } else { vec![0.0; n] };
        let soil: Vec<f64> = (0..n)
            .map(|k| {
                let s = gx[k].hypot(gy[k]);
                // Soil thins with roughness as with extra slope.
                if s.is_nan() { 0.0 } else if k_rough > 0.0 { soil_depth(s + k_rough * rough[k], p) } else { soil_depth(s, p) }
            })
            .collect();
        let area: Vec<f64> = area.iter().map(|&a| if a.is_nan() { 0.0 } else { a }).collect();
        let channel: Vec<u8> = area.iter().map(|&a| (a >= p.a_min) as u8).collect();
        let is_channel: Vec<bool> = channel.iter().map(|&c| c != 0).collect();
        let chdist = terrain::distance_to(&is_channel, nr, nc, 1.0, 1.0);
        let hp_slots = hairpin_slots(p, res);
        let bend_geo = build_bends(p.n_theta, p.r_min / res)?;
        let lat = lattice_arrays(p.n_theta, p.r_min / res, Some(&hp_slots), p.l_cross_max / res, Some(&bend_geo),
                                 p.move_max_cells, p.kink_max_deg)?;
        let (lattice, actions) = lat.to_search();
        let (table, table_ng, ds, dg) = cost_table(p);
        let table_cells: Vec<i64> = (0..n).filter(|&k| open[k] != 0).map(|k| k as i64).collect();
        let mut macro_lookup = vec![-1i32; n];
        for (i, &k) in table_cells.iter().enumerate() {
            macro_lookup[k as usize] = i as i32;
        }
        let nh = p.n_theta;
        let n_bend_slots = bend_geo.ns;
        let (n_hp, n_bend) = (table_cells.len() * nh * N_SLOTS, table_cells.len() * nh * n_bend_slots);
        let vecs = heading_vectors(nh);
        Ok(Problem {
            nr, nc, res,
            hp_params: hairpin_params(p, res),
            bend_params: bend_params(p, res),
            lam_fill: p.c_borrow,
            hp: if tables { vec![f32::INFINITY; n_hp] } else { Vec::new() },
            bend: if tables { vec![f32::INFINITY; n_bend] } else { Vec::new() },
            hp_cache: if lazy { macros::lazy_cache(n_hp) } else { Vec::new() },
            bend_cache: if lazy { macros::lazy_cache(n_bend) } else { Vec::new() },
            lazy,
            tables_done: if tables { 0 } else { table_cells.len() },
            vx: vecs.iter().map(|v| v.0 as f64).collect(),
            vy: vecs.iter().map(|v| v.1 as f64).collect(),
            open_tables: open.clone(),
            open, gx, gy, zs, soil, z_raw, channel, area, chdist, rough, k_rough, lat, lattice, actions, table, table_ng, ds,
            dg, hp_slots,
            bend_geo, table_cells, macro_lookup, n_bend_slots,
            penalty: Vec::new(),
            params,
        })
    }

    /// The whole problem at once (search.py's make_problem).
    pub fn build(inputs: FineInputs, params: Params) -> Result<Problem, String> {
        let mut prob = Problem::new(inputs, params, Macros::Tables)?;
        let n = prob.table_cells.len();
        prob.build_tables(n);
        Ok(prob)
    }

    /// Hairpin and bend costs for up to `n` more table cells. Returns the cells done so far.
    pub fn build_tables(&mut self, n: usize) -> usize {
        let (a, b) = (self.tables_done, (self.tables_done + n).min(self.table_cells.len()));
        if a == b || self.lazy {
            return b;
        }
        let nh = self.params.n_theta;
        let heads = Headings { vx: &self.vx, vy: &self.vy };
        let cells = &self.table_cells[a..b];
        let t = self.macro_terrain();
        let hp = macros::hairpin_table(cells, &heads, &self.hp_slots, self.res, &t, &self.hp_params);
        let g = &self.bend_geo;
        let shapes = BendShapes { ns: g.ns, x: &g.x, y: &g.y, l1: &g.l1, l2: &g.l2, dth: &g.dth, radius: g.r };
        let bend = macros::bend_table(cells, &heads, &shapes, self.res, &t, &self.bend_params);
        self.hp[a * nh * N_SLOTS..b * nh * N_SLOTS].copy_from_slice(&hp);
        self.bend[a * nh * self.n_bend_slots..b * nh * self.n_bend_slots].copy_from_slice(&bend);
        self.tables_done = b;
        b
    }

    /// Hairpin and bend costs of table cells a..b, as build_tables computes them:
    /// (cells, headings, slots) rows of each table.
    pub fn macro_rows(&self, a: usize, b: usize) -> (Vec<f32>, Vec<f32>) {
        let b = b.min(self.table_cells.len());
        let a = a.min(b);
        let heads = Headings { vx: &self.vx, vy: &self.vy };
        let cells = &self.table_cells[a..b];
        let t = self.macro_terrain();
        let hp = macros::hairpin_table(cells, &heads, &self.hp_slots, self.res, &t, &self.hp_params);
        let g = &self.bend_geo;
        let shapes = BendShapes { ns: g.ns, x: &g.x, y: &g.y, l1: &g.l1, l2: &g.l2, dth: &g.dth, radius: g.r };
        let bend = macros::bend_table(cells, &heads, &shapes, self.res, &t, &self.bend_params);
        (hp, bend)
    }

    /// Store macro_rows(a, ..) of an identical problem in this lazy problem's caches.
    pub fn set_macro_rows(&mut self, a: usize, hp: &[f32], bend: &[f32]) -> Result<(), String> {
        let per_hp = self.n_theta() * N_SLOTS;
        let per_bend = self.n_theta() * self.n_bend_slots;
        if !self.lazy || hp.len() % per_hp != 0 || bend.len() / per_bend != hp.len() / per_hp
            || (a + hp.len() / per_hp) > self.table_cells.len() {
            return Err("macro rows do not fit this problem".into());
        }
        for (slot, v) in self.hp_cache[a * per_hp..a * per_hp + hp.len()].iter().zip(hp) {
            slot.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
        for (slot, v) in self.bend_cache[a * per_bend..a * per_bend + bend.len()].iter().zip(bend) {
            slot.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn n_theta(&self) -> usize {
        self.params.n_theta
    }

    /// Terrain for a hairpin or bend: on the mask its table was built on.
    pub fn macro_terrain(&self) -> MacroTerrain<'_> {
        let (nr, nc) = (self.nr, self.nc);
        MacroTerrain {
            open: Grid::new(&self.open_tables, nr, nc),
            zs: Grid::new(&self.zs, nr, nc),
            gx: Grid::new(&self.gx, nr, nc),
            gy: Grid::new(&self.gy, nr, nc),
            soil: Grid::new(&self.soil, nr, nc),
            zr: Grid::new(&self.z_raw, nr, nc),
            channel: Grid::new(&self.channel, nr, nc),
            area: Grid::new(&self.area, nr, nc),
            culvert_breaks: &self.params.culvert_area_breaks,
            culvert_costs: &self.params.culvert_costs,
        }
    }

    pub fn cost_params(&self) -> CostParams {
        let p = &self.params;
        CostParams {
            res: self.res, g_max: p.g_max, c_base: p.c_base, culvert_breaks: p.culvert_area_breaks.clone(),
            culvert_costs: p.culvert_costs.clone(), lam_fill: self.lam_fill, w: p.w, f_rock: p.f_rock,
            h_fill_max: p.h_fill_max, fill_tol: p.fill_tol, c_travel: p.c_travel, c_vertical: p.c_vertical,
            c_turn: p.c_turn, k_rough: self.k_rough,
        }
    }

    /// Macro entries the lazy caches hold so far (hairpins, bends).
    pub fn lazy_computed(&self) -> (usize, usize) {
        let count = |c: &[AtomicU32]| c.iter().filter(|v| v.load(std::sync::atomic::Ordering::Relaxed) != macros::UNSET).count();
        (count(&self.hp_cache), count(&self.bend_cache))
    }

    /// Calls `f` with the cost model over this problem (its lam_fill included).
    pub fn with_model<R>(&self, f: impl FnOnce(&RoadCost) -> R) -> R {
        let cp = self.cost_params();
        let heads = Headings { vx: &self.vx, vy: &self.vy };
        let g = &self.bend_geo;
        let shapes = BendShapes { ns: g.ns, x: &g.x, y: &g.y, l1: &g.l1, l2: &g.l2, dth: &g.dth, radius: g.r };
        let lazy = |cache, shape| LazyMacro {
            cache, cells: &self.table_cells, heads, res: self.res, t: self.macro_terrain(), shape,
        };
        let hp_lazy = lazy(&self.hp_cache[..], MacroShape::Hairpin { slots: &self.hp_slots, hp: &self.hp_params });
        let bend_lazy = lazy(&self.bend_cache[..], MacroShape::Bend { shapes, bp: &self.bend_params });
        let (nr, nc) = (self.nr, self.nc);
        let look = Grid::new(&self.macro_lookup, nr, nc);
        let nh = self.n_theta();
        let n = self.table_cells.len();
        let (hp_shape, bend_shape) = ([n, nh, N_SLOTS], [n, nh, self.n_bend_slots]);
        let (hairpins, bends) = if self.lazy {
            (MacroTable::lazy(&hp_lazy, hp_shape, look), MacroTable::lazy(&bend_lazy, bend_shape, look))
        } else {
            (MacroTable::new(&self.hp, hp_shape, look), MacroTable::new(&self.bend, bend_shape, look))
        };
        let model = RoadCost {
            actions: &self.actions,
            terrain: Terrain {
                open: Grid::new(&self.open, nr, nc),
                gx: Grid::new(&self.gx, nr, nc),
                gy: Grid::new(&self.gy, nr, nc),
                zs: Grid::new(&self.zs, nr, nc),
                z_raw: Grid::new(&self.z_raw, nr, nc),
                channel: Grid::new(&self.channel, nr, nc),
                area: Grid::new(&self.area, nr, nc),
                chdist: Grid::new(&self.chdist, nr, nc),
                rough: Grid::new(&self.rough, nr, nc),
            },
            section: SectionTable { data: &self.table, ns: TABLE_NS, ng: self.table_ng, ds: self.ds, dg: self.dg },
            hairpins,
            bends,
            params: &cp,
            penalty: if self.penalty.is_empty() { None } else { Some(Grid::new(&self.penalty, nr, nc)) },
        };
        f(&model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corridors_share_a_common_stretch_and_split_where_they_part() {
        // Two lines from (50, 10): A runs east along row 50, B turns north-east after column 30.
        let a = vec![(50.0, 10.0), (50.0, 30.0), (50.0, 90.0)];
        let b = vec![(50.0, 10.0), (50.0, 30.0), (10.0, 90.0)];
        let (nr, nc, res) = (100, 100, 10.0);
        let ra = corridor_region(nr, nc, res, &a, std::slice::from_ref(&b), 1000.0, 0.0);
        let rb = corridor_region(nr, nc, res, &b, std::slice::from_ref(&a), 1000.0, 0.0);
        let at = |m: &[u8], r: usize, c: usize| m[r * nc + c] != 0;
        // The shared stretch: both.
        assert!(at(&ra, 50, 20) && at(&rb, 50, 20));
        // Near each line where they part: its own only.
        assert!(at(&ra, 52, 85) && !at(&rb, 52, 85));
        assert!(at(&rb, 13, 85) && !at(&ra, 13, 85));
        // Every cell within the buffer belongs to one of them.
        assert!((0..nr * nc).all(|k| ra[k] != 0 || rb[k] != 0));
        // Without other corridors, the plain buffer.
        let alone = corridor_region(nr, nc, res, &a, &[], 100.0, 0.0);
        assert!(at(&alone, 50, 20) && at(&alone, 59, 50) && !at(&alone, 70, 50));
    }
}
