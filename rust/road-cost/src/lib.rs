//! Road edge costs on a DEM, for [`lattice_search`].
//!
//! A port of the numba kernel in `roadsearch/search.py` (`edge_cost` and what
//! it calls), kept operation for operation so the two agree to the last bit:
//!
//! * motion primitives: earthwork per meter from a section table over
//!   (sidehill slope, |grade|), the grade checked on the profile itself,
//!   truck travel and climb, and culverts where the edge crosses channels;
//! * fill crossings: a straight run on a linear design over a channel,
//!   priced by fill volume at the current fill price;
//! * hairpins and bends: looked up in precomputed tables (left-hand ones;
//!   right-hand ones are their reverses);
//! * reversing switchbacks: a zero-length turn onto a heading roughly opposite, priced by a level
//!   landing past the junction (its samples carried by the action like a primitive's);
//! * every edge: a turning charge per radian of heading change.
//!
//! An edge is always costed in its canonical direction (the lower-numbered of
//! an action and its reverse), so an edge and its reverse cost the same.

use lattice_search::{EdgeCost, Heuristic};

pub mod macros;

pub const ACTION_PRIM: u8 = 0;
pub const ACTION_HP_LEFT: u8 = 1;
pub const ACTION_HP_RIGHT: u8 = 2;
pub const ACTION_FILL: u8 = 3;
pub const ACTION_BEND_L: u8 = 4;
pub const ACTION_BEND_R: u8 = 5;
pub const ACTION_SWITCH: u8 = 6;

/// Per-action arrays of the road lattice (roadsearch.lattice.LatticeArrays).
#[derive(Clone, Debug, Default)]
pub struct Actions {
    pub h0: Vec<usize>,
    pub dr: Vec<i64>,
    pub dc: Vec<i64>,
    /// Length in cells.
    pub length: Vec<f64>,
    pub rev: Vec<usize>,
    /// First sample and sample count of each primitive.
    pub s0: Vec<usize>,
    pub ns: Vec<usize>,
    pub kind: Vec<u8>,
    pub slot: Vec<usize>,
    /// Heading change, radians.
    pub turn: Vec<f64>,
    /// Samples: row and column offsets (cells) and unit tangent (east, north).
    pub sdr: Vec<f64>,
    pub sdc: Vec<f64>,
    pub tx: Vec<f64>,
    pub ty: Vec<f64>,
}

/// A row-major 2D view.
#[derive(Clone, Copy, Debug)]
pub struct Grid<'a, T> {
    pub data: &'a [T],
    pub nr: usize,
    pub nc: usize,
}

impl<'a, T: Copy> Grid<'a, T> {
    pub fn new(data: &'a [T], nr: usize, nc: usize) -> Self {
        assert_eq!(data.len(), nr * nc, "grid data must be nr * nc long");
        Grid { data, nr, nc }
    }

    #[inline(always)]
    pub fn at(&self, r: i64, c: i64) -> T {
        self.data[r as usize * self.nc + c as usize]
    }
}

/// Macro-action costs per (table cell, heading, slot), float32, infinite where infeasible;
/// `lookup` gives each grid cell's row in the table (-1: none). Tables hold only the cells
/// that were open when they were built.
#[derive(Clone, Copy, Debug)]
pub struct MacroTable<'a> {
    pub data: &'a [f32],
    /// (cells, headings, slots)
    pub shape: [usize; 3],
    pub lookup: Grid<'a, i32>,
    /// Entries computed when first asked for, instead of `data`.
    pub lazy: Option<&'a macros::LazyMacro<'a>>,
}

impl<'a> MacroTable<'a> {
    pub fn new(data: &'a [f32], shape: [usize; 3], lookup: Grid<'a, i32>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        MacroTable { data, shape, lookup, lazy: None }
    }

    /// A table whose entries are computed as the search asks for them.
    pub fn lazy(lazy: &'a macros::LazyMacro<'a>, shape: [usize; 3], lookup: Grid<'a, i32>) -> Self {
        assert_eq!(lazy.cache.len(), shape.iter().product::<usize>());
        MacroTable { data: &[], shape, lookup, lazy: Some(lazy) }
    }

    #[inline(always)]
    fn get(&self, r: i64, c: i64, h: usize, s: usize) -> f64 {
        let [n, nh, ns] = self.shape;
        if r < 0 || c < 0 || r as usize >= self.lookup.nr || c as usize >= self.lookup.nc || h >= nh || s >= ns {
            return f64::INFINITY;
        }
        let k = self.lookup.at(r, c);
        if k < 0 || k as usize >= n {
            return f64::INFINITY;
        }
        if let Some(l) = self.lazy {
            return l.entry(k as usize, h, s, nh, ns) as f64;
        }
        self.data[(k as usize * nh + h) * ns + s] as f64
    }
}

/// Earthwork cost per meter over (sidehill slope s, |grade| g), bilinear.
#[derive(Clone, Copy, Debug)]
pub struct SectionTable<'a> {
    pub data: &'a [f64],
    pub ns: usize,
    pub ng: usize,
    pub ds: f64,
    pub dg: f64,
}

impl SectionTable<'_> {
    /// Cost per meter at sidehill slope s and |grade| g (bilinear); infinite past the mask.
    #[inline(always)]
    pub fn cost(&self, s: f64, g: f64) -> f64 {
        let fs = s / self.ds;
        if fs >= self.ns as f64 - 1.0 {
            return f64::INFINITY;
        }
        let fg = (g / self.dg).min(self.ng as f64 - 1.0);
        let i = fs as usize;
        let j = (fg as usize).min(self.ng - 2);
        let ws = fs - i as f64;
        let wg = fg - j as f64;
        let t = |i: usize, j: usize| self.data[i * self.ng + j];
        let (a, b, c, d) = (t(i, j), t(i, j + 1), t(i + 1, j), t(i + 1, j + 1));
        if !(a < f64::INFINITY && b < f64::INFINITY && c < f64::INFINITY && d < f64::INFINITY) {
            return f64::INFINITY;
        }
        (1.0 - ws) * ((1.0 - wg) * a + wg * b) + ws * ((1.0 - wg) * c + wg * d)
    }
}

/// The terrain rasters, all the same shape.
#[derive(Clone, Copy, Debug)]
pub struct Terrain<'a> {
    /// 1 where the search may go.
    pub open: Grid<'a, u8>,
    /// Gradient of the smoothed DEM (x east, y north).
    pub gx: Grid<'a, f64>,
    pub gy: Grid<'a, f64>,
    /// Smoothed elevation.
    pub zs: Grid<'a, f64>,
    /// Working DEM (road elevations, fill heights).
    pub z_raw: Grid<'a, f64>,
    pub channel: Grid<'a, u8>,
    /// Contributing area, ha.
    pub area: Grid<'a, f64>,
    /// Distance to the nearest channel cell, cells.
    pub chdist: Grid<'a, f64>,
    /// Roughness (m), thinning the soil as extra slope (k_rough per meter); zeros if none.
    pub rough: Grid<'a, f64>,
}

/// Scalars of the cost model.
#[derive(Clone, Debug)]
pub struct CostParams {
    pub res: f64,
    pub g_max: f64,
    pub c_base: f64,
    /// Culvert price by contributing area: costs[i] below breaks[i], the last above all.
    pub culvert_breaks: Vec<f64>,
    pub culvert_costs: Vec<f64>,
    /// Price of crossing fill per m3.
    pub lam_fill: f64,
    pub w: f64,
    pub f_rock: f64,
    pub h_fill_max: f64,
    pub fill_tol: f64,
    pub c_travel: f64,
    pub c_vertical: f64,
    pub c_turn: f64,
    /// Slope added to the soil-thinning slope per meter of roughness (S0 / rough_r0); 0: none.
    pub k_rough: f64,
}

/// The cost model: `EdgeCost` for lattice-search.
pub struct RoadCost<'a> {
    pub actions: &'a Actions,
    pub terrain: Terrain<'a>,
    pub section: SectionTable<'a>,
    pub hairpins: MacroTable<'a>,
    pub bends: MacroTable<'a>,
    pub params: &'a CostParams,
    /// A cost factor per cell (>= 1), for the search only: an edge costs the larger factor of
    /// its two end cells times its cost (the same both ways). `edge_cost` stays unpenalized, for
    /// pricing found routes. None: no penalty.
    pub penalty: Option<Grid<'a, f32>>,
}

#[inline(always)]
fn bilinear(a: &Grid<f64>, fr: f64, fc: f64) -> f64 {
    let (nr, nc) = (a.nr as i64, a.nc as i64);
    if fr < 0.0 || fc < 0.0 || fr > (nr - 1) as f64 || fc > (nc - 1) as f64 {
        return f64::NAN;
    }
    let r0 = if nr > 1 { (fr as i64).min(nr - 2) } else { 0 };
    let c0 = if nc > 1 { (fc as i64).min(nc - 2) } else { 0 };
    let wr = fr - r0 as f64;
    let wc = fc - c0 as f64;
    let (r1, c1) = ((r0 + 1).min(nr - 1), (c0 + 1).min(nc - 1));
    (1.0 - wr) * ((1.0 - wc) * a.at(r0, c0) + wc * a.at(r0, c1))
        + wr * ((1.0 - wc) * a.at(r1, c0) + wc * a.at(r1, c1))
}

#[inline(always)]
fn nearest(f: f64) -> i64 {
    (f + 0.5).floor() as i64
}

/// Culvert price for a contributing area: costs[i] below breaks[i], the last above all.
#[inline(always)]
pub fn culvert_cost(area: f64, breaks: &[f64], costs: &[f64]) -> f64 {
    for (i, &b) in breaks.iter().enumerate() {
        if area < b {
            return costs[i];
        }
    }
    costs[costs.len() - 1]
}

/// Channel-run state along an edge (culvert::_run_step).
#[derive(Clone, Copy)]
struct Run {
    in_run: bool,
    max: f64,
    touches_start: bool,
}

impl Run {
    /// Advance by one point; returns the culvert charge of a run that just ended.
    #[inline(always)]
    fn step(&mut self, is_ch: bool, a: f64, first: bool, breaks: &[f64], costs: &[f64]) -> f64 {
        let mut charge = 0.0;
        if is_ch {
            if self.in_run {
                self.max = self.max.max(a);
            } else {
                self.in_run = true;
                self.max = a;
                self.touches_start = first;
            }
        } else if self.in_run {
            charge = culvert_cost(self.max, breaks, costs) * if self.touches_start { 0.5 } else { 1.0 };
            self.in_run = false;
        }
        charge
    }
}

impl<'a> RoadCost<'a> {
    #[inline(always)]
    fn cell_open(&self, r: i64, c: i64) -> bool {
        let o = &self.terrain.open;
        r >= 0 && c >= 0 && (r as usize) < o.nr && (c as usize) < o.nc && o.at(r, c) != 0
    }

    /// Moving between neighboring cells: both open, and a diagonal step needs
    /// one of the two side cells open.
    #[inline(always)]
    fn step_ok(&self, pr: i64, pc: i64, r: i64, c: i64) -> bool {
        if !self.cell_open(r, c) {
            return false;
        }
        if pr != r && pc != c {
            return self.cell_open(pr, c) || self.cell_open(r, pc);
        }
        true
    }

    /// Reversing switchback at (r, c): a level landing along the action's samples (the bisector of
    /// the arriving heading and the reverse of the departing one), priced as road on the sidehill
    /// across it plus the prism of cut or fill that keeps it level, at the fill price.
    fn eval_switch(&self, r: i64, c: i64, p: usize) -> f64 {
        let (a, t, q) = (self.actions, &self.terrain, self.params);
        if !self.cell_open(r, c) {
            return f64::INFINITY;
        }
        let n = a.ns[p];
        let w_s = a.length[p] * q.res / n as f64;
        let z0 = t.zs.at(r, c);
        let (mut total, mut vol) = (0.0, 0.0);
        for k in a.s0[p]..a.s0[p] + n {
            let fr = r as f64 + a.sdr[k];
            let fc = c as f64 + a.sdc[k];
            if !self.cell_open(nearest(fr), nearest(fc)) {
                return f64::INFINITY;
            }
            let zk = bilinear(&t.zs, fr, fc);
            let ax = bilinear(&t.gx, fr, fc);
            let ay = bilinear(&t.gy, fr, fc);
            if !(ax.abs() < f64::INFINITY && ay.abs() < f64::INFINITY && zk.abs() < f64::INFINITY) {
                return f64::INFINITY;
            }
            let h = z0 - zk;
            if h.abs() > q.h_fill_max {
                return f64::INFINITY;
            }
            vol += (q.w * h.abs() + h * h / q.f_rock) * w_s;
            // The ground's slope along the landing enters through soil depth, as grade does on a road.
            let mut g = (ax * a.tx[k] + ay * a.ty[k]).abs();
            let s = (-ax * a.ty[k] + ay * a.tx[k]).abs();
            if q.k_rough > 0.0 {
                let rho = bilinear(&t.rough, fr, fc);
                if rho > 0.0 {
                    let se = (s * s + g * g).sqrt() + q.k_rough * rho;
                    g = (se * se - s * s).max(0.0).sqrt();
                }
            }
            let cpm = self.section.cost(s, g);
            if cpm == f64::INFINITY {
                return f64::INFINITY;
            }
            total += cpm * w_s;
        }
        total + q.lam_fill * vol
    }

    /// Fill crossing: a straight macro-action over a channel on a linear design.
    fn eval_fill(&self, r: i64, c: i64, p: usize) -> f64 {
        let (a, t, q) = (self.actions, &self.terrain, self.params);
        let lc = a.length[p];
        if t.chdist.at(r, c) > lc {
            return f64::INFINITY;
        }
        let (r1, c1) = (r + a.dr[p], c + a.dc[p]);
        if !(self.cell_open(r, c) && self.cell_open(r1, c1)) {
            return f64::INFINITY;
        }
        if t.channel.at(r, c) != 0 || t.channel.at(r1, c1) != 0 {
            return f64::INFINITY;
        }
        let l = lc * q.res;
        let z0 = t.z_raw.at(r, c);
        let z1 = t.z_raw.at(r1, c1);
        let gl = q.g_max * l * (1.0 + 1e-12);
        if (z1 - z0).abs() > gl || (t.zs.at(r1, c1) - t.zs.at(r, c)).abs() > gl {
            return f64::INFINITY;
        }
        let n = 2.max((2.0 * lc).ceil() as i64);
        let wseg = l / n as f64;
        let mut vol = 0.0;
        let mut amax = -1.0f64;
        let (mut pr, mut pc) = (r, c);
        for k in 0..n {
            let f = (k as f64 + 0.5) / n as f64;
            let fr = r as f64 + f * a.dr[p] as f64;
            let fc = c as f64 + f * a.dc[p] as f64;
            let (cr, cc) = (nearest(fr), nearest(fc));
            if cr != pr || cc != pc {
                if (cr - pr).abs() > 1 || (cc - pc).abs() > 1 || !self.step_ok(pr, pc, cr, cc) {
                    return f64::INFINITY;
                }
                pr = cr;
                pc = cc;
            }
            if t.channel.at(cr, cc) != 0 {
                amax = amax.max(t.area.at(cr, cc));
            }
            let zg = bilinear(&t.z_raw, fr, fc);
            let h = z0 + f * (z1 - z0) - zg;
            if !(h >= -q.fill_tol) || h > q.h_fill_max {
                return f64::INFINITY;
            }
            if h > 0.0 {
                vol += (q.w * h + h * h / q.f_rock) * wseg;
            }
        }
        if (r1 != pr || c1 != pc) && ((r1 - pr).abs() > 1 || (c1 - pc).abs() > 1 || !self.step_ok(pr, pc, r1, c1)) {
            return f64::INFINITY;
        }
        if amax < 0.0 {
            return f64::INFINITY;
        }
        (q.c_base + q.c_travel) * l
            + q.c_vertical * (t.zs.at(r1, c1) - t.zs.at(r, c)).abs()
            + q.lam_fill * vol
            + culvert_cost(amax, &q.culvert_breaks, &q.culvert_costs)
    }

    /// At-grade culvert charge of a primitive edge, and the largest area crossed.
    pub fn culvert_charges(&self, r: i64, c: i64, p: usize) -> (f64, f64) {
        let (a, t, q) = (self.actions, &self.terrain, self.params);
        let (br, co) = (&q.culvert_breaks[..], &q.culvert_costs[..]);
        let (r1, c1) = (r + a.dr[p], c + a.dc[p]);
        let mut total = 0.0;
        let mut amax = 0.0f64;
        let mut run = Run { in_run: false, max: 0.0, touches_start: false };
        run.step(t.channel.at(r, c) != 0, t.area.at(r, c), true, br, co);
        if t.channel.at(r, c) != 0 {
            amax = t.area.at(r, c);
        }
        for k in a.s0[p]..a.s0[p] + a.ns[p] {
            let cr = nearest(r as f64 + a.sdr[k]);
            let cc = nearest(c as f64 + a.sdc[k]);
            total += run.step(t.channel.at(cr, cc) != 0, t.area.at(cr, cc), false, br, co);
            if t.channel.at(cr, cc) != 0 {
                amax = amax.max(t.area.at(cr, cc));
            }
        }
        total += run.step(t.channel.at(r1, c1) != 0, t.area.at(r1, c1), false, br, co);
        if t.channel.at(r1, c1) != 0 {
            amax = amax.max(t.area.at(r1, c1));
        }
        if run.in_run {
            let ts = if run.touches_start { 1.0 } else { 0.0 };
            total += culvert_cost(run.max, br, co) * (1.0 - 0.5 * ts - 0.5);
        }
        (total, amax)
    }

    /// Cost of action p from node (r, c) in its own direction; infinite if infeasible.
    pub fn eval_action(&self, r: i64, c: i64, p: usize) -> f64 {
        let (a, t, q) = (self.actions, &self.terrain, self.params);
        let kind = a.kind[p];
        if kind == ACTION_HP_LEFT || kind == ACTION_BEND_L {
            // Tables are built once; the end nodes are checked against the current mask.
            if !(self.cell_open(r, c) && self.cell_open(r + a.dr[p], c + a.dc[p])) {
                return f64::INFINITY;
            }
            let tab = if kind == ACTION_HP_LEFT { &self.hairpins } else { &self.bends };
            return tab.get(r, c, a.h0[p], a.slot[p]);
        }
        if kind == ACTION_FILL {
            return self.eval_fill(r, c, p);
        }
        if kind == ACTION_SWITCH {
            return self.eval_switch(r, c, p);
        }
        let (r1, c1) = (r + a.dr[p], c + a.dc[p]);
        if !(self.cell_open(r, c) && self.cell_open(r1, c1)) {
            return f64::INFINITY;
        }
        let l = a.length[p] * q.res;
        let n = a.ns[p];
        let w_s = l / n as f64;
        let gtol = q.g_max * (1.0 + 1e-12);
        let mut zprev = t.zs.at(r, c);
        let mut step = 0.5 * w_s;
        let mut total = 0.0;
        let mut vert = 0.0;
        let (mut pr, mut pc) = (r, c);
        for k in a.s0[p]..a.s0[p] + n {
            let fr = r as f64 + a.sdr[k];
            let fc = c as f64 + a.sdc[k];
            let (cr, cc) = (nearest(fr), nearest(fc));
            if cr != pr || cc != pc {
                if (cr - pr).abs() > 1 || (cc - pc).abs() > 1 || !self.step_ok(pr, pc, cr, cc) {
                    return f64::INFINITY;
                }
                pr = cr;
                pc = cc;
            }
            let zk = bilinear(&t.zs, fr, fc);
            let ax = bilinear(&t.gx, fr, fc);
            let ay = bilinear(&t.gy, fr, fc);
            if !(ax.abs() < f64::INFINITY && ay.abs() < f64::INFINITY && zk.abs() < f64::INFINITY) {
                return f64::INFINITY;
            }
            if (zk - zprev).abs() > gtol * step {
                return f64::INFINITY;
            }
            vert += (zk - zprev).abs();
            zprev = zk;
            step = w_s;
            // Section geometry from the smoothed gradient; grade enters only through soil depth.
            let mut g = (ax * a.tx[k] + ay * a.ty[k]).abs().min(q.g_max);
            let s = (-ax * a.ty[k] + ay * a.tx[k]).abs();
            if q.k_rough > 0.0 {
                // Roughness thins the soil as extra slope: the soil-effective grade.
                let rho = bilinear(&t.rough, fr, fc);
                if rho > 0.0 {
                    let se = (s * s + g * g).sqrt() + q.k_rough * rho;
                    g = (se * se - s * s).max(0.0).sqrt();
                }
            }
            let cpm = self.section.cost(s, g);
            if cpm == f64::INFINITY {
                return f64::INFINITY;
            }
            total += cpm * w_s;
        }
        if (t.zs.at(r1, c1) - zprev).abs() > gtol * 0.5 * w_s {
            return f64::INFINITY;
        }
        vert += (t.zs.at(r1, c1) - zprev).abs();
        if (r1 != pr || c1 != pc) && ((r1 - pr).abs() > 1 || (c1 - pc).abs() > 1 || !self.step_ok(pr, pc, r1, c1)) {
            return f64::INFINITY;
        }
        total + q.c_travel * l + q.c_vertical * vert + self.culvert_charges(r, c, p).0
    }

    /// Cost of action p from (r, c), evaluated in canonical direction, plus its turning charge.
    #[inline]
    pub fn edge_cost(&self, r: i64, c: i64, p: usize) -> f64 {
        let a = self.actions;
        let q = a.rev[p];
        let cost = if q < p { self.eval_action(r + a.dr[p], c + a.dc[p], q) } else { self.eval_action(r, c, p) };
        cost + self.params.c_turn * a.turn[p]
    }
}

impl EdgeCost for RoadCost<'_> {
    #[inline]
    fn cost(&self, r: i32, c: i32, action: usize) -> f64 {
        let cost = self.edge_cost(r as i64, c as i64, action);
        match &self.penalty {
            None => cost,
            Some(p) => {
                let (r, c) = (r as i64, c as i64);
                let (r1, c1) = (r + self.actions.dr[action], c + self.actions.dc[action]);
                let inside = |r: i64, c: i64| r >= 0 && c >= 0 && (r as usize) < p.nr && (c as usize) < p.nc;
                let f0 = if inside(r, c) { p.at(r, c) } else { 1.0 };
                let f1 = if inside(r1, c1) { p.at(r1, c1) } else { 1.0 };
                cost * f0.max(f1) as f64
            }
        }
    }
}

/// Lower bound on the cost to the goal: every meter costs at least C_base + C_travel, the
/// road is at least as long as the straight distance and as |dz| / g_max, and it climbs |dz|.
pub struct ClimbHeuristic<'a> {
    pub goal: (i32, i32),
    pub goal_z: f64,
    pub zs: Grid<'a, f64>,
    pub res: f64,
    pub g_max: f64,
    pub c_len: f64,
    pub c_vertical: f64,
}

impl Heuristic for ClimbHeuristic<'_> {
    #[inline]
    fn estimate(&self, r: i32, c: i32) -> f64 {
        let dist = ((r - self.goal.0) as f64).hypot((c - self.goal.1) as f64) * self.res;
        let dz = (self.goal_z - self.zs.at(r as i64, c as i64)).abs();
        self.c_len * dist.max(dz / self.g_max) + self.c_vertical * dz
    }
}
