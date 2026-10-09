//! The half-bench sidecast cross-section (a port of roadsearch/section.py), the soil depth rule,
//! and the cost table the search interpolates.
//!
//! Section plane: x is horizontal distance from the hinge, y vertical. Uphill the ground is
//! y = s x and the rock surface y = s x - d; downhill the ground is y = -s x.

use crate::params::Params;

pub const INFEASIBLE: i8 = -1;
pub const FLAT: i8 = 0;
pub const HALF_BENCH: i8 = 1;
pub const FULL_BENCH: i8 = 2;

/// Rows and columns of the cost table over (sidehill slope, |grade|).
pub const TABLE_NS: usize = 4097;
pub const TABLE_NG: usize = 21;

const BISECT_ITERS: usize = 60;
const S_ZERO: f64 = 1e-12;

/// The parameters the section uses (section.py's SectionParams).
#[derive(Clone, Copy, Debug)]
pub struct SectionParams {
    pub w: f64,
    pub c_soil: f64,
    pub c_rock: f64,
    pub f_soil: f64,
    pub f_rock: f64,
    pub k_soil: f64,
    pub k_rock: f64,
    pub b_min: f64,
    pub s_mask: f64,
    pub s_flat: f64,
    pub c_base: f64,
    pub c_soil_cost: f64,
    pub c_rock_cost: f64,
    pub catch_max: f64,
}

impl SectionParams {
    pub fn new(p: &Params) -> Self {
        SectionParams {
            w: p.w,
            c_soil: p.c_soil,
            c_rock: p.c_rock,
            f_soil: p.f_soil,
            f_rock: p.f_rock,
            k_soil: p.k_soil,
            k_rock: p.k_rock,
            b_min: p.b_min,
            s_mask: p.s_mask(),
            s_flat: p.s_flat,
            c_base: p.c_base,
            c_soil_cost: p.c_soil_cost,
            c_rock_cost: p.c_rock_cost,
            catch_max: p.catch_max,
        }
    }
}

/// One cross-section: areas per meter of road (m2 = m3/m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Section {
    pub kind: i8,
    /// Cut width and fill width.
    pub a: f64,
    pub b: f64,
    pub a_soil: f64,
    pub a_rock: f64,
    pub a_fill: f64,
    /// Loose volume per meter left over (full bench only).
    pub spoil: f64,
    pub cut_catch: f64,
    pub fill_toe: f64,
    /// Spoil fill slope.
    pub f: f64,
    /// Cost per meter.
    pub cost: f64,
}

fn shoelace4(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
    let acc = (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3);
    0.5 * acc.abs()
}

fn shoelace3(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let acc = (x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) + (x2 * y0 - x0 * y2);
    0.5 * acc.abs()
}

/// Shoelace area of the polygon clipped to x <= xmax (one Sutherland-Hodgman pass).
fn clipped_area(xs: &[f64], ys: &[f64], xmax: f64) -> f64 {
    let n = xs.len();
    let mut cx = [0.0; 8];
    let mut cy = [0.0; 8];
    let mut m = 0;
    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi, xj, yj) = (xs[i], ys[i], xs[j], ys[j]);
        let (in_i, in_j) = (xi <= xmax, xj <= xmax);
        if in_i {
            cx[m] = xi;
            cy[m] = yi;
            m += 1;
        }
        if in_i != in_j {
            let t = (xmax - xi) / (xj - xi);
            cx[m] = xmax;
            cy[m] = yi + t * (yj - yi);
            m += 1;
        }
    }
    let mut acc = 0.0;
    for i in 0..m {
        let j = (i + 1) % m;
        acc += cx[i] * cy[j] - cx[j] * cy[i];
    }
    0.5 * acc.abs()
}

/// Cut of bench width `a` on sidehill `s` with vertical soil depth `d`: (A_soil, A_rock, catch),
/// catch being the horizontal distance from the hinge to where the backslope meets the ground,
/// clipped at catch_max as if ended by a vertical face.
pub fn cut_geometry(a: f64, s: f64, d: f64, c_soil: f64, c_rock: f64, catch_max: f64) -> (f64, f64, f64) {
    if a <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if s <= S_ZERO {
        return (0.0, 0.0, a.min(catch_max));
    }
    if a * s <= d {
        // Bench floor stays in soil: one triangle, backslope at c_soil.
        let x2 = c_soil * a / (c_soil - s);
        if x2 <= catch_max {
            return (shoelace3(0.0, 0.0, a, 0.0, x2, s * x2), 0.0, x2);
        }
        return (clipped_area(&[0.0, a, x2], &[0.0, 0.0, s * x2], catch_max), 0.0, catch_max);
    }
    // Floor enters rock at xr = d/s; backslope at c_rock to the rock surface at x1, then at
    // c_soil through the soil to the ground at x2.
    let xr = d / s;
    let x1 = (c_rock * a - d) / (c_rock - s);
    let y1 = c_rock * (x1 - a);
    let x2 = if d > 0.0 { x1 + d / (c_soil - s) } else { x1 };
    let y2 = s * x2;
    if x2 <= catch_max {
        let total = shoelace4(0.0, 0.0, a, 0.0, x1, y1, x2, y2);
        let rock = shoelace3(xr, 0.0, a, 0.0, x1, y1);
        return ((total - rock).max(0.0), rock, x2);
    }
    let total = clipped_area(&[0.0, a, x1, x2], &[0.0, 0.0, y1, y2], catch_max);
    let rock = clipped_area(&[xr, a, x1], &[0.0, 0.0, y1], catch_max);
    ((total - rock).max(0.0), rock, catch_max)
}

/// Volume-weighted fill slope of the sidecast spoil.
fn spoil_fill_slope(vs: f64, vr: f64, d: f64, s: f64, sp: &SectionParams) -> f64 {
    let tot = vs + vr;
    if tot > 0.0 {
        return (vs * sp.f_soil + vr * sp.f_rock) / tot;
    }
    // No cut yet: the first spoil is soil if there is soil at the hinge.
    if d > 0.0 && s < sp.c_soil { sp.f_soil } else { sp.f_rock }
}

fn fill_area(b: f64, s: f64, f: f64) -> f64 {
    if b <= 0.0 {
        return 0.0;
    }
    if s >= f {
        return f64::INFINITY;
    }
    0.5 * b * b * s * f / (f - s)
}

/// Cross-section on sidehill slope s with soil depth d (section.py's section_core).
pub fn section(s: f64, d: f64, sp: &SectionParams) -> Section {
    let s = s.abs();
    let w = sp.w;
    if s >= sp.s_mask {
        return Section {
            kind: INFEASIBLE, a: 0.0, b: 0.0, a_soil: 0.0, a_rock: 0.0, a_fill: 0.0, spoil: 0.0, cut_catch: 0.0,
            fill_toe: 0.0, f: 0.0, cost: f64::INFINITY,
        };
    }
    // Soil cannot stand steeper than its own cut slope.
    let d = if d < 0.0 || s >= sp.c_soil { 0.0 } else { d };
    if s <= S_ZERO {
        let k = if d > 0.0 { sp.k_soil } else { sp.k_rock };
        let f = if d > 0.0 { sp.f_soil } else { sp.f_rock };
        let a = w / (1.0 + k.sqrt()); // the balance's limit as s -> 0
        return Section {
            kind: FLAT, a, b: w - a, a_soil: 0.0, a_rock: 0.0, a_fill: 0.0, spoil: 0.0, cut_catch: a, fill_toe: w - a,
            f, cost: sp.c_base,
        };
    }
    let mut full = s >= sp.f_soil.max(sp.f_rock);
    let mut a = w;
    if !full {
        // Bisection on r(a) = loose cut - fill(w - a); r(0) < 0 < r(w).
        let (mut lo, mut hi) = (0.0, w);
        for _ in 0..BISECT_ITERS {
            let mid = 0.5 * (lo + hi);
            let (a_s, a_r, _) = cut_geometry(mid, s, d, sp.c_soil, sp.c_rock, sp.catch_max);
            let (vs, vr) = (sp.k_soil * a_s, sp.k_rock * a_r);
            let f = spoil_fill_slope(vs, vr, d, s, sp);
            if vs + vr - fill_area(w - mid, s, f) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        a = 0.5 * (lo + hi);
    }
    let (mut a_s, mut a_r, mut catch) = cut_geometry(a, s, d, sp.c_soil, sp.c_rock, sp.catch_max);
    let (mut vs, mut vr) = (sp.k_soil * a_s, sp.k_rock * a_r);
    let mut f = spoil_fill_slope(vs, vr, d, s, sp);
    let b = w - a;
    if !full && (s >= f || b < sp.b_min) {
        full = true;
    }
    if full {
        (a_s, a_r, catch) = cut_geometry(w, s, d, sp.c_soil, sp.c_rock, sp.catch_max);
        vs = sp.k_soil * a_s;
        vr = sp.k_rock * a_r;
        f = spoil_fill_slope(vs, vr, d, s, sp);
        let cost = sp.c_base + sp.c_soil_cost * a_s + sp.c_rock_cost * a_r;
        return Section {
            kind: FULL_BENCH, a: w, b: 0.0, a_soil: a_s, a_rock: a_r, a_fill: 0.0, spoil: vs + vr, cut_catch: catch,
            fill_toe: 0.0, f, cost,
        };
    }
    let a_fill = fill_area(b, s, f);
    let toe = f * b / (f - s);
    let kind = if s < sp.s_flat { FLAT } else { HALF_BENCH };
    let cost = sp.c_base + sp.c_soil_cost * a_s + sp.c_rock_cost * a_r;
    Section { kind, a, b, a_soil: a_s, a_rock: a_r, a_fill, spoil: 0.0, cut_catch: catch, fill_toe: toe, f, cost }
}

/// Vertical soil depth for terrain slope magnitude S: d0 exp(-S/S0), none at c_soil or steeper.
pub fn soil_depth(slope: f64, p: &Params) -> f64 {
    if slope >= p.c_soil { 0.0 } else { p.d0 * (-slope / p.s0).exp() }
}

/// Largest cut catch distance at the mask threshold slope (a full bench), at most catch_max: the
/// slope mask's dilation radius.
pub fn max_catch_at_mask(p: &Params) -> f64 {
    let s = p.s_mask();
    let mut worst: f64 = 0.0;
    for d in [0.0, soil_depth(s, p)] {
        let d = if s < p.c_soil { d } else { 0.0 };
        let (_, _, catch) = cut_geometry(p.w, s, d, p.c_soil, p.c_rock, p.catch_max);
        worst = worst.max(catch);
    }
    worst
}

/// Cost per meter over (sidehill slope, |grade|), TABLE_NS rows by the returned number of grade
/// columns (TABLE_NG; with roughness on, on at the same step to c_soil, for the soil-effective
/// grade), row-major, with its steps (ds, dg): search.py's build_cost_table. The last row
/// (s = s_mask) is infeasible.
pub fn cost_table(p: &Params) -> (Vec<f64>, usize, f64, f64) {
    let sp = SectionParams::new(p);
    let s_mask = p.s_mask();
    // np.linspace: i * step, the last value exactly the end.
    let lin = |stop: f64, n: usize| -> Vec<f64> {
        let step = stop / (n - 1) as f64;
        let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        v[n - 1] = stop;
        v
    };
    let (s, mut g) = (lin(s_mask, TABLE_NS), lin(p.g_max, TABLE_NG));
    let dg = g[1] - g[0];
    if p.rough_r0 > 0.0 {
        let end = (p.c_soil / dg).ceil() as usize + 2;
        g.extend((TABLE_NG..end).map(|i| i as f64 * dg));
    }
    let ng = g.len();
    let mut table = vec![0.0; TABLE_NS * ng];
    for i in 0..TABLE_NS {
        for j in 0..ng {
            let d = soil_depth(s[i].hypot(g[j]), p);
            table[i * ng + j] = if i == TABLE_NS - 1 { f64::INFINITY } else { section(s[i], d, &sp).cost };
        }
    }
    (table, ng, s[1] - s[0], dg)
}
