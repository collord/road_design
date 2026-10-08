//! Hairpin and bend macro-actions: design, earthwork and cost (a port of `roadsearch/hairpin.py`
//! and `roadsearch/bend.py`, kept operation for operation).
//!
//! A left hairpin from node P0 (local frame: x along the entry heading, y to its left) ends at
//! P1 = (0, D) heading back: entry leg, a reverse curve outward (keyholes only), the loop, the
//! mirror reverse curve and the exit leg. A left bend is a straight lead-in, an arc of radius
//! R_min and a straight lead-out. Both are priced by the earthwork over their footprint (cut split
//! into soil and rock, cut backslopes and fill slopes along both edges), culverts where the
//! centerline crosses channels, and truck travel and climb. Right-hand macros are the reverses of
//! left ones, so tables of left macros price both directions.

use crate::{culvert_cost, Grid, Run};

const BIG: f64 = 1e6;
/// Samples across the footprint at each station.
pub const N_ACROSS: usize = 6;
/// Leg spacings per heading in the hairpin table.
pub const N_SLOTS: usize = 2;
/// Loop radii tried, as multiples of max(R_hp, D/2).
pub const R_FACTORS: [f64; 2] = [1.0, 1.5];
/// Length of the hairpin detail vector (hairpin.py's HP_FIELDS).
pub const HP_N: usize = 24;
/// Length of the bend detail vector (bend.py's BEND_FIELDS).
pub const BEND_N: usize = 18;

/// Hairpin design and cost parameters (hairpin.py's HairpinParams, same order).
#[derive(Clone, Debug)]
pub struct HairpinParams {
    pub w_hp: f64,
    pub g_hp: f64,
    pub g_max: f64,
    pub leg_max: f64,
    pub c_soil: f64,
    pub c_rock: f64,
    pub f_soil: f64,
    pub f_rock: f64,
    pub k_soil: f64,
    pub k_rock: f64,
    pub cost_base: f64,
    pub cost_soil: f64,
    pub cost_rock: f64,
    pub cost_borrow: f64,
    pub catch_max: f64,
    pub delta: f64,
    pub r_hp: f64,
    pub rho: f64,
    pub cost_wall: f64,
    pub c_travel: f64,
    pub c_vertical: f64,
}

/// Bend design and cost parameters (bend.py's BendParams, same order).
#[derive(Clone, Debug)]
pub struct BendParams {
    pub w: f64,
    pub g_max: f64,
    pub c_soil: f64,
    pub c_rock: f64,
    pub f_soil: f64,
    pub f_rock: f64,
    pub k_soil: f64,
    pub k_rock: f64,
    pub cost_base: f64,
    pub cost_soil: f64,
    pub cost_rock: f64,
    pub cost_borrow: f64,
    pub catch_max: f64,
    pub delta: f64,
    pub c_travel: f64,
    pub c_vertical: f64,
}

/// The rasters macro costs are computed on (the mask the tables are built on).
#[derive(Clone, Copy, Debug)]
pub struct MacroTerrain<'a> {
    pub open: Grid<'a, u8>,
    /// Smoothed elevation and its gradient (x east, y north).
    pub zs: Grid<'a, f64>,
    pub gx: Grid<'a, f64>,
    pub gy: Grid<'a, f64>,
    /// Soil depth (NaN replaced by 0).
    pub soil: Grid<'a, f64>,
    /// Working DEM (design elevations, fill heights).
    pub zr: Grid<'a, f64>,
    pub channel: Grid<'a, u8>,
    pub area: Grid<'a, f64>,
    pub culvert_breaks: &'a [f64],
    pub culvert_costs: &'a [f64],
}

impl MacroTerrain<'_> {
    fn nr(&self) -> i64 {
        self.zr.nr as i64
    }

    fn nc(&self) -> i64 {
        self.zr.nc as i64
    }

    fn channel_at(&self, r: i64, c: i64) -> (bool, f64) {
        if r < 0 || c < 0 || r >= self.nr() || c >= self.nc() {
            return (false, 0.0);
        }
        (self.channel.at(r, c) != 0, self.area.at(r, c))
    }
}

/// max as numba (and Python) compute it: the first argument unless the second is greater, so a
/// NaN in the first argument propagates (f64::max would drop it). An off-grid sample makes a cost
/// NaN, and NaN costs are infeasible, as in the numba kernel.
#[inline(always)]
fn pmax(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// min as numba (and Python) compute it (see pmax).
#[inline(always)]
fn pmin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Bilinear interpolation (hairpin.py's `_bil`): NaN outside the grid.
#[inline(always)]
fn bil(a: &Grid<f64>, fr: f64, fc: f64) -> f64 {
    let (nr, nc) = (a.nr as i64, a.nc as i64);
    if fr < 0.0 || fc < 0.0 || fr > (nr - 1) as f64 || fc > (nc - 1) as f64 {
        return f64::NAN;
    }
    let r0 = (fr as i64).min(nr - 2);
    let c0 = (fc as i64).min(nc - 2);
    let wr = fr - r0 as f64;
    let wc = fc - c0 as f64;
    (1.0 - wr) * ((1.0 - wc) * a.at(r0, c0) + wc * a.at(r0, c0 + 1))
        + wr * ((1.0 - wc) * a.at(r0 + 1, c0) + wc * a.at(r0 + 1, c0 + 1))
}

/// Shoelace area of a polygon clipped to x <= xmax (one Sutherland-Hodgman pass).
fn clipped_area(xs: &[f64], ys: &[f64], xmax: f64) -> f64 {
    let n = xs.len();
    let mut cx = [0.0f64; 8];
    let mut cy = [0.0f64; 8];
    let mut m = 0;
    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi, xj, yj) = (xs[i], ys[i], xs[j], ys[j]);
        let in_i = xi <= xmax;
        let in_j = xj <= xmax;
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

/// Cut wedge outside a footprint edge whose face is h high: (A_soil, A_rock, catch) per meter.
fn edge_cut(h: f64, d: f64, s_o: f64, c_soil: f64, c_rock: f64, catch_max: f64) -> (f64, f64, f64) {
    if h <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if d >= h {
        let den = c_soil - s_o;
        let x2 = if den > 0.0 { h / den } else { BIG };
        let area = clipped_area(&[0.0, x2, 0.0], &[0.0, c_soil * x2, h], catch_max);
        return (area, 0.0, pmin(x2, catch_max));
    }
    let hr = h - d;
    let den_r = c_rock - s_o;
    let x1 = if den_r > 0.0 { hr / den_r } else { BIG };
    let y1 = c_rock * x1;
    let x2 = if d > 0.0 {
        let den_s = c_soil - s_o;
        x1 + if den_s > 0.0 { d / den_s } else { BIG }
    } else {
        x1
    };
    let y2 = h + s_o * x2;
    let total = clipped_area(&[0.0, x1, x2, 0.0], &[0.0, y1, y2, h], catch_max);
    let rock = clipped_area(&[0.0, x1, 0.0], &[0.0, y1, hr], catch_max);
    (pmax(total - rock, 0.0), rock, pmin(x2, catch_max))
}

/// Fill wedge outside an edge standing h above ground: (area, toe), infinite if out of reach.
fn edge_fill(h: f64, s_o: f64, f: f64, catch_max: f64) -> (f64, f64) {
    if h <= 0.0 {
        return (0.0, 0.0);
    }
    let den = f + s_o;
    if den <= 0.0 {
        return (f64::INFINITY, f64::INFINITY);
    }
    let toe = h / den;
    if toe > catch_max {
        return (f64::INFINITY, f64::INFINITY);
    }
    (0.5 * h * toe, toe)
}

/// A centerline segment: (x0, y0, heading0, curvature, length), local frame, meters.
pub type Seg = [f64; 5];

/// The five segments of a left hairpin: (segments, alpha, R), R = D/2 for a semicircle.
pub fn hairpin_segments(d: f64, r: f64, rho: f64, l: f64) -> ([Seg; 5], f64, f64) {
    let mut seg = [[0.0f64; 5]; 5];
    let (r, alpha, phi, dx) = if r <= 0.5 * d + 1e-9 {
        (0.5 * d, 0.0, 0.5 * std::f64::consts::PI, 0.0)
    } else {
        let dx = ((r + rho).powi(2) - (0.5 * d + rho).powi(2)).sqrt();
        let phi = (0.5 * d + rho).atan2(dx);
        (r, 0.5 * std::f64::consts::PI - phi, phi, dx)
    };
    let xc = l + dx;
    let q1x = xc - r * phi.cos();
    let q1y = 0.5 * d - r * phi.sin();
    let q2y = 0.5 * d + r * phi.sin();
    let pi = std::f64::consts::PI;
    seg[0] = [0.0, 0.0, 0.0, 0.0, l];
    seg[1] = [l, 0.0, 0.0, -1.0 / rho, rho * alpha];
    seg[2] = [q1x, q1y, -alpha, 1.0 / r, r * (pi + 2.0 * alpha)];
    seg[3] = [q1x, q2y, pi + alpha, -1.0 / rho, rho * alpha];
    seg[4] = [l, d, pi, 0.0, l];
    (seg, alpha, r)
}

/// (x, y, heading) at distance s along a segment.
#[inline(always)]
pub fn seg_point(seg: &Seg, s: f64) -> (f64, f64, f64) {
    let th0 = seg[2];
    let k = seg[3];
    if k == 0.0 {
        return (seg[0] + s * th0.cos(), seg[1] + s * th0.sin(), th0);
    }
    let th = th0 + k * s;
    (seg[0] + (th.sin() - th0.sin()) / k, seg[1] - (th.cos() - th0.cos()) / k, th)
}

/// Leg length and grades (leg, curves, loop) for a climb dz between the ends.
pub fn hairpin_grades(dz: f64, a_loop: f64, a_curves: f64, g_hp: f64, g_max: f64) -> (f64, f64, f64, f64) {
    let a = dz.abs();
    let sgn = if dz >= 0.0 { 1.0 } else { -1.0 };
    if a <= g_hp * (a_loop + a_curves) {
        let g = dz / (a_loop + a_curves);
        return (0.0, 0.0, g, g);
    }
    if a_curves > 0.0 && a <= g_hp * a_loop + g_max * a_curves {
        return (0.0, 0.0, (dz - sgn * g_hp * a_loop) / a_curves, sgn * g_hp);
    }
    let l = (a - g_hp * a_loop - g_max * a_curves) / (2.0 * g_max);
    (l, sgn * g_max, sgn * g_max, sgn * g_hp)
}

/// Design minus ground over a centerline buffered by width/2:
/// (v_soil, v_rock, v_fill, max_cut, max_fill), or None if a sample is off the open cells.
#[allow(clippy::too_many_arguments)]
fn centerline_footprint(
    seg: &[Seg],
    nst: &[usize],
    grades: &[f64],
    zstart: &[f64],
    r: i64,
    c: i64,
    ux: f64,
    uy: f64,
    res: f64,
    width: f64,
    t: &MacroTerrain,
) -> Option<(f64, f64, f64, f64, f64)> {
    let nx = -uy;
    let ny = ux;
    let w2 = 0.5 * width;
    let dt = width / N_ACROSS as f64;
    let (mut v_soil, mut v_rock, mut v_fill, mut max_cut, mut max_fill) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (nr, nc) = (t.nr(), t.nc());
    for i in 0..seg.len() {
        let n = nst[i];
        if n == 0 {
            continue;
        }
        let ds = seg[i][4] / n as f64;
        let kap = seg[i][3];
        for q in 0..n {
            let s = (q as f64 + 0.5) * ds;
            let (x, y, th) = seg_point(&seg[i], s);
            let zd = zstart[i] + grades[i] * s;
            let mx = -th.sin();
            let my = th.cos();
            for j in 0..N_ACROSS {
                let tt = -w2 + (j as f64 + 0.5) * dt;
                let lx = x + tt * mx;
                let ly = y + tt * my;
                let fr = r as f64 - (lx * uy + ly * ny) / res;
                let fc = c as f64 + (lx * ux + ly * nx) / res;
                let cr = (fr + 0.5).floor() as i64;
                let cc = (fc + 0.5).floor() as i64;
                if !(0 <= cr && cr < nr && 0 <= cc && cc < nc && t.open.at(cr, cc) != 0) {
                    return None;
                }
                let zg = bil(&t.zr, fr, fc);
                let da = ds * dt * (1.0 - kap * tt);
                let e = zd - zg;
                if e > 0.0 {
                    v_fill += e * da;
                    max_fill = pmax(max_fill, e);
                } else {
                    let dsoil = bil(&t.soil, fr, fc);
                    let so = pmin(-e, dsoil);
                    v_soil += so * da;
                    v_rock += (-e - so) * da;
                    max_cut = pmax(max_cut, -e);
                }
            }
        }
    }
    let ok = v_soil < f64::INFINITY && v_rock < f64::INFINITY && v_fill < f64::INFINITY;
    if ok {
        Some((v_soil, v_rock, v_fill, max_cut, max_fill))
    } else {
        None
    }
}

/// Cut or fill wedge outside one footprint edge (side sn = +1 left, -1 right):
/// (A_soil, A_rock, A_fill, reach, face height, edge length factor, local x, local y).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn edge_wedge(
    seg: &Seg,
    s: f64,
    sn: f64,
    zd: f64,
    r: i64,
    c: i64,
    ux: f64,
    uy: f64,
    res: f64,
    w2: f64,
    t: &MacroTerrain,
    f: f64,
    c_soil: f64,
    c_rock: f64,
    catch_max: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let nx = -uy;
    let ny = ux;
    let (x, y, th) = seg_point(seg, s);
    let mx = -th.sin();
    let my = th.cos();
    let lx = x + sn * w2 * mx;
    let ly = y + sn * w2 * my;
    let fr = r as f64 - (lx * uy + ly * ny) / res;
    let fc = c as f64 + (lx * ux + ly * nx) / res;
    let zg = bil(&t.zr, fr, fc);
    let ox = sn * (mx * ux + my * nx);
    let oy = sn * (mx * uy + my * ny);
    let s_o = bil(&t.gx, fr, fc) * ox + bil(&t.gy, fr, fc) * oy;
    let h = zg - zd;
    let (mut a_s, mut a_r, mut a_f) = (0.0, 0.0, 0.0);
    let xr;
    if h > 0.0 {
        let (s_, r_, x_) = edge_cut(h, bil(&t.soil, fr, fc), s_o, c_soil, c_rock, catch_max);
        a_s = s_;
        a_r = r_;
        xr = x_;
    } else {
        let (f_, x_) = edge_fill(-h, s_o, f, catch_max);
        a_f = f_;
        xr = x_;
    }
    (a_s, a_r, a_f, xr, h.abs(), 1.0 - seg[3] * sn * w2, lx, ly)
}

/// Culverts where a centerline crosses channels: (cost, largest contributing area crossed).
#[allow(clippy::too_many_arguments)]
fn centerline_culverts(
    seg: &[Seg],
    nst: &[usize],
    r: i64,
    c: i64,
    r1: i64,
    c1: i64,
    ux: f64,
    uy: f64,
    res: f64,
    t: &MacroTerrain,
) -> (f64, f64) {
    let nx = -uy;
    let ny = ux;
    let (br, co) = (t.culvert_breaks, t.culvert_costs);
    let mut culv = 0.0;
    let mut camax = 0.0f64;
    let mut run = Run { in_run: false, max: 0.0, touches_start: false };
    let (ch0, a0) = t.channel_at(r, c);
    run.step(ch0, a0, true, br, co);
    for i in 0..seg.len() {
        let n = nst[i];
        for q in 0..n {
            let s = (q as f64 + 0.5) * seg[i][4] / n as f64;
            let (x, y, _th) = seg_point(&seg[i], s);
            let cr = (r as f64 - (x * uy + y * ny) / res + 0.5).floor() as i64;
            let cc = (c as f64 + (x * ux + y * nx) / res + 0.5).floor() as i64;
            let (ch, a) = t.channel_at(cr, cc);
            culv += run.step(ch, a, false, br, co);
            if ch {
                camax = pmax(camax, a);
            }
        }
    }
    let (ch1, a1) = t.channel_at(r1, c1);
    culv += run.step(ch1, a1, false, br, co);
    if run.in_run {
        let ts = if run.touches_start { 1.0 } else { 0.0 };
        culv += culvert_cost(run.max, br, co) * (1.0 - 0.5 * ts - 0.5);
    }
    (culv, camax)
}

/// Left hairpin from node (r, c), unit heading (ux, uy), leg spacing k*vlen cells, loop radius R
/// (m). Fills `out` (HP_N values, hairpin.py's HP_FIELDS) and returns the cost (inf if infeasible).
#[allow(clippy::too_many_arguments)]
pub fn hairpin_core(
    r: i64,
    c: i64,
    ux: f64,
    uy: f64,
    k: i64,
    vlen: f64,
    radius: f64,
    res: f64,
    t: &MacroTerrain,
    hp: &HairpinParams,
    out: &mut [f64; HP_N],
) -> f64 {
    out.fill(0.0);
    out[0] = f64::INFINITY;
    let nx = -uy;
    let ny = ux;
    let d = k as f64 * vlen * res;
    let w2 = 0.5 * hp.w_hp;
    let rho = hp.rho;
    let r1 = r - (k as f64 * vlen * ny).round_ties_even() as i64;
    let c1 = c + (k as f64 * vlen * nx).round_ties_even() as i64;
    let (nr, nc) = (t.nr(), t.nc());
    if !(0 <= r1 && r1 < nr && 0 <= c1 && c1 < nc && t.open.at(r, c) != 0 && t.open.at(r1, c1) != 0) {
        return f64::INFINITY;
    }
    let (mut seg, mut alpha, mut big_r) = hairpin_segments(d, radius, rho, 0.0);
    if big_r - w2 <= 0.0 || (alpha > 0.0 && rho - w2 <= 0.0) || d - hp.w_hp <= 0.0 {
        return f64::INFINITY;
    }
    let a_loop = seg[2][4];
    let a_curves = 2.0 * seg[1][4];
    let z0 = t.zr.at(r, c);
    let z1 = t.zr.at(r1, c1);
    let (l, g_leg, g_curve, g_loop) = hairpin_grades(z1 - z0, a_loop, a_curves, hp.g_hp, hp.g_max);
    if l > hp.leg_max {
        return f64::INFINITY;
    }
    let total = 2.0 * l + a_curves + a_loop;
    // The smoothed climb is bounded too, which keeps the A* heuristic consistent.
    if (t.zs.at(r1, c1) - t.zs.at(r, c)).abs() > hp.g_max * total {
        return f64::INFINITY;
    }
    if l > 0.0 {
        (seg, alpha, big_r) = hairpin_segments(d, big_r, rho, l);
    }
    let grades = [g_leg, g_curve, g_loop, g_curve, g_leg];
    let mut zstart = [0.0f64; 5];
    let mut acc = z0;
    for i in 0..5 {
        zstart[i] = acc;
        acc += grades[i] * seg[i][4];
    }
    let mut nst = [0usize; 5];
    for i in 0..5 {
        nst[i] = if seg[i][4] > 0.0 { (seg[i][4] / hp.delta).ceil() as usize } else { 0 };
    }
    let Some((mut v_soil, mut v_rock, mut v_fill, max_cut, max_fill)) =
        centerline_footprint(&seg, &nst, &grades, &zstart, r, c, ux, uy, res, hp.w_hp, t)
    else {
        return f64::INFINITY;
    };
    let loose_s = hp.k_soil * v_soil;
    let loose_r = hp.k_rock * v_rock;
    let f = if loose_s + loose_r > 0.0 {
        (loose_s * hp.f_soil + loose_r * hp.f_rock) / (loose_s + loose_r)
    } else {
        hp.f_soil
    };

    // Edges. The left (+m) edge faces the loop's center and, on the two arms (legs and reverse
    // curves), the other arm across the gap. Arm inner edges are paired with their mirror images.
    let n_arm = nst[0] + nst[1];
    // [arm][station]: soil, rock, fill, reach, face, ds, x, y
    let mut arm = [vec![[0.0f64; 8]; n_arm.max(1)], vec![[0.0f64; 8]; n_arm.max(1)]];
    let mut max_catch = 0.0f64;
    let mut max_toe = 0.0f64;
    for i in 0..5 {
        let n = nst[i];
        if n == 0 {
            continue;
        }
        let ds = seg[i][4] / n as f64;
        for q in 0..n {
            let s = (q as f64 + 0.5) * ds;
            let zd = zstart[i] + grades[i] * s;
            for side in 0..2 {
                let sn = if side == 0 { 1.0 } else { -1.0 };
                let (a_s, a_r, a_f, xr, face, lf, lx, ly) =
                    edge_wedge(&seg[i], s, sn, zd, r, c, ux, uy, res, w2, t, f, hp.c_soil, hp.c_rock, hp.catch_max);
                let dl = ds * lf;
                if a_s > 0.0 || a_r > 0.0 {
                    max_catch = pmax(max_catch, xr);
                } else if a_f < f64::INFINITY {
                    max_toe = pmax(max_toe, xr);
                }
                if side == 0 && i != 2 {
                    let a_ = if i < 2 { 0 } else { 1 };
                    let idx = if i < 2 {
                        if i == 0 { q } else { nst[0] + q }
                    } else if i == 3 {
                        q
                    } else {
                        nst[3] + q
                    };
                    arm[a_][idx] = [
                        a_s * dl,
                        a_r * dl,
                        if a_f < f64::INFINITY { a_f * dl } else { f64::INFINITY },
                        xr,
                        face,
                        dl,
                        lx,
                        ly,
                    ];
                    continue;
                }
                if a_f == f64::INFINITY {
                    return f64::INFINITY;
                }
                v_soil += a_s * dl;
                v_rock += a_r * dl;
                v_fill += a_f * dl;
            }
        }
    }
    let mut conflict = 0.0;
    let mut wall = 0.0;
    for e in 0..n_arm {
        let x = n_arm - 1 - e; // mirror station on the exit arm
        let gap = (arm[0][e][6] - arm[1][x][6]).hypot(arm[0][e][7] - arm[1][x][7]);
        let reach = arm[0][e][3] + arm[1][x][3];
        let overlap = reach > gap || arm[0][e][2] == f64::INFINITY || arm[1][x][2] == f64::INFINITY;
        if overlap && hp.cost_wall < f64::INFINITY {
            wall += arm[0][e][4] * arm[0][e][5] + arm[1][x][4] * arm[1][x][5];
            continue;
        }
        if arm[0][e][2] == f64::INFINITY || arm[1][x][2] == f64::INFINITY {
            return f64::INFINITY;
        }
        if overlap {
            conflict = 1.0;
        }
        for a_ in 0..2 {
            let st = if a_ == 0 { e } else { x };
            v_soil += arm[a_][st][0];
            v_rock += arm[a_][st][1];
            v_fill += arm[a_][st][2];
        }
    }
    let (culv, camax) = centerline_culverts(&seg, &nst, r, c, r1, c1, ux, uy, res, t);
    let loose = hp.k_soil * v_soil + hp.k_rock * v_rock;
    let borrow = pmax(v_fill - loose, 0.0);
    let spoil = pmax(loose - v_fill, 0.0);
    let cost = hp.cost_base * total
        + hp.cost_soil * v_soil
        + hp.cost_rock * v_rock
        + hp.cost_borrow * borrow
        + culv
        + (if wall > 0.0 { hp.cost_wall * wall } else { 0.0 })
        + hp.c_travel * total
        + hp.c_vertical * (t.zs.at(r1, c1) - t.zs.at(r, c)).abs();
    *out = [
        cost,
        l,
        big_r,
        total,
        v_soil,
        v_rock,
        v_fill,
        borrow,
        spoil,
        conflict,
        max_cut,
        max_fill,
        z0,
        z1,
        g_loop.abs(),
        g_loop.abs() * big_r / (big_r - w2),
        max_catch,
        max_toe,
        culv,
        camax,
        d,
        alpha,
        if alpha > 0.0 { rho } else { 0.0 },
        wall,
    ];
    cost
}

/// The cheaper of the loop radii R_FACTORS * max(R_hp, D/2); its detail in `out`.
#[allow(clippy::too_many_arguments)]
pub fn best_hairpin(
    r: i64,
    c: i64,
    ux: f64,
    uy: f64,
    k: i64,
    vlen: f64,
    res: f64,
    t: &MacroTerrain,
    hp: &HairpinParams,
    out: &mut [f64; HP_N],
) -> f64 {
    let d = k as f64 * vlen * res;
    let r0 = pmax(hp.r_hp, 0.5 * d);
    let mut best = f64::INFINITY;
    let mut tmp = [0.0f64; HP_N];
    for f in R_FACTORS {
        let cost = hairpin_core(r, c, ux, uy, k, vlen, f * r0, res, t, hp, &mut tmp);
        if cost < best {
            best = cost;
            *out = tmp;
        }
    }
    if best == f64::INFINITY {
        out[0] = f64::INFINITY;
    }
    best
}

/// The three segments of a left bend (lengths in meters).
pub fn bend_segments(l1: f64, radius: f64, dth: f64, l2: f64) -> [Seg; 3] {
    let mut seg = [[0.0f64; 5]; 3];
    seg[0][4] = l1;
    seg[1][0] = l1;
    seg[1][3] = (if dth > 0.0 { 1.0 } else { -1.0 }) / radius;
    seg[1][4] = radius * dth.abs();
    let (x, y, th) = seg_point(&seg[1], seg[1][4]);
    seg[2] = [x, y, th, 0.0, l2];
    seg
}

/// Left bend from (r, c), unit heading (ux, uy), ending at (r - Y, c + X); lengths in cells.
/// Fills `out` (BEND_N values, bend.py's BEND_FIELDS) and returns the cost (inf if infeasible).
#[allow(clippy::too_many_arguments)]
pub fn bend_core(
    r: i64,
    c: i64,
    ux: f64,
    uy: f64,
    bx: i64,
    by: i64,
    l1: f64,
    radius: f64,
    dth: f64,
    l2: f64,
    res: f64,
    t: &MacroTerrain,
    bp: &BendParams,
    out: &mut [f64; BEND_N],
) -> f64 {
    out.fill(0.0);
    out[0] = f64::INFINITY;
    let r1 = r - by;
    let c1 = c + bx;
    let (nr, nc) = (t.nr(), t.nc());
    if !(0 <= r1 && r1 < nr && 0 <= c1 && c1 < nc && t.open.at(r, c) != 0 && t.open.at(r1, c1) != 0) {
        return f64::INFINITY;
    }
    let seg = bend_segments(l1 * res, radius * res, dth, l2 * res);
    let total = seg[0][4] + seg[1][4] + seg[2][4];
    let z0 = t.zr.at(r, c);
    let z1 = t.zr.at(r1, c1);
    let g = (z1 - z0) / total;
    if g.abs() > bp.g_max * (1.0 + 1e-12) || (t.zs.at(r1, c1) - t.zs.at(r, c)).abs() > bp.g_max * total {
        return f64::INFINITY;
    }
    let grades = [g, g, g];
    let zstart = [z0, z0 + g * seg[0][4], z0 + g * (seg[0][4] + seg[1][4])];
    let mut nst = [0usize; 3];
    for i in 0..3 {
        nst[i] = if seg[i][4] > 0.0 { (seg[i][4] / bp.delta).ceil() as usize } else { 0 };
    }
    let Some((mut v_soil, mut v_rock, mut v_fill, max_cut, max_fill)) =
        centerline_footprint(&seg, &nst, &grades, &zstart, r, c, ux, uy, res, bp.w, t)
    else {
        return f64::INFINITY;
    };
    let loose_s = bp.k_soil * v_soil;
    let loose_r = bp.k_rock * v_rock;
    let f = if loose_s + loose_r > 0.0 {
        (loose_s * bp.f_soil + loose_r * bp.f_rock) / (loose_s + loose_r)
    } else {
        bp.f_soil
    };
    let w2 = 0.5 * bp.w;
    let mut max_catch = 0.0f64;
    let mut max_toe = 0.0f64;
    for i in 0..3 {
        let n = nst[i];
        if n == 0 {
            continue;
        }
        let ds = seg[i][4] / n as f64;
        for q in 0..n {
            let s = (q as f64 + 0.5) * ds;
            let zd = zstart[i] + grades[i] * s;
            for side in 0..2 {
                let sn = if side == 0 { 1.0 } else { -1.0 };
                let (a_s, a_r, a_f, xr, _face, lf, _lx, _ly) =
                    edge_wedge(&seg[i], s, sn, zd, r, c, ux, uy, res, w2, t, f, bp.c_soil, bp.c_rock, bp.catch_max);
                if a_f == f64::INFINITY {
                    return f64::INFINITY;
                }
                if a_s > 0.0 || a_r > 0.0 {
                    max_catch = pmax(max_catch, xr);
                } else if a_f > 0.0 {
                    max_toe = pmax(max_toe, xr);
                }
                let dl = ds * lf;
                v_soil += a_s * dl;
                v_rock += a_r * dl;
                v_fill += a_f * dl;
            }
        }
    }
    let (culv, camax) = centerline_culverts(&seg, &nst, r, c, r1, c1, ux, uy, res, t);
    let loose = bp.k_soil * v_soil + bp.k_rock * v_rock;
    let borrow = pmax(v_fill - loose, 0.0);
    let spoil = pmax(loose - v_fill, 0.0);
    let cost = bp.cost_base * total
        + bp.cost_soil * v_soil
        + bp.cost_rock * v_rock
        + bp.cost_borrow * borrow
        + culv
        + bp.c_travel * total
        + bp.c_vertical * (t.zs.at(r1, c1) - t.zs.at(r, c)).abs();
    *out = [
        cost,
        total,
        v_soil,
        v_rock,
        v_fill,
        borrow,
        spoil,
        max_cut,
        max_fill,
        z0,
        z1,
        g,
        max_catch,
        max_toe,
        culv,
        camax,
        dth.abs().to_degrees(),
        radius * res,
    ];
    cost
}

/// Per-heading geometry of the lattice, for building tables.
pub struct Headings<'a> {
    /// Heading vectors (x east, y north), in lattice cells.
    pub vx: &'a [f64],
    pub vy: &'a [f64],
}

impl Headings<'_> {
    fn unit(&self, h: usize) -> (f64, f64, f64) {
        let vlen = self.vx[h].hypot(self.vy[h]);
        (self.vx[h] / vlen, self.vy[h] / vlen, vlen)
    }
}

/// Bend lattice geometry per (heading, slot) (lattice.BendGeometry), row-major (nh, ns).
pub struct BendShapes<'a> {
    pub ns: usize,
    pub x: &'a [i64],
    pub y: &'a [i64],
    pub l1: &'a [f64],
    pub l2: &'a [f64],
    pub dth: &'a [f64],
    /// Arc radius, cells.
    pub radius: f64,
}

fn map_cells<F>(n_cells: usize, per_cell: usize, f: F) -> Vec<f32>
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    let mut table = vec![f32::INFINITY; n_cells * per_cell];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        table.par_chunks_mut(per_cell).enumerate().for_each(|(i, row)| f(i, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        table.chunks_mut(per_cell).enumerate().for_each(|(i, row)| f(i, row));
    }
    table
}

/// Left-hairpin cost per (cell, heading, slot) for the given flat cells (row * nc + col);
/// slots[h * N_SLOTS + j] is the leg spacing in lattice steps, 0 for an unused slot.
/// Infinite where infeasible; float32 like the numba table.
pub fn hairpin_table(cells: &[i64], heads: &Headings, slots: &[i64], res: f64, t: &MacroTerrain, hp: &HairpinParams) -> Vec<f32> {
    let nh = heads.vx.len();
    let nc = t.nc();
    map_cells(cells.len(), nh * N_SLOTS, |i, row| {
        let (r, c) = (cells[i] / nc, cells[i] % nc);
        if t.open.at(r, c) == 0 {
            return;
        }
        let mut out = [0.0f64; HP_N];
        for h in 0..nh {
            let (ux, uy, vlen) = heads.unit(h);
            for j in 0..N_SLOTS {
                let k = slots[h * N_SLOTS + j];
                if k == 0 {
                    continue;
                }
                let cost = best_hairpin(r, c, ux, uy, k, vlen, res, t, hp, &mut out);
                if cost < f64::INFINITY {
                    row[h * N_SLOTS + j] = cost as f32;
                }
            }
        }
    })
}

/// Left-bend cost per (cell, heading, slot) for the given flat cells. Infinite where infeasible.
pub fn bend_table(cells: &[i64], heads: &Headings, shapes: &BendShapes, res: f64, t: &MacroTerrain, bp: &BendParams) -> Vec<f32> {
    let nh = heads.vx.len();
    let ns = shapes.ns;
    let nc = t.nc();
    map_cells(cells.len(), nh * ns, |i, row| {
        let (r, c) = (cells[i] / nc, cells[i] % nc);
        if t.open.at(r, c) == 0 {
            return;
        }
        let mut out = [0.0f64; BEND_N];
        for h in 0..nh {
            let (ux, uy, _) = heads.unit(h);
            for j in 0..ns {
                let q = h * ns + j;
                let cost = bend_core(
                    r, c, ux, uy, shapes.x[q], shapes.y[q], shapes.l1[q], shapes.radius, shapes.dth[q], shapes.l2[q],
                    res, t, bp, &mut out,
                );
                if cost < f64::INFINITY {
                    row[q] = cost as f32;
                }
            }
        }
    })
}
