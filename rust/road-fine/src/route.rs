//! A found route: its edges' shapes (search.py's edge_points, hairpin and bend centerlines), the
//! exact evaluation of every segment (route.py's evaluate_route), and self-overlap detection.

use road_cost::macros::{self, BEND_N, HP_N};
use road_cost::{ACTION_BEND_L, ACTION_BEND_R, ACTION_FILL, ACTION_HP_LEFT, ACTION_HP_RIGHT, ACTION_PRIM, ACTION_SWITCH};

use crate::lattice::heading_vectors;
use crate::problem::Problem;
use crate::section::{section, soil_depth, SectionParams, FLAT, FULL_BENCH, HALF_BENCH, INFEASIBLE};

pub const SEC_FLAT: &str = "flat";
pub const SEC_HALF_BENCH: &str = "half_bench";
pub const SEC_FULL_BENCH: &str = "full_bench";
pub const SEC_INFEASIBLE: &str = "infeasible";
pub const HAIRPIN: &str = "hairpin";
pub const BEND: &str = "bend";
pub const FILL_CROSSING: &str = "fill_crossing";
pub const SWITCHBACK: &str = "switchback";

/// An edge of a route: (row, col) of its start node and the action.
pub type Edge = (i64, i64, usize);

/// One evaluated edge (route.py's Segment).
#[derive(Clone, Debug)]
pub struct Segment {
    pub seq: usize,
    pub r: i64,
    pub c: i64,
    pub action: usize,
    pub kind: &'static str,
    /// (row, col) points from the start node to the end node, and the road elevation there.
    pub pts: Vec<(f64, f64)>,
    pub z: Vec<f64>,
    pub length: f64,
    /// Mean grade in the direction of travel.
    pub grade: f64,
    pub sidehill_slope: f64,
    pub terrain_slope: f64,
    pub soil_depth: f64,
    pub cut_width: f64,
    pub fill_width: f64,
    pub cut_catch: f64,
    pub fill_toe: f64,
    pub cut_soil_m3: f64,
    pub cut_rock_m3: f64,
    pub fill_m3: f64,
    pub spoil_m3: f64,
    pub borrow_m3: f64,
    pub cost_base: f64,
    /// Excavation (+ borrow and walls for hairpins; + fill at the fill price for crossings).
    pub cost_earthwork: f64,
    /// The search's edge cost.
    pub cost_surrogate: f64,
    pub cost_culvert: f64,
    pub culvert_area_ha: f64,
    /// Rise + fall along the profile (m), heading change (radians), and the operating cost.
    pub vertical_travel: f64,
    pub turning: f64,
    pub cost_operating: f64,
}

impl Segment {
    pub fn cost_construction(&self) -> f64 {
        self.cost_base + self.cost_earthwork + self.cost_culvert
    }

    pub fn cost_total(&self) -> f64 {
        self.cost_construction() + self.cost_operating
    }
}

/// Bilinear with the corner clamped into the grid (route.py's _bil).
pub(crate) fn bil_clamped(a: &[f64], nr: usize, nc: usize, fr: f64, fc: f64) -> f64 {
    let r0 = (fr.floor() as i64).clamp(0, nr as i64 - 2) as usize;
    let c0 = (fc.floor() as i64).clamp(0, nc as i64 - 2) as usize;
    let (wr, wc) = (fr - r0 as f64, fc - c0 as f64);
    let at = |r: usize, c: usize| a[r * nc + c];
    (1.0 - wr) * ((1.0 - wc) * at(r0, c0) + wc * at(r0, c0 + 1)) + wr * ((1.0 - wc) * at(r0 + 1, c0) + wc * at(r0 + 1, c0 + 1))
}

/// Bilinear, NaN outside the grid (search.py's _bilinear).
fn bil_nan(a: &[f64], nr: usize, nc: usize, fr: f64, fc: f64) -> f64 {
    if fr < 0.0 || fc < 0.0 || fr > (nr - 1) as f64 || fc > (nc - 1) as f64 {
        return f64::NAN;
    }
    let r0 = if nr > 1 { (fr as usize).min(nr - 2) } else { 0 };
    let c0 = if nc > 1 { (fc as usize).min(nc - 2) } else { 0 };
    let (wr, wc) = (fr - r0 as f64, fc - c0 as f64);
    let at = |r: usize, c: usize| a[r * nc + c];
    (1.0 - wr) * ((1.0 - wc) * at(r0, c0) + wc * at(r0, c0 + 1)) + wr * ((1.0 - wc) * at(r0 + 1, c0) + wc * at(r0 + 1, c0 + 1))
}

/// np.linspace(0, stop, n) (n >= 2): i * step, the last exactly stop.
fn linspace(stop: f64, n: usize) -> Vec<f64> {
    let step = stop / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
    v[n - 1] = stop;
    v
}

fn unit_heading(n_theta: usize, h: usize) -> (f64, f64, f64) {
    let (x, y) = heading_vectors(n_theta)[h];
    let vlen = (x as f64).hypot(y as f64);
    (x as f64 / vlen, y as f64 / vlen, vlen)
}

/// Points along macro segments in the local frame (x along u, y to its left), every `spacing`
/// meters or closer, as (row, col); and the distance along the centerline of each.
fn macro_points(seg: &[macros::Seg], r: i64, c: i64, u: (f64, f64), res: f64) -> (Vec<(f64, f64)>, Vec<(usize, f64)>) {
    let n = (-u.1, u.0);
    let mut pts = Vec::new();
    let mut at = Vec::new();
    for (i, sg) in seg.iter().enumerate() {
        let ls = sg[4];
        if ls <= 1e-9 {
            continue;
        }
        let m = ((ls / 1.0).ceil() as usize + 1).max(2);
        let skip = if pts.is_empty() { 0 } else { 1 };
        for &s in &linspace(ls, m)[skip..] {
            let (x, y, _) = macros::seg_point(sg, s);
            let (px, py) = (x * u.0 + y * n.0, x * u.1 + y * n.1);
            pts.push((r as f64 - py / res, c as f64 + px / res));
            at.push((i, s));
        }
    }
    (pts, at)
}

impl Problem {
    /// Ground elevation (the working DEM, bilinear) at a (row, col) point.
    pub fn ground_at(&self, fr: f64, fc: f64) -> f64 {
        bil_clamped(&self.z_raw, self.nr, self.nc, fr, fc)
    }

    /// (row, col, heading, slot) of the left hairpin equivalent to action p from (r, c).
    fn canonical_hairpin(&self, r: i64, c: i64, p: usize) -> (i64, i64, usize, usize) {
        let l = &self.lat;
        if l.kind[p] as u8 == ACTION_HP_LEFT {
            (r, c, l.h0[p] as usize, l.slot[p] as usize)
        } else {
            (r + l.dr[p], c + l.dc[p], l.h0[p] as usize, l.slot[p] as usize)
        }
    }

    fn canonical_bend(&self, r: i64, c: i64, p: usize) -> (i64, i64, usize, usize) {
        let l = &self.lat;
        if l.kind[p] as u8 == ACTION_BEND_L {
            (r, c, l.h0[p] as usize, l.slot[p] as usize)
        } else {
            let q = l.rev[p] as usize;
            (r + l.dr[p], c + l.dc[p], l.h0[q] as usize, l.slot[q] as usize)
        }
    }

    /// The hairpin detail vector (hairpin.py's HP_FIELDS) of a left hairpin.
    pub fn hairpin_detail(&self, r: i64, c: i64, h: usize, slot: usize) -> [f64; HP_N] {
        let (ux, uy, vlen) = unit_heading(self.n_theta(), h);
        let k = self.hp_slots[h * macros::N_SLOTS + slot];
        let mut out = [0.0; HP_N];
        macros::best_hairpin(r, c, ux, uy, k, vlen, self.res, &self.macro_terrain(), &self.hp_params, &mut out);
        out
    }

    /// The bend detail vector (bend.py's BEND_FIELDS) of a left bend.
    pub fn bend_detail(&self, r: i64, c: i64, h: usize, slot: usize) -> [f64; BEND_N] {
        let (ux, uy, _) = unit_heading(self.n_theta(), h);
        let g = &self.bend_geo;
        let i = h * g.ns + slot;
        let mut out = [0.0; BEND_N];
        macros::bend_core(r, c, ux, uy, g.x[i], g.y[i], g.l1[i], g.r, g.dth[i], g.l2[i], self.res, &self.macro_terrain(),
                          &self.bend_params, &mut out);
        out
    }

    /// (detail, points, design elevations) of hairpin action p from (r, c), in travel order.
    fn hairpin_geometry(&self, r: i64, c: i64, p: usize) -> Result<([f64; HP_N], Vec<(f64, f64)>, Vec<f64>), String> {
        let (hr, hc, h, slot) = self.canonical_hairpin(r, c, p);
        let det = self.hairpin_detail(hr, hc, h, slot);
        if !(det[0] < f64::INFINITY) {
            return Err(format!("hairpin at row {hr}, col {hc}, heading {h}, slot {slot} is infeasible"));
        }
        let pr = &self.params;
        let (ux, uy, _) = unit_heading(self.n_theta(), h);
        let (seg, _, _) = macros::hairpin_segments(det[20], det[2], pr.r_min, det[1]);
        let (a_loop, a_curves) = (seg[2][4], 2.0 * seg[1][4]);
        let (_, g_leg, g_curve, g_loop) = macros::hairpin_grades(det[13] - det[12], a_loop, a_curves, pr.g_hp, pr.g_max);
        let grades = [g_leg, g_curve, g_loop, g_curve, g_leg];
        let (mut pts, at) = macro_points(&seg, hr, hc, (ux, uy), self.res);
        // Design elevation: the start, plus each earlier segment's climb, plus this one's so far.
        let mut z_start = [0.0; 5];
        let mut z = det[12];
        for i in 0..5 {
            z_start[i] = z;
            if seg[i][4] > 1e-9 {
                z += grades[i] * seg[i][4];
            }
        }
        let mut zs: Vec<f64> = at.iter().map(|&(i, s)| z_start[i] + grades[i] * s).collect();
        if self.lat.kind[p] as u8 != ACTION_HP_LEFT {
            pts.reverse();
            zs.reverse();
        }
        Ok((det, pts, zs))
    }

    /// (detail, points, design elevations) of bend action p from (r, c), in travel order.
    fn bend_geometry(&self, r: i64, c: i64, p: usize) -> Result<([f64; BEND_N], Vec<(f64, f64)>, Vec<f64>), String> {
        let (br, bc, h, slot) = self.canonical_bend(r, c, p);
        let det = self.bend_detail(br, bc, h, slot);
        if !(det[0] < f64::INFINITY) {
            return Err(format!("bend at row {br}, col {bc}, heading {h}, slot {slot} is infeasible"));
        }
        let g = &self.bend_geo;
        let i = h * g.ns + slot;
        let (ux, uy, _) = unit_heading(self.n_theta(), h);
        let seg = macros::bend_segments(g.l1[i] * self.res, g.r * self.res, g.dth[i], g.l2[i] * self.res);
        let (mut pts, at) = macro_points(&seg, br, bc, (ux, uy), self.res);
        let mut acc = [0.0; 3];
        for k in 1..3 {
            acc[k] = acc[k - 1] + if seg[k - 1][4] > 1e-9 { seg[k - 1][4] } else { 0.0 };
        }
        let mut zs: Vec<f64> = at.iter().map(|&(k, s)| det[9] + det[11] * (acc[k] + s)).collect();
        if self.lat.kind[p] as u8 != ACTION_BEND_L {
            pts.reverse();
            zs.reverse();
        }
        Ok((det, pts, zs))
    }

    /// (row, col) points of one edge, from its start node to its end node.
    pub fn edge_points(&self, r: i64, c: i64, p: usize) -> Result<Vec<(f64, f64)>, String> {
        let l = &self.lat;
        let (rf, cf) = (r as f64, c as f64);
        Ok(match l.kind[p] as u8 {
            ACTION_FILL => {
                let n = ((2.0 * l.length[p]).ceil() as usize).max(2);
                linspace(1.0, n + 1).iter().map(|&f| (rf + f * l.dr[p] as f64, cf + f * l.dc[p] as f64)).collect()
            }
            ACTION_PRIM => {
                let (s0, n) = (l.s0[p] as usize, l.ns[p] as usize);
                let mut v = vec![(rf, cf)];
                v.extend((s0..s0 + n).map(|k| (rf + l.sdr[k], cf + l.sdc[k])));
                v.push(((r + l.dr[p]) as f64, (c + l.dc[p]) as f64));
                v
            }
            ACTION_BEND_L | ACTION_BEND_R => self.bend_geometry(r, c, p)?.1,
            // Out along the landing and back to the junction.
            ACTION_SWITCH => {
                let k = (l.s0[p] + l.ns[p] - 1) as usize;
                let f = l.length[p] / (l.sdr[k].hypot(l.sdc[k]));
                vec![(rf, cf), (rf + f * l.sdr[k], cf + f * l.sdc[k]), (rf, cf)]
            }
            _ => self.hairpin_geometry(r, c, p)?.1,
        })
    }

    /// The route's centerline (row, col), start to goal.
    pub fn path_points(&self, edges: &[Edge], start: (i64, i64)) -> Result<Vec<(f64, f64)>, String> {
        let mut out = vec![(start.0 as f64, start.1 as f64)];
        for &(r, c, p) in edges {
            out.extend(self.edge_points(r, c, p)?.into_iter().skip(1));
        }
        Ok(out)
    }

    /// Rise + fall of the smoothed-DEM profile as the search prices it: over the samples of a
    /// primitive, end to end for macro-actions (whose design is monotone).
    fn vertical_travel(&self, r: i64, c: i64, p: usize) -> f64 {
        let l = &self.lat;
        let (nr, nc) = (self.nr, self.nc);
        let (r1, c1) = (r + l.dr[p], c + l.dc[p]);
        let zs = |r: i64, c: i64| self.zs[r as usize * nc + c as usize];
        if l.kind[p] as u8 != ACTION_PRIM {
            return (zs(r1, c1) - zs(r, c)).abs();
        }
        let (s0, n) = (l.s0[p] as usize, l.ns[p] as usize);
        let mut prof = vec![zs(r, c)];
        prof.extend((s0..s0 + n).map(|k| bil_clamped(&self.zs, nr, nc, r as f64 + l.sdr[k], c as f64 + l.sdc[k])));
        prof.push(zs(r1, c1));
        prof.windows(2).map(|w| (w[1] - w[0]).abs()).sum()
    }

    /// Fill volume, largest fill height and contributing area of a fill crossing.
    fn fill_crossing_detail(&self, r: i64, c: i64, p: usize) -> (f64, f64, f64) {
        let l = &self.lat;
        let (nr, nc) = (self.nr, self.nc);
        let lc = l.length[p];
        let (r1, c1) = (r + l.dr[p], c + l.dc[p]);
        let n = ((2.0 * lc).ceil() as usize).max(2);
        let wseg = lc * self.res / n as f64;
        let zr = |r: i64, c: i64| self.z_raw[r as usize * nc + c as usize];
        let (z0, z1) = (zr(r, c), zr(r1, c1));
        let (mut vol, mut hmax, mut amax) = (0.0, 0.0f64, 0.0f64);
        for k in 0..n {
            let f = (k as f64 + 0.5) / n as f64;
            let (fr, fc) = (r as f64 + f * l.dr[p] as f64, c as f64 + f * l.dc[p] as f64);
            let (cr, cc) = ((fr + 0.5).floor() as usize, (fc + 0.5).floor() as usize);
            if self.channel[cr * nc + cc] != 0 {
                amax = amax.max(self.area[cr * nc + cc]);
            }
            let h = z0 + f * (z1 - z0) - bil_nan(&self.z_raw, nr, nc, fr, fc);
            if h > 0.0 {
                vol += (self.params.w * h + h * h / self.params.f_rock) * wseg;
                hmax = hmax.max(h);
            }
        }
        (vol, hmax, amax)
    }

    /// Exact construction quantities and costs of one edge, plus its operating cost.
    pub fn evaluate_edge(&self, seq: usize, r: i64, c: i64, a: usize, surrogate: f64, culvert: (f64, f64)) -> Result<Segment, String> {
        let p = &self.params;
        let l = &self.lat;
        let kind = l.kind[a] as u8;
        let (nr, nc) = (self.nr, self.nc);
        let nan = f64::NAN;
        let mut seg = match kind {
            ACTION_HP_LEFT | ACTION_HP_RIGHT => {
                let (det, pts, z) = self.hairpin_geometry(r, c, a)?;
                let sign = if kind == ACTION_HP_LEFT { 1.0 } else { -1.0 };
                let wall = if det[23] > 0.0 { p.c_wall * det[23] } else { 0.0 };
                let earth = p.c_soil_cost * det[4] + p.c_rock_cost * det[5] + p.c_borrow * det[7] + wall;
                Segment {
                    seq, r, c, action: a, kind: HAIRPIN, pts, z, length: det[3], grade: sign * (det[13] - det[12]) / det[3],
                    sidehill_slope: nan, terrain_slope: nan, soil_depth: nan, cut_width: p.w_hp, fill_width: 0.0,
                    cut_catch: det[16], fill_toe: det[17], cut_soil_m3: det[4], cut_rock_m3: det[5], fill_m3: det[6],
                    spoil_m3: det[8], borrow_m3: det[7], cost_base: p.c_base * det[3], cost_earthwork: earth,
                    cost_surrogate: surrogate, cost_culvert: det[18], culvert_area_ha: det[19], vertical_travel: 0.0,
                    turning: 0.0, cost_operating: 0.0,
                }
            }
            ACTION_BEND_L | ACTION_BEND_R => {
                let (det, pts, z) = self.bend_geometry(r, c, a)?;
                let sign = if kind == ACTION_BEND_L { 1.0 } else { -1.0 };
                let earth = p.c_soil_cost * det[2] + p.c_rock_cost * det[3] + p.c_borrow * det[5];
                Segment {
                    seq, r, c, action: a, kind: BEND, pts, z, length: det[1], grade: sign * det[11], sidehill_slope: nan,
                    terrain_slope: nan, soil_depth: nan, cut_width: p.w, fill_width: 0.0, cut_catch: det[12],
                    fill_toe: det[13], cut_soil_m3: det[2], cut_rock_m3: det[3], fill_m3: det[4], spoil_m3: det[6],
                    borrow_m3: det[5], cost_base: p.c_base * det[1], cost_earthwork: earth, cost_surrogate: surrogate,
                    cost_culvert: det[14], culvert_area_ha: det[15], vertical_travel: 0.0, turning: 0.0, cost_operating: 0.0,
                }
            }
            ACTION_SWITCH => {
                // The level landing: road sections across the sidehill, plus the prism that keeps
                // it level (at the fill price, as the search prices it).
                let pts = self.edge_points(r, c, a)?;
                let zs = |r: i64, c: i64| self.zs[r as usize * nc + c as usize];
                let z0 = zs(r, c);
                let z = vec![z0; pts.len()];
                let (s0, n) = (l.s0[a] as usize, l.ns[a] as usize);
                let length = l.length[a] * self.res;
                let w = length / n as f64;
                let sp = SectionParams::new(p);
                let (mut a_soil, mut a_rock, mut a_fill, mut spoil, mut extra, mut prism, mut ss, mut dd) =
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                for k in s0..s0 + n {
                    let (fr, fc) = (r as f64 + l.sdr[k], c as f64 + l.sdc[k]);
                    let ax = bil_clamped(&self.gx, nr, nc, fr, fc);
                    let ay = bil_clamped(&self.gy, nr, nc, fr, fc);
                    let g = ax * l.tx[k] + ay * l.ty[k];
                    let s = (-ax * l.ty[k] + ay * l.tx[k]).abs();
                    let st = s.hypot(g);
                    let rho = if self.k_rough > 0.0 { bil_clamped(&self.rough, nr, nc, fr, fc) } else { 0.0 };
                    let d = if self.k_rough > 0.0 && rho > 0.0 { soil_depth(st + self.k_rough * rho, p) } else { soil_depth(st, p) };
                    let sec = section(s, d, &sp);
                    let h = z0 - bil_clamped(&self.zs, nr, nc, fr, fc);
                    prism += p.w * h.abs() + h * h / p.f_rock;
                    a_soil += sec.a_soil;
                    a_rock += sec.a_rock;
                    a_fill += sec.a_fill;
                    spoil += sec.spoil;
                    extra += sec.cost - p.c_base;
                    ss += s;
                    dd += d;
                }
                let nf = n as f64;
                Segment {
                    seq, r, c, action: a, kind: SWITCHBACK, pts, z, length, grade: 0.0, sidehill_slope: ss / nf,
                    terrain_slope: nan, soil_depth: dd / nf, cut_width: p.w, fill_width: 0.0, cut_catch: 0.0,
                    fill_toe: 0.0, cut_soil_m3: a_soil * w, cut_rock_m3: a_rock * w, fill_m3: a_fill * w + prism * w,
                    spoil_m3: spoil * w, borrow_m3: 0.0, cost_base: p.c_base * length,
                    cost_earthwork: extra * w + self.lam_fill * prism * w, cost_surrogate: surrogate, cost_culvert: 0.0,
                    culvert_area_ha: 0.0, vertical_travel: 0.0, turning: 0.0, cost_operating: 0.0,
                }
            }
            ACTION_FILL => {
                let pts = self.edge_points(r, c, a)?;
                let (vol, hmax, amax) = self.fill_crossing_detail(r, c, a);
                let length = l.length[a] * self.res;
                let zr = |r: i64, c: i64| self.z_raw[r as usize * nc + c as usize];
                let (z0, z1) = (zr(r, c), zr(r + l.dr[a], c + l.dc[a]));
                let z = linspace(1.0, pts.len()).iter().map(|&f| z0 + (z1 - z0) * f).collect();
                Segment {
                    seq, r, c, action: a, kind: FILL_CROSSING, pts, z, length, grade: (z1 - z0) / length,
                    sidehill_slope: nan, terrain_slope: nan, soil_depth: nan, cut_width: p.w, fill_width: 0.0,
                    cut_catch: 0.0, fill_toe: hmax / p.f_rock, cut_soil_m3: 0.0, cut_rock_m3: 0.0, fill_m3: vol,
                    spoil_m3: 0.0, borrow_m3: 0.0, cost_base: p.c_base * length, cost_earthwork: self.lam_fill * vol,
                    cost_surrogate: surrogate, cost_culvert: p.culvert_cost(amax), culvert_area_ha: amax,
                    vertical_travel: 0.0, turning: 0.0, cost_operating: 0.0,
                }
            }
            _ => {
                let pts = self.edge_points(r, c, a)?;
                let z = pts.iter().map(|&(fr, fc)| bil_clamped(&self.z_raw, nr, nc, fr, fc)).collect();
                let (s0, n) = (l.s0[a] as usize, l.ns[a] as usize);
                let sp = SectionParams::new(p);
                let (mut ss, mut s_terr, mut dd) = (0.0, 0.0, 0.0);
                let (mut aw, mut bw, mut catch, mut toe): (f64, f64, f64, f64) = (0.0, 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY);
                let (mut a_soil, mut a_rock, mut a_fill, mut spoil, mut extra) = (0.0, 0.0, 0.0, 0.0, 0.0);
                let mut counts = [0usize; 4]; // infeasible, flat, half bench, full bench
                for k in s0..s0 + n {
                    let (fr, fc) = (r as f64 + l.sdr[k], c as f64 + l.sdc[k]);
                    let ax = bil_clamped(&self.gx, nr, nc, fr, fc);
                    let ay = bil_clamped(&self.gy, nr, nc, fr, fc);
                    let g = ax * l.tx[k] + ay * l.ty[k];
                    let s = (-ax * l.ty[k] + ay * l.tx[k]).abs();
                    let st = s.hypot(g);
                    let rho = if self.k_rough > 0.0 { bil_clamped(&self.rough, nr, nc, fr, fc) } else { 0.0 };
                    let d = if self.k_rough > 0.0 && rho > 0.0 { soil_depth(st + self.k_rough * rho, p) } else { soil_depth(st, p) };
                    let sec = section(s, d, &sp);
                    ss += s;
                    s_terr += st;
                    dd += d;
                    aw += sec.a;
                    bw += sec.b;
                    catch = catch.max(sec.cut_catch);
                    toe = toe.max(sec.fill_toe);
                    a_soil += sec.a_soil;
                    a_rock += sec.a_rock;
                    a_fill += sec.a_fill;
                    spoil += sec.spoil;
                    extra += sec.cost - p.c_base;
                    counts[match sec.kind {
                        INFEASIBLE => 0,
                        FLAT => 1,
                        HALF_BENCH => 2,
                        FULL_BENCH => 3,
                        _ => 0,
                    }] += 1;
                }
                // The most frequent section type; ties to the more severe (infeasible above all).
                let order = [(counts[0], 9, SEC_INFEASIBLE), (counts[1], 0, SEC_FLAT), (counts[2], 1, SEC_HALF_BENCH),
                             (counts[3], 2, SEC_FULL_BENCH)];
                let label = order.iter().filter(|o| o.0 > 0).max_by_key(|o| (o.0, o.1)).map_or(SEC_FLAT, |o| o.2);
                let length = l.length[a] * self.res;
                let w = length / n as f64;
                let nf = n as f64;
                let zs = |r: i64, c: i64| self.zs[r as usize * nc + c as usize];
                Segment {
                    seq, r, c, action: a, kind: label, pts, z, length,
                    grade: (zs(r + l.dr[a], c + l.dc[a]) - zs(r, c)) / length, sidehill_slope: ss / nf,
                    terrain_slope: s_terr / nf, soil_depth: dd / nf, cut_width: aw / nf, fill_width: bw / nf,
                    cut_catch: catch, fill_toe: toe, cut_soil_m3: a_soil * w, cut_rock_m3: a_rock * w, fill_m3: a_fill * w,
                    spoil_m3: spoil * w, borrow_m3: 0.0, cost_base: p.c_base * length, cost_earthwork: extra * w,
                    cost_surrogate: surrogate, cost_culvert: culvert.0, culvert_area_ha: culvert.1, vertical_travel: 0.0,
                    turning: 0.0, cost_operating: 0.0,
                }
            }
        };
        seg.vertical_travel = self.vertical_travel(r, c, a);
        seg.turning = l.turn[a];
        // A switchback's landing is turned in, not traveled along the way.
        let traveled = if kind == ACTION_SWITCH { 0.0 } else { seg.length };
        seg.cost_operating = p.c_travel * traveled + p.c_vertical * seg.vertical_travel + p.c_turn * seg.turning;
        Ok(seg)
    }

    /// Every edge of a route, evaluated (route.py's evaluate_route).
    pub fn evaluate_route(&self, edges: &[Edge]) -> Result<Vec<Segment>, String> {
        let costs: Vec<(f64, (f64, f64))> = self.with_model(|model| {
            edges
                .iter()
                .map(|&(r, c, a)| {
                    let culvert =
                        if self.lat.kind[a] as u8 == ACTION_PRIM { model.culvert_charges(r, c, a) } else { (0.0, 0.0) };
                    (model.edge_cost(r, c, a), culvert)
                })
                .collect()
        });
        edges
            .iter()
            .zip(costs)
            .enumerate()
            .map(|(i, (&(r, c, a), (cost, culvert)))| self.evaluate_edge(i, r, c, a, cost, culvert))
            .collect()
    }

    /// Cells under the edges' points (each point rounded to the nearest cell), as a mask.
    pub fn edges_mask(&self, edges: &[Edge]) -> Result<Vec<bool>, String> {
        let mut m = vec![false; self.nr * self.nc];
        for &(r, c, p) in edges {
            for (fr, fc) in self.edge_points(r, c, p)? {
                // A point within half a cell of the grid edge rounds onto the cell past it.
                let rr = ((fr + 0.5).floor() as i64).clamp(0, self.nr as i64 - 1) as usize;
                let cc = ((fc + 0.5).floor() as i64).clamp(0, self.nc as i64 - 1) as usize;
                m[rr * self.nc + cc] = true;
            }
        }
        Ok(m)
    }
}

fn seg_point_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}

fn orient(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn segments_distance(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> f64 {
    let (o1, o2, o3, o4) = (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b));
    if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
        return 0.0;
    }
    seg_point_distance(a, c, d).min(seg_point_distance(b, c, d)).min(seg_point_distance(c, a, b)).min(seg_point_distance(d, a, b))
}

/// Shortest distance between two polylines (x, y).
fn polyline_distance(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..a.len().saturating_sub(1) {
        for j in 0..b.len().saturating_sub(1) {
            best = best.min(segments_distance(a[i], a[i + 1], b[j], b[j + 1]));
            if best == 0.0 {
                return 0.0;
            }
        }
    }
    best
}

fn bbox(pl: &[(f64, f64)]) -> [f64; 4] {
    pl.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, &(x, y)| {
        [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)]
    })
}

/// Pairs (i, j, separation m) of non-adjacent route pieces whose road surfaces overlap:
/// centerlines closer than the road width, with the road between them at least min_ratio times
/// longer than that (the route comes back on itself, unlike an ordinary curve). polylines are
/// (row, col) per piece, in travel order (route.py's self_overlaps).
pub fn self_overlaps(polylines: &[Vec<(f64, f64)>], res: f64, width: f64, min_ratio: f64) -> Vec<(usize, usize, f64)> {
    let lines: Vec<Vec<(f64, f64)>> = polylines.iter().map(|pl| pl.iter().map(|&(r, c)| (c * res, r * res)).collect()).collect();
    let mut station = vec![0.0];
    for ln in &lines {
        let len: f64 = ln.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum();
        station.push(station.last().unwrap() + len);
    }
    let boxes: Vec<[f64; 4]> = lines.iter().map(|l| bbox(l)).collect();
    let mut out = Vec::new();
    for i in 0..lines.len() {
        let bi = boxes[i];
        for j in i + 2..lines.len() {
            let bj = boxes[j];
            if bj[0] > bi[2] + width || bj[2] < bi[0] - width || bj[1] > bi[3] + width || bj[3] < bi[1] - width {
                continue;
            }
            let sep = polyline_distance(&lines[i], &lines[j]);
            let gap = station[j] - station[i + 1];
            if sep < width && gap >= min_ratio * sep && gap > width {
                out.push((i, j, sep));
            }
        }
    }
    out
}
