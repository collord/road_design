//! State-lattice motion primitives (a port of roadsearch/lattice.py).
//!
//! Headings are the lattice directions of small integer vectors, so straight motions land exactly
//! on nodes. A turning primitive joins two lattice poses with a straight segment and a circular
//! arc (or arc then straight): the arc is the largest circle tangent to both heading lines, so the
//! curve is determined by its end poses alone, and the reverse of a primitive is the same curve.
//!
//! Coordinates are in cells with x east and y north; a displacement (X, Y) is (drow, dcol) = (-Y, X).
//! Every floating-point operation follows the Python in order, so the arrays agree bit for bit.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use lattice_search::{Action, Lattice};
use road_cost::{
    Actions, ACTION_BEND_L, ACTION_BEND_R, ACTION_FILL, ACTION_HP_LEFT, ACTION_HP_RIGHT, ACTION_PRIM, ACTION_SWITCH,
};

const HEADINGS_8: [(i64, i64); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
const HEADINGS_16: [(i64, i64); 16] = [
    (1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
    (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1),
];
/// 32 bins: per octant 0, atan(1/4), atan(1/2), atan(3/4), rotated by quarter turns.
const Q32: [(i64, i64); 8] = [(1, 0), (4, 1), (2, 1), (4, 3), (1, 1), (3, 4), (1, 2), (1, 4)];

/// Cells between samples at most; the brief requires no more than half a cell.
const SAMPLE_SPACING: f64 = 0.5;

/// Bend turns in degrees, rounded to whole heading bins.
pub const BEND_ANGLES: [f64; 2] = [90.0, 135.0];

fn check_n_theta(n_theta: usize) -> Result<(), String> {
    match n_theta {
        8 | 16 | 32 => Ok(()),
        _ => Err(format!("N_theta must be one of [8, 16, 32], got {n_theta}")),
    }
}

/// Lattice vectors (east, north) of the headings; panics unless n_theta is 8, 16 or 32.
pub fn heading_vectors(n_theta: usize) -> Vec<(i64, i64)> {
    match n_theta {
        8 => HEADINGS_8.to_vec(),
        16 => HEADINGS_16.to_vec(),
        32 => (0..4)
            .flat_map(|k| {
                Q32.iter().map(move |&(x, y)| match k {
                    0 => (x, y),
                    1 => (-y, x),
                    2 => (-x, -y),
                    _ => (y, -x),
                })
            })
            .collect(),
        _ => panic!("N_theta must be one of [8, 16, 32], got {n_theta}"),
    }
}

pub fn heading_angles(n_theta: usize) -> Vec<f64> {
    heading_vectors(n_theta).iter().map(|&(x, y)| (y as f64).atan2(x as f64)).collect()
}

/// `(x + pi) % (2 pi) - pi` with Python's floored modulo.
fn wrap_pi(x: f64) -> f64 {
    (x + PI).rem_euclid(2.0 * PI) - PI
}

/// math.hypot of integers: the sum of squares is exact, so its square root is correctly rounded.
fn hypot_int(x: i64, y: i64) -> f64 {
    ((x * x + y * y) as f64).sqrt()
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn unit(a: f64) -> (f64, f64) {
    (a.cos(), a.sin())
}

/// Python's `round(x, 9)` as an exact count of 1e-9: the decimal rounding of the binary value,
/// ties to even. For |x| well under 2^53 * 1e-9 distinct counts are distinct rounded floats, so
/// tuples of counts order exactly as Python's tuples of rounded floats.
fn round9(x: f64) -> i128 {
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1u64 << 52) - 1)) as i128;
    let (m, e) = if exp == 0 { (frac, -1074) } else { (frac | (1i128 << 52), exp - 1075) };
    assert!(exp != 0x7ff && e < 40, "round9 needs a moderate finite value, got {x}");
    // x * 1e9 = q * 2^e exactly, with q below 2^83.
    let q = m * 1_000_000_000;
    let n = if e >= 0 {
        q << e
    } else if -e > 100 {
        0
    } else {
        let sh = -e;
        let fl = q >> sh;
        let rem = q & ((1i128 << sh) - 1);
        let half = 1i128 << (sh - 1);
        if rem > half || (rem == half && fl & 1 == 1) { fl + 1 } else { fl }
    };
    if x < 0.0 { -n } else { n }
}

#[derive(Clone, Debug)]
struct Primitive {
    h0: usize,
    h1: usize,
    /// End displacement, cells east and north.
    x: i64,
    y: i64,
    /// Arc length in cells.
    length: f64,
    /// Arc radius in cells (inf for straight).
    radius: f64,
    /// Sample positions (x, y) in cells, midpoints of n equal pieces, and unit tangents there.
    pts: Samples,
    tan: Samples,
}

type Key = (usize, i64, i64, usize);
/// Points or unit tangents (east, north).
type Samples = Vec<(f64, f64)>;
/// A candidate's (length, |a - b|, forward, lateral), each rounded to 1e-9 as Python's round(v, 9).
type RoundedKey = [i128; 4];

impl Primitive {
    fn key(&self) -> Key {
        (self.h0, self.x, self.y, self.h1)
    }
}

/// Tangent-arc construction between poses (0, 0, th0) and (X, Y, th1): the heading lines meet at
/// P = a u0 and E = P + b u1, and R = min(a, b) / tan(|dth| / 2). None if invalid.
struct TurnGeometry {
    a: f64,
    b: f64,
    r: f64,
    dth: f64,
}

fn turn_geometry(th0: f64, th1: f64, x: f64, y: f64) -> Option<TurnGeometry> {
    let dth = wrap_pi(th1 - th0);
    if dth.abs() < 1e-12 || (dth.abs() - PI).abs() < 1e-12 {
        return None;
    }
    let (u0, u1) = (unit(th0), unit(th1));
    let det = u0.0 * u1.1 - u0.1 * u1.0;
    let a = (x * u1.1 - y * u1.0) / det;
    let b = (u0.0 * y - u0.1 * x) / det;
    if a <= 1e-12 || b <= 1e-12 {
        return None;
    }
    // Python's min returns the first of equals; the values are equal then, so either will do.
    let r = if b < a { b } else { a } / (dth.abs() / 2.0).tan();
    Some(TurnGeometry { a, b, r, dth })
}

fn n_samples(length: f64) -> usize {
    ((length / SAMPLE_SPACING - 1e-9).ceil() as i64).max(1) as usize
}

/// Points and tangents at the midpoints of n equal arc-length pieces, and the length.
fn sample_turn(th0: f64, g: &TurnGeometry, n: usize) -> (Samples, Samples, f64) {
    let (a, b, r, dth) = (g.a, g.b, g.r, g.dth);
    let t = r * (dth.abs() / 2.0).tan();
    let (l1, la, l2) = (a - t, r * dth.abs(), b - t);
    let l = l1 + la + l2;
    let u0 = unit(th0);
    let sgn = if dth > 0.0 { 1.0 } else { -1.0 };
    let left = (-u0.1 * sgn, u0.0 * sgn);
    let p1 = (l1 * u0.0, l1 * u0.1);
    let center = (p1.0 + r * left.0, p1.1 + r * left.1);
    let rel = (p1.0 - center.0, p1.1 - center.1);
    let mut pts = Vec::with_capacity(n);
    let mut tans = Vec::with_capacity(n);
    for k in 0..n {
        let s = (k as f64 + 0.5) * l / n as f64;
        if s <= l1 {
            pts.push((s * u0.0, s * u0.1));
            tans.push(u0);
        } else if s <= l1 + la {
            // Rotate (p1 - center) by phi about the center.
            let phi = sgn * (s - l1) / r;
            let (c, s_) = (phi.cos(), phi.sin());
            pts.push((center.0 + (c * rel.0 - s_ * rel.1), center.1 + (s_ * rel.0 + c * rel.1)));
            tans.push(unit(th0 + phi));
        } else {
            let u1 = unit(th0 + dth);
            let (c, s_) = (dth.cos(), dth.sin());
            let e2 = (center.0 + (c * rel.0 - s_ * rel.1), center.1 + (s_ * rel.0 + c * rel.1));
            let d = s - l1 - la;
            pts.push((e2.0 + d * u1.0, e2.1 + d * u1.1));
            tans.push(u1);
        }
    }
    (pts, tans, l)
}

/// Straight move along v (direction th) from heading h0 to heading h1.
fn straight(h0: usize, v: (i64, i64), th: f64, h1: usize) -> Primitive {
    let l = hypot_int(v.0, v.1);
    let n = n_samples(l);
    let u = unit(th);
    let pts = (0..n)
        .map(|k| {
            let s = (k as f64 + 0.5) * l / n as f64;
            (s * u.0, s * u.1)
        })
        .collect();
    Primitive { h0, h1, x: v.0, y: v.1, length: l, radius: f64::INFINITY, pts, tan: vec![u; n] }
}

fn turn(h0: usize, h1: usize, th0: f64, th1: f64, x: i64, y: i64) -> Option<Primitive> {
    let g = turn_geometry(th0, th1, x as f64, y as f64)?;
    let t = g.r * (g.dth.abs() / 2.0).tan();
    let l = (g.a - t) + g.r * g.dth.abs() + (g.b - t);
    let (pts, tan, length) = sample_turn(th0, &g, n_samples(l));
    Some(Primitive { h0, h1, x, y, length, radius: g.r, pts, tan })
}

/// Straight moves (h0, X, Y, h1) to lattice points with coprime coordinates and
/// max(|X|, |Y|) <= max_cells, binned to lattice headings: a move in direction a may start in h0
/// and end in h1 when both are within kink_max_deg of a and one of them is the heading nearest a.
/// Moves along an exact lattice heading are kept only when they change heading. Closed under reversal.
fn binned_moves(n_theta: usize, max_cells: i64, kink_max_deg: f64) -> Vec<Key> {
    if max_cells <= 0 {
        return Vec::new();
    }
    let ths = heading_angles(n_theta);
    let kmax = kink_max_deg * (PI / 180.0);
    let mut out = Vec::new();
    for x in -max_cells..=max_cells {
        for y in -max_cells..=max_cells {
            if (x, y) == (0, 0) || gcd(x, y) != 1 {
                continue;
            }
            let a = (y as f64).atan2(x as f64);
            let d: Vec<f64> = ths.iter().map(|&th| wrap_pi(a - th)).collect();
            // np.argmin: the first minimum.
            let mut near = 0;
            for h in 1..n_theta {
                if d[h].abs() < d[near].abs() {
                    near = h;
                }
            }
            let ok: Vec<usize> = (0..n_theta).filter(|&h| d[h].abs() <= kmax + 1e-12).collect();
            let exact = d[near].abs() < 1e-12;
            for &h0 in &ok {
                for &h1 in &ok {
                    if h0 != near && h1 != near {
                        continue;
                    }
                    if exact && h0 == h1 {
                        continue;
                    }
                    out.push((h0, x, y, h1));
                }
            }
        }
    }
    out
}

/// Straight plus the shortest valid left and right turn for each heading change in `turns`, plus
/// binned straight moves, closed under reversal, in key order (h0, X, Y, h1).
fn build_primitives(
    n_theta: usize,
    r_min_cells: f64,
    turns: &[usize],
    move_max_cells: i64,
    kink_max_deg: f64,
) -> Result<Vec<Primitive>, String> {
    check_n_theta(n_theta)?;
    let vecs = heading_vectors(n_theta);
    let ths = heading_angles(n_theta);
    let m = (3.0 * r_min_cells).ceil() as i64 + 4;
    let mut prims: BTreeMap<Key, Primitive> = BTreeMap::new();
    for h0 in 0..n_theta {
        let p = straight(h0, vecs[h0], ths[h0], h0);
        prims.insert(p.key(), p);
        for &k in turns {
            for sgn in [1i64, -1] {
                let h1 = (h0 as i64 + sgn * k as i64).rem_euclid(n_theta as i64) as usize;
                let u0 = unit(ths[h0]);
                let mut best: Option<((i64, i64), RoundedKey)> = None;
                for x in -m..=m {
                    for y in -m..=m {
                        let (xf, yf) = (x as f64, y as f64);
                        let Some(g) = turn_geometry(ths[h0], ths[h1], xf, yf) else { continue };
                        if g.r < r_min_cells - 1e-9 {
                            continue;
                        }
                        let t = g.r * (g.dth.abs() / 2.0).tan();
                        let l = (g.a - t) + g.r * g.dth.abs() + (g.b - t);
                        let fwd = xf * u0.0 + yf * u0.1;
                        let lat = (xf * u0.1 - yf * u0.0).abs();
                        let key = [round9(l), round9((g.a - g.b).abs()), round9(fwd), round9(lat)];
                        if best.is_none_or(|(_, bk)| key < bk) {
                            best = Some(((x, y), key));
                        }
                    }
                }
                let Some(((x, y), _)) = best else {
                    return Err(format!("no turn primitive for heading {h0} -> {h1} within {m} cells"));
                };
                let p = turn(h0, h1, ths[h0], ths[h1], x, y).expect("the chosen turn is valid");
                prims.insert(p.key(), p);
            }
        }
    }
    for (h0, x, y, h1) in binned_moves(n_theta, move_max_cells, kink_max_deg) {
        let p = straight(h0, (x, y), (y as f64).atan2(x as f64), h1);
        prims.insert(p.key(), p);
    }
    // Close under reversal: (h0, X, Y, h1) reversed is (h1 + pi, -X, -Y, h0 + pi).
    let half = n_theta / 2;
    let existing: Vec<Primitive> = prims.values().cloned().collect();
    for p in existing {
        let rk = ((p.h1 + half) % n_theta, -p.x, -p.y, (p.h0 + half) % n_theta);
        if prims.contains_key(&rk) {
            continue;
        }
        let q = if p.radius.is_infinite() {
            straight(rk.0, (rk.1, rk.2), (rk.2 as f64).atan2(rk.1 as f64), rk.3)
        } else {
            turn(rk.0, rk.3, ths[rk.0], ths[rk.3], rk.1, rk.2)
                .ok_or_else(|| format!("the reverse of primitive {:?} is not a valid turn", p.key()))?
        };
        prims.insert(rk, q);
    }
    Ok(prims.into_values().collect())
}

/// Bend turns in heading bins (8 and 12 for 32 headings, 4 and 6 for 16).
pub fn bend_steps(n_theta: usize) -> Vec<i64> {
    BEND_ANGLES.iter().map(|a| (a / 360.0 * n_theta as f64).round_ties_even() as i64).collect()
}

/// Left bends: straight l1, arc of radius r through dth, straight l2 (cells). Arrays are
/// row-major (heading, slot); slot j turns bend_steps(n_theta)[j] bins left.
#[derive(Clone, Debug, PartialEq)]
pub struct BendGeometry {
    pub ns: usize,
    /// End displacement east and north (cells).
    pub x: Vec<i64>,
    pub y: Vec<i64>,
    /// End heading.
    pub h1: Vec<i64>,
    pub l1: Vec<f64>,
    pub l2: Vec<f64>,
    /// Signed turn (radians, > 0 left).
    pub dth: Vec<f64>,
    /// Arc radius (cells).
    pub r: f64,
}

/// Shortest left bend for each heading and turn, ending on a lattice node with the exact end
/// heading: straight, arc of radius r_cells, straight.
pub fn build_bends(n_theta: usize, r_cells: f64) -> Result<BendGeometry, String> {
    check_n_theta(n_theta)?;
    build_bends_with(n_theta, r_cells, &bend_steps(n_theta))
}

/// build_bends with the turns given in heading bins.
pub fn build_bends_with(n_theta: usize, r_cells: f64, steps: &[i64]) -> Result<BendGeometry, String> {
    check_n_theta(n_theta)?;
    if steps.iter().any(|&k| !(0 < k && k < (n_theta / 2) as i64)) {
        return Err(format!(
            "bend turns must be between 0 and 180 degrees, got {steps:?} of {n_theta} bins"
        ));
    }
    let ths = heading_angles(n_theta);
    let ns = steps.len();
    let size = n_theta * ns;
    let mut bg = BendGeometry {
        ns,
        x: vec![0; size],
        y: vec![0; size],
        h1: vec![0; size],
        l1: vec![0.0; size],
        l2: vec![0.0; size],
        dth: vec![0.0; size],
        r: r_cells,
    };
    for h0 in 0..n_theta {
        for (j, &k) in steps.iter().enumerate() {
            let h1 = (h0 + k as usize) % n_theta;
            let dth = wrap_pi(ths[h1] - ths[h0]);
            let t = r_cells * (dth.abs() / 2.0).tan();
            let m = ((3.0 * (r_cells + t)).ceil() as i64 + 4).min(200);
            let u0 = unit(ths[h0]);
            let mut best: Option<((i64, i64, f64, f64), RoundedKey)> = None;
            for x in -m..=m {
                for y in -m..=m {
                    let (xf, yf) = (x as f64, y as f64);
                    let Some(g) = turn_geometry(ths[h0], ths[h1], xf, yf) else { continue };
                    if g.a < t - 1e-9 || g.b < t - 1e-9 || g.dth * dth <= 0.0 {
                        continue;
                    }
                    let length = (g.a - t) + r_cells * dth.abs() + (g.b - t);
                    let fwd = xf * u0.0 + yf * u0.1;
                    let lat = (xf * u0.1 - yf * u0.0).abs();
                    let key = [round9(length), round9((g.a - g.b).abs()), round9(fwd), round9(lat)];
                    if best.is_none_or(|(_, bk)| key < bk) {
                        best = Some(((x, y, g.a - t, g.b - t), key));
                    }
                }
            }
            let Some(((x, y, l1, l2), _)) = best else {
                return Err(format!("no bend for heading {h0} turning {k} bins within {m} cells"));
            };
            let i = h0 * ns + j;
            bg.x[i] = x;
            bg.y[i] = y;
            bg.l1[i] = if l1 < 1e-9 { 0.0 } else { l1 };
            bg.l2[i] = if l2 < 1e-9 { 0.0 } else { l2 };
            bg.h1[i] = h1 as i64;
            bg.dth[i] = dth;
        }
    }
    Ok(bg)
}

/// Flat action arrays, as roadsearch.lattice.LatticeArrays (without the primitives themselves).
///
/// Actions are the primitives (in key order), then hairpins (a left/right pair per heading and
/// leg-spacing slot), then fill crossings (straight runs of m >= 2 lattice steps), then left bends
/// per heading and slot followed by right bends, each the reverse of a left bend.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeArrays {
    pub n_theta: usize,
    pub n_prims: usize,
    pub h0: Vec<i64>,
    pub h1: Vec<i64>,
    pub dr: Vec<i64>,
    pub dc: Vec<i64>,
    /// Cells.
    pub length: Vec<f64>,
    pub rev: Vec<i64>,
    /// First sample index and sample count (0 for macro actions).
    pub s0: Vec<i64>,
    pub ns: Vec<i64>,
    /// Sample row and column offsets (cells) and unit tangent (east, north).
    pub sdr: Vec<f64>,
    pub sdc: Vec<f64>,
    pub tx: Vec<f64>,
    pub ty: Vec<f64>,
    /// Row-major (n_theta, max_out) action indices leaving each heading, padded with -1.
    pub by_heading: Vec<i64>,
    pub max_out: usize,
    /// ACTION_PRIM, ACTION_HP_*, ACTION_FILL or ACTION_BEND_*.
    pub kind: Vec<i64>,
    /// Hairpin leg-spacing slot or bend slot (0 for other actions).
    pub slot: Vec<i64>,
    /// Change in heading, radians (pi for a hairpin), the same for an action and its reverse.
    pub turn: Vec<f64>,
}

/// Primitive arrays, plus left/right hairpin actions for each leg-spacing slot if hp_slots
/// (row-major (n_theta, n_slots) lateral lattice steps, 0 = unused) is given, plus fill-crossing
/// actions at most cross_max_cells long, plus left and right bends if bends is given.
pub fn lattice_arrays(
    n_theta: usize,
    r_min_cells: f64,
    hp_slots: Option<&[i64]>,
    cross_max_cells: f64,
    bends: Option<&BendGeometry>,
    move_max_cells: i64,
    kink_max_deg: f64,
) -> Result<LatticeArrays, String> {
    let prims = build_primitives(n_theta, r_min_cells, &[1, 2], move_max_cells, kink_max_deg)?;
    let idx: HashMap<Key, usize> = prims.iter().enumerate().map(|(i, p)| (p.key(), i)).collect();
    let half = n_theta / 2;
    let n_prims = prims.len();
    let mut out: Vec<Vec<i64>> = vec![Vec::new(); n_theta];
    let (mut h0, mut h1, mut dr, mut dc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut length, mut rev, mut kind, mut slot) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut s0, mut ns) = (Vec::new(), Vec::new());
    let n_samples: usize = prims.iter().map(|p| p.pts.len()).sum();
    let (mut sdr, mut sdc) = (Vec::with_capacity(n_samples), Vec::with_capacity(n_samples));
    let (mut tx, mut ty) = (Vec::with_capacity(n_samples), Vec::with_capacity(n_samples));
    for (i, p) in prims.iter().enumerate() {
        let rk = ((p.h1 + half) % n_theta, -p.x, -p.y, (p.h0 + half) % n_theta);
        h0.push(p.h0 as i64);
        h1.push(p.h1 as i64);
        dr.push(-p.y);
        dc.push(p.x);
        length.push(p.length);
        rev.push(idx[&rk] as i64);
        kind.push(ACTION_PRIM as i64);
        slot.push(0);
        s0.push(sdr.len() as i64);
        ns.push(p.pts.len() as i64);
        for (&(px, py), &(ux, uy)) in p.pts.iter().zip(&p.tan) {
            sdr.push(-py);
            sdc.push(px);
            tx.push(ux);
            ty.push(uy);
        }
        out[p.h0].push(i as i64);
    }
    let vecs = heading_vectors(n_theta);
    if let Some(slots) = hp_slots {
        if slots.len() % n_theta != 0 {
            return Err(format!("hp_slots must be (n_theta, n_slots), got {} values", slots.len()));
        }
        let n_slots = slots.len() / n_theta;
        for (h, &(x, y)) in vecs.iter().enumerate() {
            for j in 0..n_slots {
                let k = slots[h * n_slots + j];
                if k == 0 {
                    continue;
                }
                // Left normal of (x, y) is (-y, x); as (drow, dcol) that is (-x, -y).
                for (side, sgn) in [(ACTION_HP_LEFT, 1i64), (ACTION_HP_RIGHT, -1)] {
                    let i = h0.len() as i64;
                    h0.push(h as i64);
                    h1.push(((h + half) % n_theta) as i64);
                    dr.push(-sgn * k * x);
                    dc.push(-sgn * k * y);
                    length.push(PI * 0.5 * k as f64 * hypot_int(x, y)); // nominal arc, cells
                    rev.push(if side == ACTION_HP_LEFT { i + 1 } else { i - 1 });
                    kind.push(side as i64);
                    slot.push(j as i64);
                    s0.push(0);
                    ns.push(0);
                    out[h].push(i);
                }
            }
        }
    }
    let mut fill_index: HashMap<(usize, i64), usize> = HashMap::new();
    for (h, &(x, y)) in vecs.iter().enumerate() {
        let vlen = hypot_int(x, y);
        let m_max = (cross_max_cells / vlen + 1e-9).floor() as i64;
        for m in 2..=m_max {
            let i = h0.len();
            fill_index.insert((h, m), i);
            h0.push(h as i64);
            h1.push(h as i64);
            dr.push(-m * y);
            dc.push(m * x);
            length.push(m as f64 * vlen);
            rev.push(-1);
            kind.push(ACTION_FILL as i64);
            slot.push(0);
            s0.push(0);
            ns.push(0);
            out[h].push(i as i64);
        }
    }
    for (&(h, m), &i) in &fill_index {
        rev[i] = fill_index[&((h + half) % n_theta, m)] as i64;
    }
    if let Some(b) = bends {
        if b.x.len() != n_theta * b.ns {
            return Err(format!("bends must be ({n_theta}, {}), got {} values", b.ns, b.x.len()));
        }
        let mut left = Vec::with_capacity(n_theta * b.ns);
        for (h, leaving) in out.iter_mut().enumerate() {
            for j in 0..b.ns {
                let (i, k) = (h0.len(), h * b.ns + j);
                left.push((h, j, i));
                h0.push(h as i64);
                h1.push(b.h1[k]);
                dr.push(-b.y[k]);
                dc.push(b.x[k]);
                length.push(b.l1[k] + b.r * b.dth[k].abs() + b.l2[k]);
                rev.push(-1);
                kind.push(ACTION_BEND_L as i64);
                slot.push(j as i64);
                s0.push(0);
                ns.push(0);
                leaving.push(i as i64);
            }
        }
        for (h, j, il) in left {
            let (i, k) = (h0.len(), h * b.ns + j);
            let hr = (b.h1[k] as usize + half) % n_theta;
            h0.push(hr as i64);
            h1.push(((h + half) % n_theta) as i64);
            dr.push(b.y[k]);
            dc.push(-b.x[k]);
            length.push(length[il]);
            rev.push(il as i64);
            rev[il] = i as i64;
            kind.push(ACTION_BEND_R as i64);
            slot.push(j as i64);
            s0.push(0);
            ns.push(0);
            out[hr].push(i as i64);
        }
    }
    let max_out = out.iter().map(Vec::len).max().unwrap_or(0);
    let mut by_heading = vec![-1i64; n_theta * max_out];
    for (h, lst) in out.iter().enumerate() {
        by_heading[h * max_out..h * max_out + lst.len()].copy_from_slice(lst);
    }
    let ang = heading_angles(n_theta);
    let mut turn: Vec<f64> = (0..h0.len())
        .map(|i| {
            if kind[i] == ACTION_HP_LEFT as i64 || kind[i] == ACTION_HP_RIGHT as i64 {
                PI
            } else {
                wrap_pi(ang[h1[i] as usize] - ang[h0[i] as usize]).abs()
            }
        })
        .collect();
    // Bit-identical both ways (the search relies on it); np.minimum of the original values.
    let raw = turn.clone();
    for (i, t) in turn.iter_mut().enumerate() {
        *t = t.min(raw[rev[i] as usize]);
    }
    Ok(LatticeArrays {
        n_theta,
        n_prims,
        h0,
        h1,
        dr,
        dc,
        length,
        rev,
        s0,
        ns,
        sdr,
        sdc,
        tx,
        ty,
        by_heading,
        max_out,
        kind,
        slot,
        turn,
    })
}

impl LatticeArrays {
    /// Add reversing switchbacks: from each heading h to every heading u whose reverse is within
    /// 90 degrees of h (but not h itself: the legs must part), at the same cell. Each carries the
    /// samples of its level landing, `landing_cells` long, along the bisector of h and the reverse
    /// of u (the way past the junction from both legs, so an action and its reverse share it).
    pub fn add_switchbacks(&mut self, landing_cells: f64) {
        let n = self.n_theta;
        let half = n / 2;
        let ang = heading_angles(n);
        let n_samples = ((2.0 * landing_cells).ceil() as usize).max(2);
        let mut out: Vec<Vec<i64>> = (0..n)
            .map(|h| self.by_heading[h * self.max_out..(h + 1) * self.max_out].iter().copied().take_while(|&a| a >= 0).collect())
            .collect();
        let mut index = HashMap::new();
        for h in 0..n {
            for u in 0..n {
                let back = (u + half) % n;
                let part = wrap_pi(ang[back] - ang[h]);
                if back == h || part.abs() > 0.5 * PI + 1e-9 {
                    continue;
                }
                let i = self.h0.len();
                index.insert((h, u), i);
                let bis = ang[h] + 0.5 * part;
                let (bx, by) = (bis.cos(), bis.sin());
                self.h0.push(h as i64);
                self.h1.push(u as i64);
                self.dr.push(0);
                self.dc.push(0);
                self.length.push(landing_cells);
                self.rev.push(-1);
                self.kind.push(ACTION_SWITCH as i64);
                self.slot.push(0);
                self.s0.push(self.sdr.len() as i64);
                self.ns.push(n_samples as i64);
                for k in 0..n_samples {
                    let t = (k as f64 + 0.5) / n_samples as f64 * landing_cells;
                    self.sdr.push(-t * by);
                    self.sdc.push(t * bx);
                    self.tx.push(bx);
                    self.ty.push(by);
                }
                self.turn.push(wrap_pi(ang[u] - ang[h]).abs());
                out[h].push(i as i64);
            }
        }
        // The reverse of h -> u is reverse(u) -> reverse(h).
        for (&(h, u), &i) in &index {
            self.rev[i] = index[&((u + half) % n, (h + half) % n)] as i64;
        }
        self.max_out = out.iter().map(Vec::len).max().unwrap_or(0);
        self.by_heading = vec![-1i64; n * self.max_out];
        for (h, lst) in out.iter().enumerate() {
            self.by_heading[h * self.max_out..h * self.max_out + lst.len()].copy_from_slice(lst);
        }
    }

    /// The search lattice and the cost model's action table, keeping the by_heading expansion order.
    pub fn to_search(&self) -> (Lattice, Actions) {
        let usz = |v: &[i64]| v.iter().map(|&x| x as usize).collect::<Vec<_>>();
        let by_heading = (0..self.n_theta)
            .map(|h| {
                self.by_heading[h * self.max_out..(h + 1) * self.max_out]
                    .iter()
                    .take_while(|&&a| a >= 0)
                    .map(|&a| a as u32)
                    .collect()
            })
            .collect();
        let actions: Vec<Action> = (0..self.h0.len())
            .map(|i| Action {
                h0: self.h0[i] as u16,
                h1: self.h1[i] as u16,
                dr: self.dr[i] as i32,
                dc: self.dc[i] as i32,
            })
            .collect();
        let acts = Actions {
            h0: usz(&self.h0),
            dr: self.dr.clone(),
            dc: self.dc.clone(),
            length: self.length.clone(),
            rev: usz(&self.rev),
            s0: usz(&self.s0),
            ns: usz(&self.ns),
            kind: self.kind.iter().map(|&k| k as u8).collect(),
            slot: usz(&self.slot),
            turn: self.turn.clone(),
            sdr: self.sdr.clone(),
            sdc: self.sdc.clone(),
            tx: self.tx.clone(),
            ty: self.ty.clone(),
        };
        (Lattice::with_order(self.n_theta, actions, by_heading), acts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(n_theta: usize, r: f64, move_max: i64) -> LatticeArrays {
        let slots: Vec<i64> = (0..n_theta).flat_map(|h| [2 + (h % 2) as i64, if h % 3 == 0 { 0 } else { 4 }]).collect();
        let bends = build_bends(n_theta, r).unwrap();
        lattice_arrays(n_theta, r, Some(&slots), 6.0, Some(&bends), move_max, 10.0).unwrap()
    }

    #[test]
    fn round9_matches_python() {
        // Python: round(0.0009765625, 9) == 0.000976562, round(0.0029296875, 9) == 0.002929688.
        assert_eq!(round9(0.0009765625), 976_562);
        assert_eq!(round9(0.0029296875), 2_929_688);
        assert_eq!(round9(-0.0029296875), -2_929_688);
        assert_eq!(round9(1.0), 1_000_000_000);
        assert_eq!(round9(-0.0), 0);
        assert_eq!(round9(1e-300), 0);
        assert_eq!(round9(0.1 + 0.2), 300_000_000);
    }

    #[test]
    fn heading_vectors_are_ordered_by_angle() {
        for n in [8, 16, 32] {
            let ang = heading_angles(n);
            assert_eq!(ang.len(), n);
            assert_eq!(ang[0], 0.0);
            let unwrapped: Vec<f64> = ang.iter().map(|&a| if a < 0.0 { a + 2.0 * PI } else { a }).collect();
            assert!(unwrapped.windows(2).all(|w| w[0] < w[1]));
        }
        assert_eq!(heading_vectors(32)[8], (0, 1));
        assert_eq!(bend_steps(32), vec![8, 12]);
        assert_eq!(bend_steps(16), vec![4, 6]);
    }

    #[test]
    fn counts_match_python() {
        // From roadsearch.lattice: (n_theta, R_min cells, move_max_cells) -> (primitives, samples).
        for (n, r, mm, np_, nsamp) in [
            (8, 2.0, 0, 40, 212),
            (16, 2.0, 0, 80, 444),
            (16, 1.0, 0, 80, 428),
            (32, 20.0 / 15.0, 0, 160, 1204),
            (16, 2.0, 7, 200, 2132),
            (32, 2.0, 7, 464, 5420),
        ] {
            let l = lattice_arrays(n, r, None, 0.0, None, mm, 10.0).unwrap();
            assert_eq!((l.n_prims, l.sdr.len()), (np_, nsamp), "N_theta {n}, R {r}, moves {mm}");
        }
    }

    #[test]
    fn reverse_is_an_involution_with_equal_turn() {
        for (n, mm) in [(16, 0), (16, 7), (32, 7)] {
            let l = full(n, 2.0, mm);
            let half = (n / 2) as i64;
            for i in 0..l.h0.len() {
                let j = l.rev[i] as usize;
                assert_eq!(l.rev[j] as usize, i);
                assert_eq!(l.turn[i].to_bits(), l.turn[j].to_bits());
                assert_eq!(l.h0[j], (l.h1[i] + half) % n as i64);
                assert_eq!(l.h1[j], (l.h0[i] + half) % n as i64);
                if l.kind[i] != ACTION_HP_LEFT as i64 && l.kind[i] != ACTION_HP_RIGHT as i64 {
                    assert_eq!((l.dr[j], l.dc[j]), (-l.dr[i], -l.dc[i]));
                    // A reversed turn is rebuilt in its own frame, so its length may differ in the last bit.
                    assert!((l.length[j] - l.length[i]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn by_heading_lists_every_action_once_under_its_heading() {
        let l = full(32, 2.0, 7);
        let mut seen = vec![0; l.h0.len()];
        for h in 0..l.n_theta {
            let row = &l.by_heading[h * l.max_out..(h + 1) * l.max_out];
            let n = row.iter().take_while(|&&a| a >= 0).count();
            assert!(row[n..].iter().all(|&a| a == -1));
            for &a in &row[..n] {
                assert_eq!(l.h0[a as usize], h as i64);
                seen[a as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        let (lat, acts) = l.to_search();
        assert_eq!(lat.actions().len(), acts.h0.len());
        let row3 = &l.by_heading[3 * l.max_out..4 * l.max_out];
        assert_eq!(lat.leaving(3).len(), row3.iter().filter(|&&a| a >= 0).count());
    }

    #[test]
    fn samples_cover_each_primitive() {
        let l = full(16, 2.0, 7);
        for i in 0..l.n_prims {
            let (s0, ns) = (l.s0[i] as usize, l.ns[i] as usize);
            let step = l.length[i] / ns as f64;
            assert!(step <= SAMPLE_SPACING + 1e-9);
            // The first and last samples are half a piece from the end nodes (chord, not arc: 1e-3).
            let first = l.sdr[s0].hypot(l.sdc[s0]);
            let last = (l.dr[i] as f64 - l.sdr[s0 + ns - 1]).hypot(l.dc[i] as f64 - l.sdc[s0 + ns - 1]);
            assert!((first - step / 2.0).abs() < 1e-3, "primitive {i}: first sample {first} from the start");
            assert!((last - step / 2.0).abs() < 1e-3, "primitive {i}: last sample {last} from the end");
            for k in s0..s0 + ns {
                assert!((l.tx[k].hypot(l.ty[k]) - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn bends_reach_lattice_nodes_with_the_end_heading() {
        let b = build_bends(32, 2.0).unwrap();
        assert_eq!((b.x[0], b.y[0], b.x[1], b.y[1]), (2, 2, 1, 4));
        for (h, &th) in heading_angles(32).iter().enumerate() {
            for j in 0..b.ns {
                let k = h * b.ns + j;
                assert_eq!(b.h1[k] as usize, (h + bend_steps(32)[j] as usize) % 32);
                assert!(b.l1[k] >= 0.0 && b.l2[k] >= 0.0 && b.dth[k] > 0.0);
                // Straight l1, arc, straight l2 ends at (X, Y).
                let (u0, u1) = (unit(th), unit(th + b.dth[k]));
                let (ca, sa) = (b.r * b.dth[k].sin(), b.r * (1.0 - b.dth[k].cos()));
                let ex = b.l1[k] * u0.0 + ca * u0.0 - sa * u0.1 + b.l2[k] * u1.0;
                let ey = b.l1[k] * u0.1 + ca * u0.1 + sa * u0.0 + b.l2[k] * u1.1;
                assert!((ex - b.x[k] as f64).abs() < 1e-9 && (ey - b.y[k] as f64).abs() < 1e-9);
            }
        }
        assert!(build_bends(12, 2.0).is_err());
    }
}
