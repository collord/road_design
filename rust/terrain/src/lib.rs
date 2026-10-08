//! DEM preprocessing for road search (a port of `roadsearch/preprocess.py` and `flow.py`).
//!
//! Rasters are row-major `nr x nc` slices of f64 with NaN for nodata; row 0 is north.
//!
//! * [`smooth`]: Gaussian smoothing that ignores NaN cells (normalized convolution), following
//!   scipy's `gaussian_filter` (truncate 4, mode "nearest", rows then columns, the same kernel
//!   and accumulation order).
//! * [`gradient`]: central differences, one-sided at the edges (numpy's `gradient`).
//! * [`slope_mask`]: ground too steep for any road at grade g_max, dilated by a distance (exact
//!   Euclidean distance transform, Felzenszwalb and Huttenlocher, with unequal row and column
//!   spacing).
//! * [`contributing_area`]: D8 flow accumulation after Priority-Flood depression filling with an
//!   epsilon rise (Barnes, Lehman and Mulla 2014), so every cell has a strictly downhill path to the
//!   grid edge or nodata. pysheds (the Python path) resolves flats differently, so channels can
//!   differ on flats and in filled depressions; elsewhere they agree.
//! * [`reachable`]: the 4-connected region of open cells containing a cell.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// numpy's pairwise summation for up to 128 values (what `phi_x.sum()` does for a kernel).
fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n > 128 {
        let h = n / 2;
        let h = h - h % 8;
        return np_sum(&a[..h]) + np_sum(&a[h..]);
    }
    let mut r = [0.0f64; 8];
    r.copy_from_slice(&a[..8]);
    let mut i = 8;
    while i < n - (n % 8) {
        for j in 0..8 {
            r[j] += a[i + j];
        }
        i += 8;
    }
    let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    while i < n {
        res += a[i];
        i += 1;
    }
    res
}

/// scipy's Gaussian kernel: radius int(4 sigma + 0.5), weights exp(-x^2 / 2 sigma^2), normalized.
pub fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (4.0 * sigma + 0.5) as i64;
    let sigma2 = sigma * sigma;
    let w: Vec<f64> = (-radius..=radius).map(|x| (-0.5 / sigma2 * (x * x) as f64).exp()).collect();
    let s = np_sum(&w);
    w.into_iter().map(|v| v / s).collect()
}

/// One pass of a symmetric correlation along rows (axis 0) or columns (axis 1), mode "nearest",
/// accumulated as scipy's NI_Correlate1D does for symmetric kernels.
fn correlate_axis(src: &[f64], nr: usize, nc: usize, w: &[f64], axis: usize) -> Vec<f64> {
    let r = (w.len() / 2) as i64;
    let mut out = vec![0.0; src.len()];
    let (len, lines) = if axis == 0 { (nr, nc) } else { (nc, nr) };
    let at = |line: usize, i: i64| -> f64 {
        let i = i.clamp(0, len as i64 - 1) as usize;
        if axis == 0 {
            src[i * nc + line]
        } else {
            src[line * nc + i]
        }
    };
    for line in 0..lines {
        for i in 0..len as i64 {
            let mut acc = at(line, i) * w[r as usize];
            for jj in -r..0 {
                acc += (at(line, i + jj) + at(line, i - jj)) * w[(r + jj) as usize];
            }
            let k = if axis == 0 { i as usize * nc + line } else { line * nc + i as usize };
            out[k] = acc;
        }
    }
    out
}

fn gaussian_filter(a: &[f64], nr: usize, nc: usize, w: &[f64]) -> Vec<f64> {
    let rows = correlate_axis(a, nr, nc, w, 0);
    correlate_axis(&rows, nr, nc, w, 1)
}

/// Gaussian smoothing ignoring NaN cells: smooth(z * valid) / smooth(valid), NaN where z is NaN.
pub fn smooth(z: &[f64], nr: usize, nc: usize, sigma_cells: f64) -> Vec<f64> {
    assert_eq!(z.len(), nr * nc);
    let w = gaussian_kernel(sigma_cells);
    let zf: Vec<f64> = z.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
    let valid: Vec<f64> = z.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }).collect();
    let num = gaussian_filter(&zf, nr, nc, &w);
    let den = gaussian_filter(&valid, nr, nc, &w);
    (0..z.len()).map(|k| if z[k].is_nan() { f64::NAN } else { num[k] / den[k] }).collect()
}

/// dz/dx (east) and dz/dy (north): numpy's gradient over rows and columns divided by the cell
/// size (dx = transform.a, dy = transform.e, negative for a north-up grid).
pub fn gradient(z: &[f64], nr: usize, nc: usize, dx: f64, dy: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(nr >= 2 && nc >= 2, "gradient needs at least 2 rows and 2 columns");
    let mut gx = vec![0.0; z.len()];
    let mut gy = vec![0.0; z.len()];
    for r in 0..nr {
        for c in 0..nc {
            let k = r * nc + c;
            let d_col = if c == 0 {
                z[k + 1] - z[k]
            } else if c == nc - 1 {
                z[k] - z[k - 1]
            } else {
                (z[k + 1] - z[k - 1]) / 2.0
            };
            let d_row = if r == 0 {
                z[k + nc] - z[k]
            } else if r == nr - 1 {
                z[k] - z[k - nc]
            } else {
                (z[k + nc] - z[k - nc]) / 2.0
            };
            gx[k] = d_col / dx;
            gy[k] = d_row / dy;
        }
    }
    (gx, gy)
}

/// 1-D squared distance transform (Felzenszwalb and Huttenlocher) of f at positions q * s.
fn dt1d(f: &[f64], s: f64, out: &mut [f64], v: &mut [usize], zb: &mut [f64]) {
    let n = f.len();
    let pos = |q: usize| q as f64 * s;
    let mut k = 0usize;
    let mut first = None;
    for q in 0..n {
        if f[q].is_finite() {
            first = Some(q);
            break;
        }
    }
    let Some(q0) = first else {
        out.fill(f64::INFINITY);
        return;
    };
    v[0] = q0;
    zb[0] = f64::NEG_INFINITY;
    zb[1] = f64::INFINITY;
    for q in q0 + 1..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            let p = v[k];
            let sx = ((f[q] + pos(q) * pos(q)) - (f[p] + pos(p) * pos(p))) / (2.0 * (pos(q) - pos(p)));
            if sx <= zb[k] && k > 0 {
                k -= 1;
                continue;
            }
            if sx <= zb[k] {
                // k == 0: replace the only parabola
                v[0] = q;
                zb[0] = f64::NEG_INFINITY;
                zb[1] = f64::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            zb[k] = sx;
            zb[k + 1] = f64::INFINITY;
            break;
        }
    }
    k = 0;
    for q in 0..n {
        while zb[k + 1] < pos(q) {
            k += 1;
        }
        let d = pos(q) - pos(v[k]);
        out[q] = d * d + f[v[k]];
    }
}

/// Euclidean distance from every cell to the nearest `target` cell, with row spacing sy and
/// column spacing sx (scipy's distance_transform_edt(~target, sampling=(sy, sx))).
pub fn distance_to(target: &[bool], nr: usize, nc: usize, sy: f64, sx: f64) -> Vec<f64> {
    let mut d2 = vec![f64::INFINITY; nr * nc];
    let n = nr.max(nc);
    let (mut f, mut out, mut v, mut zb) = (vec![0.0; n], vec![0.0; n], vec![0usize; n], vec![0.0; n + 1]);
    for c in 0..nc {
        for r in 0..nr {
            f[r] = if target[r * nc + c] { 0.0 } else { f64::INFINITY };
        }
        dt1d(&f[..nr], sy, &mut out[..nr], &mut v, &mut zb);
        for r in 0..nr {
            d2[r * nc + c] = out[r];
        }
    }
    for r in 0..nr {
        f[..nc].copy_from_slice(&d2[r * nc..(r + 1) * nc]);
        dt1d(&f[..nc], sx, &mut out[..nc], &mut v, &mut zb);
        d2[r * nc..(r + 1) * nc].copy_from_slice(&out[..nc]);
    }
    d2.into_iter().map(f64::sqrt).collect()
}

/// Cells too steep for any road at grade <= g_max (smallest sidehill slope over the headings,
/// sqrt(S^2 - g_max^2), at least s_mask; or NaN slope), and that mask dilated by `radius` meters.
/// Returns (raw, dilated).
#[allow(clippy::too_many_arguments)]
pub fn slope_mask(slope: &[f64], nr: usize, nc: usize, g_max: f64, s_mask: f64, radius: f64, sy: f64, sx: f64) -> (Vec<bool>, Vec<bool>) {
    let raw: Vec<bool> = slope
        .iter()
        .map(|&s| s.is_nan() || ((s * s - g_max * g_max).max(0.0)).sqrt() >= s_mask)
        .collect();
    if !raw.iter().any(|&b| b) {
        return (raw.clone(), raw);
    }
    let dist = distance_to(&raw, nr, nc, sy, sx);
    let dilated = dist.iter().map(|&d| d <= radius).collect();
    (raw, dilated)
}

#[derive(Clone, Copy, PartialEq)]
struct Cell(f64, usize);

impl Eq for Cell {}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        // A min-heap on (elevation, index).
        other.0.total_cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const NB: [(i64, i64); 8] = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];

/// Depressions filled by Priority-Flood with an epsilon rise: every valid cell gets a strictly
/// downhill path to a seed (a cell on the grid edge or next to nodata). NaN stays NaN.
pub fn fill_depressions(z: &[f64], nr: usize, nc: usize) -> Vec<f64> {
    let mut filled = z.to_vec();
    let mut done = vec![false; z.len()];
    let mut heap = BinaryHeap::new();
    let valid = |r: i64, c: i64| r >= 0 && c >= 0 && r < nr as i64 && c < nc as i64 && !z[r as usize * nc + c as usize].is_nan();
    for r in 0..nr as i64 {
        for c in 0..nc as i64 {
            if !valid(r, c) {
                continue;
            }
            if NB.iter().any(|&(dr, dc)| !valid(r + dr, c + dc)) {
                let k = r as usize * nc + c as usize;
                done[k] = true;
                heap.push(Cell(z[k], k));
            }
        }
    }
    while let Some(Cell(h, k)) = heap.pop() {
        let (r, c) = ((k / nc) as i64, (k % nc) as i64);
        for &(dr, dc) in &NB {
            let (r2, c2) = (r + dr, c + dc);
            if !valid(r2, c2) {
                continue;
            }
            let k2 = r2 as usize * nc + c2 as usize;
            if done[k2] {
                continue;
            }
            done[k2] = true;
            let rise = h.next_up();
            filled[k2] = if z[k2] > rise { z[k2] } else { rise };
            heap.push(Cell(filled[k2], k2));
        }
    }
    filled
}

/// D8 receivers on a filled DEM: the neighbor of steepest descent (drop over distance), or
/// usize::MAX where there is none (outlets and nodata). Ties go to the first in NB order.
pub fn d8_receivers(filled: &[f64], nr: usize, nc: usize, sy: f64, sx: f64) -> Vec<usize> {
    let diag = sx.hypot(sy);
    let mut rec = vec![usize::MAX; filled.len()];
    for r in 0..nr as i64 {
        for c in 0..nc as i64 {
            let k = r as usize * nc + c as usize;
            if filled[k].is_nan() {
                continue;
            }
            let mut best = 0.0;
            for &(dr, dc) in &NB {
                let (r2, c2) = (r + dr, c + dc);
                if r2 < 0 || c2 < 0 || r2 >= nr as i64 || c2 >= nc as i64 {
                    continue;
                }
                let k2 = r2 as usize * nc + c2 as usize;
                if filled[k2].is_nan() {
                    continue;
                }
                let dist = if dr != 0 && dc != 0 { diag } else if dr != 0 { sy } else { sx };
                let s = (filled[k] - filled[k2]) / dist;
                if s > best {
                    best = s;
                    rec[k] = k2;
                }
            }
        }
    }
    rec
}

/// Upslope contributing area including the cell itself, in units of `cell_area` (pass the cell
/// area in hectares for hectares). NaN where z is NaN.
pub fn contributing_area(z: &[f64], nr: usize, nc: usize, sy: f64, sx: f64, cell_area: f64) -> Vec<f64> {
    let filled = fill_depressions(z, nr, nc);
    let rec = d8_receivers(&filled, nr, nc, sy, sx);
    // Upstream before downstream: by filled elevation, highest first (receivers are strictly lower).
    let mut order: Vec<usize> = (0..z.len()).filter(|&k| !z[k].is_nan()).collect();
    order.sort_by(|&a, &b| filled[b].total_cmp(&filled[a]).then(a.cmp(&b)));
    let mut acc = vec![0.0f64; z.len()];
    for &k in &order {
        acc[k] += 1.0;
        if rec[k] != usize::MAX {
            let a = acc[k];
            acc[rec[k]] += a;
        }
    }
    (0..z.len()).map(|k| if z[k].is_nan() { f64::NAN } else { acc[k] * cell_area }).collect()
}

/// The 4-connected region of open cells containing (r, c); all false if that cell is not open.
pub fn reachable(open: &[bool], nr: usize, nc: usize, r: usize, c: usize) -> Vec<bool> {
    let mut out = vec![false; open.len()];
    let k0 = r * nc + c;
    if !open[k0] {
        return out;
    }
    let mut q = VecDeque::from([k0]);
    out[k0] = true;
    while let Some(k) = q.pop_front() {
        let (r, c) = (k / nc, k % nc);
        let mut visit = |k2: usize| {
            if open[k2] && !out[k2] {
                out[k2] = true;
                q.push_back(k2);
            }
        };
        if r > 0 {
            visit(k - nc);
        }
        if r + 1 < nr {
            visit(k + nc);
        }
        if c > 0 {
            visit(k - 1);
        }
        if c + 1 < nc {
            visit(k + 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_transform_matches_brute_force() {
        let (nr, nc) = (17, 23);
        let target: Vec<bool> = (0..nr * nc).map(|k| (k * 7919) % 31 == 0).collect();
        let d = distance_to(&target, nr, nc, 2.0, 3.0);
        for r in 0..nr {
            for c in 0..nc {
                let mut best = f64::INFINITY;
                for r2 in 0..nr {
                    for c2 in 0..nc {
                        if target[r2 * nc + c2] {
                            let (dy, dx) = ((r as f64 - r2 as f64) * 2.0, (c as f64 - c2 as f64) * 3.0);
                            best = best.min((dy * dy + dx * dx).sqrt());
                        }
                    }
                }
                assert!((d[r * nc + c] - best).abs() < 1e-9, "({r},{c}): {} vs {best}", d[r * nc + c]);
            }
        }
    }

    #[test]
    fn a_pit_drains_and_area_adds_up() {
        // A bowl: everything drains through the rim's lowest point, so the cell there collects it all.
        let (nr, nc) = (9, 9);
        let mut z: Vec<f64> = (0..nr * nc)
            .map(|k| {
                let (r, c) = ((k / nc) as f64 - 4.0, (k % nc) as f64 - 4.0);
                10.0 + r * r + c * c
            })
            .collect();
        z[4 * nc] = 5.0; // a notch on the west edge, the outlet
        let a = contributing_area(&z, nr, nc, 1.0, 1.0, 1.0);
        let total: f64 = (0..nr * nc)
            .filter(|&k| {
                let filled = fill_depressions(&z, nr, nc);
                d8_receivers(&filled, nr, nc, 1.0, 1.0)[k] == usize::MAX
            })
            .map(|k| a[k])
            .sum();
        assert_eq!(total, (nr * nc) as f64); // every cell reaches exactly one outlet
        assert!(a[4 * nc] > 20.0); // the bowl's interior drains through the notch
    }

    #[test]
    fn reachable_is_four_connected() {
        let open = [true, true, false, false, true, false, true, false, true];
        let r = reachable(&open, 3, 3, 0, 0);
        assert_eq!(r, vec![true, true, false, false, true, false, false, false, false]);
    }
}
