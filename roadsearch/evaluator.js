/* Costing a free-drawn road centerline, in the browser.
 *
 * A port of the parts of roadsearch that price an ordinary (primitive) edge:
 * section.py's section_core / cut_geometry (half-bench sidecast section) and
 * the sample integration in route.py's _construction and search.py's
 * _eval_action. Same rules, same sample spacing:
 *
 *   - samples every half cell or less, at the midpoints of equal pieces;
 *   - slope from the smoothed DEM's central-difference gradient, bilinear at
 *     each sample; sidehill s and grade g from the gradient and the tangent;
 *   - soil depth d0 * exp(-hypot(s, g) / S0), none on ground steeper than c_soil;
 *   - cost per meter from the exact section; grade checked on the profile;
 *   - one culvert per run of channel cells, sized by the largest drainage area;
 *   - operating cost C_travel per meter plus C_vertical per meter of rise + fall.
 *
 * Unlike the search, nothing is rejected: steep grades, excluded ground, side
 * slopes past the mask and corners tighter than R_min are flagged and reported.
 * Hairpins and bends are costed as ordinary road (the drawn line follows the
 * ground), and crossing fill is not modeled.
 *
 * Works as a plain <script> (defines window.RoadEval) and under Node (module.exports).
 */
(function (root) {
  "use strict";

  const INFEASIBLE = -1, FLAT = 0, HALF_BENCH = 1, FULL_BENCH = 2;
  const KIND_NAMES = { "-1": "infeasible", 0: "flat", 1: "half_bench", 2: "full_bench" };
  const BISECT_ITERS = 60, S_ZERO = 1e-12;

  // ---------------------------------------------------------------- section

  function shoelace(xs, ys) {
    let acc = 0;
    for (let i = 0; i < xs.length; i++) {
      const j = (i + 1) % xs.length;
      acc += xs[i] * ys[j] - xs[j] * ys[i];
    }
    return 0.5 * Math.abs(acc);
  }

  function clippedArea(xs, ys, xmax) {
    const cx = [], cy = [], n = xs.length;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, xi = xs[i], yi = ys[i], xj = xs[j], yj = ys[j];
      const inI = xi <= xmax, inJ = xj <= xmax;
      if (inI) { cx.push(xi); cy.push(yi); }
      if (inI !== inJ) { const t = (xmax - xi) / (xj - xi); cx.push(xmax); cy.push(yi + t * (yj - yi)); }
    }
    return cx.length ? shoelace(cx, cy) : 0;
  }

  /** [A_soil, A_rock, catch] of a cut of bench width a (see section.py cut_geometry). */
  function cutGeometry(a, s, d, cSoil, cRock, catchMax) {
    if (a <= 0) return [0, 0, 0];
    if (s <= S_ZERO) return [0, 0, Math.min(a, catchMax)];
    if (a * s <= d) {
      const x2 = cSoil * a / (cSoil - s);
      if (x2 <= catchMax) return [shoelace([0, a, x2], [0, 0, s * x2]), 0, x2];
      return [clippedArea([0, a, x2], [0, 0, s * x2], catchMax), 0, catchMax];
    }
    const xr = d / s;
    const x1 = (cRock * a - d) / (cRock - s);
    const y1 = cRock * (x1 - a);
    const x2 = d > 0 ? x1 + d / (cSoil - s) : x1;
    const y2 = s * x2;
    if (x2 <= catchMax) {
      const total = shoelace([0, a, x1, x2], [0, 0, y1, y2]);
      const rock = shoelace([xr, a, x1], [0, 0, y1]);
      return [Math.max(total - rock, 0), rock, x2];
    }
    const total = clippedArea([0, a, x1, x2], [0, 0, y1, y2], catchMax);
    const rock = clippedArea([xr, a, x1], [0, 0, y1], catchMax);
    return [Math.max(total - rock, 0), rock, catchMax];
  }

  function spoilFillSlope(vs, vr, d, s, sp) {
    const tot = vs + vr;
    if (tot > 0) return (vs * sp.f_soil + vr * sp.f_rock) / tot;
    return d > 0 && s < sp.c_soil ? sp.f_soil : sp.f_rock;
  }

  function fillArea(b, s, f) {
    if (b <= 0) return 0;
    if (s >= f) return Infinity;
    return 0.5 * b * b * s * f / (f - s);
  }

  /** Cross-section on sidehill slope s with soil depth d (see section.py section_core). */
  function sectionCore(s, d, sp) {
    s = Math.abs(s);
    const w = sp.w;
    const out = (kind, a, b, As, Ar, Af, spoil, cat, toe, f, cost) =>
      ({ kind, a, b, As, Ar, Af, spoil, catch: cat, toe, f, cost });
    if (s >= sp.s_mask) return out(INFEASIBLE, 0, 0, 0, 0, 0, 0, 0, 0, 0, Infinity);
    if (d < 0 || s >= sp.c_soil) d = 0;
    if (s <= S_ZERO) {
      const k = d > 0 ? sp.k_soil : sp.k_rock, f = d > 0 ? sp.f_soil : sp.f_rock;
      const a = w / (1 + Math.sqrt(k));
      return out(FLAT, a, w - a, 0, 0, 0, 0, a, w - a, f, sp.C_base);
    }
    let full = s >= Math.max(sp.f_soil, sp.f_rock);
    let a = w;
    if (!full) {
      let lo = 0, hi = w;
      for (let it = 0; it < BISECT_ITERS; it++) {
        const mid = 0.5 * (lo + hi);
        const [As, Ar] = cutGeometry(mid, s, d, sp.c_soil, sp.c_rock, sp.catch_max);
        const vs = sp.k_soil * As, vr = sp.k_rock * Ar;
        const f = spoilFillSlope(vs, vr, d, s, sp);
        if (vs + vr - fillArea(w - mid, s, f) < 0) lo = mid; else hi = mid;
      }
      a = 0.5 * (lo + hi);
    }
    let [As, Ar, cat] = cutGeometry(a, s, d, sp.c_soil, sp.c_rock, sp.catch_max);
    let vs = sp.k_soil * As, vr = sp.k_rock * Ar;
    let f = spoilFillSlope(vs, vr, d, s, sp);
    const b = w - a;
    if (!full && (s >= f || b < sp.b_min)) full = true;
    if (full) {
      [As, Ar, cat] = cutGeometry(w, s, d, sp.c_soil, sp.c_rock, sp.catch_max);
      vs = sp.k_soil * As; vr = sp.k_rock * Ar;
      f = spoilFillSlope(vs, vr, d, s, sp);
      return out(FULL_BENCH, w, 0, As, Ar, 0, vs + vr, cat, 0, f, sp.C_base + sp.C_soil * As + sp.C_rock * Ar);
    }
    const Af = fillArea(b, s, f);
    const toe = f * b / (f - s);
    return out(s < sp.s_flat ? FLAT : HALF_BENCH, a, b, As, Ar, Af, 0, cat, toe, f,
               sp.C_base + sp.C_soil * As + sp.C_rock * Ar);
  }

  function soilDepth(S, P) {
    return S >= P.sp.c_soil ? 0 : P.d0 * Math.exp(-S / P.S0);
  }

  function culvertCost(area, P) {
    for (let i = 0; i < P.culvert_area_breaks.length; i++) if (area < P.culvert_area_breaks[i]) return P.culvert_costs[i];
    return P.culvert_costs[P.culvert_costs.length - 1];
  }

  // ---------------------------------------------------------------- terrain

  /** Terrain from a run's data: zs (smoothed DEM, Float32Array, row-major),
   *  w, h, res, x0, y0 (top-left), flags (bit 1 excluded, 2 channel), area (ha). */
  function makeTerrain(T) {
    const { w, h, res } = T, zs = T.zs, n = w * h;
    const gx = new Float64Array(n), gy = new Float64Array(n);
    // np.gradient: central differences inside, one-sided at the edges.
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const j = r * w + c;
        const dc = w === 1 ? 0 : c === 0 ? zs[j + 1] - zs[j] : c === w - 1 ? zs[j] - zs[j - 1] : 0.5 * (zs[j + 1] - zs[j - 1]);
        const dr = h === 1 ? 0 : r === 0 ? zs[j + w] - zs[j] : r === h - 1 ? zs[j] - zs[j - w] : 0.5 * (zs[j + w] - zs[j - w]);
        gx[j] = dc / res;
        gy[j] = -dr / res;
      }
    }
    return Object.assign({}, T, { gx, gy });
  }

  function bilinear(a, w, h, fr, fc) {
    if (!(fr >= 0 && fc >= 0 && fr <= h - 1 && fc <= w - 1)) return NaN;
    const r0 = h > 1 ? Math.min(Math.floor(fr), h - 2) : 0, c0 = w > 1 ? Math.min(Math.floor(fc), w - 2) : 0;
    const wr = fr - r0, wc = fc - c0, j = r0 * w + c0;
    return (1 - wr) * ((1 - wc) * a[j] + wc * a[j + 1]) + wr * ((1 - wc) * a[j + w] + wc * a[j + w + 1]);
  }

  // ---------------------------------------------------------------- geometry

  /** World (x, y) -> grid (row, col) floats, cell centers at integers. */
  const toRC = (T, x, y) => [(T.y0 - y) / T.res - 0.5, (x - T.x0) / T.res - 0.5];
  const toXY = (T, r, c) => [T.x0 + (c + 0.5) * T.res, T.y0 - (r + 0.5) * T.res];

  /** Round the corners of a polyline [[x, y], ...] with arcs of radius R.
   *  A corner whose arc does not fit between its neighbors gets the largest
   *  radius that does, and is reported. Returns { pts, corners: [{i, angle, radius, tight}] }. */
  function fillet(P, R, step) {
    step = step || 2;
    const n = P.length, corners = [];
    if (n < 3) return { pts: P.map(p => p.slice()), corners };
    const seg = [];
    for (let i = 0; i + 1 < n; i++) seg.push(Math.hypot(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]));
    // Tangent length each corner may use: all of an end segment, half of an inner one.
    const avail = i => Math.min(i === 1 ? seg[0] : 0.5 * seg[i - 1], i === n - 2 ? seg[n - 2] : 0.5 * seg[i]);
    const out = [P[0].slice()];
    for (let i = 1; i < n - 1; i++) {
      const ax = P[i][0] - P[i - 1][0], ay = P[i][1] - P[i - 1][1], bx = P[i + 1][0] - P[i][0], by = P[i + 1][1] - P[i][1];
      const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
      if (la < 1e-9 || lb < 1e-9) continue;
      const ux = ax / la, uy = ay / la, vx = bx / lb, vy = by / lb;
      const cross = ux * vy - uy * vx, dot = ux * vx + uy * vy;
      const delta = Math.atan2(Math.abs(cross), dot);          // turn angle, 0..pi
      if (delta < 1e-6) { out.push(P[i].slice()); continue; }
      let T = R * Math.tan(delta / 2), radius = R;
      const tmax = avail(i);
      const tight = T > tmax;
      if (tight) { T = tmax; radius = T / Math.tan(delta / 2); }
      corners.push({ i, angle: delta * 180 / Math.PI, radius, tight });
      const p1 = [P[i][0] - ux * T, P[i][1] - uy * T];
      const sgn = cross > 0 ? 1 : -1;                         // left turn: center on the left
      const cx = p1[0] - uy * radius * sgn, cy = p1[1] + ux * radius * sgn;
      const a0 = Math.atan2(p1[1] - cy, p1[0] - cx);
      const m = Math.max(2, Math.ceil(radius * delta / step));
      for (let k = 0; k <= m; k++) {
        const a = a0 + sgn * delta * k / m;
        out.push([cx + radius * Math.cos(a), cy + radius * Math.sin(a)]);
      }
    }
    out.push(P[n - 1].slice());
    return { pts: out, corners };
  }

  /** Sample points of a world polyline: midpoints of equal pieces no longer
   *  than half a cell. Returns [{fr, fc, tx, ty, w, station}] plus node stations. */
  function samplePolyline(T, pts) {
    const S = [];
    let station = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
      const L = Math.hypot(x1 - x0, y1 - y0);
      if (L < 1e-9) continue;
      const n = Math.max(1, Math.ceil(L / (0.5 * T.res) - 1e-9)), ws = L / n;
      const tx = (x1 - x0) / L, ty = (y1 - y0) / L;
      for (let k = 0; k < n; k++) {
        const f = (k + 0.5) / n, [fr, fc] = toRC(T, x0 + f * (x1 - x0), y0 + f * (y1 - y0));
        S.push({ fr, fc, tx, ty, w: ws, station: station + (k + 0.5) * ws });
      }
      station += L;
    }
    return { samples: S, length: station };
  }

  // ---------------------------------------------------------------- costing

  /** Cost samples [{fr, fc, tx, ty, w, station}] between end points rc0 and rc1
   *  (row, col floats) of total length `length` (m). P: run parameters. */
  function costSamples(T, P, samples, rc0, rc1, length) {
    const sp = P.sp, gtol = P.g_max * (1 + 1e-12);
    const zAt = (fr, fc) => bilinear(T.zs, T.w, T.h, fr, fc);
    const cellOf = (fr, fc) => {
      const r = Math.floor(fr + 0.5), c = Math.floor(fc + 0.5);
      return r >= 0 && r < T.h && c >= 0 && c < T.w ? r * T.w + c : -1;
    };
    const n = samples.length;
    const per = { station: new Float64Array(n), z: new Float64Array(n), grade: new Float64Array(n),
                  cpm: new Float64Array(n), kind: new Int8Array(n), flag: new Uint8Array(n), x: new Float64Array(n), y: new Float64Array(n) };
    const tot = { length, base: 0, earthwork: 0, soil_m3: 0, rock_m3: 0, fill_m3: 0, spoil_m3: 0,
                  culverts: 0, culvert_cost: 0, vertical: 0, climb: 0, descent: 0,
                  over_grade_m: 0, excluded_m: 0, infeasible_m: 0, outside_m: 0, max_grade: 0,
                  full_bench_m: 0, max_catch: 0 };
    let zPrev = zAt(rc0[0], rc0[1]), stPrev = 0;
    let inRun = false, runMax = 0;
    const closeRun = () => { if (inRun) { tot.culverts++; tot.culvert_cost += culvertCost(runMax, P); inRun = false; } };
    const visitCell = j => {
      if (j < 0) { closeRun(); return; }
      if (T.flags[j] & 2) { if (inRun) runMax = Math.max(runMax, T.area[j]); else { inRun = true; runMax = T.area[j]; } }
      else closeRun();
    };
    let lastCell = cellOf(rc0[0], rc0[1]);
    visitCell(lastCell);
    for (let k = 0; k < n; k++) {
      const q = samples[k], zk = zAt(q.fr, q.fc);
      const [x, y] = toXY(T, q.fr, q.fc);
      per.x[k] = x; per.y[k] = y; per.station[k] = q.station; per.z[k] = zk;
      const j = cellOf(q.fr, q.fc);
      if (j !== lastCell) { visitCell(j); lastCell = j; }
      const ax = bilinear(T.gx, T.w, T.h, q.fr, q.fc), ay = bilinear(T.gy, T.w, T.h, q.fr, q.fc);
      if (!(Number.isFinite(zk) && Number.isFinite(ax) && Number.isFinite(ay))) {
        per.flag[k] |= 8; per.kind[k] = INFEASIBLE; per.cpm[k] = NaN; tot.outside_m += q.w;
        zPrev = NaN; stPrev = q.station; continue;
      }
      if (Number.isFinite(zPrev)) {
        const dz = zk - zPrev, ds = q.station - stPrev;
        per.grade[k] = ds > 0 ? dz / ds : 0;
        tot.vertical += Math.abs(dz);
        if (dz > 0) tot.climb += dz; else tot.descent -= dz;
        if (Math.abs(dz) > gtol * ds) { per.flag[k] |= 1; tot.over_grade_m += q.w; }
        tot.max_grade = Math.max(tot.max_grade, Math.abs(per.grade[k]));
      }
      zPrev = zk; stPrev = q.station;
      if (j >= 0 && (T.flags[j] & 1)) { per.flag[k] |= 2; tot.excluded_m += q.w; }
      const g = ax * q.tx + ay * q.ty, s = Math.abs(-ax * q.ty + ay * q.tx);
      const sec = sectionCore(s, soilDepth(Math.hypot(s, g), P), sp);
      per.kind[k] = sec.kind;
      tot.base += sp.C_base * q.w;
      if (sec.kind === INFEASIBLE) { per.flag[k] |= 4; per.cpm[k] = NaN; tot.infeasible_m += q.w; continue; }
      per.cpm[k] = sec.cost;
      tot.earthwork += (sec.cost - sp.C_base) * q.w;
      tot.soil_m3 += sec.As * q.w; tot.rock_m3 += sec.Ar * q.w; tot.fill_m3 += sec.Af * q.w; tot.spoil_m3 += sec.spoil * q.w;
      if (sec.kind === FULL_BENCH) tot.full_bench_m += q.w;
      tot.max_catch = Math.max(tot.max_catch, sec.catch);
    }
    const z1 = zAt(rc1[0], rc1[1]);
    if (Number.isFinite(z1) && Number.isFinite(zPrev)) {
      const dz = z1 - zPrev;
      tot.vertical += Math.abs(dz);
      if (dz > 0) tot.climb += dz; else tot.descent -= dz;
      if (n && Math.abs(dz) > gtol * (length - stPrev)) { per.flag[n - 1] |= 1; }
    }
    const endCell = cellOf(rc1[0], rc1[1]);
    if (endCell !== lastCell) visitCell(endCell);
    closeRun();
    // Turning: the change in direction between consecutive samples (at corners and along arcs).
    tot.turning = 0;
    for (let k = 1; k < n; k++) {
      const d = Math.atan2(samples[k].ty, samples[k].tx) - Math.atan2(samples[k - 1].ty, samples[k - 1].tx);
      tot.turning += Math.abs(((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
    }
    tot.operating = P.C_travel * length + P.C_vertical * tot.vertical + (P.C_turn || 0) * tot.turning;
    tot.construction = tot.base + tot.earthwork + tot.culvert_cost;
    tot.total = tot.construction + tot.operating;
    return { per, tot };
  }

  /** Cost a drawn route: control points [[x, y], ...] in the run's working CRS. */
  function evaluateRoute(T, P, control) {
    const f = fillet(control, P.R_min, Math.min(2, T.res / 4));
    const { samples, length } = samplePolyline(T, f.pts);
    const a = f.pts[0], b = f.pts[f.pts.length - 1];
    const res = costSamples(T, P, samples, toRC(T, a[0], a[1]), toRC(T, b[0], b[1]), length);
    res.pts = f.pts;
    res.corners = f.corners;
    res.tot.tight_corners = f.corners.filter(c => c.tight).length;
    return res;
  }

  const api = { sectionCore, cutGeometry, soilDepth, makeTerrain, bilinear, fillet, samplePolyline,
                costSamples, evaluateRoute, toRC, toXY, KIND_NAMES };
  root.RoadEval = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
