// Drill holes laid out over an area: rows along the contours, then infill.
//
// Pass 1: rows on contours about `spacing` apart across the slope (the elevation step is the
// spacing times the slope at the area's center), holes every `spacing` along each contour from
// the point nearest the center, alternate rows offset by half; a hole is skipped if another is
// within `spacing` (rows bunch where the contours tighten). Pass 2: wherever a point of the area is
// `spacing` or more from every hole, a hole goes at the farthest such point, until none is left.
import type { Grid } from "./source";

export interface Layout {
  holes: [number, number][];
  /** Holes from the rows; the rest are infill. */
  rows: number;
  /** Elevation step between rows (m) and the slope it came from. */
  dz: number;
  slope: number;
}

type Pt = [number, number];

/** Whether p is inside the polygon (even-odd). */
export function inside(poly: Pt[], p: Pt): boolean {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/** Gaussian smoothing (sigma in cells) that ignores NaN cells. */
export function smooth(z: Float32Array, w: number, h: number, sigma: number): Float32Array {
  const rad = Math.ceil(3 * sigma), k = new Float64Array(2 * rad + 1);
  for (let i = -rad; i <= rad; i++) k[i + rad] = Math.exp(-(i * i) / (2 * sigma * sigma));
  const pass = (src: Float64Array, wt: Float64Array, horiz: boolean) => {
    const out = new Float64Array(w * h), ow = new Float64Array(w * h);
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        let s = 0, sw = 0;
        for (let i = -rad; i <= rad; i++) {
          const rr = horiz ? r : r + i, cc = horiz ? c + i : c;
          if (rr < 0 || rr >= h || cc < 0 || cc >= w) continue;
          const q = rr * w + cc;
          s += k[i + rad] * src[q];
          sw += k[i + rad] * wt[q];
        }
        out[r * w + c] = s;
        ow[r * w + c] = sw;
      }
    }
    return [out, ow];
  };
  const v = new Float64Array(w * h), wt = new Float64Array(w * h);
  for (let q = 0; q < w * h; q++) if (Number.isFinite(z[q])) { v[q] = z[q]; wt[q] = 1; }
  const [a, aw] = pass(v, wt, true);
  const [b, bw] = pass(a, aw, false);
  const out = new Float32Array(w * h);
  for (let q = 0; q < w * h; q++) out[q] = Number.isFinite(z[q]) && bw[q] > 0 ? b[q] / bw[q] : NaN;
  return out;
}

/** Contour lines of z at `level` (marching squares over the cell centers), in map coordinates. */
export function contours(z: Float32Array, g: Grid, level: number): Pt[][] {
  const { w, h } = g;
  const at = (r: number, c: number) => z[r * w + c];
  const xy = (r: number, c: number): Pt => [g.x0 + (c + 0.5) * g.res, g.y1 - (r + 0.5) * g.res];
  // Where the contour crosses each cell edge, keyed by edge.
  const cross = (r0: number, c0: number, r1: number, c1: number): Pt => {
    const a = at(r0, c0), b = at(r1, c1), t = (level - a) / (b - a);
    const [x0, y0] = xy(r0, c0), [x1, y1] = xy(r1, c1);
    return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)];
  };
  const point = new Map<string, Pt>();
  const links = new Map<string, string[]>();
  const link = (a: string, b: string) => {
    for (const [p, q] of [[a, b], [b, a]]) {
      const l = links.get(p);
      if (l) l.push(q); else links.set(p, [q]);
    }
  };
  for (let r = 0; r + 1 < h; r++) {
    for (let c = 0; c + 1 < w; c++) {
      const v = [at(r, c), at(r, c + 1), at(r + 1, c + 1), at(r + 1, c)];
      if (v.some((x) => !Number.isFinite(x))) continue;
      const up = v.map((x) => x >= level);
      // Edges: top (r,c)-(r,c+1), right (r,c+1)-(r+1,c+1), bottom (r+1,c)-(r+1,c+1), left (r,c)-(r+1,c).
      const edges: [string, () => Pt][] = [
        [`h${r},${c}`, () => cross(r, c, r, c + 1)],
        [`v${r},${c + 1}`, () => cross(r, c + 1, r + 1, c + 1)],
        [`h${r + 1},${c}`, () => cross(r + 1, c, r + 1, c + 1)],
        [`v${r},${c}`, () => cross(r, c, r + 1, c)],
      ];
      const hit: number[] = [];
      for (let e = 0; e < 4; e++) if (up[e] !== up[(e + 1) % 4]) hit.push(e);
      if (!hit.length) continue;
      for (const e of hit) if (!point.has(edges[e][0])) point.set(edges[e][0], edges[e][1]());
      if (hit.length === 2) link(edges[hit[0]][0], edges[hit[1]][0]);
      else {
        // A saddle: pair the crossings by the cell's mean.
        const mid = (v[0] + v[1] + v[2] + v[3]) / 4 >= level;
        if (mid === up[0]) { link(edges[0][0], edges[1][0]); link(edges[2][0], edges[3][0]); }
        else { link(edges[0][0], edges[3][0]); link(edges[1][0], edges[2][0]); }
      }
    }
  }
  // Chain the crossings into lines, from their ends first, then the closed loops.
  const used = new Set<string>(), out: Pt[][] = [];
  const walk = (start: string) => {
    const line = [start];
    used.add(start);
    for (let cur = start; ;) {
      const next = (links.get(cur) ?? []).find((n) => !used.has(n));
      if (!next) break;
      used.add(next);
      line.push(next);
      cur = next;
    }
    if (line.length > 1) out.push(line.map((k) => point.get(k)!));
  };
  for (const [k, l] of links) if (l.length === 1 && !used.has(k)) walk(k);
  for (const k of links.keys()) if (!used.has(k)) walk(k);
  return out;
}

/** Holes over `poly` (map coordinates) on the DEM z of grid g (already smoothed). */
export function layoutHoles(z: Float32Array, g: Grid, poly: Pt[], spacing: number): Layout {
  const bil = (x: number, y: number) => {
    const fc = Math.min(Math.max((x - g.x0) / g.res - 0.5, 0), g.w - 1.001);
    const fr = Math.min(Math.max((g.y1 - y) / g.res - 0.5, 0), g.h - 1.001);
    const r0 = Math.floor(fr), c0 = Math.floor(fc), wr = fr - r0, wc = fc - c0, q = (r: number, c: number) => z[r * g.w + c];
    return (1 - wr) * ((1 - wc) * q(r0, c0) + wc * q(r0, c0 + 1)) + wr * ((1 - wc) * q(r0 + 1, c0) + wc * q(r0 + 1, c0 + 1));
  };
  // The center: the polygon's centroid; the slope there sets the rows' elevation step.
  let a2 = 0, cx = 0, cy = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const cr = poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
    a2 += cr;
    cx += (poly[j][0] + poly[i][0]) * cr;
    cy += (poly[j][1] + poly[i][1]) * cr;
  }
  [cx, cy] = [cx / (3 * a2), cy / (3 * a2)];
  const d = g.res;
  const slope = Math.hypot((bil(cx + d, cy) - bil(cx - d, cy)) / (2 * d), (bil(cx, cy + d) - bil(cx, cy - d)) / (2 * d));
  const dz = spacing * Math.max(slope, 0.02);
  const z0 = bil(cx, cy);
  let zmin = Infinity, zmax = -Infinity;
  for (let r = 0; r < g.h; r++) {
    for (let c = 0; c < g.w; c++) {
      const p: Pt = [g.x0 + (c + 0.5) * g.res, g.y1 - (r + 0.5) * g.res], v = z[r * g.w + c];
      if (Number.isFinite(v) && inside(poly, p)) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); }
    }
  }
  const holes: Pt[] = [];
  const tooClose = (p: Pt) => holes.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < spacing - 1e-6);
  for (let k = Math.ceil((zmin - dz - z0) / dz); z0 + k * dz <= zmax + dz; k++) {
    for (const line of contours(z, g, z0 + k * dz)) {
      if (!line.some((p) => inside(poly, p))) continue;
      const st = [0];
      for (let i = 1; i < line.length; i++) st.push(st[i - 1] + Math.hypot(line[i][0] - line[i - 1][0], line[i][1] - line[i - 1][1]));
      let k0 = 0;
      line.forEach((p, i) => { if (Math.hypot(p[0] - cx, p[1] - cy) < Math.hypot(line[k0][0] - cx, line[k0][1] - cy)) k0 = i; });
      const off = (k % 2 ? 0.5 : 0) * spacing, end = st[st.length - 1];
      for (let n = Math.floor((-st[k0] - off) / spacing); st[k0] + off + n * spacing <= end; n++) {
        const t = st[k0] + off + n * spacing;
        if (t < 0) continue;
        let i = 1;
        while (i < st.length - 1 && st[i] < t) i++;
        const f = (t - st[i - 1]) / Math.max(st[i] - st[i - 1], 1e-12);
        const p: Pt = [line[i - 1][0] + f * (line[i][0] - line[i - 1][0]), line[i - 1][1] + f * (line[i][1] - line[i - 1][1])];
        if (inside(poly, p) && !tooClose(p)) holes.push(p);
      }
    }
  }
  const rows = holes.length;
  // Infill: farthest first on a 2 m sampling of the area, keeping each sample's nearest-hole distance.
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]), step = 2;
  const samples: Pt[] = [];
  for (let x = Math.min(...xs); x <= Math.max(...xs); x += step) {
    for (let y = Math.min(...ys); y <= Math.max(...ys); y += step) if (inside(poly, [x, y])) samples.push([x, y]);
  }
  const near = new Float64Array(samples.length).fill(Infinity);
  const update = (h: Pt) => samples.forEach((s, i) => { near[i] = Math.min(near[i], Math.hypot(s[0] - h[0], s[1] - h[1])); });
  holes.forEach(update);
  for (;;) {
    let best = -1;
    near.forEach((v, i) => { if (best < 0 || v > near[best]) best = i; });
    if (best < 0 || near[best] < spacing - 1e-6) break;
    holes.push(samples[best]);
    update(samples[best]);
  }
  return { holes, rows, dz, slope };
}
