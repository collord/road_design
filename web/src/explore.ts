// The fine search as it runs: the cells it settles (its wavefront) painted on a canvas over the
// map, the route to the state it just settled (the leading candidate), and its key over time.
import L from "leaflet";
import { CanvasLayer } from "./layers";
import type { Grid } from "./source";

/** Viridis-like ramp: 0 (cheap, near the start) violet, 1 (at the frontier) yellow. */
function ramp(t: number): [number, number, number] {
  const u = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const x = u * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const [a, b] = [stops[i], stops[i + 1]];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}

/** One corridor's search wavefront: settled cells colored by their cost to come relative to the
 *  search's key when they were settled (the share of the route's cost already spent there). */
export class Wavefront {
  readonly layer: CanvasLayer;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private pixels: ImageData;
  private dirty = false;

  constructor(readonly grid: Grid) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = grid.w;
    this.canvas.height = grid.h;
    this.ctx = this.canvas.getContext("2d")!;
    this.pixels = this.ctx.createImageData(grid.w, grid.h);
    const g = grid;
    this.layer = new CanvasLayer((ctx, px) => {
      if (this.dirty) {
        this.ctx.putImageData(this.pixels, 0, 0);
        this.dirty = false;
      }
      const [x0, y0] = px(g.x0, g.y1), [x1, y1] = px(g.x0 + g.w * g.res, g.y1 - g.h * g.res);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.canvas, x0, y0, x1 - x0, y1 - y0);
    });
  }

  /** A new search: start the picture again. */
  clear() {
    this.pixels.data.fill(0);
    this.dirty = true;
    this.layer.redraw();
  }

  /** Cells (row * w + col) settled since the last batch, their cost to come, the search's key. */
  add(cells: Uint32Array, g: Float32Array, key: number) {
    const d = this.pixels.data;
    for (let i = 0; i < cells.length; i++) {
      const [r, gr, b] = ramp(key > 0 ? g[i] / key : 0);
      const k = 4 * cells[i];
      d[k] = r; d[k + 1] = gr; d[k + 2] = b; d[k + 3] = 150;
    }
    this.dirty = true;
  }
}

/** Where switchbacks come cheap, as the helpers compute the designs: each cell colored by its
 *  cheapest feasible hairpin, deep red for the cheapest and pale for the dearest of those seen so
 *  far (5th to 95th percentile, so the colors keep their meaning as cells arrive); none: clear. */
export class HairpinSites {
  readonly layer: CanvasLayer;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private pixels: ImageData;
  private cells: number[] = [];
  private cost: number[] = [];
  private dirty = false;
  designed = 0;
  /** The color scale's ends, after the last repaint. */
  low = NaN;
  high = NaN;

  constructor(readonly grid: Grid) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = grid.w;
    this.canvas.height = grid.h;
    this.ctx = this.canvas.getContext("2d")!;
    this.pixels = this.ctx.createImageData(grid.w, grid.h);
    const g = grid;
    this.layer = new CanvasLayer((ctx, px) => {
      if (this.dirty) {
        this.repaint();
        this.dirty = false;
      }
      const [x0, y0] = px(g.x0, g.y1), [x1, y1] = px(g.x0 + g.w * g.res, g.y1 - g.h * g.res);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.canvas, x0, y0, x1 - x0, y1 - y0);
    });
  }

  add(cells: Uint32Array, cheapest: Float32Array) {
    this.designed += cells.length;
    for (let i = 0; i < cells.length; i++) {
      if (cheapest[i] < Infinity) {
        this.cells.push(cells[i]);
        this.cost.push(cheapest[i]);
      }
    }
    this.dirty = true;
  }

  /** Share of the designed cells where some hairpin fits. */
  get fits(): number {
    return this.designed ? this.cost.length / this.designed : 0;
  }

  private repaint() {
    const n = this.cost.length;
    if (!n) return;
    const sample: number[] = [];
    for (let i = 0; i < n; i += Math.max(1, Math.floor(n / 4000))) sample.push(this.cost[i]);
    sample.sort((a, b) => a - b);
    this.low = sample[Math.floor(0.05 * (sample.length - 1))];
    this.high = sample[Math.floor(0.95 * (sample.length - 1))];
    const span = Math.max(this.high - this.low, 1e-9);
    const d = this.pixels.data;
    for (let i = 0; i < n; i++) {
      const t = 1 - Math.max(0, Math.min(1, (this.cost[i] - this.low) / span));   // 1: cheapest
      const k = 4 * this.cells[i];
      d[k] = 253 - 70 * t; d[k + 1] = 219 - 199 * t; d[k + 2] = 199 - 170 * t; d[k + 3] = 60 + 170 * t;
    }
    this.ctx.putImageData(this.pixels, 0, 0);
  }
}

/** The leading candidate: a thin line with a dark casing, moved with every report. */
export class Candidate {
  readonly layer = L.layerGroup();
  private casing = L.polyline([], { color: "#111", weight: 4, opacity: 0.6, interactive: false });
  private line = L.polyline([], { color: "#fff", weight: 2, opacity: 0.95, interactive: false });

  constructor() {
    this.casing.addTo(this.layer);
    this.line.addTo(this.layer);
  }

  set(x: Float64Array, y: Float64Array) {
    const pts = Array.from(x, (xx, i) => [y[i], xx] as [number, number]);
    this.casing.setLatLngs(pts);
    this.line.setLatLngs(pts);
  }
}

/** The searches' keys over time, one series per search, as an SVG sparkline. */
export class KeyChart {
  private series: { name: string; pts: [number, number][] }[] = [];

  reset() {
    this.series = [];
  }

  add(name: string, t: number, key: number) {
    let s = this.series[this.series.length - 1];
    if (!s || s.name !== name) this.series.push((s = { name, pts: [] }));
    s.pts.push([t, key]);
  }

  svg(width = 260, height = 70): string {
    const all = this.series.flatMap((s) => s.pts);
    if (all.length < 2) return "";
    let [t0, t1, k0, k1] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const [t, k] of all) { t0 = Math.min(t0, t); t1 = Math.max(t1, t); k0 = Math.min(k0, k); k1 = Math.max(k1, k); }
    if (t1 <= t0) t1 = t0 + 1;
    if (k1 <= k0) k1 = k0 + 1;
    const px = (t: number) => 2 + ((t - t0) / (t1 - t0)) * (width - 4);
    const py = (k: number) => height - 12 - ((k - k0) / (k1 - k0)) * (height - 16);
    const colors = ["var(--accent)", "var(--alt)", "var(--start)", "var(--goal)"];
    const lines = this.series.map((s, i) =>
      `<polyline fill="none" stroke="${colors[i % colors.length]}" stroke-width="1.5" points="` +
      s.pts.map(([t, k]) => `${px(t).toFixed(1)},${py(k).toFixed(1)}`).join(" ") + `"><title>${s.name}</title></polyline>`);
    const last = all[all.length - 1][1];
    return `<svg viewBox="0 0 ${width} ${height}" class="status-chart" role="img" aria-label="search key over time">` +
      `<text x="2" y="${height - 1}">search key ${Math.round(last).toLocaleString()}</text>` +
      lines.join("") + `</svg>`;
  }
}
