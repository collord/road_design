// The pipeline, off the page's thread: the GeoTIFF reader and the Rust core (WebAssembly).
// The page sends "open", "load" and "run"; the worker answers with layers as each stage finishes.
import init, * as wasm from "./wasm/roadsearch_wasm.js";
import { Source, Z_UNIT_SCALE, type Grid } from "./source";
import { get, hairpinAllowance, maxCatchAtMask, sectionTable, sigmaM, sMask, TABLE_NG, TABLE_NS,
         type Params } from "./params";
import type { FromWorker, ToWorker } from "./messages";

const ready = init();
let source: Source | null = null;
let area: { grid: Grid; session: wasm.Session; start: [number, number]; goal: [number, number] } | null = null;

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

function timed<T>(name: string, f: () => T): T {
  post({ type: "stage", name, state: "start" });
  const t0 = performance.now();
  const r = f();
  post({ type: "stage", name, state: "end", ms: performance.now() - t0 });
  return r;
}

async function timedAsync<T>(name: string, f: () => Promise<T>): Promise<T> {
  post({ type: "stage", name, state: "start" });
  const t0 = performance.now();
  const r = await f();
  post({ type: "stage", name, state: "end", ms: performance.now() - t0 });
  return r;
}

/** Gray RGBA of a hillshade (transparent where there is no data). */
function grayRgba(hs: Uint8Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray(hs.length * 4);
  for (let i = 0; i < hs.length; i++) {
    const v = hs[i];
    out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = v;
    out[4 * i + 3] = v === 0 ? 0 : 255;
  }
  return out;
}

function cellOf(g: Grid, x: number, y: number): [number, number] | null {
  const r = Math.floor((g.y1 - y) / g.res), c = Math.floor((x - g.x0) / g.res);
  return r >= 0 && c >= 0 && r < g.h && c < g.w ? [r, c] : null;
}

/** The bytes at a URL, refusing a missing file or a page served in its place (a dev server's fallback). */
async function fetchTiff(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`could not fetch ${url}: ${r.status} ${r.statusText}`);
  const buf = await r.arrayBuffer();
  const b = new Uint8Array(buf, 0, Math.min(4, buf.byteLength));
  const tiff = (b[0] === 0x49 && b[1] === 0x49) || (b[0] === 0x4d && b[1] === 0x4d);
  if (!tiff) throw new Error(`${url} is not a GeoTIFF (got ${r.headers.get("content-type") ?? "unknown content"})`);
  return buf;
}

async function open(m: Extract<ToWorker, { type: "open" }>) {
  await ready;
  const input = m.url ? await fetchTiff(m.url) : (m.file as Blob);
  source = await timedAsync("read the GeoTIFF header", () => Source.open(input, m.name));
  area?.session.free();
  area = null;
  const ends = m.endpointsDem?.map(([x, y]) => source!.utm(x, y));
  post({ type: "info", info: source.info, endpoints: ends });
  const zScale = Z_UNIT_SCALE[m.zUnits] ?? 1;
  const pv = await timedAsync("preview", () => source!.preview(wasm.resample, zScale));
  const rgba = grayRgba(wasm.hillshade(pv.z, pv.grid.w, pv.grid.h, pv.grid.res));
  post({ type: "raster", layer: "preview", grid: pv.grid, rgba }, [rgba.buffer]);
}

async function load(m: Extract<ToWorker, { type: "load" }>) {
  if (!source) throw new Error("open a DEM first");
  const [sx, sy] = m.start, [gx, gy] = m.goal;
  const [bx0, by0, bx1, by1] = source.info.bbox;
  const res = m.res;
  const x0 = Math.floor(Math.max(Math.min(sx, gx) - m.margin, bx0) / res) * res;
  const y1 = Math.ceil(Math.min(Math.max(sy, gy) + m.margin, by1) / res) * res;
  const xe = Math.min(Math.max(sx, gx) + m.margin, bx1), ye = Math.max(Math.min(sy, gy) - m.margin, by0);
  const grid: Grid = { x0, y1, res, w: Math.ceil((xe - x0) / res), h: Math.ceil((y1 - ye) / res) };
  const cells = grid.w * grid.h;
  if (cells > m.maxCells) {
    const fit = Math.sqrt(cells / m.maxCells) * res;
    throw new Error(`the area is ${(cells / 1e6).toFixed(1)} million cells at ${res} m, over the limit of ` +
      `${(m.maxCells / 1e6).toFixed(0)} million: use cells of ${Math.ceil(fit)} m or more, or a smaller margin`);
  }
  const zScale = Z_UNIT_SCALE[m.zUnits] ?? 1;
  const z = await timedAsync("read and reproject the area", () => source!.resampleTo(grid, wasm.resample, zScale));
  const s = cellOf(grid, sx, sy), g = cellOf(grid, gx, gy);
  if (!s || !g) throw new Error("start or finish is outside the DEM");
  area?.session.free();
  area = { grid, session: new wasm.Session(z, grid.w, grid.h, res), start: s, goal: g };
  const rgba = grayRgba(wasm.hillshade(z, grid.w, grid.h, res));
  post({ type: "raster", layer: "hillshade", grid, rgba }, [rgba.buffer]);
  post({ type: "area", grid, cells });
}

function run(p: Params) {
  if (!area) throw new Error("load an area first");
  const { grid, session } = area;
  const res = grid.res;
  timed("smooth, gradients, slope mask, flow", () =>
    session.prepare(sigmaM(p) / res, get(p, "g_max"), sMask(p), maxCatchAtMask(p)));
  const open = session.open_mask();
  const slope = new Uint8ClampedArray(open.length * 4);
  for (let i = 0; i < open.length; i++) if (!open[i]) { slope[4 * i] = 215; slope[4 * i + 1] = 48; slope[4 * i + 2] = 31; slope[4 * i + 3] = 110; }
  post({ type: "raster", layer: "slope", grid, rgba: slope }, [slope.buffer]);
  const acc = session.contributing_area();
  const aMin = get(p, "A_min");
  const ch = new Uint8ClampedArray(acc.length * 4);
  for (let i = 0; i < acc.length; i++) {
    if (acc[i] >= aMin) {
      ch[4 * i] = 30; ch[4 * i + 1] = 120; ch[4 * i + 2] = 220;
      ch[4 * i + 3] = Math.min(255, 90 + 40 * Math.log10(acc[i] / aMin));
    }
  }
  post({ type: "raster", layer: "channels", grid, rgba: ch }, [ch.buffer]);
  const sad = session.saddles();
  post({ type: "points", layer: "saddles", grid, cells: sad }, [sad.buffer]);
  const nTri = timed("triangulate", () => session.triangulate(get(p, "tin_tolerance")));
  const tp = session.tri_points(), tt = session.tri_triangles();
  post({ type: "tin", grid, points: tp, triangles: tt, n: nTri }, [tp.buffer, tt.buffer]);
  const { table, ds, dg } = timed("section cost table", () => sectionTable(p));
  const edge = new Float64Array([get(p, "g_max"), get(p, "C_travel"), get(p, "C_vertical"), get(p, "A_min"),
                                 hairpinAllowance(p), get(p, "C_turn")]);
  const tin = new Float64Array([get(p, "tin_tolerance"), get(p, "tin_k"), get(p, "tin_edge_max") / res,
                                get(p, "tin_block") / res, get(p, "tin_channel_ha"), get(p, "tin_alternatives"),
                                get(p, "tin_penalty"), get(p, "tin_stretch"), get(p, "tin_max_shared")]);
  const breaks = new Float64Array(p.culvert_area_breaks as number[]);
  const costs = new Float64Array(p.culvert_costs as number[]);
  const [sr, sc] = area.start, [gr, gc] = area.goal;
  const routes = timed("graph, two-way search, corridors", () =>
    session.corridors(sr, sc, gr, gc, edge, breaks, costs, tin, table, TABLE_NS, TABLE_NG, ds, dg));
  const rows = session.node_rows(), cols = session.node_cols(), kinds = session.node_kinds();
  const field = session.field(), pairs = session.edge_pairs(), cpm = session.edge_cost_per_m();
  post({ type: "graph", grid, rows, cols, kinds, field, pairs, costPerM: cpm },
       [rows.buffer, cols.buffer, kinds.buffer, field.buffer, pairs.buffer, cpm.buffer]);
  post({ type: "corridors", grid, routes });
}

// One message at a time: a "load" must not start while an "open" is still reading its file.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent<ToWorker>) => {
  queue = queue.then(async () => {
    try {
      const m = e.data;
      if (m.type === "open") await open(m);
      else if (m.type === "load") await load(m);
      else if (m.type === "run") run(m.params);
    } catch (err) {
      post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    }
  });
};
