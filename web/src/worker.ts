// The pipeline, off the page's thread: the GeoTIFF reader and the Rust core (WebAssembly).
// The page sends "open", "load" and "run"; the worker answers with layers as each stage finishes,
// progress within the long ones, and "busy" when a task starts and ends. "stop" ends a task at
// its next checkpoint (between bands of a read, slices of the edges, and corridors).
import init, * as wasm from "./wasm/roadsearch_wasm.js";
import { Source, Z_UNIT_SCALE, type Grid } from "./source";
import { get, hairpinAllowance, maxCatchAtMask, sectionTable, sigmaM, sMask, TABLE_NG, TABLE_NS,
         type Params } from "./params";
import type { FromWorker, RouteOut, ToWorker } from "./messages";

const ready = init();
let source: Source | null = null;
let area: { grid: Grid; session: wasm.Session; start: [number, number]; goal: [number, number];
             startXY: [number, number]; goalXY: [number, number]; zScale: number;
             acc?: Float32Array; routes?: RouteOut[] } | null = null;

const post = (m: FromWorker, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

function timed<T>(name: string, f: () => T): T {
  post({ type: "stage", name, state: "start" });
  const t0 = performance.now();
  const r = f();
  post({ type: "stage", name, state: "end", ms: performance.now() - t0 });
  return r;
}

class Stopped extends Error {}
let stopRequested = false;

/** Lets messages (a "stop") in, then throws if one asked to stop. */
async function checkpoint(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
  if (stopRequested) throw new Stopped("stopped");
}

const progress = (name: string, done: number, total: number) => post({ type: "progress", name, done, total });

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

/** Channels: cells draining at least aMin ha, more opaque downstream. */
function channelRgba(acc: Float32Array, aMin: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(acc.length * 4);
  for (let i = 0; i < acc.length; i++) {
    if (acc[i] >= aMin) {
      out[4 * i] = 30; out[4 * i + 1] = 120; out[4 * i + 2] = 220;
      out[4 * i + 3] = Math.min(255, 90 + 40 * Math.log10(acc[i] / aMin));
    }
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
  const pv = await timedAsync("preview", () => source!.preview(wasm.resample, zScale, 1200, async (done, total) => {
    progress("preview", done, total);
    await checkpoint();
  }));
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
  const z = await timedAsync("read and reproject the area", () =>
    source!.resampleBands(grid, wasm.resample, zScale, -1, async (done, total) => {
      progress("read and reproject the area", done, total);
      await checkpoint();
    }));
  const s = cellOf(grid, sx, sy), g = cellOf(grid, gx, gy);
  if (!s || !g) throw new Error("start or finish is outside the DEM");
  area?.session.free();
  area = { grid, session: new wasm.Session(z, grid.w, grid.h, res), start: s, goal: g, startXY: m.start, goalXY: m.goal,
           zScale };
  const rgba = grayRgba(wasm.hillshade(z, grid.w, grid.h, res));
  post({ type: "raster", layer: "hillshade", grid, rgba }, [rgba.buffer]);
  post({ type: "area", grid, cells });
}

/** Edges are costed in slices of about this long, between checkpoints. */
const SLICE_MS = 150;

async function run(p: Params) {
  if (!area) throw new Error("load an area first");
  const { grid, session } = area;
  const res = grid.res;
  timed("smooth, gradients, slope mask, flow", () =>
    session.prepare(sigmaM(p) / res, get(p, "g_max"), sMask(p), maxCatchAtMask(p)));
  const open = session.open_mask();
  const slope = new Uint8ClampedArray(open.length * 4);
  for (let i = 0; i < open.length; i++) if (!open[i]) { slope[4 * i] = 215; slope[4 * i + 1] = 48; slope[4 * i + 2] = 31; slope[4 * i + 3] = 110; }
  post({ type: "raster", layer: "slope", grid, rgba: slope }, [slope.buffer]);
  area.acc = session.contributing_area();
  const ch = channelRgba(area.acc, get(p, "A_min"));
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
  await checkpoint();
  const nPairs = timed("graph nodes and candidate edges", () =>
    session.graph_begin(sr, sc, gr, gc, edge, breaks, costs, tin, table, TABLE_NS, TABLE_NG, ds, dg));
  // Cost the edges in slices sized to take about SLICE_MS each.
  const name = "cost the edges on the DEM";
  post({ type: "stage", name, state: "start" });
  const t0 = performance.now();
  let done = 0, slice = 2000;
  while (done < nPairs) {
    const ts = performance.now();
    const next = session.graph_cost(slice);
    const ms = performance.now() - ts;
    slice = Math.max(200, Math.min(200_000, Math.round(slice * SLICE_MS / Math.max(ms, 1) * (next - done) / slice)));
    done = next;
    progress(name, done, nPairs);
    await checkpoint();
  }
  post({ type: "stage", name, state: "end", ms: performance.now() - t0 });
  timed("assemble the graph", () => session.graph_finish());
  await checkpoint();
  const best = timed("search from both ends: best route", () => session.search_begin()) as RouteOut | null;
  const rows = session.node_rows(), cols = session.node_cols(), kinds = session.node_kinds();
  const field = session.field(), pairs = session.edge_pairs(), cpm = session.edge_cost_per_m();
  post({ type: "graph", grid, rows, cols, kinds, field, pairs, costPerM: cpm },
       [rows.buffer, cols.buffer, kinds.buffer, field.buffer, pairs.buffer, cpm.buffer]);
  if (!best) throw new Error("no route: the finish cannot be reached from the start on this graph");
  const routes: RouteOut[] = [best];
  const nAlt = get(p, "tin_alternatives");
  post({ type: "corridors", grid, routes, final: nAlt === 0 });
  // Alternatives, each as it is found.
  for (;;) {
    progress("alternatives", routes.length - 1, nAlt);
    await checkpoint();
    const r = timed(`look for alternative ${routes.length}`, () => session.search_next()) as RouteOut | null;
    if (!r) break;
    routes.push(r);
    post({ type: "corridors", grid, routes, final: false });
  }
  post({ type: "corridors", grid, routes, final: true });
  area.routes = routes;
}

/**
 * The fine search of each corridor (or the best): the DEM read at `res` over the corridor's
 * buffer, preprocessed there, and searched on the heading lattice with hairpins, bends and fill
 * crossings, as the command line's fine stage does over its corridors.
 */
async function refine(p: Params, res: number, which: "best" | "all") {
  if (!area?.routes?.length || !source) throw new Error("find corridors first");
  const { grid: cg, routes, startXY, goalXY, zScale } = area;
  const buffer = get(p, "tin_buffer");
  const [bx0, by0, bx1, by1] = source.info.bbox;
  const todo = which === "best" ? [0] : routes.map((_, i) => i);
  for (const k of todo) {
    const t0 = performance.now();
    const name = `corridor ${k + 1}`;
    const r = routes[k];
    // The corridor's line in UTM, and a grid at `res` over it plus the buffer, within the DEM.
    const xs = r.cols.map((c) => cg.x0 + (c + 0.5) * cg.res), ys = r.rows.map((row) => cg.y1 - (row + 0.5) * cg.res);
    let [xmin, xmax, ymin, ymax] = [Infinity, -Infinity, Infinity, -Infinity];
    for (let i = 0; i < xs.length; i++) {
      xmin = Math.min(xmin, xs[i]); xmax = Math.max(xmax, xs[i]); ymin = Math.min(ymin, ys[i]); ymax = Math.max(ymax, ys[i]);
    }
    const x0 = Math.floor(Math.max(xmin - buffer, bx0) / res) * res, y1 = Math.ceil(Math.min(ymax + buffer, by1) / res) * res;
    const fg: Grid = { x0, y1, res, w: Math.ceil((Math.min(xmax + buffer, bx1) - x0) / res),
                       h: Math.ceil((y1 - Math.max(ymin - buffer, by0)) / res) };
    const z = await timedAsync(`${name}: read the DEM at ${res} m`, () =>
      source!.resampleBands(fg, wasm.resample, zScale, -1, async (done, total) => {
        progress(`${name}: read the DEM at ${res} m`, done, total);
        await checkpoint();
      }));
    const lr = new Float64Array(ys.map((y) => (fg.y1 - y) / res - 0.5));
    const lc = new Float64Array(xs.map((x) => (x - fg.x0) / res - 0.5));
    const fs = timed(`${name}: preprocess ${fg.w.toLocaleString()} x ${fg.h.toLocaleString()} cells`, () =>
      new wasm.FineSession(z, fg.w, fg.h, res, JSON.stringify(p), lr, lc, buffer));
    await checkpoint();
    try {
      const cell = (x: number, y: number) => [Math.floor((fg.y1 - y) / res), Math.floor((x - fg.x0) / res)];
      const [sr, sc] = cell(startXY[0], startXY[1]), [gr, gc] = cell(goalXY[0], goalXY[1]);
      let stage = "", tStage = 0;
      const out = fs.search(sr, sc, gr, gc, (st: string, settled: number) => {
        if (st !== stage) {
          if (stage) post({ type: "stage", name: `${name}: ${stage}`, state: "end", ms: performance.now() - tStage });
          stage = st;
          tStage = performance.now();
          post({ type: "stage", name: `${name}: ${st}`, state: "start" });
        }
        progress(`${name}: ${st}, states settled`, settled, 0);
      });
      if (stage) post({ type: "stage", name: `${name}: ${stage} and the route's costs`, state: "end", ms: performance.now() - tStage });
      if (!out) throw new Error(`${name}: no route at ${res} m within ${buffer} m of the corridor`);
      post({ type: "fine", route: {
        index: k, res, x: out.cols.map((c: number) => fg.x0 + (c + 0.5) * res), y: out.rows.map((row: number) => fg.y1 - (row + 0.5) * res),
        z: out.z, cost: out.cost, length: out.length, hairpins: out.hairpins, bends: out.bends,
        fill_crossings: out.fill_crossings, segments: out.segments, history: out.history,
        seconds: (performance.now() - t0) / 1000, cells: fs.cells(), macros_computed: out.macros_computed,
      } });
    } finally {
      fs.free();
    }
    await checkpoint();
  }
}

// One message at a time: a "load" must not start while an "open" is still reading its file.
// "stop" is the exception: it is read at once, and the running task ends at its next checkpoint.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === "stop") {
    stopRequested = true;
    return;
  }
  queue = queue.then(async () => {
    stopRequested = false;
    const task = m.type === "channels" ? null : m.type;
    if (task) post({ type: "busy", task });
    try {
      if (m.type === "open") await open(m);
      else if (m.type === "load") await load(m);
      else if (m.type === "run") await run(m.params);
      else if (m.type === "refine") await refine(m.params, m.res, m.which);
      else if (m.type === "channels" && area?.acc) {
        const ch = channelRgba(area.acc, m.aMin);
        post({ type: "raster", layer: "channels", grid: area.grid, rgba: ch }, [ch.buffer]);
      }
    } catch (err) {
      if (err instanceof Stopped) post({ type: "stopped", task: task ?? "" });
      else post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    } finally {
      if (task) post({ type: "busy", task: null });
    }
  });
};
