// The pipeline, off the page's thread: the GeoTIFF reader and the Rust core (WebAssembly).
// The page sends "open", "load" and "run"; the worker answers with layers as each stage finishes,
// progress within the long ones, and "busy" when a task starts and ends. "stop" ends a task at
// its next checkpoint (between bands of a read, slices of the edges, and corridors).
import init, * as wasm from "./wasm/roadsearch_wasm.js";
import { layoutHoles, smooth } from "./layout";
import { Source, Z_UNIT_SCALE, type Grid } from "./source";
import { get, hairpinAllowance, maxCatchAtMask, roughK, roughR0, sectionTable, sigmaM, sMask, TABLE_NS,
         type Params } from "./params";
import type { FineOut, FromWorker, RouteOut, ToWorker } from "./messages";
import type { FromMacro, ToMacro } from "./macroWorker";

const ready = init();
let source: Source | null = null;
let area: { grid: Grid; session: wasm.Session; start: [number, number]; goal: [number, number];
             startXY: [number, number]; goalXY: [number, number]; zScale: number;
             pads: ([number, number] | null)[]; padsXY: [number, number][];
             /** The coarse pad network: its edges (UTM) and the order pads joined it (page pads). */
             network?: { lines: [number, number][][]; order: number[] };
             acc?: Float32Array; routes?: RouteOut[];
             /** Whether the area has a roughness (a DEM of fine enough pixels). */
             rough: boolean } | null = null;

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
  // The area covers the start, the finish (a route) or the pads (a network), plus the margin.
  const pts = [m.start, ...(m.goal ? [m.goal] : []), ...m.pads];
  const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
  const [bx0, by0, bx1, by1] = source.info.bbox;
  const res = m.res;
  const x0 = Math.floor(Math.max(Math.min(...xs) - m.margin, bx0) / res) * res;
  const y1 = Math.ceil(Math.min(Math.max(...ys) + m.margin, by1) / res) * res;
  const xe = Math.min(Math.max(...xs) + m.margin, bx1), ye = Math.max(Math.min(...ys) - m.margin, by0);
  const grid: Grid = { x0, y1, res, w: Math.ceil((xe - x0) / res), h: Math.ceil((y1 - ye) / res) };
  const cells = grid.w * grid.h;
  if (cells > m.maxCells) {
    const fit = Math.sqrt(cells / m.maxCells) * res;
    throw new Error(`the area is ${(cells / 1e6).toFixed(1)} million cells at ${res} m, over the limit of ` +
      `${(m.maxCells / 1e6).toFixed(0)} million: use cells of ${Math.ceil(fit)} m or more, or a smaller margin`);
  }
  roughLimit = m.roughMaxRes;
  const zScale = Z_UNIT_SCALE[m.zUnits] ?? 1;
  const { z, rough } = await timedAsync("read and reproject the area", () =>
    readDem(grid, true, zScale, async (done, total) => {
      progress("read and reproject the area", done, total);
      await checkpoint();
    }));
  const s = cellOf(grid, m.start[0], m.start[1]);
  const g = m.goal ? cellOf(grid, m.goal[0], m.goal[1]) : s;
  if (!s || !g) throw new Error(`the ${m.goal ? "start or finish" : "access point"} is outside the DEM`);
  const pads = m.pads.map(([x, y]) => cellOf(grid, x, y));
  area?.session.free();
  area = { grid, session: new wasm.Session(z, grid.w, grid.h, res), start: s, goal: g, startXY: m.start,
           goalXY: m.goal ?? m.start, zScale, pads, padsXY: m.pads, rough: rough.length > 0 };
  if (rough.length) area.session.set_rough(rough);
  const rgba = grayRgba(wasm.hillshade(z, grid.w, grid.h, res));
  post({ type: "raster", layer: "hillshade", grid, rgba }, [rgba.buffer]);
  post({ type: "area", grid, cells, roughness: roughNote(rough) });
}

/** Roughness (m) of the DEM's native pixels (roughness.py ROUGH_SIGMA_PX = 2). */
const roughness = (z: Float32Array, w: number, h: number) => wasm.roughness(z, w, h, 2);

/** Coarsest DEM pixels (m) roughness is read from (rough_max_res), as of the last area loaded. */
let roughLimit = 3;

/** The DEM on a grid, with its roughness when asked for and the DEM's pixels are fine enough for
 *  it to mean rock (else empty). */
async function readDem(g: Grid, withRough: boolean, zScale: number,
                       onBand: (done: number, total: number) => Promise<void>): Promise<{ z: Float32Array; rough: Float32Array }> {
  if (withRough && source!.info.resM <= roughLimit) {
    return source!.resampleWithRoughness(g, wasm.resample, roughness, zScale, onBand);
  }
  return { z: await source!.resampleBands(g, wasm.resample, zScale, -1, onBand), rough: new Float32Array(0) };
}

/** What became of the roughness (roughness.py). */
function roughNote(rough: Float32Array): string {
  const px = source!.info.resM;
  if (!rough.length) {
    return `roughness: the DEM's ${px.toFixed(1)} m pixels are coarser than rough_max_res ${roughLimit} m: their ` +
      "detrended relief is mostly landform, not rock; roughness is ignored";
  }
  const v = Array.from(rough).filter((x) => !Number.isNaN(x)).sort((a, b) => a - b);
  return `roughness from the DEM's ${px.toFixed(2)} m pixels: median ${v.length ? v[v.length >> 1].toFixed(2) : "-"} m ` +
    "(thins the soil when rough_r0 > 0)";
}

/** The DEM on a fine grid, with its roughness when the parameters use it. */
const readFine = (fg: Grid, p: Params, zScale: number, onBand: (done: number, total: number) => Promise<void>) =>
  readDem(fg, roughR0(p) > 0, zScale, onBand);

/** Drill holes over an area (map coordinates): the DEM read over it and a margin at 5 m,
 *  smoothed (sigma 25 m), then rows along the contours and infill (layout.ts). */
async function layout(m: Extract<ToWorker, { type: "layout" }>) {
  if (!source) throw new Error("open a DEM first");
  if (m.area.length < 3) throw new Error("the drilling area needs at least three corners");
  const res = 5, pad = 2 * m.spacing;
  const xs = m.area.map((p) => p[0]), ys = m.area.map((p) => p[1]);
  const x0 = Math.floor((Math.min(...xs) - pad) / res) * res, y1 = Math.ceil((Math.max(...ys) + pad) / res) * res;
  const g: Grid = { x0, y1, res, w: Math.ceil((Math.max(...xs) + pad - x0) / res), h: Math.ceil((y1 - Math.min(...ys) + pad) / res) };
  if (g.w * g.h > 4e6) throw new Error("the drilling area is too large to lay out at 5 m");
  const z = await timedAsync("layout: read the DEM at 5 m", () =>
    source!.resampleBands(g, wasm.resample, Z_UNIT_SCALE[m.zUnits] ?? 1, -1, async (done, total) => {
      progress("layout: read the DEM", done, total);
      await checkpoint();
    }));
  const zs = timed("layout: smooth", () => smooth(z, g.w, g.h, 25 / res));
  const out = timed("layout: rows along the contours, then infill", () => layoutHoles(zs, g, m.area, m.spacing));
  post({ type: "holes", holes: out.holes, rows: out.rows, row: out.row, station: out.station, dz: out.dz, slope: out.slope });
}

// --- the fine search's helpers -----------------------------------------------------------------

/** Helper workers computing hairpin and bend costs ahead of the search: all but two cores, at
 *  most 8, and within HELPER_BUDGET bytes in all (each holds a copy of the corridor's rasters,
 *  about HELPER_BYTES_PER_CELL). Fewer than 2: the search computes them itself as it reaches
 *  them. The helpers are ended after each refine, which hands their memory back. */
const MAX_HELPERS = Math.min(8, Math.max(0, (navigator.hardwareConcurrency || 4) - 2));
const HELPER_BUDGET = 1.2e9;
const HELPER_BYTES_PER_CELL = 150;
/** Cells per helper task: about a second or two of designs. */
const MACRO_CHUNK = 1024;
let pool: Worker[] = [];

function stopPool() {
  for (const w of pool) w.terminate();
  pool = [];
}

/** One message from a helper. */
function reply(w: Worker): Promise<FromMacro> {
  return new Promise((resolve) => {
    w.onmessage = (e: MessageEvent<FromMacro>) => resolve(e.data);
    w.onerror = (e) => resolve({ type: "error", message: e.message || "a helper worker failed" });
  });
}

/**
 * Hairpin and bend costs of every cell of the fine session, computed by the helpers in parallel
 * and stored in the session's cache (the values it would compute itself as the search reaches
 * them). Stops at a checkpoint if asked to, terminating the helpers.
 */
async function precomputeMacros(fs: wasm.FineSession, setup: Extract<ToMacro, { type: "setup" }>, name: string,
                                index: number, grid: Grid) {
  const size = Math.min(MAX_HELPERS, Math.floor(HELPER_BUDGET / (setup.w * setup.h * HELPER_BYTES_PER_CELL)));
  if (size < 2) return;
  while (pool.length < size) pool.push(new Worker(new URL("./macroWorker.ts", import.meta.url), { type: "module" }));
  const n = fs.cells();
  const tableCells = fs.table_cells();
  // Reversing switchbacks need no hairpin designs: no hairpin sites to show.
  const switchbacks = Number(JSON.parse(setup.params).switchback_landing) > 0;
  const stage = `${name}: ${switchbacks ? "bend" : "hairpin and bend"} designs on ${pool.length} workers`;
  post({ type: "stage", name: stage, state: "start" });
  const t0 = performance.now();
  try {
    await Promise.all(pool.map(async (w) => {
      w.postMessage({ ...setup, z: setup.z.slice() } satisfies ToMacro);
      const r = await reply(w);
      if (r.type !== "ready") throw new Error(r.type === "error" ? r.message : "a helper did not start");
    }));
    let next = 0, done = 0;
    await Promise.all(pool.map(async (w) => {
      while (next < n) {
        const a = next;
        next = Math.min(n, next + MACRO_CHUNK);
        w.postMessage({ type: "rows", a, b: next } satisfies ToMacro);
        const r = await reply(w);
        if (r.type === "error") throw new Error(r.message);
        if (r.type !== "rows") continue;
        fs.set_macro_rows(r.a, r.hp, r.bend);
        done += Math.min(MACRO_CHUNK, n - r.a);
        progress(stage + ", cells", Math.min(done, n), n);
        await checkpoint();
        if (switchbacks) continue;
        // Where switchbacks come cheap: the cheapest feasible hairpin per cell, for the page.
        const m = Math.min(MACRO_CHUNK, n - r.a), per = r.hp.length / m;
        const cheapest = new Float32Array(m);
        for (let i = 0; i < m; i++) {
          let c = Infinity;
          for (let j = i * per; j < (i + 1) * per; j++) if (r.hp[j] < c) c = r.hp[j];
          cheapest[i] = c;
        }
        const cells = tableCells.slice(r.a, r.a + m);
        post({ type: "sites", index, grid, cells, cheapest }, [cells.buffer, cheapest.buffer]);
      }
    }));
  } catch (err) {
    stopPool();
    throw err;
  }
  post({ type: "stage", name: stage, state: "end", ms: performance.now() - t0 });
}

/** The fine search's memory: per cell of a corridor's buffer (hairpin and bend caches, 128
 *  float32 per cell at 32 headings, and the search's cost, parent and settled flags per heading),
 *  per cell of its bounding box (the rasters), and the most one corridor may take. */
const FINE_BYTES_PER_CELL = 900;
const FINE_BYTES_PER_GRID_CELL = 80;
const FINE_BUDGET = 1.5e9;

/** Edges are costed in slices of about this long, between checkpoints. */
const SLICE_MS = 150;

/** Preprocess the area and build the coarse graph with these terminals (row, col): the first is
 *  the start or access point. */
async function buildGraph(p: Params, terminals: [number, number][], extra: [number, number][] = [], prepared = false) {
  const { grid, session } = area!;
  const res = grid.res;
  if (!prepared) {
    timed("smooth, gradients, slope mask, flow", () =>
      session.prepare(sigmaM(p) / res, get(p, "g_max"), sMask(p), maxCatchAtMask(p)));
  }
  const open = session.open_mask();
  const slope = new Uint8ClampedArray(open.length * 4);
  for (let i = 0; i < open.length; i++) if (!open[i]) { slope[4 * i] = 215; slope[4 * i + 1] = 48; slope[4 * i + 2] = 31; slope[4 * i + 3] = 110; }
  post({ type: "raster", layer: "slope", grid, rgba: slope }, [slope.buffer]);
  const acc = (area!.acc = session.contributing_area());
  const ch = channelRgba(acc, get(p, "A_min"));
  post({ type: "raster", layer: "channels", grid, rgba: ch }, [ch.buffer]);
  const sad = session.saddles();
  post({ type: "points", layer: "saddles", grid, cells: sad }, [sad.buffer]);
  const nTri = timed("triangulate", () => session.triangulate(get(p, "tin_tolerance")));
  const tp = session.tri_points(), tt = session.tri_triangles();
  post({ type: "tin", grid, points: tp, triangles: tt, n: nTri }, [tp.buffer, tt.buffer]);
  // Without a roughness the table needs no grades past g_max (nor the time to build them).
  const pt = area!.rough ? p : { ...p, rough_r0: 0 };
  const { table, ng, ds, dg } = timed("section cost table", () => sectionTable(pt));
  const edge = new Float64Array([get(p, "g_max"), get(p, "C_travel"), get(p, "C_vertical"), get(p, "A_min"),
                                 hairpinAllowance(p), get(p, "C_turn"), roughK(pt)]);
  const tin = new Float64Array([get(p, "tin_tolerance"), get(p, "tin_k"), get(p, "tin_edge_max") / res,
                                get(p, "tin_block") / res, get(p, "tin_channel_ha"), get(p, "tin_alternatives"),
                                get(p, "tin_penalty"), get(p, "tin_stretch"), get(p, "tin_max_shared"),
                                get(p, "tin_separation") / res]);
  const breaks = new Float64Array(p.culvert_area_breaks as number[]);
  const costs = new Float64Array(p.culvert_costs as number[]);
  await checkpoint();
  const tr = new Uint32Array(terminals.map((t) => t[0])), tc = new Uint32Array(terminals.map((t) => t[1]));
  const er = new Uint32Array(extra.map((t) => t[0])), ec = new Uint32Array(extra.map((t) => t[1]));
  const nPairs = timed("graph nodes and candidate edges", () =>
    session.graph_begin_terminals(tr, tc, er, ec, edge, breaks, costs, tin, table, TABLE_NS, ng, ds, dg));
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
}

/** The graph's nodes and edges to the page (with the corridor field, if a search made one). */
function postGraph() {
  const { grid, session } = area!;
  const rows = session.node_rows(), cols = session.node_cols(), kinds = session.node_kinds();
  const field = session.field(), pairs = session.edge_pairs(), cpm = session.edge_cost_per_m();
  post({ type: "graph", grid, rows, cols, kinds, field, pairs, costPerM: cpm },
       [rows.buffer, cols.buffer, kinds.buffer, field.buffer, pairs.buffer, cpm.buffer]);
}

async function run(p: Params) {
  if (!area) throw new Error("load an area first");
  const { grid, session } = area;
  await buildGraph(p, [area.start, area.goal]);
  const best = timed("search from both ends: best route", () => session.search_begin()) as RouteOut | null;
  postGraph();
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
 * A road network from the access point (the start) to the pads, on the coarse graph: grown pad by
 * pad (the pad cheapest to reach from the network so far), then improved (each pad's end branch
 * reconnected at least cost) until a pass changes nothing. A pad may lie on the way to at most
 * `b` others. The network is posted after every step.
 */
/** Cells (row, col) every `step` cells within `buffer` cells of the convex hull of the points:
 *  a denser mesh for the coarse graph where the pads are. */
function meshNear(points: [number, number][], step: number, buffer: number, h: number, w: number): [number, number][] {
  // Convex hull (monotone chain) of the points as (col, row).
  const pts = [...new Set(points.map(([r, c]) => `${c},${r}`))].map((s) => s.split(",").map(Number) as [number, number])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: [number, number][]) => {
    const out: [number, number][] = [];
    for (const p of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
      out.push(p);
    }
    return out;
  };
  const lower = half(pts), upper = half([...pts].reverse());
  const hull = pts.length < 3 ? pts : [...lower.slice(0, -1), ...upper.slice(0, -1)];
  const segDist = (x: number, y: number, a: number[], b: number[]) => {
    const [dx, dy] = [b[0] - a[0], b[1] - a[1]], l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / l2)) : 0;
    return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
  };
  const inside = (x: number, y: number) => hull.length >= 3 && hull.every((a, i) => cross(a, hull[(i + 1) % hull.length], [x, y]) >= 0);
  const near = (x: number, y: number) => inside(x, y) ||
    hull.some((a, i) => segDist(x, y, a, hull[(i + 1) % hull.length]) <= buffer) || hull.some((a) => Math.hypot(x - a[0], y - a[1]) <= buffer);
  const xs = hull.map((q) => q[0]), ys = hull.map((q) => q[1]);
  const out: [number, number][] = [];
  for (let y = Math.max(0, Math.min(...ys) - buffer); y <= Math.min(h - 1, Math.max(...ys) + buffer); y += step) {
    for (let x = Math.max(0, Math.min(...xs) - buffer); x <= Math.min(w - 1, Math.max(...xs) + buffer); x += step) {
      if (near(x, y)) out.push([Math.round(y), Math.round(x)]);
    }
  }
  return out;
}

async function network(p: Params, b: number, refine: number) {
  if (!area) throw new Error("load an area first");
  const { grid, session } = area;
  const pads = area.pads;
  if (!pads.length) throw new Error("add pads first");
  const outside = pads.map((q, i) => (q ? -1 : i + 1)).filter((i) => i > 0);
  if (outside.length) throw new Error(`pads ${outside.join(", ")} are outside the area`);
  // One terminal per cell: pads in the access point's cell are connected with it, pads sharing a
  // cell share a terminal.
  const key = (q: [number, number]) => q[0] * grid.w + q[1];
  const terminals: [number, number][] = [area.start];
  const termOf = new Map<number, number>([[key(area.start), 0]]);
  const padTerm = pads.map((q) => {
    const k = key(q!);
    if (!termOf.has(k)) {
      termOf.set(k, terminals.length);
      terminals.push(q!);
    }
    return termOf.get(k)!;
  });
  // A denser mesh over the pads (their convex hull and 300 m around it): background nodes
  // `refine` times closer than tin_block, and a triangulation `refine` times tighter than
  // tin_tolerance there, so nodes land on the benches and breaks in slope near the pads.
  const zone = 300 / grid.res;
  const extra = refine > 1 ? meshNear(terminals, Math.max(1, get(p, "tin_block") / refine / grid.res), zone, grid.h, grid.w) : [];
  if (refine > 1) {
    timed("smooth, gradients, slope mask, flow", () =>
      session.prepare(sigmaM(p) / grid.res, get(p, "g_max"), sMask(p), maxCatchAtMask(p)));
    const rows = terminals.map((t) => t[0]), cols = terminals.map((t) => t[1]);
    const r0 = Math.max(0, Math.floor(Math.min(...rows) - zone)), c0 = Math.max(0, Math.floor(Math.min(...cols) - zone));
    const r1 = Math.min(grid.h, Math.ceil(Math.max(...rows) + zone) + 1), c1 = Math.min(grid.w, Math.ceil(Math.max(...cols) + zone) + 1);
    const v = timed("triangulate the pad zone", () => session.tin_vertices_in(r0, c0, r1, c1, get(p, "tin_tolerance") / refine));
    const inZone = new Set(meshNear(terminals, 1, zone, grid.h, grid.w).map(([r, c]) => r * grid.w + c));
    for (let i = 0; i < v.length; i += 2) if (inZone.has(v[i] * grid.w + v[i + 1])) extra.push([v[i], v[i + 1]]);
  }
  await buildGraph(p, terminals, extra, refine > 1);
  if (extra.length) post({ type: "stage", name: `mesh near the pads: ${extra.length.toLocaleString()} extra nodes`, state: "end", ms: 0 });
  // Pads lie along the contour.
  const axes = session.contour_angle(new Uint32Array(pads.map((q) => q![0])), new Uint32Array(pads.map((q) => q![1])));
  post({ type: "padAxes", angles: Array.from(axes) });
  postGraph();
  session.network_begin(b);
  const n = terminals.length - 1;
  const send = (phase: string, final: boolean) => {
    const st = session.network_state() as { pairs: number[]; uses: number[]; connected: boolean[];
      unreachable: boolean[]; beyond: number[]; order: number[]; construction: number; operating: number };
    // Per pad in the page's order (terminal t >= 1 is pad t - 1 of the graph's).
    const per = <T>(a: T[], atAccess: T) => padTerm.map((t) => (t === 0 ? atAccess : a[t - 1]));
    post({ type: "network", grid, pairs: st.pairs, uses: st.uses, connected: per(st.connected, true),
           unreachable: per(st.unreachable, false), beyond: per(st.beyond, 0),
           // The graph's pad order as page pads (the first page pad of each terminal).
           order: st.order.map((t) => padTerm.indexOf(t + 1)), construction: st.construction,
           operating: st.operating, phase, final });
  };
  const grow = "connect the pads, cheapest first";
  post({ type: "stage", name: grow, state: "start" });
  let t0 = performance.now();
  for (let k = 0; ; k++) {
    progress(grow, k, n);
    await checkpoint();
    if (session.network_grow() < 0) break;
    send(`${k + 1} of ${n} pads connected`, false);
  }
  post({ type: "stage", name: grow, state: "end", ms: performance.now() - t0 });
  send("connected", false);
  for (let pass = 1; pass <= 5; pass++) {
    const name = `improve the network, pass ${pass}`;
    post({ type: "stage", name, state: "start" });
    t0 = performance.now();
    await checkpoint();
    const moved = session.network_improve();
    post({ type: "stage", name, state: "end", ms: performance.now() - t0 });
    send(`pass ${pass}: ${moved.length} pads rerouted`, false);
    if (!moved.length) break;
  }
  // Kept for the fine stage: the edges in UTM, and the order the pads joined.
  const st = session.network_state() as { pairs: number[]; order: number[] };
  const nr = session.node_rows(), nc = session.node_cols();
  const xy = (v: number): [number, number] => [grid.x0 + (nc[v] + 0.5) * grid.res, grid.y1 - (nr[v] + 0.5) * grid.res];
  const lines: [number, number][][] = [];
  for (let i = 0; i < st.pairs.length; i += 2) lines.push([xy(st.pairs[i]), xy(st.pairs[i + 1])]);
  const order = st.order.map((t) => padTerm.indexOf(t + 1)).filter((i) => i >= 0);
  area.network = { lines, order: [...order, ...pads.map((_, i) => i).filter((i) => !order.includes(i))] };
  send("done", true);
}

/**
 * The pad network on the fine lattice: the DEM read at `res` over the coarse network's edges
 * plus `buffer` m, the pads as level footprints along the contour (the road arriving along their
 * long axis), each pad's branch searched from the network so far in the coarse network's order,
 * then improvement passes. The network is posted after every pad; the searches show as the
 * wavefront.
 */
/** Lay a rigs network out as row runs (see refineNetwork). */
const RIG_RUNS = false;

async function refineNetwork(p: Params, res: number, buffer: number, b: number, length: number, width: number,
                             beside: boolean, rigs: number, byRoad: boolean, approach: number, padOrder: string,
                             strategy: string, padRows: { row: number[]; station: number[] } | null) {
  if (!area?.network || !source) throw new Error("find the network first");
  const { network: net, startXY, zScale, padsXY } = area;
  const [bx0, by0, bx1, by1] = source.info.bbox;
  const pts = [startXY, ...padsXY, ...net.lines.flat()];
  const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
  let len = 0;
  for (const [a, c] of net.lines) len += Math.hypot(c[0] - a[0], c[1] - a[1]);
  const bufferCells = (2 * buffer * len + Math.PI * buffer * buffer * (padsXY.length + 1)) / (res * res);
  const boxCells = (Math.max(...xs) - Math.min(...xs) + 2 * buffer) * (Math.max(...ys) - Math.min(...ys) + 2 * buffer) / (res * res);
  const need = Math.min(bufferCells, boxCells) * FINE_BYTES_PER_CELL + boxCells * FINE_BYTES_PER_GRID_CELL;
  if (need > FINE_BUDGET) {
    throw new Error(`the network at ${res} m needs about ${(need / 1e9).toFixed(1)} GB, over the ${FINE_BUDGET / 1e9} GB ` +
      `the browser search allows: use cells of ${Math.ceil(res * Math.sqrt(need / FINE_BUDGET))} m or more, or a smaller buffer`);
  }
  const x0 = Math.floor(Math.max(Math.min(...xs) - buffer, bx0) / res) * res;
  const y1 = Math.ceil(Math.min(Math.max(...ys) + buffer, by1) / res) * res;
  const fg: Grid = { x0, y1, res, w: Math.ceil((Math.min(Math.max(...xs) + buffer, bx1) - x0) / res),
                     h: Math.ceil((y1 - Math.max(Math.min(...ys) - buffer, by0)) / res) };
  const { z, rough } = await timedAsync(`network: read the DEM at ${res} m`, () =>
    readFine(fg, p, zScale, async (done, total) => {
      progress(`network: read the DEM at ${res} m`, done, total);
      await checkpoint();
    }));
  const toR = (y: number) => (fg.y1 - y) / res - 0.5, toC = (x: number) => (x - fg.x0) / res - 0.5;
  const lrA: number[] = [], lcA: number[] = [];
  for (const line of net.lines) {
    for (const [x, y] of line) { lrA.push(toR(y)); lcA.push(toC(x)); }
    lrA.push(NaN); lcA.push(NaN);
  }
  const lr = new Float64Array(lrA), lc = new Float64Array(lcA), none = new Float64Array(0);
  const paramsJson = JSON.stringify(p);
  const fs = timed(`network: preprocess ${fg.w.toLocaleString()} x ${fg.h.toLocaleString()} cells`, () =>
    new wasm.FineSession(z, fg.w, fg.h, res, paramsJson, lr, lc, buffer, none, none, 1, false, rough));
  try {
    await precomputeMacros(fs, { type: "setup", z, w: fg.w, h: fg.h, res, params: paramsJson, lr, lc, buffer, or: none,
                                 oc: none, apart: 1, rough }, "network", 0, fg);
    const cell = (q: [number, number]) => [Math.floor((fg.y1 - q[1]) / res), Math.floor((q[0] - fg.x0) / res)];
    const [ar, ac] = cell(startXY);
    const axes = fs.network_begin(ar, ac, new Float64Array(padsXY.map((q) => toR(q[1]))),
                                  new Float64Array(padsXY.map((q) => toC(q[0]))), length, width, b, beside, rigs, byRoad, approach);
    post({ type: "padAxes", angles: Array.from(axes) });
    const n = padsXY.length;
    let stage = "", tStage = 0;
    const onProgress = (o: { stage: string; settled: number; key: number; cells: Uint32Array; g: Float32Array;
                             rows: Float64Array; cols: Float64Array }) => {
      try {
        const x = o.cols.map((c) => fg.x0 + (c + 0.5) * res), y = o.rows.map((row) => fg.y1 - (row + 0.5) * res);
        post({ type: "explore", index: 0, grid: fg, stage: o.stage, settled: o.settled, key: o.key, cells: o.cells,
               g: o.g, x, y }, [o.cells.buffer, o.g.buffer, x.buffer, y.buffer]);
      } catch (err) {
        post({ type: "error", message: `progress report: ${err instanceof Error ? err.message : String(err)}` });
      }
      if (o.stage !== stage) {
        if (stage) post({ type: "stage", name: `network: ${stage}`, state: "end", ms: performance.now() - tStage });
        stage = o.stage;
        tStage = performance.now();
        post({ type: "stage", name: `network: ${o.stage}`, state: "start" });
      }
      progress(`network: ${o.stage}, states settled`, o.settled, 0);
    };
    const closeStage = () => {
      if (stage) post({ type: "stage", name: `network: ${stage}`, state: "end", ms: performance.now() - tStage });
      stage = "";
    };
    const sendState = (phase: string, final: boolean) => {
      const st = fs.network_state() as { runs: { pad: number; uses: number; rows: number[]; cols: number[]; z: number[] }[];
        connected: boolean[]; unreachable: boolean[]; beyond: number[]; ends: number[]; construction: number;
        operating: number; switchbacks: number; beside: number; pad_cost: number; dead_ends: number; axes: number[] };
      post({ type: "fineNetwork", res, phase, final, connected: st.connected, unreachable: st.unreachable, beyond: st.beyond,
             ends: st.ends, switchbacks: st.switchbacks, beside: st.beside, pad_cost: st.pad_cost, dead_ends: st.dead_ends,
             axes: st.axes,
             construction: st.construction, operating: st.operating,
             runs: st.runs.map((r) => ({ pad: r.pad, uses: r.uses, z: r.z, x: r.cols.map((c) => fg.x0 + (c + 0.5) * res),
                                         y: r.rows.map((row) => fg.y1 - (row + 0.5) * res) })) });
    };
    let k = 0;
    // RIG_RUNS: with rigs, the grid's rows as a serpentine cut into one run per rig (each run's
    // first pad joins where cheapest, the rest carry its branch on). Off: on ground where the rows
    // cross the contours (pads turned to their own contour) it costs more than the coarse
    // network's order. Otherwise the coarse network's order.
    // Along the rows: the pads as rows along the contours, a snake; each row's first pad joins where
    // cheapest, the rest carry the row's road on.
    if (strategy === "rows" && padRows) fs.network_set_rows(new Int32Array(padRows.row), new Float64Array(padRows.station));
    const runs = strategy === "rows" ? Array.from(fs.network_row_runs())
      : rigs > 0 && RIG_RUNS ? Array.from(fs.network_rig_pieces()) : [];
    // Lowest first: lower pads are on the network before the pads above, which may then switch
    // back on them; otherwise cheapest first, the coarse network's order.
    const elev = (q: [number, number]) => {
      const r = Math.min(fg.h - 1, Math.max(0, Math.round(toR(q[1])))), c = Math.min(fg.w - 1, Math.max(0, Math.round(toC(q[0]))));
      return z[r * fg.w + c];
    };
    const coarseOrder = padOrder === "lowest" ? [...net.order].sort((a, b) => elev(padsXY[a]) - elev(padsXY[b])) : net.order;
    const order: [number, number][] = runs.length
      ? runs.map((p, i) => [p, i > 0 ? runs[i - 1] : -1] as [number, number]).filter(([p]) => p >= 0)
      : coarseOrder.map((p) => [p, -1] as [number, number]);
    for (const [i, [pad, prev]] of order.entries()) {
      progress("network: connect the pads", k, n);
      await checkpoint();
      // Along the rows, a row's road carries on past each pad but its last.
      const keepRoad = strategy === "rows" && i + 1 < order.length && order[i + 1][1] === pad;
      if (prev >= 0) fs.network_connect_after(pad, prev, keepRoad, onProgress);
      else {
        fs.network_beside_only(keepRoad);
        fs.network_connect(pad, onProgress);
        fs.network_beside_only(false);
      }
      closeStage();
      k++;
      sendState(`${k} of ${n} pads searched`, false);
    }
    // Along the rows, the passes (pad by pad) would undo the rows.
    for (let pass = 1; pass <= (strategy === "rows" ? 0 : 3); pass++) {
      await checkpoint();
      const moved = fs.network_improve(onProgress);
      closeStage();
      sendState(`improvement pass ${pass}: ${moved.length} pads rerouted`, false);
      if (!moved.length) break;
    }
    sendState("done", true);
  } finally {
    fs.free();
    stopPool();
  }
}

/**
 * The fine search of each corridor (or the best): the DEM read at `res` over the corridor's
 * buffer, preprocessed there, and searched on the heading lattice with hairpins, bends and fill
 * crossings, as the command line's fine stage does over its corridors.
 */
async function refine(p: Params, res: number, which: "best" | "all", apart: number) {
  if (!area?.routes?.length || !source) throw new Error("find corridors first");
  const buffer = get(p, "tin_buffer");
  const todo = which === "best" ? [0] : area.routes.map((_, i) => i);
  try {
    for (const k of todo) await refineOne(k, p, res, buffer, apart);
  } finally {
    stopPool();
  }
}

async function refineOne(k: number, p: Params, res: number, buffer: number, apart: number) {
  const { grid: cg, startXY, goalXY, zScale } = area!;
  const routes = area!.routes!;
  const [bx0, by0, bx1, by1] = source!.info.bbox;
  const t0 = performance.now();
  const name = `corridor ${k + 1}`;
  const r = routes[k];
  // The corridor's line in UTM, and a grid at `res` over it plus the buffer, within the DEM.
  const xs = r.cols.map((c) => cg.x0 + (c + 0.5) * cg.res), ys = r.rows.map((row) => cg.y1 - (row + 0.5) * cg.res);
  let [xmin, xmax, ymin, ymax] = [Infinity, -Infinity, Infinity, -Infinity];
  for (let i = 0; i < xs.length; i++) {
    xmin = Math.min(xmin, xs[i]); xmax = Math.max(xmax, xs[i]); ymin = Math.min(ymin, ys[i]); ymax = Math.max(ymax, ys[i]);
  }
  // Memory: about FINE_BYTES_PER_CELL per cell of the buffer (macro caches, search state) and
  // FINE_BYTES_PER_GRID_CELL over its bounding box (rasters). Refuse rather than exhaust the tab.
  let lineLength = 0;
  for (let i = 1; i < xs.length; i++) lineLength += Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
  const bufferCells = (2 * buffer * lineLength + Math.PI * buffer * buffer) / (res * res);
  const boxCells = (xmax - xmin + 2 * buffer) * (ymax - ymin + 2 * buffer) / (res * res);
  const need = bufferCells * FINE_BYTES_PER_CELL + boxCells * FINE_BYTES_PER_GRID_CELL;
  if (need > FINE_BUDGET) {
    const fit = res * Math.sqrt(need / FINE_BUDGET);
    throw new Error(`${name} at ${res} m needs about ${(need / 1e9).toFixed(1)} GB (${(bufferCells / 1e6).toFixed(1)} million ` +
      `cells in its ${buffer} m buffer), over the ${FINE_BUDGET / 1e9} GB the browser search allows: use cells of ` +
      `${Math.ceil(fit)} m or more, or a smaller tin_buffer. (Long corridors will be searched in pieces later.)`);
  }
  const x0 = Math.floor(Math.max(xmin - buffer, bx0) / res) * res, y1 = Math.ceil(Math.min(ymax + buffer, by1) / res) * res;
  const fg: Grid = { x0, y1, res, w: Math.ceil((Math.min(xmax + buffer, bx1) - x0) / res),
                     h: Math.ceil((y1 - Math.max(ymin - buffer, by0)) / res) };
  const { z, rough } = await timedAsync(`${name}: read the DEM at ${res} m`, () =>
    readFine(fg, p, zScale, async (done, total) => {
      progress(`${name}: read the DEM at ${res} m`, done, total);
      await checkpoint();
    }));
  const lr = new Float64Array(ys.map((y) => (fg.y1 - y) / res - 0.5));
  const lc = new Float64Array(xs.map((x) => (x - fg.x0) / res - 0.5));
  // The other corridors' lines on this grid (NaN between lines): the ground nearer to them than to
  // this corridor's line costs `apart` times as much in the search (or is closed).
  const or: number[] = [], oc: number[] = [];
  routes.forEach((o, j) => {
    if (j === k) return;
    o.rows.forEach((row, i) => {
      or.push((fg.y1 - (cg.y1 - (row + 0.5) * cg.res)) / res - 0.5);
      oc.push((cg.x0 + (o.cols[i] + 0.5) * cg.res - fg.x0) / res - 0.5);
    });
    or.push(NaN);
    oc.push(NaN);
  });
  const otherRows = new Float64Array(or), otherCols = new Float64Array(oc);
  // The best corridor searches its whole buffer: its route is the best there, whatever the others.
  const own = k === 0 ? 1 : apart;
  const paramsJson = JSON.stringify(p);
  const fs = timed(`${name}: preprocess ${fg.w.toLocaleString()} x ${fg.h.toLocaleString()} cells`, () =>
    new wasm.FineSession(z, fg.w, fg.h, res, paramsJson, lr, lc, buffer, otherRows, otherCols, own, false, rough));
  await checkpoint();
  try {
    await precomputeMacros(fs, { type: "setup", z, w: fg.w, h: fg.h, res, params: paramsJson, lr, lc, buffer,
                                 or: otherRows, oc: otherCols, apart: own, rough }, name, k, fg);
    const cell = (x: number, y: number) => [Math.floor((fg.y1 - y) / res), Math.floor((x - fg.x0) / res)];
    const [sr, sc] = cell(startXY[0], startXY[1]), [gr, gc] = cell(goalXY[0], goalXY[1]);
    let stage = "", tStage = 0;
    const cells = fs.cells(); // the session is busy (borrowed) while it searches
    const toUtm = (o: any, final: boolean): FineOut => ({
      index: k, res, final, x: o.cols.map((c: number) => fg.x0 + (c + 0.5) * res),
      y: o.rows.map((row: number) => fg.y1 - (row + 0.5) * res), z: o.z, ground: o.ground ?? [], cost: o.cost, length: o.length,
      hairpins: o.hairpins, switchbacks: o.switchbacks ?? 0, bends: o.bends, fill_crossings: o.fill_crossings,
      segments: o.segments, history: o.history,
      seconds: (performance.now() - t0) / 1000, cells, macros_computed: o.macros_computed,
    });
    const out = fs.search(sr, sc, gr, gc, (o: { stage: string; settled: number; key: number; cells: Uint32Array;
                                                 g: Float32Array; rows: Float64Array; cols: Float64Array }) => {
      const { stage: st, settled } = o;
      // The search ignores what this callback throws, so report it here.
      try {
        const x = o.cols.map((c) => fg.x0 + (c + 0.5) * res), y = o.rows.map((row) => fg.y1 - (row + 0.5) * res);
        post({ type: "explore", index: k, grid: fg, stage: st, settled, key: o.key, cells: o.cells, g: o.g, x, y },
             [o.cells.buffer, o.g.buffer, x.buffer, y.buffer]);
      } catch (err) {
        post({ type: "error", message: `progress report: ${err instanceof Error ? err.message : String(err)}` });
      }
      if (st !== stage) {
        if (stage) post({ type: "stage", name: `${name}: ${stage}`, state: "end", ms: performance.now() - tStage });
        stage = st;
        tStage = performance.now();
        post({ type: "stage", name: `${name}: ${st}`, state: "start" });
      }
      progress(`${name}: ${st}, states settled`, settled, 0);
    }, (o: any) => post({ type: "fine", route: toUtm(o, false) }));
    if (stage) post({ type: "stage", name: `${name}: ${stage} and the route's costs`, state: "end", ms: performance.now() - tStage });
    if (!out) throw new Error(`${name}: no route at ${res} m within ${buffer} m of the corridor`);
    post({ type: "fine", route: toUtm(out, true) });
  } finally {
    fs.free();
  }
  await checkpoint();
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
      else if (m.type === "network") await network(m.params, m.b, m.refine);
      else if (m.type === "refineNetwork") {
        await refineNetwork(m.params, m.res, m.buffer, m.b, m.length, m.width, m.beside, m.rigs, m.byRoad, m.approach, m.padOrder,
                            m.strategy, m.padRows);
      }
      else if (m.type === "refine") await refine(m.params, m.res, m.which, m.apart);
      else if (m.type === "layout") await layout(m);
      else if (m.type === "channels" && area?.acc) {
        const ch = channelRgba(area.acc, m.aMin);
        post({ type: "raster", layer: "channels", grid: area.grid, rgba: ch }, [ch.buffer]);
      }
    } catch (err) {
      if (m.type === "refine") stopPool();
      if (err instanceof Stopped) post({ type: "stopped", task: task ?? "" });
      else post({ type: "error", message: err instanceof Error ? err.message : String(err) });
    } finally {
      if (task) post({ type: "busy", task: null });
    }
  });
};
