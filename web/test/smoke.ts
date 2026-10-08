// The browser pipeline under Node: the GeoTIFF reader, the parameters and the WebAssembly core,
// as the worker drives them, on the bundled examples. Run: node test/smoke.ts (from web/).
import { readFileSync } from "node:fs";
import init, * as wasm from "../src/wasm/roadsearch_wasm.js";
import { Source, Z_UNIT_SCALE } from "../src/source.ts";
import { get, hairpinAllowance, maxCatchAtMask, sectionTable, sigmaM, sMask, TABLE_NG, TABLE_NS } from "../src/params.ts";

await init({ module_or_path: readFileSync(new URL("../src/wasm/roadsearch_wasm_bg.wasm", import.meta.url)) });
const examples = JSON.parse(readFileSync(new URL("../public/examples.json", import.meta.url), "utf-8"));
const defaults = JSON.parse(readFileSync(new URL("../public/params.json", import.meta.url), "utf-8"));
let failed = false;
const check = (ok: boolean, what: string) => { console.log(`  ${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed = true; };

for (const ex of examples) {
  console.log(`${ex.name}: ${ex.dem}`);
  const t0 = performance.now();
  const buf = readFileSync(new URL(`../public/${ex.dem}`, import.meta.url));
  const src = await Source.open(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), ex.title);
  const zs = Z_UNIT_SCALE[ex.params.z_units] ?? 1;
  console.log(`  ${src.info.width} x ${src.info.height}, zone ${src.info.zone}, ${src.info.units}, ~${src.info.resM.toFixed(1)} m`);
  const pv = await src.preview(wasm.resample, zs, 600);
  const finite = pv.z.filter(Number.isFinite).length;
  check(finite > 0.5 * pv.z.length, `preview ${pv.grid.w} x ${pv.grid.h}, ${(100 * finite / pv.z.length).toFixed(0)}% with data`);
  const [s, g] = [src.utm(ex.start[0], ex.start[1]), src.utm(ex.goal[0], ex.goal[1])];
  const p = { ...defaults, ...ex.params };
  const res = 10, m = 1500;
  const x0 = Math.floor(Math.max(Math.min(s[0], g[0]) - m, src.info.bbox[0]) / res) * res;
  const y1 = Math.ceil(Math.min(Math.max(s[1], g[1]) + m, src.info.bbox[3]) / res) * res;
  const w = Math.ceil((Math.min(Math.max(s[0], g[0]) + m, src.info.bbox[2]) - x0) / res);
  const h = Math.ceil((y1 - Math.max(Math.min(s[1], g[1]) - m, src.info.bbox[1])) / res);
  const grid = { x0, y1, res, w, h };
  const z = await src.resampleTo(grid, wasm.resample, zs);
  const zf = z.filter(Number.isFinite);
  check(zf.length > 0.9 * z.length, `area ${w} x ${h} at ${res} m, ${(100 * zf.length / z.length).toFixed(0)}% with data`);
  const session = new wasm.Session(z, w, h, res);
  session.prepare(sigmaM(p) / res, get(p, "g_max"), sMask(p), maxCatchAtMask(p));
  const open = session.open_mask();
  check(open.some((v: number) => v === 1), `slope mask: ${(100 * open.filter((v: number) => !v).length / open.length).toFixed(1)}% excluded`);
  const area = session.contributing_area();
  let amax = 0;
  for (const v of area) if (v > amax) amax = v;
  check(amax > 10, `flow accumulation: up to ${amax.toFixed(0)} ha`);
  const nt = session.triangulate(get(p, "tin_tolerance"));
  check(nt > 100, `triangulation: ${nt} triangles`);
  const { table, ds, dg } = sectionTable(p);
  const edge = new Float64Array([get(p, "g_max"), get(p, "C_travel"), get(p, "C_vertical"), get(p, "A_min"),
                                 hairpinAllowance(p), get(p, "C_turn")]);
  const tin = new Float64Array([get(p, "tin_tolerance"), get(p, "tin_k"), get(p, "tin_edge_max") / res,
                                get(p, "tin_block") / res, get(p, "tin_channel_ha"), get(p, "tin_alternatives"),
                                get(p, "tin_penalty"), get(p, "tin_stretch"), get(p, "tin_max_shared"),
                                get(p, "tin_separation") / res]);
  const sr = Math.floor((y1 - s[1]) / res), sc = Math.floor((s[0] - x0) / res);
  const gr = Math.floor((y1 - g[1]) / res), gc = Math.floor((g[0] - x0) / res);
  const routes = session.corridors(sr, sc, gr, gc, edge, new Float64Array(p.culvert_area_breaks),
                                   new Float64Array(p.culvert_costs), tin, table, TABLE_NS, TABLE_NG, ds, dg);
  check(routes.length >= 1 && Number.isFinite(routes[0].cost),
        `corridors: ${routes.length}, best ${Math.round(routes[0]?.cost ?? NaN).toLocaleString()} ` +
        `(${((routes[0]?.length ?? 0) / 1000).toFixed(1)} km), ${session.node_rows().length} nodes`);
  console.log(`  ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  session.free();
}
process.exit(failed ? 1 : 0);
