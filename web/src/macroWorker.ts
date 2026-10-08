// One of the fine search's helpers: on the same corridor grid as the searching worker, computes
// hairpin and bend costs for slices of its cells, in parallel with the other helpers.
import init, * as wasm from "./wasm/roadsearch_wasm.js";

export type ToMacro =
  | { type: "setup"; z: Float32Array; w: number; h: number; res: number; params: string; lr: Float64Array;
      lc: Float64Array; buffer: number; or: Float64Array; oc: Float64Array; apart: number }
  | { type: "rows"; a: number; b: number };

export type FromMacro =
  | { type: "ready"; cells: number }
  | { type: "rows"; a: number; hp: Float32Array; bend: Float32Array }
  | { type: "error"; message: string };

const ready = init();
let session: wasm.FineSession | null = null;
const post = (m: FromMacro, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = async (e: MessageEvent<ToMacro>) => {
  await ready;
  const m = e.data;
  try {
    if (m.type === "setup") {
      session?.free();
      session = new wasm.FineSession(m.z, m.w, m.h, m.res, m.params, m.lr, m.lc, m.buffer, m.or, m.oc, m.apart, true);
      post({ type: "ready", cells: session.cells() });
    } else if (m.type === "rows" && session) {
      const o = session.macro_rows(m.a, m.b) as { hp: Float32Array; bend: Float32Array };
      post({ type: "rows", a: m.a, hp: o.hp, bend: o.bend }, [o.hp.buffer, o.bend.buffer]);
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
