// Messages between the page and the worker.
import type { Grid, SourceInfo } from "./source";
import type { Params } from "./params";

export type ToWorker =
  | { type: "open"; file?: Blob; url?: string; name: string; zUnits: string; endpointsDem?: [number, number][] }
  | { type: "load"; start: [number, number]; goal: [number, number]; margin: number; res: number; maxCells: number;
      zUnits: string }
  | { type: "run"; params: Params }
  | { type: "channels"; aMin: number }
  | { type: "refine"; params: Params; res: number; which: "best" | "all" }
  | { type: "stop" };

export interface RouteOut {
  rows: number[];
  cols: number[];
  cost: number;
  construction: number;
  operating: number;
  length: number;
  hairpins: number;
  turning: number;
  how: string;
}

/** A corridor's fine route: centerline in UTM with the road's design elevation, and its costs. */
export interface FineOut {
  index: number;
  res: number;
  /** False for a route of one of the searches, before the last. */
  final: boolean;
  x: number[];
  y: number[];
  z: number[];
  /** Ground elevation at each point (the final route; empty while refining). */
  ground: number[];
  cost: { total: number; construction: number; operating: number; base: number; excavation: number; culverts: number;
          crossing_fill: number; fill_m3: number; borrow_m3: number; surrogate: number };
  length: number;
  hairpins: number;
  bends: number;
  fill_crossings: number;
  segments: { kind: string; first: number; length: number; grade: number; cost: number }[];
  history: { iteration: number; lam_fill: number; true_cost: number; settled: number }[];
  seconds: number;
  cells: number;
  macros_computed: number;
}

export type FromWorker =
  | { type: "stage"; name: string; state: "start" | "end"; ms?: number }
  | { type: "info"; info: SourceInfo; endpoints?: [number, number][] }
  | { type: "raster"; layer: "preview" | "hillshade" | "slope" | "channels"; grid: Grid; rgba: Uint8ClampedArray }
  | { type: "points"; layer: "saddles"; grid: Grid; cells: Uint32Array }
  | { type: "tin"; grid: Grid; points: Uint32Array; triangles: Uint32Array; n: number }
  | { type: "graph"; grid: Grid; rows: Uint32Array; cols: Uint32Array; kinds: Uint8Array; field: Float32Array;
      pairs: Uint32Array; costPerM: Float32Array }
  | { type: "corridors"; grid: Grid; routes: RouteOut[]; final: boolean }
  | { type: "progress"; name: string; done: number; total: number }
  | { type: "busy"; task: "open" | "load" | "run" | "refine" | null }
  | { type: "fine"; route: FineOut }
  | { type: "stopped"; task: string }
  | { type: "area"; grid: Grid; cells: number }
  | { type: "error"; message: string };
