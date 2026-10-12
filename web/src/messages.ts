// Messages between the page and the worker.
import type { Grid, SourceInfo } from "./source";
import type { Params } from "./params";

export type ToWorker =
  | { type: "open"; file?: Blob; url?: string; name: string; zUnits: string; endpointsDem?: [number, number][] }
  | { type: "load"; start: [number, number]; goal: [number, number] | null; pads: [number, number][]; margin: number;
      res: number; maxCells: number; zUnits: string; roughMaxRes: number }
  | { type: "network"; params: Params; b: number; refine: number }
  | { type: "refineNetwork"; params: Params; res: number; buffer: number; b: number; length: number; width: number;
      beside: boolean; rigs: number; byRoad: boolean; approach: number; padOrder: string; strategy: string }
  | { type: "run"; params: Params }
  | { type: "channels"; aMin: number }
  | { type: "refine"; params: Params; res: number; which: "best" | "all"; apart: number }
  | { type: "layout"; area: [number, number][]; spacing: number; zUnits: string }
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
  switchbacks: number;
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
  | { type: "busy"; task: "open" | "load" | "run" | "refine" | "network" | "refineNetwork" | "layout" | null }
  /** Drill holes laid out over an area: rows along the contours, then infill. */
  | { type: "holes"; holes: [number, number][]; rows: number; dz: number; slope: number }
  | { type: "fine"; route: FineOut }
  /** The pad network on the graph: its edges (graph node pairs) with the pads each serves, and per
   *  pad (in the page's order) whether it is connected or unreachable and how many pads lie beyond. */
  /** The fine pad network: stretches of road (UTM, design elevation) with the pads using each,
   *  and per pad whether it is connected or unreachable and how many pads lie beyond. */
  | { type: "fineNetwork"; runs: { pad: number; uses: number; x: number[]; y: number[]; z: number[] }[];
      connected: boolean[]; unreachable: boolean[]; beyond: number[]; ends: number[]; construction: number;
      operating: number; switchbacks: number; beside: number; pad_cost: number; dead_ends: number; axes: number[];
      phase: string;
      final: boolean; res: number }
  /** Each pad's long axis (radians, east = 0, counterclockwise): along the contour at its center. */
  | { type: "padAxes"; angles: number[] }
  | { type: "network"; grid: Grid; pairs: number[]; uses: number[]; connected: boolean[]; unreachable: boolean[];
      beyond: number[]; order: number[]; construction: number; operating: number; phase: string; final: boolean }
  /** Hairpin designs computed for some cells (row * w + col of `grid`): the cheapest feasible left
   *  hairpin at each (any heading and leg spacing), Infinity where none fits. */
  | { type: "sites"; index: number; grid: Grid; cells: Uint32Array; cheapest: Float32Array }
  /** A running fine search: cells (row * w + col of `grid`) settled first since the last report and
   *  their cost to come, the search's key, and the route to the state just settled (UTM). */
  | { type: "explore"; index: number; grid: Grid; stage: string; settled: number; key: number;
      cells: Uint32Array; g: Float32Array; x: Float64Array; y: Float64Array }
  | { type: "stopped"; task: string }
  | { type: "area"; grid: Grid; cells: number; roughness: string }
  | { type: "error"; message: string };
