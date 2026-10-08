// Messages between the page and the worker.
import type { Grid, SourceInfo } from "./source";
import type { Params } from "./params";

export type ToWorker =
  | { type: "open"; file?: Blob; url?: string; name: string; zUnits: string; endpointsDem?: [number, number][] }
  | { type: "load"; start: [number, number]; goal: [number, number]; margin: number; res: number; maxCells: number;
      zUnits: string }
  | { type: "run"; params: Params }
  | { type: "channels"; aMin: number };

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

export type FromWorker =
  | { type: "stage"; name: string; state: "start" | "end"; ms?: number }
  | { type: "info"; info: SourceInfo; endpoints?: [number, number][] }
  | { type: "raster"; layer: "preview" | "hillshade" | "slope" | "channels"; grid: Grid; rgba: Uint8ClampedArray }
  | { type: "points"; layer: "saddles"; grid: Grid; cells: Uint32Array }
  | { type: "tin"; grid: Grid; points: Uint32Array; triangles: Uint32Array; n: number }
  | { type: "graph"; grid: Grid; rows: Uint32Array; cols: Uint32Array; kinds: Uint8Array; field: Float32Array;
      pairs: Uint32Array; costPerM: Float32Array }
  | { type: "corridors"; grid: Grid; routes: RouteOut[] }
  | { type: "area"; grid: Grid; cells: number }
  | { type: "error"; message: string };
