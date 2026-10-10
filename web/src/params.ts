// Parameters for the pipeline: the full set from params.json / an example's set, and what the
// search derives from them (as roadsearch/params.py, section.py and search.py do).
import "../../roadsearch/evaluator.js";

export type Params = Record<string, number | number[] | string | null>;

const num = (p: Params, k: string): number => {
  const v = p[k];
  if (v === null) return Infinity; // JSON has no infinity: null stands for it
  if (typeof v !== "number") throw new Error(`parameter ${k} is not a number`);
  return v;
};

export const sMask = (p: Params) => num(p, "mask_margin") * num(p, "c_rock");
export const sigmaM = (p: Params) => (num(p, "smooth_sigma") > 0 ? num(p, "smooth_sigma") : num(p, "w"));

export function sectionParams(p: Params): Record<string, number> {
  const keys = ["w", "c_soil", "c_rock", "f_soil", "f_rock", "k_soil", "k_rock", "b_min", "s_flat", "C_base",
                "C_soil", "C_rock", "catch_max"];
  const sp: Record<string, number> = {};
  for (const k of keys) sp[k] = num(p, k);
  sp.s_mask = sMask(p);
  return sp;
}

/** Vertical soil depth at terrain slope S (section.py soil_depth). */
export function soilDepth(S: number, p: Params): number {
  return S >= num(p, "c_soil") ? 0 : num(p, "d0") * Math.exp(-S / num(p, "S0"));
}

/** Largest cut catch distance at the mask threshold (section.py max_catch_at_mask): the slope
 *  mask is dilated by it. */
export function maxCatchAtMask(p: Params): number {
  const s = sMask(p);
  let worst = 0;
  for (const d of [0, soilDepth(s, p)]) {
    const [, , c] = RoadEval.cutGeometry(num(p, "w"), s, s < num(p, "c_soil") ? d : 0, num(p, "c_soil"),
                                         num(p, "c_rock"), num(p, "catch_max"));
    worst = Math.max(worst, c);
  }
  return worst;
}

export const TABLE_NS = 4097;
export const TABLE_NG = 21;

/** Roughness (m) over which soil thins by e (roughness.py); 0, or absent from a parameter set: off. */
export const roughR0 = (p: Params) => (typeof p.rough_r0 === "number" ? p.rough_r0 : 0);

/** Coarsest DEM pixels (m) roughness is read from; 3 when a parameter set lacks it. */
export const roughMaxRes = (p: Params) => (typeof p.rough_max_res === "number" ? p.rough_max_res : 3);

/** Soil slope added per meter of roughness (roughness.py rough_k); 0: off. */
export const roughK = (p: Params) => (roughR0(p) > 0 ? num(p, "S0") / roughR0(p) : 0);

/** Cost per meter over (sidehill slope, |grade|), TABLE_NS x ng (search.py build_cost_table):
 *  with roughness on, the grade axis runs on at the same step to c_soil, for the soil-effective
 *  grade. */
export function sectionTable(p: Params): { table: Float64Array; ng: number; ds: number; dg: number } {
  const sm = sMask(p), gm = num(p, "g_max"), sp = sectionParams(p);
  const ds = sm / (TABLE_NS - 1), dg = gm / (TABLE_NG - 1);
  const ng = roughR0(p) > 0 ? Math.max(TABLE_NG, Math.ceil(num(p, "c_soil") / dg) + 2) : TABLE_NG;
  const table = new Float64Array(TABLE_NS * ng);
  for (let i = 0; i < TABLE_NS; i++) {
    for (let j = 0; j < ng; j++) {
      const s = i * ds, g = j * dg;
      table[i * ng + j] = i === TABLE_NS - 1 ? Infinity
        : RoadEval.sectionCore(s, soilDepth(Math.hypot(s, g), p), sp).cost;
    }
  }
  return { table, ng, ds, dg };
}

/** Landing length (m) of reversing switchbacks; 0, or absent from a parameter set: loop hairpins. */
export const switchbackLanding = (p: Params) => (typeof p.switchback_landing === "number" ? p.switchback_landing : 0);

/** Allowance per hairpin a steep graph edge implies (tin.py hairpin_allowance); with reversing
 *  switchbacks, the landing's base cost and the turn. */
export function hairpinAllowance(p: Params): number {
  const h = num(p, "tin_hairpin_cost");
  if (h >= 0) return h;
  if (switchbackLanding(p) > 0) return num(p, "C_base") * switchbackLanding(p) + num(p, "C_turn") * Math.PI;
  return (num(p, "C_base") + num(p, "C_travel")) * (Math.PI + 2) * num(p, "R_hp") + num(p, "C_turn") * Math.PI;
}

export const get = num;
