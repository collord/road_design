// The browser section model (roadsearch/evaluator.js), loaded for its side effect.
export {};

declare global {
  interface SectionResult {
    cost: number;
    catch: number;
  }
  interface RoadEvalApi {
    sectionCore(s: number, d: number, sp: Record<string, number>): SectionResult;
    cutGeometry(a: number, s: number, d: number, cSoil: number, cRock: number, catchMax: number): [number, number, number];
  }
  // eslint-disable-next-line no-var
  var RoadEval: RoadEvalApi;
}
