// A DEM in a GeoTIFF, read a window at a time: never the whole file.
//
// The georeferencing (any CRS a GeoTIFF declares: latitude/longitude, State Plane, UTM...) is
// turned into a proj4 definition. The working grid is UTM (WGS 84 zones, as the command line
// uses) at the zone of the DEM's center. Resampling onto it goes through a coarse lattice of
// projected positions (the Rust core interpolates between them and samples the DEM).
import { fromArrayBuffer, fromBlob, type GeoTIFF, type GeoTIFFImage } from "geotiff";
import proj4, { type Converter } from "proj4";
import { toProj4 } from "geotiff-geokeys-to-proj4";

export const LATTICE_STEP = 16;

/** Meters per unit of the DEM's elevations (roadsearch/params.py Z_UNIT_SCALE). */
export const Z_UNIT_SCALE: Record<string, number> = { m: 1, ft: 0.3048, us_ft: 1200 / 3937 };

export interface SourceInfo {
  name: string;
  width: number;
  height: number;
  geographic: boolean;
  units: string;
  zone: number;
  north: boolean;
  /** The DEM's footprint in the working UTM zone: [xmin, ymin, xmax, ymax] (m). */
  bbox: [number, number, number, number];
  /** Approximate native cell size (m). */
  resM: number;
  overviews: number;
  tiled: boolean;
  warnings: string[];
}

export interface Grid {
  /** Top-left corner and cell size (m); w x h cells. */
  x0: number;
  y1: number;
  res: number;
  w: number;
  h: number;
}

type Resample = (src: Float32Array, sw: number, sh: number, lc: Float32Array, lr: Float32Array, lw: number,
                 lh: number, step: number, w: number, h: number, k: number) => Float32Array;

/** Roughness (m) of the DEM's native pixels, w x h (roughness.py). */
type Roughness = (z: Float32Array, w: number, h: number) => Float32Array;

/** Extra source pixels read around a window for roughness: its smoothing's reach (sigma 2 px). */
const ROUGH_MARGIN = 10;


export class Source {
  info!: SourceInfo;
  private tiff!: GeoTIFF;
  private image!: GeoTIFFImage;
  private levels: GeoTIFFImage[] = [];
  private origin!: number[];
  private resolution!: number[];
  private conv = { x: 1, y: 1, z: 1 };
  private toUtm!: Converter;
  private nodata: number | null = null;

  static async open(input: Blob | ArrayBuffer, name: string): Promise<Source> {
    const s = new Source();
    s.tiff = input instanceof ArrayBuffer ? await fromArrayBuffer(input) : await fromBlob(input);
    s.image = await s.tiff.getImage(0);
    const n = await s.tiff.getImageCount();
    for (let i = 1; i < n; i++) s.levels.push(await s.tiff.getImage(i));
    s.origin = s.image.getOrigin();
    s.resolution = s.image.getResolution();
    const warnings: string[] = [];
    const keys = s.image.getGeoKeys();
    if (!keys) throw new Error(`${name} has no georeferencing (no GeoTIFF keys): reproject it first`);
    const pj = toProj4(keys);
    for (const [k, v] of Object.entries(pj.errors)) warnings.push(`georeferencing: ${k} ${v}`);
    s.conv = pj.conversionParameters;
    const nd = s.image.getGDALNoData();
    s.nodata = nd === null || nd === undefined ? null : Number(nd);
    if (s.image.getSamplesPerPixel() > 1) warnings.push("several bands: using band 1");
    const [w, h] = [s.image.getWidth(), s.image.getHeight()];
    // UTM zone at the center.
    const toWgs = proj4(pj.proj4, "WGS84");
    const center = toWgs.forward(s.srcPoint(w / 2, h / 2));
    const zone = Math.floor((center[0] + 180) / 6) % 60 + 1;
    const north = center[1] >= 0;
    s.toUtm = proj4(pj.proj4, `+proj=utm +zone=${zone}${north ? "" : " +south"} +datum=WGS84 +units=m +no_defs`);
    // Footprint: the projected outline, sampled along the edges.
    let [xmin, ymin, xmax, ymax] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i <= 32; i++) {
      for (const [c, r] of [[i * w / 32, 0], [i * w / 32, h], [0, i * h / 32], [w, i * h / 32]]) {
        const [x, y] = s.toUtm.forward(s.srcPoint(c, r));
        xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
      }
    }
    const lonSpan = Math.abs(toWgs.forward(s.srcPoint(0, h / 2))[0] - toWgs.forward(s.srcPoint(w, h / 2))[0]);
    if (lonSpan > 6) warnings.push(`the DEM spans ${lonSpan.toFixed(1)} degrees of longitude: more than one UTM zone; using zone ${zone}`);
    s.info = {
      name, width: w, height: h, geographic: pj.isGCS, units: String(pj.coordinatesUnits).replace("metre", "meter"), zone, north,
      bbox: [xmin, ymin, xmax, ymax], resM: (xmax - xmin) / w, overviews: s.levels.length,
      tiled: s.image.isTiled, warnings,
    };
    return s;
  }

  /** The source CRS coordinates (in proj4's units) of a pixel position (col, row; edges at integers). */
  private srcPoint(col: number, row: number): [number, number] {
    const x = this.origin[0] + col * this.resolution[0];
    const y = this.origin[1] + row * this.resolution[1];
    return [x * this.conv.x, y * this.conv.y];
  }

  /** A UTM point in the working zone. */
  utm(x: number, y: number): [number, number] {
    const p = this.toUtm.forward([x * this.conv.x, y * this.conv.y]);
    return [p[0], p[1]];
  }

  /** Longitude and latitude to UTM in the working zone. */
  lonlatToUtm(lon: number, lat: number): [number, number] {
    const p = proj4("WGS84", this.toUtmDef(), [lon, lat]);
    return [p[0], p[1]];
  }

  utmToLonlat(x: number, y: number): [number, number] {
    const p = proj4(this.toUtmDef(), "WGS84", [x, y]);
    return [p[0], p[1]];
  }

  private toUtmDef(): string {
    return `+proj=utm +zone=${this.info.zone}${this.info.north ? "" : " +south"} +datum=WGS84 +units=m +no_defs`;
  }

  /** Source pixel position (col, row; pixel centers at integers) of a UTM point. */
  private pixelOf(e: number, n: number, origin: number[], resolution: number[]): [number, number] {
    const p = this.toUtm.inverse([e, n]);
    const x = p[0] / this.conv.x, y = p[1] / this.conv.y;
    return [(x - origin[0]) / resolution[0] - 0.5, (y - origin[1]) / resolution[1] - 0.5];
  }

  private async read(image: GeoTIFFImage, window: [number, number, number, number]): Promise<Float32Array> {
    const r = await image.readRasters({ window, samples: [0], interleave: true });
    const a = r as unknown as ArrayLike<number>;
    const out = new Float32Array(a.length);
    const nd = this.nodata;
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      out[i] = nd !== null && v === nd ? NaN : v;
    }
    return out;
  }

  /** Resample (part of) a level onto a UTM grid, reading only the pixels it needs. */
  async resampleTo(g: Grid, resample: Resample, zScale: number, level = -1): Promise<Float32Array> {
    return (await this.sample(g, resample, zScale, level)).z;
  }

  /** As resampleTo, and with `roughness` the roughness of the pixels read, averaged onto the grid. */
  private async sample(g: Grid, resample: Resample, zScale: number, level: number,
                       roughness?: Roughness): Promise<{ z: Float32Array; rough: Float32Array | null }> {
    const image = level < 0 ? this.image : this.levels[level];
    const scale = this.image.getWidth() / image.getWidth();
    const origin = this.origin, res = [this.resolution[0] * scale, this.resolution[1] * scale];
    const [iw, ih] = [image.getWidth(), image.getHeight()];
    const step = LATTICE_STEP;
    const lw = Math.ceil((g.w - 1) / step) + 2, lh = Math.ceil((g.h - 1) / step) + 2;
    const lc = new Float32Array(lw * lh), lr = new Float32Array(lw * lh);
    let [c0, r0, c1, r1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < lh; i++) {
      for (let j = 0; j < lw; j++) {
        const e = g.x0 + (j * step + 0.5) * g.res, n = g.y1 - (i * step + 0.5) * g.res;
        const [c, r] = this.pixelOf(e, n, origin, res);
        lc[i * lw + j] = c; lr[i * lw + j] = r;
        c0 = Math.min(c0, c); c1 = Math.max(c1, c); r0 = Math.min(r0, r); r1 = Math.max(r1, r);
      }
    }
    // The source window: the lattice's extent plus a margin, within the image.
    const m = roughness ? ROUGH_MARGIN : 0;
    const W: [number, number, number, number] = [
      Math.max(0, Math.floor(c0) - 2 - m), Math.max(0, Math.floor(r0) - 2 - m),
      Math.min(iw, Math.ceil(c1) + 3 + m), Math.min(ih, Math.ceil(r1) + 3 + m),
    ];
    if (W[2] <= W[0] || W[3] <= W[1]) {
      const none = () => new Float32Array(g.w * g.h).fill(NaN);
      return { z: none(), rough: roughness ? none() : null };
    }
    const src = await this.read(image, W);
    for (let q = 0; q < lc.length; q++) { lc[q] -= W[0]; lr[q] -= W[1]; }
    if (zScale !== 1) for (let q = 0; q < src.length; q++) src[q] *= zScale;
    const srcM = this.info.resM * scale;
    const k = Math.max(1, Math.min(8, Math.ceil(g.res / srcM)));
    const [sw, sh] = [W[2] - W[0], W[3] - W[1]];
    const rough = roughness ? resample(roughness(src, sw, sh), sw, sh, lc, lr, lw, lh, step, g.w, g.h, k) : null;
    return { z: resample(src, sw, sh, lc, lr, lw, lh, step, g.w, g.h, k), rough };
  }

  /**
   * The DEM on grid g, read and resampled in horizontal bands so each read covers only its rows,
   * calling onBand(rows done, rows) after each (it may throw to stop). Bands are whole multiples
   * of the projection lattice, so the cells are those of a single read. About `pixelsPerBand`
   * source pixels per band, and at least `minBands` bands when the grid is tall enough.
   */
  async resampleBands(g: Grid, resample: Resample, zScale: number, level = -1,
                      onBand?: (done: number, total: number) => Promise<void> | void,
                      pixelsPerBand = 4_000_000, minBands = 10): Promise<Float32Array> {
    return (await this.bands(g, resample, zScale, level, onBand, pixelsPerBand, minBands)).z;
  }

  /** As resampleBands at the native resolution, with the roughness of the native pixels (m)
   *  averaged onto the grid as well. */
  async resampleWithRoughness(g: Grid, resample: Resample, roughness: Roughness, zScale: number,
                              onBand?: (done: number, total: number) => Promise<void> | void)
                              : Promise<{ z: Float32Array; rough: Float32Array }> {
    const r = await this.bands(g, resample, zScale, -1, onBand, 4_000_000, 10, roughness);
    return { z: r.z, rough: r.rough! };
  }

  private async bands(g: Grid, resample: Resample, zScale: number, level: number,
                      onBand: ((done: number, total: number) => Promise<void> | void) | undefined,
                      pixelsPerBand: number, minBands: number, roughness?: Roughness)
                      : Promise<{ z: Float32Array; rough: Float32Array | null }> {
    const image = level < 0 ? this.image : this.levels[level];
    const srcM = this.info.resM * this.image.getWidth() / image.getWidth();
    const ratio = Math.max(g.res / srcM, 1e-9);
    const byPixels = pixelsPerBand / Math.max(1, g.w * ratio * ratio);
    const rows = Math.min(byPixels, Math.ceil(g.h / minBands));
    const band = Math.max(1, Math.ceil(rows / LATTICE_STEP)) * LATTICE_STEP;
    const z = new Float32Array(g.w * g.h), rough = roughness ? new Float32Array(g.w * g.h) : null;
    for (let i0 = 0; i0 < g.h; i0 += band) {
      const bh = Math.min(band, g.h - i0);
      const s = await this.sample({ ...g, y1: g.y1 - i0 * g.res, h: bh }, resample, zScale, level, roughness);
      z.set(s.z, i0 * g.w);
      if (rough) rough.set(s.rough!, i0 * g.w);
      await onBand?.(i0 + bh, g.h);
    }
    return { z, rough };
  }

  /** A preview grid over the whole DEM about `maxSide` cells across, from the coarsest overview
   *  that is fine enough, or the full image (read band by band through the resampler's window). */
  async preview(resample: Resample, zScale: number, maxSide = 1200,
                onBand?: (done: number, total: number) => Promise<void> | void): Promise<{ grid: Grid; z: Float32Array }> {
    const [xmin, ymin, xmax, ymax] = this.info.bbox;
    const res = Math.max((xmax - xmin) / maxSide, (ymax - ymin) / maxSide, this.info.resM);
    const g: Grid = { x0: xmin, y1: ymax, res, w: Math.ceil((xmax - xmin) / res), h: Math.ceil((ymax - ymin) / res) };
    // The coarsest overview still at least as fine as the preview.
    let level = -1;
    this.levels.forEach((im, i) => {
      if (this.info.resM * this.image.getWidth() / im.getWidth() <= res) level = i;
    });
    if (level >= 0) return { grid: g, z: await this.resampleTo(g, resample, zScale, level) };
    // No overview: the full image, band by band (about 4 million source pixels, 16 MB, each).
    return { grid: g, z: await this.resampleBands(g, resample, zScale, -1, onBand) };
  }
}
