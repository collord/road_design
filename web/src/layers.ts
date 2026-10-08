// Map layers in UTM meters (Leaflet CRS.Simple, lat = northing, lng = easting).
import L from "leaflet";
import type { Grid } from "./source";

export const boundsOf = (g: Grid): L.LatLngBoundsExpression =>
  [[g.y1 - g.h * g.res, g.x0], [g.y1, g.x0 + g.w * g.res]];

/** An RGBA raster on the grid, as an image overlay. */
export function rasterLayer(g: Grid, rgba: Uint8ClampedArray, opacity = 1,
                            options: L.ImageOverlayOptions = {}): L.ImageOverlay {
  const c = document.createElement("canvas");
  c.width = g.w;
  c.height = g.h;
  c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), g.w, g.h), 0, 0);
  return L.imageOverlay(c.toDataURL(), boundsOf(g), { opacity, interactive: false, ...options });
}

/**
 * Many segments or points on one canvas, redrawn when the map moves. Coordinates are UTM;
 * CRS.Simple is linear, so each point is projected with one multiply-add, and anything outside
 * the view is skipped. Colors are drawn in batches, one path each.
 */
export class CanvasLayer extends L.Layer {
  private canvas!: HTMLCanvasElement;
  private mapRef!: L.Map;
  constructor(
    private draw: (ctx: CanvasRenderingContext2D, px: (x: number, y: number) => [number, number],
                   view: [number, number, number, number], metersPerPixel: number) => void,
  ) {
    super();
  }

  onAdd(map: L.Map): this {
    this.mapRef = map;
    this.canvas = L.DomUtil.create("canvas", "leaflet-zoom-hide") as HTMLCanvasElement;
    this.canvas.style.position = "absolute";
    map.getPanes().overlayPane.appendChild(this.canvas);
    map.on("moveend zoomend resize", this.redraw, this);
    this.redraw();
    return this;
  }

  onRemove(map: L.Map): this {
    map.off("moveend zoomend resize", this.redraw, this);
    this.canvas.remove();
    return this;
  }

  redraw(): void {
    const map = this.mapRef;
    const size = map.getSize();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = size.x * dpr;
    this.canvas.height = size.y * dpr;
    this.canvas.style.width = `${size.x}px`;
    this.canvas.style.height = `${size.y}px`;
    L.DomUtil.setPosition(this.canvas, map.containerPointToLayerPoint([0, 0]));
    const ctx = this.canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);
    // The linear map from UTM to container pixels, from two reference points.
    const b = map.getBounds();
    const p0 = map.latLngToContainerPoint([b.getSouth(), b.getWest()]);
    const p1 = map.latLngToContainerPoint([b.getNorth(), b.getEast()]);
    const sx = (p1.x - p0.x) / (b.getEast() - b.getWest()), sy = (p1.y - p0.y) / (b.getNorth() - b.getSouth());
    const ox = p0.x - sx * b.getWest(), oy = p0.y - sy * b.getSouth();
    const px = (x: number, y: number): [number, number] => [ox + sx * x, oy + sy * y];
    this.draw(ctx, px, [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], 1 / Math.abs(sx));
  }
}

/** A color ramp, green (best) to red (twice the best and beyond). */
export function rampColor(t: number): string {
  const u = Math.max(0, Math.min(1, t));
  const r = Math.round(40 + 200 * u), g = Math.round(170 - 120 * u), b = Math.round(60 - 30 * u);
  return `rgb(${r},${g},${b})`;
}
