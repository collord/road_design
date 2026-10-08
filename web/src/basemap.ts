// Base maps from ArcGIS map services, drawn by the server straight into the DEM's UTM zone
// (the export operation reprojects), so no Web Mercator tiles need warping. One image per view.
import L from "leaflet";

export interface Basemap { name: string; url: string; attribution: string; format: "jpg" | "png32" }

const USGS = "https://basemap.nationalmap.gov/arcgis/rest/services";
const ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services";
const USGS_CREDIT = '<a href="https://www.usgs.gov/programs/national-geospatial-program/national-map">USGS The National Map</a>';

export const BASEMAPS: Basemap[] = [
  { name: "USGS imagery", url: `${USGS}/USGSImageryOnly/MapServer`, attribution: USGS_CREDIT, format: "jpg" },
  { name: "USGS imagery and topo", url: `${USGS}/USGSImageryTopo/MapServer`, attribution: USGS_CREDIT, format: "jpg" },
  { name: "USGS topo", url: `${USGS}/USGSTopo/MapServer`, attribution: USGS_CREDIT, format: "jpg" },
  { name: "Esri streets", url: `${ESRI}/World_Street_Map/MapServer`, format: "jpg",
    attribution: "Esri, HERE, Garmin, USGS, NGA, &copy; OpenStreetMap contributors" },
  { name: "Esri imagery", url: `${ESRI}/World_Imagery/MapServer`, format: "jpg",
    attribution: "Esri, Maxar, Earthstar Geographics, and the GIS User Community" },
];

/** EPSG code of WGS 84 / UTM: 326zz north, 327zz south. */
export const utmEpsg = (zone: number, north: boolean) => (north ? 32600 : 32700) + zone;

/** The map service's image of the current view, refreshed when the view settles. */
export class ExportLayer extends L.Layer {
  private mapRef?: L.Map;
  private shown?: L.ImageOverlay;
  private loading?: L.ImageOverlay;
  private timer?: ReturnType<typeof setTimeout>;
  constructor(private base: Basemap, private epsg: number, private pane: string, private onError: (msg: string) => void) {
    super();
    L.setOptions(this, { attribution: base.attribution });
  }

  onAdd(map: L.Map): this {
    this.mapRef = map;
    map.on("moveend", this.schedule, this);
    this.update();
    return this;
  }

  onRemove(map: L.Map): this {
    map.off("moveend", this.schedule, this);
    clearTimeout(this.timer);
    this.shown?.remove();
    this.loading?.remove();
    this.shown = this.loading = undefined;
    return this;
  }

  private schedule(): void {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.update(), 150);
  }

  private update(): void {
    const map = this.mapRef;
    if (!map) return;
    const b = map.getBounds(), size = map.getSize();
    if (!size.x || !size.y) return;
    // Device pixels, up to 2x, within the 4096 px that map services allow.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const scale = Math.min(dpr, 4096 / Math.max(size.x, size.y));
    const w = Math.round(size.x * scale), h = Math.round(size.y * scale);
    const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(1)).join(",");
    const url = `${this.base.url}/export?bbox=${bbox}&bboxSR=${this.epsg}&imageSR=${this.epsg}` +
      `&size=${w},${h}&format=${this.base.format}&transparent=false&f=image`;
    // The new image loads over the old one, which stays until it is ready.
    this.loading?.remove();
    const next = L.imageOverlay(url, b, { pane: this.pane, zIndex: 1, interactive: false });
    this.loading = next;
    next.once("load", () => {
      if (this.loading !== next) return;
      this.shown?.remove();
      this.shown = next;
      this.loading = undefined;
    });
    next.once("error", () => {
      if (this.loading !== next) return;
      next.remove();
      this.loading = undefined;
      this.onError(`${this.base.name}: the map service did not answer`);
    });
    next.addTo(map);
  }
}
