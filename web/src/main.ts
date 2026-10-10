// The page: controls, the map (Leaflet in UTM meters) and the worker that runs the pipeline.
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import proj4 from "proj4";
import "./style.css";
import { BASEMAPS, ExportLayer, utmEpsg } from "./basemap";
import { Candidate, HairpinSites, KeyChart, Wavefront } from "./explore";
import { profileSummary, profileSvg } from "./profile";
import { boundsOf, CanvasLayer, rampColor, rasterLayer } from "./layers";
import type { FineOut, FromWorker, RouteOut, ToWorker } from "./messages";
import { roughMaxRes, type Params } from "./params";
import type { Grid, SourceInfo } from "./source";

const MAX_CELLS = 12_000_000;

interface Example { name: string; title: string; heading: string; dem: string; start: number[]; goal: number[]; params: Params }

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const map = L.map("map", { crs: L.CRS.Simple, minZoom: -10, maxZoom: 5, zoomSnap: 0.25 });
map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
map.setView([0, 0], 0);
// The base map and the hillshades share a pane, so the hillshade can multiply onto the base map.
map.createPane("terrain").style.zIndex = "250";
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const send = (m: ToWorker) => worker.postMessage(m);

let defaults: Params = {};
let base: Params = {};             // the parameter set before the controls (an example's, or the defaults)
let info: SourceInfo | null = null;
let start: [number, number] | null = null;
let goal: [number, number] | null = null;
let picking: "start" | "goal" | "pads" | null = null;
let mode: "route" | "network" = "route";
/** Pad centers (UTM). */
const pads: [number, number][] = [];
const padLayer = L.layerGroup();
let padStatus: { connected: boolean[]; unreachable: boolean[]; beyond: number[]; ends?: number[] } | null = null;
/** Each pad's long axis (radians, east = 0), along the contour, once an area is prepared. */
let padAxes: number[] | null = null;
let lastGraph: { grid: Grid; rows: Uint32Array; cols: Uint32Array } | null = null;
let areaGrid: Grid | null = null;
let haveCorridors = false;
let haveNetwork = false;
let fineNet: Extract<FromWorker, { type: "fineNetwork" }> | null = null;
const fineRoutes: FineOut[] = [];
let profileIndex: number | null = null;   // corridor whose fine route the profile shows
// The running fine search: its wavefront (one per corridor), leading candidate and key chart.
const wavefronts = new Map<number, Wavefront>();
const sites = new Map<number, HairpinSites>();
let sitesPending = false;
const candidate = new Candidate();
const keyChart = new KeyChart();
let exploreStage = "";
let exploreT0 = 0;
let explorePending = false;
const layers = new Map<string, L.Layer>();
const markers: { start?: L.CircleMarker; goal?: L.CircleMarker } = {};

// ---------------------------------------------------------------- layers panel

function setLayer(name: string, layer: L.Layer | null, visible = true) {
  const old = layers.get(name);
  if (old) map.removeLayer(old);
  layers.delete(name);
  let box = document.querySelector<HTMLInputElement>(`#layers input[data-name="${CSS.escape(name)}"]`);
  if (!layer) {
    box?.parentElement?.remove();
    return;
  }
  layers.set(name, layer);
  if (!box) {
    const label = document.createElement("label");
    box = document.createElement("input");
    box.type = "checkbox";
    box.dataset.name = name;
    box.addEventListener("change", () => {
      const l = layers.get(name);
      if (!l) return;
      if (box!.checked) l.addTo(map); else map.removeLayer(l);
    });
    label.append(box, document.createTextNode(name));
    $("layers").append(label);
  }
  box.checked = visible;
  if (visible) layer.addTo(map);
}

function log(text: string, cls = "") {
  const li = document.createElement("li");
  li.textContent = text;
  if (cls) li.className = cls;
  $("log").append(li);
  li.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- status: what the worker is doing

const TASK_TEXT = { open: "Opening the DEM", load: "Loading the area", run: "Finding corridors", refine: "Refining corridors",
                    network: "Building the pad network", refineNetwork: "Refining the pad network" } as const;
let busy: keyof typeof TASK_TEXT | null = null;
let busySince = 0;
let ticker: ReturnType<typeof setInterval> | undefined;

const statusBox = L.DomUtil.create("div", "status");
statusBox.hidden = true;
statusBox.innerHTML = '<div class="status-head"><span class="spinner"></span><strong class="status-task"></strong>' +
  '<span class="status-time"></span><button type="button" class="status-stop">Stop</button></div>' +
  '<div class="status-step"></div><progress class="status-bar"></progress><div class="status-plot"></div>';
L.DomEvent.disableClickPropagation(statusBox);
const statusEl = <T extends Element>(cls: string) => statusBox.querySelector(`.${cls}`) as T;
new (L.Control.extend({ onAdd: () => statusBox }))({ position: "topright" }).addTo(map);
statusEl<HTMLButtonElement>("status-stop").addEventListener("click", () => {
  send({ type: "stop" });
  statusEl("status-step").textContent = "stopping at the next checkpoint...";
});

function setBusy(task: keyof typeof TASK_TEXT | null) {
  busy = task;
  const buttons = { load: $<HTMLButtonElement>("load"), run: $<HTMLButtonElement>("run"),
                    refine: $<HTMLButtonElement>("refine") };
  const clicked = task === "refineNetwork" ? "refine" : task === "network" ? "run" : task;
  for (const [k, b] of Object.entries(buttons)) b.classList.toggle("working", k === clicked);
  clearInterval(ticker);
  if (task) {
    busySince = performance.now();
    buttons.load.disabled = buttons.run.disabled = buttons.refine.disabled = true;
    statusEl("status-task").textContent = TASK_TEXT[task];
    statusEl("status-step").textContent = "";
    statusEl<HTMLProgressElement>("status-bar").removeAttribute("value");
    statusBox.hidden = false;
    const tick = () => { statusEl("status-time").textContent = `${((performance.now() - busySince) / 1000).toFixed(1)} s`; };
    tick();
    ticker = setInterval(tick, 200);
  } else {
    statusBox.hidden = true;
    statusEl("status-plot").innerHTML = "";
    estimate();
    buttons.run.disabled = !areaGrid;
    buttons.refine.disabled = mode === "network" ? !haveNetwork : !haveCorridors;
  }
}

function setStep(name: string, done?: number, total?: number) {
  const bar = statusEl<HTMLProgressElement>("status-bar");
  const step = name.charAt(0).toUpperCase() + name.slice(1);
  if (done === undefined) {
    bar.removeAttribute("value");
    statusEl("status-step").textContent = `${step}...`;
  } else if (!total) {
    bar.removeAttribute("value");
    statusEl("status-step").textContent = `${step}: ${done.toLocaleString()}`;
  } else {
    bar.max = total;
    bar.value = done;
    statusEl("status-step").textContent = `${step}: ${done.toLocaleString()} of ${total.toLocaleString()}`;
  }
}

// ---------------------------------------------------------------- base map and hillshade

const TERRAIN_LAYERS = ["Terrain (whole DEM)", "Terrain (area)"];
let basemap: ExportLayer | null = null;
for (const b of BASEMAPS) {
  const o = document.createElement("option");
  o.value = o.textContent = b.name;
  $("basemap").append(o);
}

function hillshadeOpacity(): number { return Number($<HTMLInputElement>("hillshade").value) / 100; }

function applyBasemap() {
  basemap?.remove();
  basemap = null;
  const b = BASEMAPS.find((x) => x.name === $<HTMLSelectElement>("basemap").value);
  map.getPane("terrain")!.classList.toggle("blend", !!b);
  if (!b || !info) return;
  basemap = new ExportLayer(b, utmEpsg(info.zone, info.north), "terrain", (msg) => log(msg, "err")).addTo(map);
}

$("basemap").addEventListener("change", () => {
  // Full-strength relief over a photo is dark; start the blend at half.
  const slider = $<HTMLInputElement>("hillshade");
  if ($<HTMLSelectElement>("basemap").value && slider.value === "100") {
    slider.value = "50";
    slider.dispatchEvent(new Event("input"));
  }
  applyBasemap();
});
$("hillshade").addEventListener("input", () => {
  for (const name of TERRAIN_LAYERS) (layers.get(name) as L.ImageOverlay | undefined)?.setOpacity(hillshadeOpacity());
});

// ---------------------------------------------------------------- coordinates

function utmDef(): string {
  return info ? `+proj=utm +zone=${info.zone}${info.north ? "" : " +south"} +datum=WGS84 +units=m +no_defs` : "";
}

function formatPoint(p: [number, number]): string {
  if ($<HTMLSelectElement>("coord-crs").value === "lonlat" && info) {
    const [lon, lat] = proj4(utmDef(), "WGS84", p);
    return `${lon.toFixed(6)}, ${lat.toFixed(6)}`;
  }
  return `${p[0].toFixed(1)}, ${p[1].toFixed(1)}`;
}

function parsePoint(text: string): [number, number] | null {
  const v = text.split(/[,\s]+/).filter(Boolean).map(Number);
  if (v.length !== 2 || v.some((x) => !Number.isFinite(x))) return null;
  if ($<HTMLSelectElement>("coord-crs").value === "lonlat") {
    if (!info) return null;
    const [x, y] = proj4("WGS84", utmDef(), [v[0], v[1]]);
    return [x, y];
  }
  return [v[0], v[1]];
}

function setEnd(which: "start" | "goal", p: [number, number]) {
  if (which === "start") start = p; else goal = p;
  const color = getComputedStyle(document.documentElement).getPropertyValue(which === "start" ? "--start" : "--goal");
  markers[which]?.remove();
  markers[which] = L.circleMarker([p[1], p[0]], { radius: 7, color: "#000", weight: 1.5, fillColor: color.trim(),
                                                  fillOpacity: 1 }).addTo(map)
    .bindTooltip(which === "start" ? "start" : "finish");
  $<HTMLInputElement>(which === "start" ? "start-text" : "goal-text").value = formatPoint(p);
  estimate();
}

// ---------------------------------------------------------------- pads

function padSize(): [number, number] {
  return [Math.max(1, Number($<HTMLInputElement>("pad-w").value) || 15), Math.max(1, Number($<HTMLInputElement>("pad-h").value) || 10)];
}

/** The pads: each a rectangle of the pad size and a marker colored by its state in the network. */
function drawPads() {
  padLayer.clearLayers();
  const [pw, ph] = padSize();
  const css = getComputedStyle(document.documentElement);
  pads.forEach(([x, y], i) => {
    const st = padStatus;
    const color = !st ? "#9ca3af" : st.unreachable[i] ? css.getPropertyValue("--goal").trim()
      : st.connected[i] ? css.getPropertyValue("--start").trim() : "#9ca3af";
    // Long axis along the contour (once known), width across it.
    const a = padAxes?.[i] ?? 0, [ux, uy] = [Math.cos(a), Math.sin(a)], [vx, vy] = [-uy, ux];
    const corner = (s: number, t: number) => [y + s * uy * pw / 2 + t * vy * ph / 2, x + s * ux * pw / 2 + t * vx * ph / 2] as [number, number];
    L.polygon([corner(1, 1), corner(1, -1), corner(-1, -1), corner(-1, 1)], { color: "#111", weight: 1, fillColor: color,
      fillOpacity: 0.8, interactive: false }).addTo(padLayer);
    const why = st?.ends ? (st.ends[i] === 0 ? " (both ends on ground too steep for a road)" : " (no road within the buffer)") : "";
    const note = !st ? "" : st.unreachable[i] ? `: unreachable${why}` : st.connected[i]
      ? `: connected, ${st.beyond[i]} pad${st.beyond[i] === 1 ? "" : "s"} beyond` : ": waiting";
    L.circleMarker([y, x], { radius: 5, color: "#111", weight: 1, fillColor: color, fillOpacity: 1 })
      .bindTooltip(`pad ${i + 1}${note}`)
      .on("click", () => {
        if (picking !== "pads") return;
        pads.splice(i, 1);
        padStatus = null;
        drawPads();
      })
      .addTo(padLayer);
  });
  if (pads.length && !map.hasLayer(padLayer)) padLayer.addTo(map);
  $("pads-info").textContent = pads.length
    ? `${pads.length} pad${pads.length === 1 ? "" : "s"}` + (picking === "pads" ? " (click the map to add, a pad to remove)" : "")
    : "No pads yet. While adding, click a pad again to remove it. A CSV needs x and y (or lon and lat) columns; GeoJSON, points.";
  estimate();
}

/** Points from a CSV (x/y, easting/northing or lon/lat columns; or two numbers per line, in the
 *  coordinate system chosen above) or GeoJSON (points, longitude and latitude). In UTM. */
function parsePads(text: string, name: string): [number, number][] {
  if (!info) throw new Error("open a DEM first");
  const toUtm = (lon: number, lat: number): [number, number] => proj4("WGS84", utmDef(), [lon, lat]) as [number, number];
  if (/\.(geo)?json$/i.test(name) || text.trimStart().startsWith("{")) {
    const gj = JSON.parse(text);
    const out: [number, number][] = [];
    const visit = (geom: any) => {
      if (!geom) return;
      if (geom.type === "Point") out.push(toUtm(geom.coordinates[0], geom.coordinates[1]));
      else if (geom.type === "MultiPoint") for (const c of geom.coordinates) out.push(toUtm(c[0], c[1]));
      else if (geom.type === "GeometryCollection") geom.geometries.forEach(visit);
    };
    for (const f of gj.type === "FeatureCollection" ? gj.features : [gj]) visit(f.type === "Feature" ? f.geometry : f);
    return out;
  }
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const cells = (l: string) => l.split(/[,;\t]/).map((c) => c.trim());
  const head = cells(lines[0]).map((h) => h.toLowerCase());
  const col = (names: string[]) => head.findIndex((h) => names.includes(h));
  const [ix, iy] = [col(["x", "easting", "east", "e"]), col(["y", "northing", "north", "n"])];
  const [ilon, ilat] = [col(["lon", "long", "longitude", "lng"]), col(["lat", "latitude"])];
  const out: [number, number][] = [];
  if ((ix >= 0 && iy >= 0) || (ilon >= 0 && ilat >= 0)) {
    for (const l of lines.slice(1)) {
      const c = cells(l).map(Number);
      if (ix >= 0 && iy >= 0 && Number.isFinite(c[ix]) && Number.isFinite(c[iy])) out.push([c[ix], c[iy]]);
      else if (Number.isFinite(c[ilon]) && Number.isFinite(c[ilat])) out.push(toUtm(c[ilon], c[ilat]));
    }
    return out;
  }
  for (const l of lines) {
    const p = parsePoint(cells(l).slice(0, 2).join(","));
    if (p) out.push(p);
  }
  return out;
}

function buttonsRefineState() {
  if (!busy) $<HTMLButtonElement>("refine").disabled = mode === "network" ? !haveNetwork : !haveCorridors;
}

/** Drill pad roads' maximum grade (%), the network mode's default; the route grade waits meanwhile. */
const PAD_ROAD_GRADE = "15";
let routeGrade: string | null = null;

function setMode(m: "route" | "network") {
  const switching = m !== mode;
  mode = m;
  const net = m === "network";
  $("pads-box").hidden = !net;
  $("goal-row").hidden = net;
  $("pick-goal").hidden = net;
  // The fine section serves both: the corridor choices only for routes, the buffer only for networks.
  for (const id of ["fine-which", "fine-apart"]) ($(id).closest("label") as HTMLElement).hidden = net;
  $("fine-buffer-row").hidden = !net;
  buttonsRefineState();
  $("start-label").textContent = net ? "Access point" : "Start";
  $("pick-start").textContent = net ? "Click to set the access point" : "Click to set start";
  $("run").textContent = net ? "Find network" : "Find corridors";
  // A network's area is its pads': a small margin, so finer cells.
  const margin = $<HTMLInputElement>("margin");
  if (net && margin.value === "10") margin.value = "2";
  if (!net && margin.value === "2") margin.value = "10";
  // Pad roads at their own grade; the route's back when leaving.
  const grade = $<HTMLInputElement>("g_max");
  if (switching && net) {
    routeGrade = grade.value;
    grade.value = PAD_ROAD_GRADE;
  } else if (switching && routeGrade !== null) {
    grade.value = routeGrade;
    routeGrade = null;
  }
  markers.goal?.setStyle({ opacity: net ? 0 : 1, fillOpacity: net ? 0 : 1 });
  estimate();
}

$<HTMLSelectElement>("mode").addEventListener("change", (e) => setMode((e.target as HTMLSelectElement).value as "route" | "network"));
$("pick-pads").addEventListener("click", () => {
  picking = picking === "pads" ? null : "pads";
  $("pick-pads").classList.toggle("active", picking === "pads");
  $("pick-start").classList.remove("active");
  $("pick-goal").classList.remove("active");
  drawPads();
});
$("clear-pads").addEventListener("click", () => {
  pads.length = 0;
  padAxes = null;
  padStatus = null;
  drawPads();
});
$<HTMLInputElement>("pads-file").addEventListener("change", async (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (!f) return;
  try {
    const got = parsePads(await f.text(), f.name);
    if (!got.length) throw new Error("no points found");
    pads.push(...got);
    padStatus = null;
    drawPads();
    log(`${f.name}: ${got.length} pads`);
  } catch (err) {
    log(`${f.name}: ${err instanceof Error ? err.message : String(err)}`, "err");
  }
  (e.target as HTMLInputElement).value = "";
});
for (const id of ["pad-w", "pad-h"]) $(id).addEventListener("input", drawPads);

/** The pad network: roads weighted by the pads they serve, the pads colored by state, and a summary. */
function showNetwork(m: Extract<FromWorker, { type: "network" }>) {
  padStatus = { connected: m.connected, unreachable: m.unreachable, beyond: m.beyond };
  drawPads();
  if (!lastGraph) return;
  const { grid: g, rows, cols } = lastGraph;
  const xy = (v: number) => [g.y1 - (rows[v] + 0.5) * g.res, g.x0 + (cols[v] + 0.5) * g.res] as [number, number];
  const group = L.layerGroup();
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  for (let i = 0; i < m.uses.length; i++) {
    const pts = [xy(m.pairs[2 * i]), xy(m.pairs[2 * i + 1])];
    const wgt = 2 + 1.5 * Math.log2(1 + m.uses[i]);
    L.polyline(pts, { color: "#111", weight: wgt + 3, opacity: 0.5, interactive: false }).addTo(group);
    L.polyline(pts, { color: accent, weight: wgt, opacity: 0.95 })
      .bindTooltip(`serves ${m.uses[i]} pad${m.uses[i] === 1 ? "" : "s"}`).addTo(group);
  }
  setLayer("Pad network", group);
  const nConn = m.connected.filter(Boolean).length, nUnr = m.unreachable.filter(Boolean).length;
  const maxBeyond = Math.max(0, ...m.beyond);
  const fmt = (v: number) => Math.round(v).toLocaleString();
  $("corridors").innerHTML = `<tr><th>pad network</th><th></th></tr>` +
    `<tr><td>${m.phase}</td><td></td></tr>` +
    `<tr><td>pads connected</td><td>${nConn} of ${pads.length}</td></tr>` +
    (nUnr ? `<tr><td>unreachable</td><td>${m.unreachable.map((u, i) => (u ? i + 1 : 0)).filter(Boolean).join(", ")}</td></tr>` : "") +
    `<tr><td>construction</td><td>${fmt(m.construction)}</td></tr>` +
    `<tr><td>operating (all pads' traffic)</td><td>${fmt(m.operating)}</td></tr>` +
    `<tr><td>total</td><td>${fmt(m.construction + m.operating)}</td></tr>` +
    `<tr><td>most pads beyond a pad</td><td>${maxBeyond}</td></tr>`;
  if (m.final) {
    haveNetwork = m.connected.some(Boolean);
    if (start) map.fitBounds(L.latLngBounds([start, ...pads].map(([x, y]) => [y, x] as [number, number])).pad(0.15));
    log(`network: ${nConn} of ${pads.length} pads, construction ${fmt(m.construction)}, operating ${fmt(m.operating)}` +
        (nUnr ? `; unreachable: ${nUnr}` : ""), nUnr ? "err" : "");
  }
}

/** The fine pad network: roads weighted by the pads they serve, the pads, a summary. */
function showFineNetwork(m: Extract<FromWorker, { type: "fineNetwork" }>) {
  fineNet = m;
  padStatus = { connected: m.connected, unreachable: m.unreachable, beyond: m.beyond, ends: m.ends };
  // Pads by road lie the way their road arrives; until connected, along the contour.
  if (m.axes.length) padAxes = m.axes.map((a, i) => (Number.isFinite(a) ? a : padAxes?.[i] ?? 0));
  drawPads();
  const group = L.layerGroup();
  const color = getComputedStyle(document.documentElement).getPropertyValue("--fine").trim();
  for (const r of m.runs) {
    const pts = r.x.map((x, i) => [r.y[i], x] as [number, number]);
    const wgt = 2.5 + 1.5 * Math.log2(1 + r.uses);
    L.polyline(pts, { color: "#111", weight: wgt + 3, opacity: 0.6, interactive: false }).addTo(group);
    L.polyline(pts, { color, weight: wgt }).bindTooltip(`pad ${r.pad + 1}'s branch: serves ${r.uses} pad${r.uses === 1 ? "" : "s"}`).addTo(group);
  }
  setLayer("Fine network", group);
  const fmt = (v: number) => Math.round(v).toLocaleString();
  const nConn = m.connected.filter(Boolean).length, nUnr = m.unreachable.filter(Boolean).length;
  let km = 0;
  for (const r of m.runs) for (let i = 1; i < r.x.length; i++) km += Math.hypot(r.x[i] - r.x[i - 1], r.y[i] - r.y[i - 1]) / 1000;
  $("fine").innerHTML = `<tr><th>fine network, ${m.res} m</th><th></th></tr>` +
    `<tr><td>${m.phase}</td><td></td></tr>` +
    `<tr><td>pads connected</td><td>${nConn} of ${pads.length}</td></tr>` +
    (nUnr ? `<tr><td>unreachable</td><td>${m.unreachable.map((u, i) => (u ? `${i + 1}` +
      (m.ends[i] === 0 ? " (ends on steep ground)" : " (none within the buffer)") : "")).filter(Boolean).join(", ")}</td></tr>` : "") +
    `<tr><td>road length</td><td>${km.toFixed(2)} km</td></tr>` +
    (m.switchbacks ? `<tr><td>switchbacks</td><td>${m.switchbacks}</td></tr>` : "") +
    (m.beside ? `<tr><td>pads beside the road</td><td>${m.beside}</td></tr>` : "") +
    `<tr><td>pad benches (in construction)</td><td>${fmt(m.pad_cost)}</td></tr>` +
    `<tr><td>dead ends (rigs at once)</td><td>${m.dead_ends}</td></tr>` +
    `<tr><td>construction</td><td>${fmt(m.construction)}</td></tr>` +
    `<tr><td>operating (all pads' traffic)</td><td>${fmt(m.operating)}</td></tr>` +
    `<tr><td>total</td><td>${fmt(m.construction + m.operating)}</td></tr>` +
    `<tr><td>most pads beyond a pad</td><td>${Math.max(0, ...m.beyond)}</td></tr>`;
  $<HTMLButtonElement>("download").disabled = !m.final;
  if (m.final) {
    log(`fine network: ${nConn} of ${pads.length} pads, ${km.toFixed(2)} km, construction ${fmt(m.construction)}, ` +
        `operating ${fmt(m.operating)}` + (nUnr ? `; unreachable: ${nUnr} (see the table)` : ""), nUnr ? "warn" : "");
  }
}

// ---------------------------------------------------------------- controls

function zUnits(): string { return $<HTMLSelectElement>("z_units").value; }

/** The points the area must cover: start and finish, or the access point and the pads. */
function areaPoints(): [number, number][] | null {
  if (!start) return null;
  if (mode === "network") return pads.length ? [start, ...pads] : null;
  return goal ? [start, goal] : null;
}

/** Standard cell sizes (m) for the coarse graph. */
const NICE_RES = [1, 2, 2.5, 3, 4, 5, 6, 7.5, 8, 10, 12, 12.5, 15, 20, 25, 30, 40, 50, 60, 75, 100];
/** The coarse graph's cells when the size is automatic. */
const AUTO_CELLS = 6_000_000;

/** The area's extent (m) for the current points and margin. */
function areaSize(): [number, number] | null {
  const pts = areaPoints();
  if (!info || !pts) return null;
  const m = Number($<HTMLInputElement>("margin").value) * 1000;
  const [bx0, by0, bx1, by1] = info.bbox;
  const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
  return [Math.max(0, Math.min(Math.max(...xs) + m, bx1) - Math.max(Math.min(...xs) - m, bx0)),
          Math.max(0, Math.min(Math.max(...ys) + m, by1) - Math.max(Math.min(...ys) - m, by0))];
}

/** The coarse cell size: as typed, or the finest standard size at or above the DEM's own that
 *  keeps the area within AUTO_CELLS. */
function coarseRes(): number {
  const typed = Number($<HTMLInputElement>("res").value);
  if (typed > 0) return typed;
  const size = areaSize();
  const native = info ? info.resM : 10;
  const area = size ? size[0] * size[1] : 0;
  return NICE_RES.find((r) => r >= native - 1e-6 && area / (r * r) <= AUTO_CELLS) ?? NICE_RES[NICE_RES.length - 1];
}

function estimate() {
  const loadBtn = $<HTMLButtonElement>("load");
  const size = areaSize();
  if (!info || !size || busy) { loadBtn.disabled = true; return; }
  const res = coarseRes();
  const cells = (size[0] / res) * (size[1] / res);
  const over = cells > MAX_CELLS;
  const auto = !(Number($<HTMLInputElement>("res").value) > 0);
  $("estimate").textContent = `${(cells / 1e6).toFixed(1)} million cells at ${res} m${auto ? " (automatic)" : ""}, about ` +
    `${Math.round(cells * 70 / 1e6)} MB` + (over ? ` — over the limit of ${MAX_CELLS / 1e6} million: use larger cells or a smaller margin` : "");
  loadBtn.disabled = over || res <= 0 || !!busy;
}

function params(): Params {
  const p: Params = { ...defaults, ...base };
  const g = Number($<HTMLInputElement>("g_max").value);
  if (g > 0) p.g_max = g / 100;
  const w = Number($<HTMLInputElement>("w").value);
  if (w > 0) p.w = w;
  for (const key of ["A_min", "tin_channel_ha"]) {
    const v = Number($<HTMLInputElement>(key).value);
    if (v > 0) p[key] = v;
  }
  const k = Number($<HTMLInputElement>("k").value);
  for (const key of ["C_travel", "C_vertical", "C_turn"]) p[key] = (base[key] as number ?? defaults[key] as number) * k;
  p.z_units = zUnits();
  p.tin_res = coarseRes();
  return p;
}

function setControls(p: Params) {
  const grade = String(+(100 * (p.g_max as number)).toFixed(2));
  // In network mode the pad roads' grade stays; the example's waits for route mode.
  if (mode === "network") routeGrade = grade;
  else $<HTMLInputElement>("g_max").value = grade;
  $<HTMLInputElement>("w").value = String(p.w);
  $<HTMLInputElement>("res").value = "";
  $<HTMLInputElement>("margin").value = String((p.tin_margin as number ?? 10000) / 1000);
  $<HTMLSelectElement>("z_units").value = String(p.z_units ?? "m");
  $<HTMLInputElement>("k").value = "1";
  $<HTMLInputElement>("A_min").value = String(p.A_min);
  $<HTMLInputElement>("tin_channel_ha").value = String(p.tin_channel_ha);
}

function openDem(m: ToWorker & { type: "open" }) {
  info = null; start = null; goal = null; areaGrid = null; haveCorridors = false;
  clearFine();
  for (const name of [...layers.keys()]) setLayer(name, null);
  markers.start?.remove(); markers.goal?.remove();
  $("corridors").innerHTML = "";
  $("log").innerHTML = "";
  $<HTMLButtonElement>("load").disabled = true;
  $<HTMLButtonElement>("run").disabled = true;
  send(m);
}

$<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) { base = {}; setControls(defaults); openDem({ type: "open", file: f, name: f.name, zUnits: zUnits() }); }
});
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const f = e.dataTransfer?.files[0];
  if (f) { base = {}; setControls(defaults); openDem({ type: "open", file: f, name: f.name, zUnits: zUnits() }); }
});

let examples: Example[] = [];
$<HTMLSelectElement>("example").addEventListener("change", (e) => {
  const ex = examples.find((x) => x.name === (e.target as HTMLSelectElement).value);
  if (!ex) return;
  base = ex.params;
  setControls(ex.params);
  // The worker resolves relative URLs against its own script, so hand it an absolute one.
  openDem({ type: "open", url: new URL(ex.dem, document.baseURI).href, name: ex.title, zUnits: String(ex.params.z_units ?? "m"),
            endpointsDem: [[ex.start[0], ex.start[1]], [ex.goal[0], ex.goal[1]]] });
});

for (const which of ["start", "goal"] as const) {
  const btn = $<HTMLButtonElement>(which === "start" ? "pick-start" : "pick-goal");
  btn.addEventListener("click", () => {
    picking = picking === which ? null : which;
    $("pick-start").classList.toggle("active", picking === "start");
    $("pick-goal").classList.toggle("active", picking === "goal");
  });
  $<HTMLInputElement>(which === "start" ? "start-text" : "goal-text").addEventListener("change", (e) => {
    const p = parsePoint((e.target as HTMLInputElement).value);
    if (p) setEnd(which, p); else log(`could not read "${(e.target as HTMLInputElement).value}" as x, y`, "err");
  });
}
$<HTMLSelectElement>("coord-crs").addEventListener("change", () => {
  if (start) $<HTMLInputElement>("start-text").value = formatPoint(start);
  if (goal) $<HTMLInputElement>("goal-text").value = formatPoint(goal);
});
map.on("click", (e: L.LeafletMouseEvent) => {
  if (picking === "pads") {
    pads.push([e.latlng.lng, e.latlng.lat]);
    padStatus = null;
    drawPads();
    return;
  }
  if (!picking) return;
  setEnd(picking, [e.latlng.lng, e.latlng.lat]);
  picking = null;
  $("pick-start").classList.remove("active");
  $("pick-goal").classList.remove("active");
});
for (const id of ["margin", "res"]) $(id).addEventListener("input", estimate);
$("A_min").addEventListener("change", () => {
  const v = Number($<HTMLInputElement>("A_min").value);
  if (!(v > 0) || !layers.has("Channels")) return;
  send({ type: "channels", aMin: v });
  if ($("corridors").innerHTML) log(`channel threshold ${v} ha: find corridors again to cost crossings with it`);
});

$("load").addEventListener("click", () => {
  if (!start || !areaPoints()) return;
  areaGrid = null;
  haveCorridors = false;
  clearFine();
  $<HTMLButtonElement>("load").disabled = true;
  $<HTMLButtonElement>("run").disabled = true;
  send({ type: "load", start, goal: mode === "route" ? goal : null, pads: mode === "network" ? [...pads] : [],
         margin: Number($<HTMLInputElement>("margin").value) * 1000,
         res: coarseRes(), maxCells: MAX_CELLS, zUnits: zUnits(), roughMaxRes: roughMaxRes(runParams()) });
});
/** The run's parameters, with the pads' traffic scaling the operating costs in network mode. */
function runParams(): Params {
  const p = params();
  if (mode === "network") {
    const tau = Math.max(0, Number($<HTMLInputElement>("pad-traffic").value) || 0);
    for (const k of ["C_travel", "C_vertical", "C_turn"]) p[k] = (p[k] as number) * tau;
    p.switchback_landing = Math.max(0, Number($<HTMLInputElement>("pad-landing").value) || 0);
  }
  return p;
}

$("refine").addEventListener("click", () => {
  const res = Number($<HTMLInputElement>("fine-res").value);
  if (!(res > 0)) return;
  for (const b of ["load", "run", "refine"]) $<HTMLButtonElement>(b).disabled = true;
  keyChart.reset();
  exploreStage = "";
  if (mode === "network") {
    clearFine();
    const [length, width] = padSize();
    send({ type: "refineNetwork", params: runParams(), res, buffer: Number($<HTMLInputElement>("fine-buffer").value) || 300,
           b: Math.max(0, Math.round(Number($<HTMLInputElement>("pad-b").value))), length, width,
           beside: $<HTMLInputElement>("pad-beside").checked,
           byRoad: $<HTMLSelectElement>("pad-orient").value === "road",
           rigs: Math.max(0, Math.round(Number($<HTMLInputElement>("pad-rigs").value) || 0)) });
    return;
  }
  send({ type: "refine", params: params(), res, which: $<HTMLSelectElement>("fine-which").value as "best" | "all",
         apart: Number($<HTMLSelectElement>("fine-apart").value) });
});

$("run").addEventListener("click", () => {
  haveCorridors = false;
  hiddenCorridors.clear();
  hiddenFine.clear();
  clearFine();
  $<HTMLButtonElement>("load").disabled = true;
  $<HTMLButtonElement>("run").disabled = true;
  for (const name of ["Slope mask", "Channels", "Saddles", "Triangulation", "Graph nodes (corridor field)",
                      "Graph edges", "Corridors"]) setLayer(name, null);
  $("corridors").innerHTML = "";
  setLayer("Pad network", null);
  padStatus = null;
  drawPads();
  if (mode === "network") {
    haveNetwork = false;
    send({ type: "network", params: runParams(), b: Math.max(0, Math.round(Number($<HTMLInputElement>("pad-b").value))),
           refine: Math.max(1, Number($<HTMLInputElement>("pad-refine").value) || 1) });
  }
  else send({ type: "run", params: params() });
});

// ---------------------------------------------------------------- results

function pointsLayer(g: Grid, rows: ArrayLike<number>, cols: ArrayLike<number>, color: (i: number) => string,
                     size: number): CanvasLayer {
  return new CanvasLayer((ctx, px, [x0, y0, x1, y1]) => {
    const groups = new Map<string, number[]>();
    for (let i = 0; i < rows.length; i++) {
      const x = g.x0 + (cols[i] + 0.5) * g.res, y = g.y1 - (rows[i] + 0.5) * g.res;
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      const c = color(i);
      let a = groups.get(c);
      if (!a) groups.set(c, (a = []));
      a.push(x, y);
    }
    for (const [c, a] of groups) {
      ctx.fillStyle = c;
      ctx.beginPath();
      for (let k = 0; k < a.length; k += 2) {
        const [u, v] = px(a[k], a[k + 1]);
        ctx.rect(u - size / 2, v - size / 2, size, size);
      }
      ctx.fill();
    }
  });
}

function segmentsLayer(g: Grid, rows: ArrayLike<number>, cols: ArrayLike<number>, pairs: ArrayLike<number>,
                       stroke: string, width: number, maxDraw = 400_000): CanvasLayer {
  return new CanvasLayer((ctx, px, [x0, y0, x1, y1]) => {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.beginPath();
    let drawn = 0;
    for (let k = 0; k < pairs.length && drawn < maxDraw; k += 2) {
      const a = pairs[k], b = pairs[k + 1];
      const xa = g.x0 + (cols[a] + 0.5) * g.res, ya = g.y1 - (rows[a] + 0.5) * g.res;
      const xb = g.x0 + (cols[b] + 0.5) * g.res, yb = g.y1 - (rows[b] + 0.5) * g.res;
      if (Math.max(xa, xb) < x0 || Math.min(xa, xb) > x1 || Math.max(ya, yb) < y0 || Math.min(ya, yb) > y1) continue;
      const [ua, va] = px(xa, ya), [ub, vb] = px(xb, yb);
      ctx.moveTo(ua, va);
      ctx.lineTo(ub, vb);
      drawn++;
    }
    ctx.stroke();
  });
}

function clearFine() {
  fineNet = null;
  setLayer("Fine network", null);
  fineRoutes.length = 0;
  profileIndex = null;
  wavefronts.clear();
  sites.clear();
  setLayer("Hairpin sites", null);
  setLayer("Search wavefront", null);
  setLayer("Leading candidate", null);
  keyChart.reset();
  statusEl("status-plot").innerHTML = "";
  setLayer("Fine routes", null);
  $("fine").innerHTML = "";
  $("profile").innerHTML = "";
  $("profile-note").textContent = "";
  $<HTMLButtonElement>("download").disabled = true;
}

/** The profile of the selected fine route (by default the cheapest finished one). */
function showProfile() {
  const done = fineRoutes.filter((f) => f.final && f.ground.length);
  const f = done.find((x) => x.index === profileIndex) ??
            [...done].sort((a, b) => a.cost.total - b.cost.total)[0];
  $("profile").innerHTML = f ? profileSvg(f) : "";
  $("profile-note").textContent = f ? profileSummary(f) + ". Shaded: hairpins and switchbacks (red), bends (blue), fill crossings (yellow)." : "";
  for (const tr of document.querySelectorAll<HTMLTableRowElement>("#fine tr[data-index]")) {
    tr.classList.toggle("selected", f !== undefined && Number(tr.dataset.index) === f.index);
  }
}

/** The finished fine routes as GeoJSON (RFC 7946: longitude, latitude, design elevation), each
 *  route with its costs, and its segments as separate features for styling by type or grade. */
function fineGeoJson(): string {
  const toLonLat = (x: number, y: number, z: number) => {
    const [lon, lat] = proj4(utmDef(), "WGS84", [x, y]);
    return [+lon.toFixed(7), +lat.toFixed(7), +z.toFixed(2)];
  };
  if (mode === "network" && fineNet) {
    const net = fineNet;
    const features: object[] = net.runs.map((r) => ({
      type: "Feature", geometry: { type: "LineString", coordinates: r.x.map((x, i) => toLonLat(x, r.y[i], r.z[i])) },
      properties: { feature: "road", branch_of_pad: r.pad + 1, pads_served: r.uses },
    }));
    pads.forEach(([x, y], i) => {
      const [lon, lat] = proj4(utmDef(), "WGS84", [x, y]);
      features.push({ type: "Feature", geometry: { type: "Point", coordinates: [+lon.toFixed(7), +lat.toFixed(7)] },
        properties: { feature: "pad", pad: i + 1, connected: net.connected[i], pads_beyond: net.beyond[i],
                      long_axis_deg: padAxes ? +(90 - (padAxes[i] * 180) / Math.PI).toFixed(1) : null } });
    });
    features.push({ type: "Feature", geometry: null, properties: { feature: "network", cell_m: net.res,
      construction: Math.round(net.construction), operating: Math.round(net.operating) } });
    return JSON.stringify({ type: "FeatureCollection", features });
  }
  const done = fineRoutes.filter((f) => f.final).sort((a, b) => a.cost.total - b.cost.total);
  const features: object[] = [];
  done.forEach((f, rank) => {
    const coords = f.x.map((x, i) => toLonLat(x, f.y[i], f.z[i]));
    features.push({
      type: "Feature", geometry: { type: "LineString", coordinates: coords },
      properties: {
        feature: "route", corridor: f.index + 1, rank: rank + 1, cell_m: f.res, length_m: Math.round(f.length),
        cost_total: Math.round(f.cost.total), construction: Math.round(f.cost.construction),
        operating: Math.round(f.cost.operating), base: Math.round(f.cost.base), excavation: Math.round(f.cost.excavation),
        culverts: Math.round(f.cost.culverts), crossing_fill: Math.round(f.cost.crossing_fill),
        fill_m3: Math.round(f.cost.fill_m3), borrow_m3: Math.round(f.cost.borrow_m3), hairpins: f.hairpins, bends: f.bends,
        fill_crossings: f.fill_crossings,
      },
    });
    f.segments.forEach((s, i) => {
      const end = i + 1 < f.segments.length ? f.segments[i + 1].first : f.x.length - 1;
      features.push({
        type: "Feature", geometry: { type: "LineString", coordinates: coords.slice(s.first, end + 1) },
        properties: { feature: "segment", corridor: f.index + 1, seq: i, kind: s.kind, length_m: +s.length.toFixed(1),
                      grade: +s.grade.toFixed(4), cost: Math.round(s.cost) },
      });
    });
  });
  return JSON.stringify({ type: "FeatureCollection", features });
}

$("download").addEventListener("click", () => {
  const blob = new Blob([fineGeoJson()], { type: "application/geo+json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const stem = (info?.name ?? "dem").replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_");
  a.download = `fine_routes_${stem}.geojson`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

$("fine").addEventListener("change", (e) => {
  const box = e.target as HTMLInputElement;
  if (box.dataset.fine === undefined) return;
  const i = Number(box.dataset.fine);
  if (box.checked) hiddenFine.delete(i); else hiddenFine.add(i);
  showFine();
});

$("fine").addEventListener("click", (e) => {
  if ((e.target as HTMLElement).tagName === "INPUT") return;
  const tr = (e.target as HTMLElement).closest<HTMLTableRowElement>("tr[data-index]");
  if (!tr) return;
  profileIndex = Number(tr.dataset.index);
  showProfile();
});

/** The fine routes, drawn over their corridors, and their table. */
/** One color per corridor, for its coarse line, its fine route and its table rows. */
const CORRIDOR_COLORS = ["#e11d48", "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#be185d", "#4d7c0f"];
const corridorColor = (i: number) => CORRIDOR_COLORS[i % CORRIDOR_COLORS.length];
const swatch = (i: number) => `<span class="swatch" style="background:${corridorColor(i)}"></span>`;
/** Corridors and fine routes switched off in their tables, by corridor index. */
const hiddenCorridors = new Set<number>();
const hiddenFine = new Set<number>();
let lastCorridors: { grid: Grid; routes: RouteOut[] } | null = null;

/** Share of route a's length within `tol` meters of route b (are two fine routes the same road?). */
function sharedWith(a: FineOut, b: FineOut, tol = 20): number {
  const key = (x: number, y: number) => `${Math.floor(x / tol)},${Math.floor(y / tol)}`;
  const cells = new Map<string, number[]>();
  b.x.forEach((x, i) => {
    const k = key(x, b.y[i]);
    (cells.get(k) ?? cells.set(k, []).get(k)!).push(i);
  });
  const near = (x: number, y: number) => {
    const cx = Math.floor(x / tol), cy = Math.floor(y / tol);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (const j of cells.get(`${cx + dx},${cy + dy}`) ?? []) if (Math.hypot(b.x[j] - x, b.y[j] - y) <= tol) return true;
      }
    }
    return false;
  };
  let shared = 0, total = 0;
  for (let i = 1; i < a.x.length; i++) {
    const len = Math.hypot(a.x[i] - a.x[i - 1], a.y[i] - a.y[i - 1]);
    total += len;
    if (near(0.5 * (a.x[i] + a.x[i - 1]), 0.5 * (a.y[i] + a.y[i - 1]))) shared += len;
  }
  return total > 0 ? shared / total : 0;
}

function showFine() {
  const group = L.layerGroup();
  const sorted = [...fineRoutes].sort((a, b) => a.cost.total - b.cost.total);
  for (const f of sorted) {
    if (hiddenFine.has(f.index)) continue;
    const pts = f.x.map((x, i) => [f.y[i], x] as [number, number]);
    // A dark casing under the line keeps it readable over imagery; a route still being refined is dashed.
    L.polyline(pts, { color: "#111", weight: 7, opacity: 0.55, interactive: false }).addTo(group);
    L.polyline(pts, { color: corridorColor(f.index), weight: 4, dashArray: f.final ? undefined : "8 6" })
      .bindTooltip(`corridor ${f.index + 1}, fine: ${Math.round(f.cost.total).toLocaleString()}, ` +
                   `${(f.length / 1000).toFixed(1)} km, ${f.hairpins} hairpins`)
      .addTo(group);
  }
  setLayer("Fine routes", group);
  // How much of each route is another's road: the largest share, and with which corridor.
  const shares = sorted.map((f) => {
    let best = 0, with_ = -1;
    for (const g of fineRoutes) {
      if (g === f || !g.final || !f.final) continue;
      const s = sharedWith(f, g);
      if (s > best) { best = s; with_ = g.index; }
    }
    return with_ < 0 ? "" : `${Math.round(100 * best)}% with ${with_ + 1}`;
  });
  const head = "<tr><th>corridor</th><th>cost</th><th>build</th><th>operate</th><th>km</th><th>hairpins</th>" +
               "<th>bends</th><th>crossings</th><th>shares</th><th>s</th></tr>";
  $("fine").innerHTML = head + sorted.map((f, i) =>
    `<tr data-index="${f.index}"><td><input type="checkbox" data-fine="${f.index}" ${hiddenFine.has(f.index) ? "" : "checked"} ` +
    `aria-label="show the fine route of corridor ${f.index + 1}">${swatch(f.index)}${f.index + 1}${f.final ? "" : " (refining)"}</td>` +
    `<td>${Math.round(f.cost.total).toLocaleString()}</td>` +
    `<td>${Math.round(f.cost.construction).toLocaleString()}</td><td>${Math.round(f.cost.operating).toLocaleString()}</td>` +
    `<td>${(f.length / 1000).toFixed(2)}</td><td>${f.hairpins}</td><td>${f.bends}</td><td>${f.fill_crossings}</td>` +
    `<td>${shares[i]}</td><td>${f.seconds.toFixed(0)}</td></tr>`).join("");
  showProfile();
  $<HTMLButtonElement>("download").disabled = !fineRoutes.some((f) => f.final);
}

function showCorridors(g: Grid, routes: RouteOut[]) {
  lastCorridors = { grid: g, routes };
  const group = L.layerGroup();
  routes.forEach((r, i) => {
    if (hiddenCorridors.has(i)) return;
    const pts = r.rows.map((row, k) => [g.y1 - (row + 0.5) * g.res, g.x0 + (r.cols[k] + 0.5) * g.res] as [number, number]);
    L.polyline(pts, { color: corridorColor(i), weight: i === 0 ? 3 : 2.5, opacity: 0.85, dashArray: i === 0 ? undefined : "6 4" })
      .bindTooltip(`corridor ${i + 1} (${r.how}): ${Math.round(r.cost).toLocaleString()}, ${(r.length / 1000).toFixed(1)} km`)
      .addTo(group);
  });
  setLayer("Corridors", group);
  const head = "<tr><th>#</th><th>found by</th><th>cost</th><th>km</th><th>hairpins</th><th>turning °</th></tr>";
  $("corridors").innerHTML = head + routes.map((r, i) =>
    `<tr><td><input type="checkbox" data-corridor="${i}" ${hiddenCorridors.has(i) ? "" : "checked"} ` +
    `aria-label="show corridor ${i + 1}">${swatch(i)}${i + 1}</td><td>${r.how}</td><td>${Math.round(r.cost).toLocaleString()}</td>` +
    `<td>${(r.length / 1000).toFixed(1)}</td><td>${r.hairpins}</td><td>${Math.round(r.turning * 180 / Math.PI).toLocaleString()}</td></tr>`).join("");
}

$("corridors").addEventListener("change", (e) => {
  const box = e.target as HTMLInputElement;
  if (box.dataset.corridor === undefined) return;
  const i = Number(box.dataset.corridor);
  if (box.checked) hiddenCorridors.delete(i); else hiddenCorridors.add(i);
  if (lastCorridors) showCorridors(lastCorridors.grid, lastCorridors.routes);
});

const stageStart = new Map<string, number>();
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "busy":
      setBusy(m.task);
      break;
    case "stage":
      if (m.state === "start") {
        stageStart.set(m.name, performance.now());
        setStep(m.name);
      } else log(`${m.name}: ${((m.ms ?? 0) / 1000).toFixed(2)} s`);
      break;
    case "progress":
      setStep(m.name, m.done, m.total);
      break;
    case "stopped":
      log(`stopped (${m.task})`);
      break;
    case "info": {
      info = m.info;
      const i = m.info;
      $("info").textContent = `${i.name}: ${i.width.toLocaleString()} x ${i.height.toLocaleString()} cells, ` +
        `${i.geographic ? "latitude/longitude" : `projected (${i.units})`}, about ${i.resM.toFixed(1)} m; ` +
        `working in UTM zone ${i.zone}${i.north ? "N" : "S"}` + (i.overviews ? `; ${i.overviews} overviews` : "") +
        (i.warnings.length ? `. Note: ${i.warnings.join("; ")}` : "");
      if (m.endpoints && m.endpoints.length === 2) { setEnd("start", m.endpoints[0]); setEnd("goal", m.endpoints[1]); }
      estimate();
      applyBasemap();
      break;
    }
    case "raster": {
      const names = { preview: "Terrain (whole DEM)", hillshade: "Terrain (area)", slope: "Slope mask",
                      channels: "Channels" } as const;
      const terrain = m.layer === "preview" || m.layer === "hillshade";
      setLayer(names[m.layer], terrain
        ? rasterLayer(m.grid, m.rgba, hillshadeOpacity(), { pane: "terrain", zIndex: 2, className: "hillshade" })
        : rasterLayer(m.grid, m.rgba));
      if (m.layer === "preview") map.fitBounds(boundsOf(m.grid));
      break;
    }
    case "area":
      areaGrid = m.grid;
      map.fitBounds(boundsOf(m.grid));
      log(`area: ${m.grid.w.toLocaleString()} x ${m.grid.h.toLocaleString()} cells at ${m.grid.res} m`);
      log(m.roughness);
      break;
    case "points": {
      const g = m.grid, rows = new Uint32Array(m.cells.length), cols = new Uint32Array(m.cells.length);
      m.cells.forEach((v, k) => { rows[k] = Math.floor(v / g.w); cols[k] = v % g.w; });
      setLayer("Saddles", pointsLayer(g, rows, cols, () => "#7c3aed", 4), false);
      break;
    }
    case "tin": {
      const g = m.grid, n = m.points.length / 2;
      const rows = new Uint32Array(n), cols = new Uint32Array(n);
      for (let i = 0; i < n; i++) { cols[i] = m.points[2 * i]; rows[i] = m.points[2 * i + 1]; }
      const seen = new Set<number>(), pairs: number[] = [];
      for (let t = 0; t < m.triangles.length; t += 3) {
        for (const [a, b] of [[m.triangles[t], m.triangles[t + 1]], [m.triangles[t + 1], m.triangles[t + 2]],
                              [m.triangles[t + 2], m.triangles[t]]]) {
          const key = Math.min(a, b) * n + Math.max(a, b);
          if (!seen.has(key)) { seen.add(key); pairs.push(a, b); }
        }
      }
      log(`triangulation: ${n.toLocaleString()} vertices, ${m.n.toLocaleString()} triangles`);
      setLayer("Triangulation", segmentsLayer(g, rows, cols, pairs, "rgba(30,30,30,0.35)", 0.6));
      break;
    }
    case "network":
      showNetwork(m);
      break;
    case "fineNetwork":
      showFineNetwork(m);
      break;
    case "padAxes":
      padAxes = m.angles;
      drawPads();
      break;
    case "graph": {
      const g = m.grid;
      lastGraph = { grid: g, rows: m.rows, cols: m.cols };
      let fmin = Infinity;
      for (const v of m.field) if (v < fmin) fmin = v;
      log(`graph: ${m.rows.length.toLocaleString()} nodes, ${(m.pairs.length / 2).toLocaleString()} edges`);
      setLayer("Graph nodes (corridor field)", pointsLayer(g, m.rows, m.cols, (i) => {
        const f = m.field[i];
        return Number.isFinite(f) ? rampColor((Math.round((f / fmin - 1) * 20) / 20)) : "rgba(120,120,120,0.4)";
      }, 3), mode === "route");
      setLayer("Graph edges", segmentsLayer(g, m.rows, m.cols, m.pairs, "rgba(80,80,160,0.25)", 0.5, 300_000), false);
      setLayer("Triangulation", layers.get("Triangulation") ?? null, false);
      break;
    }
    case "sites": {
      let st = sites.get(m.index);
      if (!st || st.grid.x0 !== m.grid.x0 || st.grid.y1 !== m.grid.y1 || st.grid.res !== m.grid.res) {
        st = new HairpinSites(m.grid);
        sites.set(m.index, st);
        setLayer("Hairpin sites", st.layer);
        setLayer("Graph nodes (corridor field)", layers.get("Graph nodes (corridor field)") ?? null, false);
      }
      st.add(m.cells, m.cheapest);
      if (!sitesPending) {
        sitesPending = true;
        requestAnimationFrame(() => {
          sitesPending = false;
          st!.layer.redraw();
          const fmt = (v: number) => Math.round(v).toLocaleString();
          statusEl("status-plot").innerHTML = `<p class="status-note">Cheapest hairpin per cell: deep red ` +
            `${fmt(st!.low)} to pale ${fmt(st!.high)}; a hairpin fits at ${(100 * st!.fits).toFixed(0)}% of the ` +
            `${st!.designed.toLocaleString()} cells designed so far.</p>`;
        });
      }
      break;
    }
    case "explore": {
      let wf = wavefronts.get(m.index);
      if (!wf || wf.grid.x0 !== m.grid.x0 || wf.grid.y1 !== m.grid.y1 || wf.grid.res !== m.grid.res) {
        wf = new Wavefront(m.grid);
        wavefronts.set(m.index, wf);
        setLayer("Search wavefront", wf.layer);
        setLayer("Leading candidate", candidate.layer);
        // The graph's nodes would hide the wavefront; they stay a click away.
        setLayer("Graph nodes (corridor field)", layers.get("Graph nodes (corridor field)") ?? null, false);
        exploreT0 = performance.now();
      }
      const stage = `corridor ${m.index + 1}: ${m.stage}`;
      if (stage !== exploreStage) {
        wf.clear();                       // a new search: watch it from the start
        exploreStage = stage;
      }
      wf.add(m.cells, m.g, m.key);
      candidate.set(m.x, m.y);
      keyChart.add(stage, (performance.now() - exploreT0) / 1000, m.key);
      // Draw at most once a frame, however fast the reports come.
      if (!explorePending) {
        explorePending = true;
        requestAnimationFrame(() => {
          explorePending = false;
          wf!.layer.redraw();
          statusEl("status-plot").innerHTML = keyChart.svg();
        });
      }
      break;
    }
    case "fine": {
      const f = m.route;
      const i = fineRoutes.findIndex((x) => x.index === f.index);
      if (i >= 0) fineRoutes[i] = f; else fineRoutes.push(f);
      showFine();
      if (!f.final) {
        log(`corridor ${f.index + 1}: a route at ${Math.round(f.cost.total).toLocaleString()}, ${(f.length / 1000).toFixed(2)} km ` +
            `(${f.seconds.toFixed(0)} s); refining`);
        break;
      }
      log(`corridor ${f.index + 1} at ${f.res} m: ${Math.round(f.cost.total).toLocaleString()}, ` +
          `${(f.length / 1000).toFixed(2)} km, ${f.hairpins} hairpins, ` +
          (f.switchbacks ? `${f.switchbacks} switchbacks, ` : "") + `${f.bends} bends, ${f.fill_crossings} crossings ` +
          `(${f.history.length} searches, ${f.cells.toLocaleString()} cells, ${f.macros_computed.toLocaleString()} macro designs)`);
      break;
    }
    case "corridors": {
      if (m.final) haveCorridors = m.routes.length > 0;
      showCorridors(m.grid, m.routes);
      const r = m.routes[m.routes.length - 1];
      if (m.final) log(`corridors: ${m.routes.length}`);
      else log(`corridor ${m.routes.length} (${r.how}): ${Math.round(r.cost).toLocaleString()}, ${(r.length / 1000).toFixed(1)} km`);
      break;
    }
    case "error":
      log(m.message, "err");
      break;
  }
};

// ---------------------------------------------------------------- start

(async () => {
  defaults = await (await fetch("params.json")).json();
  setControls(defaults);
  examples = await (await fetch("examples.json")).json();
  for (const ex of examples) {
    const o = document.createElement("option");
    o.value = ex.name;
    o.textContent = ex.title;
    $("example").append(o);
  }
  void areaGrid;
})();
