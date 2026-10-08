// The page: controls, the map (Leaflet in UTM meters) and the worker that runs the pipeline.
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import proj4 from "proj4";
import "./style.css";
import { BASEMAPS, ExportLayer, utmEpsg } from "./basemap";
import { profileSummary, profileSvg } from "./profile";
import { boundsOf, CanvasLayer, rampColor, rasterLayer } from "./layers";
import type { FineOut, FromWorker, RouteOut, ToWorker } from "./messages";
import type { Params } from "./params";
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
let picking: "start" | "goal" | null = null;
let areaGrid: Grid | null = null;
let haveCorridors = false;
const fineRoutes: FineOut[] = [];
let profileIndex: number | null = null;   // corridor whose fine route the profile shows
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

const TASK_TEXT = { open: "Opening the DEM", load: "Loading the area", run: "Finding corridors", refine: "Refining corridors" } as const;
let busy: keyof typeof TASK_TEXT | null = null;
let busySince = 0;
let ticker: ReturnType<typeof setInterval> | undefined;

const statusBox = L.DomUtil.create("div", "status");
statusBox.hidden = true;
statusBox.innerHTML = '<div class="status-head"><span class="spinner"></span><strong class="status-task"></strong>' +
  '<span class="status-time"></span><button type="button" class="status-stop">Stop</button></div>' +
  '<div class="status-step"></div><progress class="status-bar"></progress>';
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
  for (const [k, b] of Object.entries(buttons)) b.classList.toggle("working", k === task);
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
    estimate();
    buttons.run.disabled = !areaGrid;
    buttons.refine.disabled = !haveCorridors;
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

// ---------------------------------------------------------------- controls

function zUnits(): string { return $<HTMLSelectElement>("z_units").value; }

function estimate() {
  const loadBtn = $<HTMLButtonElement>("load");
  if (!info || !start || !goal || busy) { loadBtn.disabled = true; return; }
  const m = Number($<HTMLInputElement>("margin").value) * 1000, res = Number($<HTMLInputElement>("res").value);
  const [bx0, by0, bx1, by1] = info.bbox;
  const w = (Math.min(Math.max(start[0], goal[0]) + m, bx1) - Math.max(Math.min(start[0], goal[0]) - m, bx0)) / res;
  const h = (Math.min(Math.max(start[1], goal[1]) + m, by1) - Math.max(Math.min(start[1], goal[1]) - m, by0)) / res;
  const cells = Math.max(0, w) * Math.max(0, h);
  const over = cells > MAX_CELLS;
  $("estimate").textContent = `${(cells / 1e6).toFixed(1)} million cells at ${res} m, about ` +
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
  p.tin_res = Number($<HTMLInputElement>("res").value);
  return p;
}

function setControls(p: Params) {
  $<HTMLInputElement>("g_max").value = String(+(100 * (p.g_max as number)).toFixed(2));
  $<HTMLInputElement>("w").value = String(p.w);
  $<HTMLInputElement>("res").value = String(p.tin_res ?? 15);
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
  if (!start || !goal) return;
  areaGrid = null;
  haveCorridors = false;
  clearFine();
  $<HTMLButtonElement>("load").disabled = true;
  $<HTMLButtonElement>("run").disabled = true;
  send({ type: "load", start, goal, margin: Number($<HTMLInputElement>("margin").value) * 1000,
         res: Number($<HTMLInputElement>("res").value), maxCells: MAX_CELLS, zUnits: zUnits() });
});
$("refine").addEventListener("click", () => {
  const res = Number($<HTMLInputElement>("fine-res").value);
  if (!(res > 0)) return;
  for (const b of ["load", "run", "refine"]) $<HTMLButtonElement>(b).disabled = true;
  send({ type: "refine", params: params(), res, which: $<HTMLSelectElement>("fine-which").value as "best" | "all" });
});

$("run").addEventListener("click", () => {
  haveCorridors = false;
  clearFine();
  $<HTMLButtonElement>("load").disabled = true;
  $<HTMLButtonElement>("run").disabled = true;
  for (const name of ["Slope mask", "Channels", "Saddles", "Triangulation", "Graph nodes (corridor field)",
                      "Graph edges", "Corridors"]) setLayer(name, null);
  $("corridors").innerHTML = "";
  send({ type: "run", params: params() });
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
  fineRoutes.length = 0;
  profileIndex = null;
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
  $("profile-note").textContent = f ? profileSummary(f) + ". Shaded: hairpins (red), bends (blue), fill crossings (yellow)." : "";
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

$("fine").addEventListener("click", (e) => {
  const tr = (e.target as HTMLElement).closest<HTMLTableRowElement>("tr[data-index]");
  if (!tr) return;
  profileIndex = Number(tr.dataset.index);
  showProfile();
});

/** The fine routes, drawn over their corridors, and their table. */
function showFine() {
  const css = getComputedStyle(document.documentElement);
  const group = L.layerGroup();
  const sorted = [...fineRoutes].sort((a, b) => a.cost.total - b.cost.total);
  for (const f of sorted) {
    const pts = f.x.map((x, i) => [f.y[i], x] as [number, number]);
    const best = f === sorted[0];
    // A dark casing under the line keeps it readable over imagery; a route still being refined is dashed.
    L.polyline(pts, { color: "#111", weight: best ? 7 : 5, opacity: 0.55, interactive: false }).addTo(group);
    L.polyline(pts, { color: (best ? css.getPropertyValue("--fine") : css.getPropertyValue("--fine-alt")).trim(),
                      weight: best ? 4 : 3, dashArray: f.final ? undefined : "8 6" })
      .bindTooltip(`corridor ${f.index + 1}, fine: ${Math.round(f.cost.total).toLocaleString()}, ` +
                   `${(f.length / 1000).toFixed(1)} km, ${f.hairpins} hairpins`)
      .addTo(group);
  }
  setLayer("Fine routes", group);
  const head = "<tr><th>corridor</th><th>cost</th><th>build</th><th>operate</th><th>km</th><th>hairpins</th>" +
               "<th>bends</th><th>crossings</th><th>s</th></tr>";
  $("fine").innerHTML = head + sorted.map((f) =>
    `<tr data-index="${f.index}"><td>${f.index + 1}${f.final ? "" : " (refining)"}</td><td>${Math.round(f.cost.total).toLocaleString()}</td>` +
    `<td>${Math.round(f.cost.construction).toLocaleString()}</td><td>${Math.round(f.cost.operating).toLocaleString()}</td>` +
    `<td>${(f.length / 1000).toFixed(2)}</td><td>${f.hairpins}</td><td>${f.bends}</td><td>${f.fill_crossings}</td>` +
    `<td>${f.seconds.toFixed(0)}</td></tr>`).join("");
  showProfile();
  $<HTMLButtonElement>("download").disabled = !fineRoutes.some((f) => f.final);
}

function showCorridors(g: Grid, routes: RouteOut[]) {
  const css = getComputedStyle(document.documentElement);
  const group = L.layerGroup();
  routes.forEach((r, i) => {
    const pts = r.rows.map((row, k) => [g.y1 - (row + 0.5) * g.res, g.x0 + (r.cols[k] + 0.5) * g.res] as [number, number]);
    L.polyline(pts, { color: (i === 0 ? css.getPropertyValue("--route") : css.getPropertyValue("--alt")).trim(),
                      weight: i === 0 ? 4 : 2.5, opacity: i === 0 ? 1 : 0.8 })
      .bindTooltip(`${i + 1} (${r.how}): ${Math.round(r.cost).toLocaleString()}, ${(r.length / 1000).toFixed(1)} km`)
      .addTo(group);
  });
  setLayer("Corridors", group);
  const head = "<tr><th>#</th><th>found by</th><th>cost</th><th>km</th><th>hairpins</th><th>turning °</th></tr>";
  $("corridors").innerHTML = head + routes.map((r, i) =>
    `<tr><td>${i + 1}</td><td>${r.how}</td><td>${Math.round(r.cost).toLocaleString()}</td>` +
    `<td>${(r.length / 1000).toFixed(1)}</td><td>${r.hairpins}</td><td>${Math.round(r.turning * 180 / Math.PI).toLocaleString()}</td></tr>`).join("");
}

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
    case "graph": {
      const g = m.grid;
      let fmin = Infinity;
      for (const v of m.field) if (v < fmin) fmin = v;
      log(`graph: ${m.rows.length.toLocaleString()} nodes, ${(m.pairs.length / 2).toLocaleString()} edges`);
      setLayer("Graph nodes (corridor field)", pointsLayer(g, m.rows, m.cols, (i) => {
        const f = m.field[i];
        return Number.isFinite(f) ? rampColor((Math.round((f / fmin - 1) * 20) / 20)) : "rgba(120,120,120,0.4)";
      }, 3));
      setLayer("Graph edges", segmentsLayer(g, m.rows, m.cols, m.pairs, "rgba(80,80,160,0.25)", 0.5, 300_000), false);
      setLayer("Triangulation", layers.get("Triangulation") ?? null, false);
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
          `${(f.length / 1000).toFixed(2)} km, ${f.hairpins} hairpins, ${f.bends} bends, ${f.fill_crossings} crossings ` +
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
