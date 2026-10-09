// The page in headless Chrome, driven over the DevTools protocol: for each example, choose it,
// load the area, find corridors, and read the progress log and the corridor table.
// Run with the dev server up: node test/e2e.ts [url] (from web/). CHROME overrides the browser path.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "http://localhost:5173/";
const candidates = [
  process.env.CHROME,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter((p): p is string => !!p && existsSync(p));
if (!candidates.length) throw new Error("no Chrome or Edge found; set CHROME");

const profile = mkdtempSync(join(tmpdir(), "roadsearch-e2e-"));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(candidates[0], [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--window-size=1400,900", "about:blank",
], { stdio: "ignore" });

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let targets: { type: string; webSocketDebuggerUrl: string }[] = [];
for (let i = 0; i < 50 && !targets.some((t) => t.type === "page"); i++) {
  await sleep(200);
  targets = await fetch(`http://127.0.0.1:${port}/json`).then((r) => r.json()).catch(() => []);
}
const ws = new WebSocket(targets.find((t) => t.type === "page")!.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let nextId = 0;
const pending = new Map<number, (v: any) => void>();
const consoleErrors: string[] = [];
ws.addEventListener("message", (e) => {
  const m = JSON.parse(String(e.data));
  if (m.id !== undefined) pending.get(m.id)?.(m);
  else if (m.method === "Runtime.exceptionThrown") consoleErrors.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
  else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error")
    consoleErrors.push(m.params.args.map((a: any) => a.value ?? a.description).join(" "));
});
const send = (method: string, params: object = {}): Promise<any> =>
  new Promise((resolve) => { const id = ++nextId; pending.set(id, resolve); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression: string): Promise<any> => {
  const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? "evaluate failed");
  return r.result?.result?.value;
};
/** Polls until the expression is truthy; returns its value. */
const until = async (expression: string, what: string, timeoutMs: number): Promise<any> => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const v = await evaluate(expression);
    if (v) return v;
    await sleep(250);
  }
  throw new Error(`timed out waiting for ${what}`);
};

const shots = process.env.SHOTS;
/** A PNG of the page into $SHOTS, when set. */
const shot = async (name: string) => {
  if (!shots) return;
  await sleep(500);
  const r = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(shots, `${name}.png`), Buffer.from(r.result.data, "base64"));
};
const channelSrc = `[...document.querySelectorAll(".leaflet-overlay-pane img")].map((i) => i.src.length).join(",")`;
const statusShown = `getComputedStyle(document.querySelector(".status")).display !== "none"`;
const logText = `[...document.querySelectorAll("#log li")].map((li) => (li.className === "err" ? "ERROR " : "") + li.textContent)`;
const hasError = `document.querySelector("#log li.err")?.textContent`;
let failed = false;
try {
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url });
  await until(`document.querySelectorAll("#example option").length > 1`, "the example list", 15000);
  if (await evaluate(statusShown)) throw new Error("the status box shows before any task");
  const names: string[] = await evaluate(`[...document.querySelectorAll("#example option")].map((o) => o.value).filter(Boolean)`);
  for (const name of names) {
    console.log(`${name}:`);
    const t0 = Date.now();
    await evaluate(`(() => { const s = document.querySelector("#example"); s.value = ${JSON.stringify(name)};
                    s.dispatchEvent(new Event("change")); })()`);
    const loadReady = await until(`(${hasError}) || (!document.querySelector("#load").disabled && "ready")`, "the preview", 60000);
    if (loadReady !== "ready") throw new Error(loadReady);
    await evaluate(`document.querySelector("#load").click()`);
    const runReady = await until(`(${hasError}) || (!document.querySelector("#run").disabled && "ready")`, "the area", 120000);
    if (runReady !== "ready") throw new Error(runReady);
    // While it runs: the status box's steps, and when the first corridor (the best) shows.
    await evaluate(`document.querySelector("#run").click()`);
    const tRun = Date.now();
    const steps = new Set<string>();
    let firstRow = 0;
    for (;;) {
      const st = await evaluate(`({ err: ${hasError} ?? null, step: getComputedStyle(document.querySelector(".status")).display === "none" ? "" :
        document.querySelector(".status-step").textContent.replace(/[0-9,]+ of [0-9,]+/, "n of N"),
        rows: document.querySelectorAll("#corridors tr").length, idle: !document.querySelector("#run").disabled })`);
      if (st.err) throw new Error(st.err);
      if (st.step) steps.add(st.step);
      if (st.rows > 1 && !firstRow) firstRow = Date.now() - tRun;
      if (st.idle && st.rows > 1) break;
      if (Date.now() - tRun > 300000) throw new Error("timed out waiting for corridors");
      await sleep(100);
    }
    await until(`!(${statusShown})`, "the status box to close after the run", 5000);
    console.log(`  status steps seen: ${[...steps].join(" | ")}`);
    console.log(`  best corridor shown after ${(firstRow / 1000).toFixed(1)} s, all after ${((Date.now() - tRun) / 1000).toFixed(1)} s`);
    for (const line of await evaluate(logText)) console.log(`  ${line}`);
    const rows: string[] = await evaluate(`[...document.querySelectorAll("#corridors tr")].slice(0, 3).map((r) => r.textContent.trim().replace(/\\s+/g, " "))`);
    for (const r of rows) console.log(`  | ${r}`);
    const layers: string[] = await evaluate(`[...document.querySelectorAll("#layers label")].map((l) => l.textContent.trim())`);
    console.log(`  layers: ${layers.join(", ")}`);
    // The channel threshold redraws the Channels layer without a new run.
    const before: string = await evaluate(channelSrc);
    await evaluate(`(() => { const i = document.querySelector("#A_min"); i.value = "20"; i.dispatchEvent(new Event("change")); })()`);
    await until(`(${hasError}) || (${channelSrc}) !== ${JSON.stringify(before)}`, "the channels redraw", 20000);
    console.log(`  channel threshold 20 ha: Channels redrawn`);
    // Base maps, drawn by the map service in the DEM's UTM zone.
    for (const bm of ["USGS imagery", "Esri streets"]) {
      await evaluate(`(() => { const s = document.querySelector("#basemap"); s.value = ${JSON.stringify(bm)};
                      s.dispatchEvent(new Event("change")); })()`);
      const img = await until(`(${hasError}) || [...document.querySelectorAll(".leaflet-terrain-pane img")]
        .filter((i) => i.src.includes("/export?") && i.complete && i.naturalWidth > 0).map((i) => i.naturalWidth + " x " + i.naturalHeight)[0]`,
        `the ${bm} base map`, 30000);
      if (!/ x /.test(img)) throw new Error(img);
      console.log(`  base map ${bm}: ${img} px`);
      await shot(`${name}-${bm.replace(/\s+/g, "-").toLowerCase()}`);
    }
    await evaluate(`(() => { const s = document.querySelector("#basemap"); s.value = ""; s.dispatchEvent(new Event("change")); })()`);
    // The fine search of the best corridor (one example: it takes a minute or more).
    if (name === (process.env.FINE ?? "red-mountain")) {
      const tFine = Date.now();
      // FINE_WHICH=all refines every corridor; FINE_APART sets the cost factor on other corridors' ground.
      await evaluate(`(() => { document.querySelector("#fine-which").value = ${JSON.stringify(process.env.FINE_WHICH ?? "best")};
        document.querySelector("#fine-apart").value = ${JSON.stringify(process.env.FINE_APART ?? "2")}; })()`);
      await evaluate(`document.querySelector("#refine").click()`);
      // A picture of the search while it runs: a few seconds into the first A* search.
      let searching = 0, mid = false, designing = 0, sitesShot = false;
      const doneExpr = `(${hasError}) || (!document.querySelector("#refine").disabled &&
        document.querySelectorAll("#fine tr").length > 1 && "done")`;
      let fine = "";
      for (const t0 = Date.now(); Date.now() - t0 < 3600000;) {
        fine = await evaluate(doneExpr);
        if (fine) break;
        // And of the hairpin sites, about halfway through the helpers' designs.
        const note: string = await evaluate(`/hairpin and bend designs/.test(document.querySelector(".status-step").textContent)
          ? document.querySelector(".status-note")?.textContent ?? "" : ""`);
        if (note && !sitesShot && ++designing >= 60) {
          sitesShot = true;
          await shot(`${name}-sites`);
          console.log(`  while designing: ${note}`);
        }
        const plot = await evaluate(`!!document.querySelector(".status-plot svg") && /A\* search 1/.test(document.querySelector(".status-step").textContent)`);
        if (plot && !mid && ++searching >= 30) {
          mid = true;
          await shot(`${name}-searching`);
          const ok = await evaluate(`({ layers: [...document.querySelectorAll("#layers label")].map((l) => l.textContent.trim()),
            lines: document.querySelectorAll(".status-plot polyline").length })`);
          console.log(`  while searching: layers ${ok.layers.filter((l: string) => /wavefront|candidate/i.test(l)).join(", ")}; ` +
                      `chart lines ${ok.lines}`);
        }
        await sleep(250);
      }
      const table: string[] = await evaluate(`[...document.querySelectorAll("#fine tr")].map((r) => [...r.cells].map((c) => c.textContent.trim()).join(" | "))`);
      for (const t of table) console.log(`    ${t}`);
      if (fine !== "done") throw new Error(fine);
      const row: string = await evaluate(`[...document.querySelectorAll("#fine tr")][1].textContent.trim()`);
      const logLine: string = await evaluate(`[...document.querySelectorAll("#log li")].map((li) => li.textContent).filter((t) => /^corridor 1 at/.test(t)).pop()`);
      console.log(`  fine: ${logLine} in ${((Date.now() - tFine) / 1000).toFixed(0)} s`);
      const steps: string[] = await evaluate(`[...document.querySelectorAll("#log li")].map((li) => li.textContent).filter((t) => /^corridor 1/.test(t))`);
      for (const t of steps) console.log(`    ${t}`);
      const prof = await evaluate(`({ paths: document.querySelectorAll("#profile svg path").length,
        note: document.querySelector("#profile-note").textContent, download: !document.querySelector("#download").disabled })`);
      if (prof.paths !== 2 || !prof.download) throw new Error(`profile or download missing: ${JSON.stringify(prof)}`);
      console.log(`  profile: ${prof.note}`);
      // The GeoJSON download: lon/lat/z lines, a route and its segments.
      await send("Page.setDownloadBehavior", { behavior: "allow", downloadPath: profile });
      await evaluate(`document.querySelector("#download").click()`);
      let file = "";
      for (let i = 0; i < 40 && !file; i++) {
        await sleep(250);
        file = readdirSync(profile).find((f) => f.endsWith(".geojson")) ?? "";
      }
      if (!file) throw new Error("no GeoJSON downloaded");
      const gj = JSON.parse(readFileSync(join(profile, file), "utf-8"));
      const routesF = gj.features.filter((f: any) => f.properties.feature === "route");
      const segsF = gj.features.filter((f: any) => f.properties.feature === "segment");
      const [lon, lat, z] = routesF[0].geometry.coordinates[0];
      if (!(Math.abs(lon) <= 180 && Math.abs(lat) <= 90 && Number.isFinite(z))) throw new Error(`bad coordinates ${lon} ${lat} ${z}`);
      console.log(`  GeoJSON ${file}: ${routesF.length} route, ${segsF.length} segments, starts at ${lon}, ${lat}, ${z} m; ` +
                  `cost ${routesF[0].properties.cost_total.toLocaleString()}`);
      await evaluate(`document.querySelector("#profile").scrollIntoView()`);
      await shot(`${name}-fine`);
    }
    // A pad network (one example): 16 pads scattered over the start-finish box, the start as the
    // access point, loaded from a CSV through the file input; at most B = 2 pads beyond any pad.
    if (name === (process.env.NETWORK ?? "haul-road")) {
      const box: number[][] = await evaluate(`[document.querySelector("#start-text").value, document.querySelector("#goal-text").value]
        .map((t) => t.split(",").map(Number))`);
      const [[sx, sy], [gx, gy]] = box;
      let seed = 7;
      const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
      const rows = ["x,y"];
      for (let i = 0; i < 16; i++) {
        rows.push(`${(Math.min(sx, gx) - 500 + rnd() * (Math.abs(gx - sx) + 1000)).toFixed(1)},` +
                  `${(Math.min(sy, gy) - 500 + rnd() * (Math.abs(gy - sy) + 1000)).toFixed(1)}`);
      }
      const csv = join(profile, "pads.csv");
      writeFileSync(csv, rows.join(String.fromCharCode(10)));
      await evaluate(`(() => { const s = document.querySelector("#mode"); s.value = "network"; s.dispatchEvent(new Event("change")); })()`);
      const doc = await send("DOM.getDocument", {});
      const q = await send("DOM.querySelector", { nodeId: doc.result.root.nodeId, selector: "#pads-file" });
      await send("DOM.setFileInputFiles", { nodeId: q.result.nodeId, files: [csv] });
      await until(`/16 pads/.test(document.querySelector("#pads-info").textContent)`, "the pads", 10000);
      await until(`!document.querySelector("#load").disabled`, "load enabled", 10000);
      await evaluate(`document.querySelector("#load").click()`);
      await until(`(${hasError}) || (!document.querySelector("#run").disabled && "ready")`, "the network's area", 120000);
      const tNet = Date.now();
      await evaluate(`document.querySelector("#run").click()`);
      const net = await until(`(${hasError}) || (!document.querySelector("#run").disabled &&
        [...document.querySelectorAll("#corridors td")].some((td) => td.textContent === "done") && "done")`, "the network", 600000);
      if (net !== "done") throw new Error(net);
      const rowsOut: string[] = await evaluate(`[...document.querySelectorAll("#corridors tr")].map((r) => [...r.cells].map((c) => c.textContent).join(": "))`);
      console.log(`  network in ${((Date.now() - tNet) / 1000).toFixed(0)} s: ${rowsOut.slice(1).join("; ")}`);
      const most = Number(rowsOut.find((r) => r.startsWith("most pads beyond"))?.split(": ")[1]);
      if (!(most <= 2)) throw new Error(`a pad has ${most} pads beyond it`);
      await shot(`${name}-network`);
      // The fine network: the same pads on the lattice at 5 m, within the buffer (default 300 m) of the coarse roads.
      if (process.env.FINE_NETWORK !== "0") {
        const tFine = Date.now();
        await evaluate(`document.querySelector("#refine").click()`);
        const fn = await until(`(${hasError}) || (!document.querySelector("#refine").disabled &&
          [...document.querySelectorAll("#fine td")].some((td) => td.textContent === "done") && "done")`, "the fine network", 3600000);
        if (fn !== "done") throw new Error(fn);
        const fineRows: string[] = await evaluate(`[...document.querySelectorAll("#fine tr")].map((r) => [...r.cells].map((c) => c.textContent).join(": "))`);
        console.log(`  fine network in ${((Date.now() - tFine) / 1000).toFixed(0)} s: ${fineRows.slice(1).join("; ")}`);
        const mostFine = Number(fineRows.find((r) => r.startsWith("most pads beyond"))?.split(": ")[1]);
        if (!(mostFine <= 2)) throw new Error(`a pad has ${mostFine} pads beyond it`);
        await shot(`${name}-fine-network`);
      }
      await evaluate(`(() => { const s = document.querySelector("#mode"); s.value = "route"; s.dispatchEvent(new Event("change")); })()`);
    }
    // Stop: a second run, stopped while it costs the edges, leaves the page ready to run again.
    if (name === names[0]) {
      await evaluate(`document.querySelector("#run").click()`);
      await until(`(${hasError}) || /cost the edges/i.test(document.querySelector(".status-step").textContent)`, "edge costing", 60000);
      await evaluate(`document.querySelector(".status-stop").click()`);
      const stopped = await until(`(${hasError}) || (!document.querySelector("#run").disabled && getComputedStyle(document.querySelector(".status")).display === "none" &&
        [...document.querySelectorAll("#log li")].some((li) => li.textContent.startsWith("stopped")) && "stopped")`, "the stop", 30000);
      if (stopped !== "stopped" && !/^stopped/.test(stopped)) throw new Error(stopped);
      await evaluate(`document.querySelectorAll("#log li.err").forEach((li) => li.remove())`);
      console.log(`  stop: the run ended while costing edges; Find corridors is enabled again`);
    }
    console.log(`  ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
} catch (e) {
  failed = true;
  console.log(`FAIL ${(e as Error).message}`);
  for (const line of await evaluate(logText).catch(() => [])) console.log(`  ${line}`);
} finally {
  for (const e of consoleErrors) console.log(`  console: ${e}`);
  if (consoleErrors.length) failed = true;
  ws.close();
  chrome.kill();
  await sleep(500);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still hold files */ }
}
process.exit(failed ? 1 : 0);
