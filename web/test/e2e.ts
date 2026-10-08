// The page in headless Chrome, driven over the DevTools protocol: for each example, choose it,
// load the area, find corridors, and read the progress log and the corridor table.
// Run with the dev server up: node test/e2e.ts [url] (from web/). CHROME overrides the browser path.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
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

const logText = `[...document.querySelectorAll("#log li")].map((li) => (li.className === "err" ? "ERROR " : "") + li.textContent)`;
const hasError = `document.querySelector("#log li.err")?.textContent`;
let failed = false;
try {
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url });
  await until(`document.querySelectorAll("#example option").length > 1`, "the example list", 15000);
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
    await evaluate(`document.querySelector("#run").click()`);
    const done = await until(`(${hasError}) || (document.querySelectorAll("#corridors tr").length > 1 && "done")`, "corridors", 300000);
    if (done !== "done") throw new Error(done);
    for (const line of await evaluate(logText)) console.log(`  ${line}`);
    const rows: string[] = await evaluate(`[...document.querySelectorAll("#corridors tr")].slice(0, 3).map((r) => r.textContent.trim().replace(/\\s+/g, " "))`);
    for (const r of rows) console.log(`  | ${r}`);
    const layers: string[] = await evaluate(`[...document.querySelectorAll("#layers label")].map((l) => l.textContent.trim())`);
    console.log(`  layers: ${layers.join(", ")}`);
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
