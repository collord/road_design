"""Worked examples (examples.toml) and the static site that shows them.

`python -m roadsearch example NAME` runs one example's sweep and builds its
viewer; `python -m roadsearch site` builds every example's viewer into
site/<name>/ (reusing finished runs) and writes a landing page linking them.
"""

from __future__ import annotations

import html
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = REPO / "examples.toml"


def load_examples(path: str | Path = EXAMPLES_PATH) -> dict[str, dict]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def sweep_argv(name: str, ex: dict, k: str | None = None, out: str | None = None, dest: str | None = None,
               no_viewer: bool = False) -> list[str]:
    """Arguments for `roadsearch sweep` that run example `name`."""
    argv = ["sweep", "--dem", ex["dem"], "--start", ex["start"], "--goal", ex["goal"], "--params", ex["params"],
            "--out", out or f"out-{name}", "--title", ex["title"], "--heading", ex["heading"]]
    extras = ex.get("extra", [])
    only_extra = k is not None and k.startswith("extra")
    if only_extra:                      # just one of the extra runs (CI runs each as its own job)
        extras = [extras[int(k[len("extra"):])]]
        k = ""
    elif k is not None:
        extras = []
    for e in extras:
        sets = ",".join(f"{n}={v:g}" for n, v in e.items() if n != "label")
        argv += ["--extra", f"{e['label']}|{sets}"]
    if "reference" in ex:
        argv += ["--reference", ex["reference"]["file"], "--reference-label", ex["reference"]["label"]]
    if "vary" in ex:
        v = ex["vary"]
        values = k if k is not None else ",".join(f"{x:g}" for x in v["values"])
        argv += ["--vary", f"{v['param']}={values}", "--vary-label", v.get("label", v["param"]),
                 "--vary-unit", v.get("unit", "")]
    else:
        argv += ["--k", k if k is not None else ",".join(f"{x:g}" for x in ex["k"])]
    if ex.get("resolution"):
        argv += ["--resolution", str(ex["resolution"])]
    if ex.get("coarse_res"):
        argv += ["--coarse-res", str(ex["coarse_res"])]
    if dest:
        argv += ["--dest", dest]
    if no_viewer:
        argv.append("--no-viewer")
    return argv


LANDING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>roadsearch examples</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600&family=Barlow:wght@400;500&display=swap">
<style>
:root { --paper: #eef0ec; --sheet: #f8f9f6; --ink: #1d2421; --muted: #5b6660; --rule: #c9cfc9; --road: #b91c1c; }
@media (prefers-color-scheme: dark) { :root { --paper: #121614; --sheet: #191e1b; --ink: #e4e9e5; --muted: #95a39b;
  --rule: #2d3531; --road: #f87171; color-scheme: dark; } }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 "Barlow", "Helvetica Neue", Arial, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 40px 16px 56px; display: grid; gap: 20px; }
h1 { font: 600 clamp(1.8rem, 4vw, 2.6rem)/1.05 "Barlow Condensed", "Arial Narrow", sans-serif; margin: 0; }
p { margin: 0; max-width: 65ch; }
.lead { color: var(--muted); }
a.card { display: grid; gap: 6px; padding: 18px 20px; background: var(--sheet); border: 1px solid var(--rule);
         border-left: 4px solid var(--road); color: inherit; text-decoration: none; }
a.card:hover { border-color: var(--ink); border-left-color: var(--road); }
a.card h2 { font: 600 1.4rem/1.1 "Barlow Condensed", "Arial Narrow", sans-serif; margin: 0; }
a.card span { color: var(--muted); font-size: 0.95rem; }
footer { color: var(--muted); font-size: 0.85rem; }
footer a { color: inherit; }
</style></head>
<body><main>
<h1>roadsearch examples</h1>
<p class="lead">Least-cost road alignments over a DEM, with hairpins, earthwork, drainage and, optionally, truck
operating costs. Each example compares its routes, replays the search, lets you draw and cost your own route,
and explains how the search works.</p>
__CARDS__
<footer>DEMs: U.S. Geological Survey. __REPO__</footer>
</main></body></html>
"""


def write_landing(dest: str | Path, examples: dict[str, dict], repo_url: str = "") -> Path:
    cards = "\n".join(
        f'<a class="card" href="{html.escape(name)}/"><h2>{html.escape(ex["title"])}</h2>'
        f'<span>{html.escape(ex.get("description", ""))}</span></a>' for name, ex in examples.items())
    repo = f'Source: <a href="{html.escape(repo_url)}">{html.escape(repo_url)}</a>.' if repo_url else ""
    out = Path(dest) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(LANDING.replace("__CARDS__", cards).replace("__REPO__", repo), encoding="utf-8")
    return out


def run_matrix() -> str:
    """The CI job matrix, one entry per example run, as JSON (stdlib only, so it runs before installing)."""
    import json

    include = []
    for name, ex in load_examples().items():
        values = ex["vary"]["values"] if "vary" in ex else ex["k"]
        include += [{"example": name, "value": f"{v:g}"} for v in values]
        include += [{"example": name, "value": f"extra{i}"} for i in range(len(ex.get("extra", [])))]
    return json.dumps({"include": include})
