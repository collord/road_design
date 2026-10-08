"""Data for the browser app (web/): default parameters and the bundled examples.

    python -m roadsearch web-data --dest web/public

writes params.json (the defaults of params.toml) and examples.json (each example of
examples.toml with its full parameter set, endpoints in the DEM's CRS and its DEM copied to
examples/). Infinite values are written as null (JSON has no infinity).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from .examples import REPO, load_examples
from .params import load_params


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _params(path=None) -> dict:
    p = load_params(REPO / path if path else None)
    return {k: _clean(v) for k, v in p.to_dict().items()}


def write_web_data(dest) -> list[Path]:
    dest = Path(dest)
    (dest / "examples").mkdir(parents=True, exist_ok=True)
    written = [dest / "params.json"]
    (dest / "params.json").write_text(json.dumps(_params(), indent=1), encoding="utf-8")
    examples = []
    for name, ex in load_examples().items():
        dem = REPO / ex["dem"]
        shutil.copyfile(dem, dest / "examples" / dem.name)
        written.append(dest / "examples" / dem.name)
        examples.append({
            "name": name, "title": ex["title"], "heading": ex.get("heading", ex["title"]),
            "dem": f"examples/{dem.name}",
            "start": [float(v) for v in ex["start"].split(",")],
            "goal": [float(v) for v in ex["goal"].split(",")],
            "params": _params(ex.get("params")),
        })
    (dest / "examples.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")
    written.append(dest / "examples.json")
    return written
