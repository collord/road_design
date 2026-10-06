"""Run results to a GeoPackage, for GIS: one set of layers per output root.

  <prefix>_routes    one 3D line per route (best and alternatives), with its costs,
                     counts and the run's key parameters
  <prefix>_segments  the routes piece by piece: section type, earthwork, costs
  <prefix>_features  hairpins, bends, crossings and stacked-leg conflicts
                     (not the hairpin candidates, which are millions of points)
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

KEY_PARAMS = ("w", "g_max", "R_min", "R_hp", "w_hp", "g_hp", "C_base", "C_travel", "C_vertical", "C_turn", "N_theta")


def _label(params: dict, varying: list[str], exclude: list[str]) -> str:
    bits = []
    for k in varying:
        v = params[k]
        bits.append(f"{k} {v * 100:g}%" if k == "g_max" else f"{k} {v:g}")
    if exclude:
        bits.append("with exclusion zones")
    return ", ".join(bits) or "run"


def export_runs(out_root: str | Path, gpkg: str | Path, prefix: str, run_ids: list[str] | None = None,
                labels: list[str] | None = None) -> dict:
    """Write the layers for the runs in out_root (all of them, or run_ids in that order)."""
    from .route import self_overlaps

    out_root = Path(out_root)
    summ = gpd.read_parquet(out_root / "route_summary.parquet")
    run_ids = run_ids or list(dict.fromkeys(summ["run_id"]))
    recs = {r: json.loads((out_root / "runs" / r / "params.json").read_text(encoding="utf-8")) for r in run_ids}
    varying = [k for k in KEY_PARAMS if len({recs[r]["params"].get(k) for r in run_ids}) > 1]
    some_excl = len({tuple(recs[r]["inputs"].get("exclude", [])) for r in run_ids}) > 1
    if labels is None:
        labels = [_label(recs[r]["params"], varying, recs[r]["inputs"].get("exclude", []) if some_excl else [])
                  for r in run_ids]
    lab = dict(zip(run_ids, labels))
    order = {r: i for i, r in enumerate(run_ids)}

    def tag(df):
        df = df[df["run_id"].isin(run_ids)].copy()
        df.insert(0, "label", df["run_id"].map(lab))
        df.insert(1, "run_order", df["run_id"].map(order))
        return df

    routes = tag(summ)
    segs = tag(gpd.read_parquet(out_root / "routes.parquet"))
    w = {r: recs[r]["params"]["w"] for r in run_ids}
    crossings = []
    for _, row in routes.iterrows():
        pieces = [np.asarray(g.coords)[:, :2][:, ::-1]
                  for g in segs.loc[(segs["run_id"] == row["run_id"]) & (segs["route_id"] == row["route_id"])]
                  .sort_values("seq").geometry]
        crossings.append(len(self_overlaps(pieces, 1.0, w[row["run_id"]])) if len(pieces) > 2 else 0)
    routes["self_overlaps"] = crossings
    for k in KEY_PARAMS:
        routes[k] = routes["run_id"].map(lambda r: recs[r]["params"].get(k))
    routes["exclude"] = routes["run_id"].map(lambda r: "; ".join(recs[r]["inputs"].get("exclude", [])))
    routes = routes.sort_values(["run_order", "rank"])

    feats = gpd.read_parquet(out_root / "features.parquet")
    feats = tag(feats[feats["feature_type"] != "hairpin_candidate"])
    if len(feats):
        keep = [c for c in feats.columns if c in ("label", "run_order", "geometry") or feats[c].notna().any()]
        feats = feats[keep]

    layers = {f"{prefix}_routes": routes, f"{prefix}_segments": segs.sort_values(["run_order", "rank", "seq"]),
              f"{prefix}_features": feats.sort_values(["run_order", "rank", "feature_type"])}
    for name, df in layers.items():
        df = gpd.GeoDataFrame(df.drop(columns=["run_order"]), geometry="geometry", crs=summ.crs)
        # Plain types only: GeoPackage has no column type for Python objects mixing None and numbers.
        for c in df.columns:
            if c != "geometry" and df[c].dtype == object:
                df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        df.to_file(gpkg, layer=name, driver="GPKG")
    return {name: len(df) for name, df in layers.items()}
