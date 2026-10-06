"""Command-line interface.

    python -m roadsearch inspect DEM
    python -m roadsearch preprocess --dem DEM --start X,Y --goal X,Y [--exclude FILE ...]
    python -m roadsearch run --dem DEM --start X,Y --goal X,Y [--clip auto|box|none] [--resolution M]
    python -m roadsearch sweep --dem DEM --start X,Y --goal X,Y [--k 0,0.1,1,10,100] [run options]
    python -m roadsearch animate RUN_ID [RUN_ID ...] [--factors 0,1,...]
    python -m roadsearch burn RUN_ID [--rank 1] [--res 2]
    python -m roadsearch tree RUN_ID [RUN_ID ...]
    python -m roadsearch merge SRC [SRC ...] --out DEST
    python -m roadsearch example NAME [--k K] [--out DIR]     (examples.toml)
    python -m roadsearch site [--dest site]
    python -m roadsearch export OUT.gpkg --out-root DIR --prefix NAME [--run-ids ...] [--labels ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import rasterio
from pyproj import CRS

from .exclusion import ExclusionError, normalize_spec
from .params import add_param_arguments, params_from_args
from .preprocess import ConnectivityError, EndpointError, preprocess
from .run import make_run_id, run
from .terrain import format_inspection, inspect_dem, load_grid, parse_point


def cmd_inspect(args) -> int:
    p = params_from_args(args)
    info = inspect_dem(args.dem, p)
    print(json.dumps(info, indent=2) if args.json else format_inspection(info))
    return 0


def cmd_preprocess(args) -> int:
    p = params_from_args(args)
    dem_info = inspect_dem(args.dem, p)
    grid, grid_info = load_grid(args.dem, p, args.resolution)
    with rasterio.open(args.dem) as ds:
        dem_crs = CRS.from_user_input(ds.crs)
    pt_crs = CRS.from_user_input(args.endpoint_crs) if args.endpoint_crs else dem_crs
    start = parse_point(args.start, pt_crs, grid.crs)
    goal = parse_point(args.goal, pt_crs, grid.crs)

    inputs = {
        "dem": dem_info["path"], "start": start, "goal": goal,
        "exclude": [normalize_spec(e) for e in args.exclude],
        "resolution": grid.res, "params": p.to_dict(),
    }
    run_id = args.run_id or make_run_id(inputs)
    run_dir = Path(args.out) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "inputs": inputs, "params": p.to_dict(), "dem": dem_info, "grid": grid_info}

    status = 0
    try:
        terrain = preprocess(grid, start, goal, p, args.exclude, run_dir)
        record["preprocess"] = terrain.info
        print(f"run {run_id}: preprocessing done, goal reachable. Rasters in {run_dir}")
    except (EndpointError, ExclusionError, ConnectivityError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"run {run_id}: {record['error']}", file=sys.stderr)
        status = 2
    (run_dir / "params.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return status


def cmd_run(args) -> int:
    p = params_from_args(args)
    rec = run(args.dem, args.start, args.goal, p, args.exclude, args.out, args.resolution, args.clip,
              args.clip_margin, args.coarse_res, args.run_id, args.endpoint_crs)
    return 2 if "error" in rec else 0


def cmd_sweep(args) -> int:
    """One run per value of the swept quantity, reusing finished runs, then the viewer over all of them.

    By default the sweep is over the operating-cost weight k (C_travel and C_vertical times k);
    --vary NAME=v1,v2,... sweeps any other parameter instead.
    """
    from .animate import write_viewer
    from .params import Params

    p = params_from_args(args)
    variants = []          # (params, label, value)
    if args.vary:
        name, _, vals = args.vary.partition("=")
        if name not in {f.name for f in dataclasses.fields(Params)} or (not vals and not args.extra):
            raise SystemExit(f"--vary needs NAME=v1,v2,... with NAME a parameter, got {args.vary!r}")
        unit = f" {args.vary_unit}" if args.vary_unit else ""
        for v in (float(x) for x in vals.split(",") if x):
            variants.append((dataclasses.replace(p, **{name: v}), f"{v:g}{unit}", v, []))
    else:
        ks = [float(v) for v in args.k.split(",") if v]
        if p.C_travel == 0 and p.C_vertical == 0 and any(k != 0 for k in ks):
            raise SystemExit("sweep multiplies C_travel and C_vertical by k: set their k = 1 values "
                             "(params_haul.toml has them), or use --k 0 for construction cost only")
        for k in ks:
            # Rounded so that k x the unit cost hashes like the same value typed on the command line.
            variants.append((dataclasses.replace(p, C_travel=float(f"{k * p.C_travel:.12g}"),
                                                 C_vertical=float(f"{k * p.C_vertical:.12g}"),
                                                 C_turn=float(f"{k * p.C_turn:.12g}")), f"k={k:g}", k, []))
    for spec in args.extra:
        # "LABEL|NAME=V,NAME=V[|exclude=FILE[|WHERE]]": one more run beside the sweep, e.g.
        # "20 m, 8% grade|R_min=20,g_max=0.08", optionally with more exclusion zones.
        lab, _, rest = spec.partition("|")
        sets, _, excl = rest.partition("|exclude=")
        over = dict(kv.split("=", 1) for kv in sets.split(",") if kv)
        variants.append((dataclasses.replace(p, **{k: float(v) for k, v in over.items()}), lab, None,
                         [excl] if excl else []))
    if not variants:
        raise SystemExit("nothing to run: give --k or --vary values, or --extra")
    ids, labels, values = [], [], []
    for pv, lab, v, excl in variants:
        print(f"{lab}: running")
        rec = run(args.dem, args.start, args.goal, pv, list(args.exclude) + excl, args.out, args.resolution, args.clip,
                  args.clip_margin, args.coarse_res, None, args.endpoint_crs, reuse=True)
        if "error" in rec:
            print(f"{lab}: {rec['error']}", file=sys.stderr)
            continue
        ids.append(rec["run_id"])
        labels.append(lab)
        values.append(v)
    if not ids:
        return 2
    print("runs: " + ", ".join(f"{lab} {r}" for lab, r in zip(labels, ids)))
    if not args.no_viewer:
        dest = Path(args.dest) if args.dest else Path(args.out) / "viewer"
        ref = (args.reference, args.reference_label) if args.reference else None
        if args.vary:
            path = write_viewer(ids, args.out, dest, labels=labels, title=args.title, heading=args.heading,
                                column=args.vary_label or args.vary.partition("=")[0], reference=ref)
        else:
            path = write_viewer(ids, args.out, dest, labels=labels, factors=values, title=args.title,
                                heading=args.heading, reference=ref)
        print(f"wrote {path}")
    return 0 if len(ids) == len(variants) else 2


def cmd_animate(args) -> int:
    from .animate import write_viewer

    factors = [float(v) for v in args.factors.split(",")] if args.factors else None
    if factors and len(factors) != len(args.run_ids):
        raise SystemExit("--factors needs one value per run")
    dest = Path(args.dest) if args.dest else Path(args.out) / "viewer"
    labels = args.labels.split(";") if args.labels else None
    if labels and len(labels) != len(args.run_ids):
        raise SystemExit("--labels needs one label per run")
    ref = (args.reference, args.reference_label) if args.reference else None
    path = write_viewer(args.run_ids, args.out, dest, labels=labels, factors=factors, title=args.title,
                        heading=args.heading, column=args.column, reference=ref)
    print(f"wrote {path} and {len(args.run_ids)} data files in {dest / 'data'}")
    return 0


def cmd_tree(args) -> int:
    from .explore import backfill_tree

    for rid in args.run_ids:
        backfill_tree(Path(args.out) / "runs" / rid)
    return 0


def cmd_merge(args) -> int:
    from .outputs import merge_outputs

    ids = merge_outputs([Path(s) for s in args.sources], Path(args.out))
    print(f"merged {len(ids)} runs into {args.out}: {', '.join(ids)}")
    return 0


def cmd_example(args) -> int:
    from .examples import load_examples, sweep_argv

    ex = load_examples()
    if args.name not in ex:
        raise SystemExit(f"unknown example {args.name!r}; examples.toml has: {', '.join(ex)}")
    return main(sweep_argv(args.name, ex[args.name], args.k, args.out, args.dest, args.no_viewer))


def cmd_site(args) -> int:
    """Every example's viewer into DEST/<name>/, reusing finished runs, plus a landing page."""
    from .examples import load_examples, sweep_argv, write_landing

    ex = load_examples()
    status = 0
    for name, e in ex.items():
        status = max(status, main(sweep_argv(name, e, dest=str(Path(args.dest) / name))))
    print(f"wrote {write_landing(args.dest, ex, args.repo_url)}")
    return status


def cmd_export(args) -> int:
    from .export import export_runs

    labels = args.labels.split(";") if args.labels else None
    n = export_runs(args.out_root, args.gpkg, args.prefix, args.run_ids or None, labels)
    print(f"wrote {args.gpkg}: " + ", ".join(f"{k} ({v} rows)" for k, v in n.items()))
    return 0


def cmd_burn(args) -> int:
    from .burn import burn_route

    summ = burn_route(Path(args.out) / "runs" / args.run_id, args.out, args.rank, args.res, args.buffer)
    v, m = summ["burned"], summ["section_model"]
    print(f"burned rank {args.rank} of run {args.run_id} at {args.res:g} m:")
    print(f"  cut soil {v['cut_soil_m3']:,.0f} m3 (section model {m['cut_soil_m3']:,.0f}), "
          f"cut rock {v['cut_rock_m3']:,.0f} (model {m['cut_rock_m3']:,.0f}), "
          f"fill {v['fill_m3']:,.0f} (model {m['fill_m3']:,.0f}); sidecast spoil not placed {m['spoil_sidecast_m3']:,.0f}")
    print(f"  wall cells {v['wall_cells']}; rasters in {Path(args.out) / 'runs' / args.run_id / 'burn'}")
    return 0


def add_run_arguments(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--dem", required=True)
    sp.add_argument("--start", required=True, help="x,y")
    sp.add_argument("--goal", required=True, help="x,y")
    sp.add_argument("--endpoint-crs", default=None,
                    help="CRS of --start/--goal (default: the DEM's CRS; EPSG:4326 for lon,lat)")
    sp.add_argument("--exclude", action="append", default=[], help="polygon file to keep out of, optionally filtered as 'FILE|WHERE' (e.g. \"zones.gpkg|type='avoid'\"); repeatable")
    sp.add_argument("--resolution", type=float, default=None,
                    help="working resolution (m); default fits max_states over the (clipped) grid")
    sp.add_argument("--clip", choices=["none", "box", "auto"], default="auto",
                    help="none: whole DEM; box: endpoint box + --clip-margin; auto: coarse search of the endpoint "
                         "box + clip_coarse_margin, then a fine search of its corridor (default)")
    sp.add_argument("--clip-margin", type=float, default=1000.0, help="box margin (m)")
    sp.add_argument("--coarse-res", type=float, default=None,
                    help="resolution of the --clip auto coarse pass (default: finest that fits max_states "
                         "over the endpoint box + clip_coarse_margin)")
    sp.add_argument("--out", default="out")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="roadsearch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect", help="report DEM metadata and propose a working resolution")
    sp.add_argument("dem")
    sp.add_argument("--json", action="store_true")
    add_param_arguments(sp)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("preprocess", help="build terrain rasters, masks and the connectivity check")
    sp.add_argument("--dem", required=True)
    sp.add_argument("--start", required=True, help="x,y")
    sp.add_argument("--goal", required=True, help="x,y")
    sp.add_argument("--endpoint-crs", default=None,
                    help="CRS of --start/--goal (default: the DEM's CRS; EPSG:4326 for lon,lat)")
    sp.add_argument("--exclude", action="append", default=[], help="polygon file to keep out of, optionally filtered as 'FILE|WHERE' (e.g. \"zones.gpkg|type='avoid'\"); repeatable")
    sp.add_argument("--resolution", type=float, default=None, help="working resolution (m)")
    sp.add_argument("--out", default="out")
    sp.add_argument("--run-id", default=None)
    add_param_arguments(sp)
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("run", help="full run: search, alternatives, outputs, quick-look")
    add_run_arguments(sp)
    sp.add_argument("--run-id", default=None)
    add_param_arguments(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("sweep", help="one run per operating-cost weight k, reusing finished runs, then the viewer")
    add_run_arguments(sp)
    sp.add_argument("--k", default="0,0.1,1,10,100",
                    help="comma-separated weights; each run uses k x C_travel and k x C_vertical")
    sp.add_argument("--vary", default=None, metavar="NAME=V1,V2,...",
                    help="sweep this parameter instead of k, e.g. R_min=10,20,30")
    sp.add_argument("--vary-label", default=None, help="what --vary sweeps, for the viewer (e.g. 'Minimum turn radius')")
    sp.add_argument("--vary-unit", default="", help="unit of the --vary values, for labels (e.g. m)")
    sp.add_argument("--extra", action="append", default=[], metavar="LABEL|NAME=V,...",
                    help="one more run beside the sweep, with these parameters changed; repeatable")
    sp.add_argument("--reference", default=None, help="a centerline (any vector file) to show and cost for comparison")
    sp.add_argument("--reference-label", default="Reference", help="its label in the viewer")
    sp.add_argument("--dest", default=None, help="viewer folder (default: <out>/viewer)")
    sp.add_argument("--no-viewer", action="store_true")
    sp.add_argument("--title", default="Road Search", help="viewer page title (browser tab)")
    sp.add_argument("--heading", default=None, help="viewer heading (default: from the title, or a sweep heading)")
    add_param_arguments(sp)
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("animate", help="write the run comparison and wavefront viewer (static HTML site)")
    sp.add_argument("run_ids", nargs="+")
    sp.add_argument("--out", default="out")
    sp.add_argument("--dest", default=None, help="site folder (default: <out>/viewer)")
    sp.add_argument("--title", default="Road Search", help="viewer page title (browser tab)")
    sp.add_argument("--heading", default=None, help="viewer heading (default: from the title, or a sweep heading)")
    sp.add_argument("--labels", default=None, help="';'-separated label per run")
    sp.add_argument("--column", default=None, help="table heading for the labels, e.g. 'Run'")
    sp.add_argument("--reference", default=None, help="a centerline to show and cost for comparison")
    sp.add_argument("--reference-label", default="Reference", help="its label in the viewer")
    sp.add_argument("--factors", default=None,
                    help="comma-separated operating-cost weight k of each run, for labels (0 = construction only)")
    sp.set_defaults(func=cmd_animate)

    sp = sub.add_parser("tree", help="save the corridor tree of runs made before it was saved (for the viewer's "
                                     "'best route through a point')")
    sp.add_argument("run_ids", nargs="+")
    sp.add_argument("--out", default="out")
    sp.set_defaults(func=cmd_tree)

    sp = sub.add_parser("merge", help="combine output roots (run folders and GeoParquet files) into one")
    sp.add_argument("sources", nargs="+")
    sp.add_argument("--out", default="out")
    sp.set_defaults(func=cmd_merge)

    sp = sub.add_parser("example", help="run a worked example from examples.toml and build its viewer")
    sp.add_argument("name")
    sp.add_argument("--k", default=None,
                    help="only these values of the swept quantity, comma-separated (default: the example's list)")
    sp.add_argument("--out", default=None, help="output root (default: out-<name>)")
    sp.add_argument("--dest", default=None, help="viewer folder (default: <out>/viewer)")
    sp.add_argument("--no-viewer", action="store_true")
    sp.set_defaults(func=cmd_example)

    sp = sub.add_parser("site", help="build every example's viewer and a landing page (static site)")
    sp.add_argument("--dest", default="site")
    sp.add_argument("--repo-url", default="", help="link to the source repository, shown on the landing page")
    sp.set_defaults(func=cmd_site)

    sp = sub.add_parser("export", help="write runs' routes, segments and features to a GeoPackage")
    sp.add_argument("gpkg")
    sp.add_argument("--out-root", default="out", help="the output root holding the runs")
    sp.add_argument("--prefix", default="runs", help="layer name prefix, e.g. red_mountain")
    sp.add_argument("--run-ids", nargs="*", default=None, help="these runs, in this order (default: all in the root)")
    sp.add_argument("--labels", default=None, help="';'-separated label per run (default: from the parameters that differ)")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("burn", help="burn a route into the DEM: design surface and cut/fill rasters")
    sp.add_argument("run_id")
    sp.add_argument("--rank", type=int, default=1)
    sp.add_argument("--res", type=float, default=2.0, help="cell size of the burned rasters (m)")
    sp.add_argument("--buffer", type=float, default=150.0, help="corridor either side of the route (m)")
    sp.add_argument("--out", default="out")
    sp.set_defaults(func=cmd_burn)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
