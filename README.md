# roadsearch

Finds the cheapest road between two points over a DEM. "Cheapest" covers the cost to build it (cut and fill from the cross slope, hairpins and bends, culverts, crossing fill, retaining walls) and, optionally, the cost of trucks driving it for years. The search runs over a network of short buildable pieces of road. It returns the best route and ranked alternatives, as GeoParquet and GeoTIFF, plus an interactive viewer.

**Example site:** worked examples on two included USGS DEMs, built by GitHub Actions and published to GitHub Pages. It has four tabs: compare runs, watch a search, draw your own route and cost it, and how the search works.

The method is explained from first principles in [roadsearch/reference.md](roadsearch/reference.md), which is also the site's "How the search works" tab. [road-search-brief.md](road-search-brief.md) is the original specification. Where this implementation departs from it, the departures are listed below.

## Install

Python 3.11 or later; developed and tested on 3.14.

```
pip install -c constraints.txt -e ".[test]"
python -m pytest              # about 20 minutes; the evaluator tests also need Node.js
```

## Examples

`examples.toml` lists the worked examples. Run one by name, or build them all into a static site:

```
python -m roadsearch example red-mountain        # its runs, then out-red-mountain/viewer
python -m roadsearch site                        # every example's viewer under site/, plus a landing page
python -m http.server -d site                    # then open http://localhost:8000
```

Finished runs are reused, so `site` only runs what is missing. Opening `index.html` straight from disk doesn't work, because the page fetches its data files.

| Example | DEM | Road | Swept | Time on one machine |
|---|---|---|---|---|
| `haul-road` | `data/topo.tif`: x-y in US survey feet, z in meters | 20 m, 8% max grade, R_min 40 m | truck-cost weight k = 0, 0.1, 1, 10, 100 | about 4 hours |
| `red-mountain` | `data/topo_redmtn.tif`: NAD83 / UTM 13N, meters | 9 m, 6% max grade, construction cost only | minimum turn radius R_min = 10, 20, 30 m, plus 20 m at 8% and 9% and 16 m at 10% | about 1 hour per run |

`sweep --vary NAME=v1,v2,...` sweeps any parameter the same way, for example `--vary R_min=10,20,30`. Two more options:
- `--extra "LABEL|NAME=V,..."` adds a run outside the sweep, for example `--extra "20 m, 8% grade|R_min=20,g_max=0.08"`.
- `--reference FILE` shows a given centerline, for example the as-built road, on the maps and in the table, with a metrics panel. It is costed with the same model as the routes, on each run's terrain (`roadsearch/polyline.py`). The panel gives grades from the DEM averaged over 100 m and 200 m, curve radii and switchbacks.

The Red Mountain example uses US 550 over Red Mountain Pass as its reference. The DEM shows that highway already built, so the ground along it is the finished bench rather than the original hillside, and its earthwork cost is a lower bound.

## Building the site on GitHub

`.github/workflows/site.yml` reads `examples.toml` and runs each example run as its own job, in parallel. It then merges each example's runs (`python -m roadsearch merge`), builds the site (`python -m roadsearch site`) and deploys it to GitHub Pages.
- **Time:** the slowest job takes 1 to 2 hours on a standard runner.
- **Caching:** finished runs are cached under a key built from the code, the parameter files, `examples.toml` and the DEMs. A rebuild with none of those changed only regenerates the viewers.
- **Triggers:** the workflow runs on pushes that change those files, or by hand from the Actions tab.
- **Setup:** set Settings → Pages → Source to "GitHub Actions" once.

`.github/workflows/tests.yml` runs the test suite on every push and pull request.

## Usage

```
python -m roadsearch inspect DEM
python -m roadsearch run --dem DEM --start X,Y --goal X,Y [--exclude zones.shp ...]
                         [--clip auto|box|none] [--resolution M] [--out out] [--<param> VALUE ...]
python -m pytest
```

- Endpoints are in the DEM's CRS unless `--endpoint-crs` is given (`EPSG:4326` for lon,lat).
- Every parameter in `params.toml` can be overridden as `--<name> VALUE`, or from a partial TOML file passed with `--params`.
- Horizontal units come from the DEM's CRS. A DEM that isn't in meters is reprojected to its UTM zone. Vertical units come from `z_units`.
- `--clip auto` (the default) runs in two passes:
  - **Coarse pass.** Searches the endpoint box plus `clip_coarse_margin` (2 km), at the finest resolution that fits the limits below (or `--coarse-res`). If the coarse route reaches the edge of that box, the margin is doubled. A whole-DEM coarse pass was tried and dropped: at ~30 m the lattice cannot follow contours on this terrain.
  - **Fine pass.** Limited to the coarse corridor: cells whose cheapest route through them costs at most `(1 + eps_alt)(1 + clip_slack)` times the best, plus `clip_buffer` around them. Switchbacks well outside the endpoint box therefore remain possible. If the fine route reaches the clip edge, the run records a warning and retries once with a doubled buffer.
  - **Resolution.** The fine resolution (unless `--resolution` is given) is the finest whose corridor states fit `max_states` while the whole grid's search memory (about 26 bytes per state) fits `max_memory_gb`. Cells outside the corridor are never searched but are still allocated.

## Operating costs

The brief minimizes construction cost. For roads that trucks will use for years, two lifetime operating costs can be added, in the same relative units:

- `C_travel`: cost per meter of route length, for the trucks' whole life on the road.
- `C_vertical`: cost per meter of **total vertical travel** (rise plus fall along the profile).
- `C_turn`: cost per radian of **change in heading**: every curve, bend and hairpin, for the braking and re-accelerating it costs each vehicle. A wiggle pays for both its turns, so needless S-curves stop being free.

Only the total vertical travel matters. Trucks may pay different rates climbing and descending, loaded and empty, but a route's climb minus its descent is fixed by its endpoints, so for any route:

    a·climb + b·descent = (a + b)/2 · (climb + descent) + (a − b)/2 · (net elevation change)

The second term is the same for every route. So, over N round trips:

    C_vertical = N · (c_up,loaded + c_down,loaded + c_up,empty + c_down,empty) / 2
    C_travel   = N · (c_per_m,loaded + c_per_m,empty)

This penalizes routes that lose elevation and regain it: diesel to climb back, brakes to come down. It also keeps the cost the same in both directions, which the bidirectional search needs. All three default to 0, and `sweep --k` scales all three. Segments and route summaries report `cost_construction`, `cost_operating`, `vertical_travel` and `turning_deg` separately, and routes are ranked by the sum. `params_haul.toml` gives a crude haul-truck value for `C_turn` (8,000 per radian at k = 1), commented out so the haul-road example is unchanged.

`params_haul.toml` sets both for one truck over 5 years (diesel only, unconfirmed $4 per cost unit). Road maintenance, tires, brakes and fleet size are all taken to scale with these per-meter costs, so they are covered by a single weight k:

```
python -m roadsearch sweep --dem DEM --start X,Y --goal X,Y --params params_haul.toml [--k 0,0.1,1,10,100] [run options]
```

This does one run per k, with `k × C_travel` and `k × C_vertical`, then writes the viewer over all of them. The default k values are 0, 0.1, 1, 10 and 100. A run with the same inputs that has already finished is reused, so adding a k only runs the new one.

## Haul roads

`params_haul.toml` holds placeholder values for a 20 m mine haul road:

| Parameter | Value |
|---|---|
| Width (`w`) | 20 m |
| Max grade (`g_max`) | 8% |
| Curve radius (`R_min`) | 40 m |
| Hairpin loop radius (`R_hp`) | 25 m |
| Width through hairpins (`w_hp`) | 25 m |
| Hairpin grade (`g_hp`) | 3% |
| Base cost (`C_base`) | 40 per meter (unconfirmed) |

Use it with `--params params_haul.toml`. With a road this wide, 10 m cells are adequate (`--resolution 10 --coarse-res 15`).

## Outputs

- **Output root**, accumulating across runs and keyed by `run_id`: `routes.parquet`, `route_summary.parquet` and `features.parquet` (GeoParquet).
- **`runs/<run_id>/`**:
  - rasters: working and smoothed DEM, dz/dx, dz/dy, slope, soil depth, contributing area, coded exclusion mask, reachable region, corridor field, forward and backward cost-to-come, A* explored set, and for `--clip auto` the coarse corridor and the clip region
  - `params.json`
  - `quicklook.png`
  - `corridor_tree.npz`: the forward and backward search trees, kept only for routes through cells within twice the best cost

  The default `run_id` is a hash of the inputs and parameters, so identical runs coincide.

## Viewing runs

`python -m roadsearch animate RUN_ID [RUN_ID ...] [--factors 0,0.1,...] [--dest out/viewer]` writes a static site: `index.html` plus `data/<run_id>.json` for each run. It has three tabs:

- **Compare runs.** Every run's best route over one hillshade, with a cost table.
- **Watch a search.** For one run, the A* search, the two Dijkstra fronts meeting, and the corridor with the ranked routes.
- **Draw a route.** Draw your own line from start to goal and see it costed against the optimum, with a profile of elevation and cost per meter.
  - The costing is `roadsearch/evaluator.js`, a port of the section model and edge integration. `tests/test_evaluator_js.py` checks it against Python under Node, to 1e-9.
  - Corners are rounded to `R_min`. Grades over the limit, excluded ground, unbuildable side slopes and tight corners are flagged, not refused.
  - Hairpins are costed as plain road, and crossing fill and spoil reuse are left out.
  - The drawing is kept in the browser and carries over between runs, so one line can be compared across k.
  - "Copy GeoJSON" gives the line in the working CRS.
  - **Best route through a point** shows, for any cell, the cost of the cheapest route through it (from the corridor field) and draws that route. Drawing it uses the run's corridor tree (`corridor_tree.npz`). Runs made before the tree was saved need `python -m roadsearch tree RUN_ID ...` once, which rebuilds the search and checks it against the recorded best cost.
- **How the search works.** The method write-up, from `roadsearch/reference.md` and its diagram `reference_pipeline.svg`. Edit those files to change it.

The animation needs no search trace: it thresholds the saved cost-to-come and corridor rasters frame by frame. `--factors` labels each run with its operating-cost weight. Pale gray ground in a run's map was never searched: it is inside the fine grid but outside the coarse corridor.

## Burning a route into the DEM

`python -m roadsearch burn RUN_ID [--rank 1] [--res 2] [--buffer 150]` builds the terrain with the road built, in a corridor around one saved route, at fine resolution. Each cell takes its nearest point on the centerline, which is the section hinge. Its new elevation is:

    new = road + clip(ground − road, −f·d, +backslope(d))

where d is the cell's distance outside the road edge. The road is level across its width. Cut faces rise at the rock slope up to the rock surface and at the soil slope above it, clipped at `catch_max`. Fill falls at the slope of the segment's own spoil mix.

- **Ordinary sections** are half benches about the hinge: the cut width uphill, the fill width downhill, and no fill on a full bench.
- **Hairpins, bends and fill crossings** are symmetric about the centerline and keep their designed profiles.

Outputs, in `runs/<run_id>/burn/rank<k>/`:
- `dem_existing.tif` and `dem_design.tif`
- `cut_fill_depth.tif` (+ fill, − cut)
- `cut_material.tif` (1 soil, 2 rock, 3 fill)
- `road_surface.tif`
- `wall_cells.tif`: places where two non-adjacent parts of the route meet with a step of more than 1 m
- `burn_summary.json`: burned volumes against the section model, by section type
- a quick-look PNG

On a plane the burned cut and fill match the section model within 4%. On real terrain, burned fill typically runs 15–20% above the model, because the model reads slope from ground smoothed at the road width. Spoil sidecast from full-bench cuts is not placed; its volume is reported instead.

## Departures from the brief, and additions

- **Headings** — 32 by default (the brief says 16). With 16, contour following fails at some aspects on slopes over 0.43; with 32, over 0.82. Headings are exact lattice directions, built from (4,1), (2,1) and (4,3) per octant.
- **Binned straight moves** (`move_max_cells`, default 7; `kink_max_deg`, default 10°) — with exact lattice headings only, the climb rate on a uniform slope S is S·sin(lattice angle), so on many slopes the best feasible climb is well below `g_max` (0.061 on a 0.25 slope). Straight moves to every coprime lattice point within 7 cells are therefore added, assigned to lattice headings. A move may start and end in any heading within `kink_max_deg` of its direction, provided one of them is its nearest heading, so the road can hold in-between directions and pass from one heading class to the next. Straights then climb at about 0.098 on that 0.25 slope, and the test route there is 25% cheaper. This follows Stückelberger et al. (2007), who rounded 48-link patterns to 16 classes with up to 18° of error. `move_max_cells = 0` gives the brief's exact-heading lattice.
- **Grade check** — measured on the profile, from smoothed-DEM elevation differences between consecutive samples. These sum to the node-to-node climb, which keeps the A* climb heuristic admissible.
- **Hairpin shapes: semicircles and keyholes.** The brief's hairpin is a semicircle whose exit is 2R across the slope, linear from ground to ground at grade ≤ `g_hp`. That is only feasible where the cross slope is under `g_hp`·π/2 ≈ 0.063.
  - Hairpins may add **approach legs** (`hp_leg_max`), on which the design leaves the ground at up to `g_max`. All footprint earthwork is priced.
  - Hairpins may also be **keyholes**: the legs run closer than the loop's diameter (clear gap ≥ `hp_gap_min`), and each swings outward on a reverse curve of radius `R_min` before a loop of radius ≥ `R_hp` that turns 180° + 2α. The two ends then differ less in elevation and the loop is longer to climb on.
  - Per heading there are two leg spacings, each a separate search action: the narrowest the gap allows and the semicircle spacing. Each tries two loop radii and keeps the cheaper.
  - Between the legs, overlapping slopes are replaced by retaining walls priced at `C_wall` per m² of face. The default `C_wall = inf` means no walls: the hairpin is flagged as a conflict instead.
  - `hp_leg_max = 0` with a semicircle-only leg spacing gives the brief's hairpin.
- **Bends** — 90° and about 135° turns (8 and 12 heading bins) are also macro-actions. Each is a straight lead-in, an arc of radius `R_min` and a straight lead-out, with the design at one grade (≤ `g_max`) between the end nodes and the footprint earthwork priced. They let a road turn through the fall line on ground slightly steeper than `g_max`, where ordinary curves cannot. Their average grade is the slope times chord/arc (0.90 for 90°, 0.78 for 135°), so they only help up to about 1.1–1.3 × `g_max`. Beyond that, turning through the fall line takes a hairpin.
- **Self-overlapping routes** — nothing in the search state stops a route from crossing or running back over itself. Where a hairpin is expensive and flat ground is nearby, the cheapest lattice path can be a balloon loop: overshoot, loop round, return over its own track. After the fill-price search, such overlaps are detected: non-adjacent pieces closer than the road width, with the road between them clearly longer than that. The excursion is then blocked (buffered by half the road width) and the search repeated, up to 5 rounds. If blocking leaves no route, the previous route is kept and the overlap reported. Alternatives that overlap themselves are rejected.
- **Inside of a hairpin turn** — `g_hp` limits the centerline grade. The design is level across the width, so the inside edge of the loop is steeper by R/(R − w_hp/2). That grade is reported per hairpin.
- **Cut catch cap** — `catch_max`, default 50 m. A cut that would daylight beyond 50 m is clipped there. This bounds the cost spike as s → `c_soil`, where a soil backslope never reaches the ground.
- **Culverts** — one culvert per run of channel cells crossed. A run that starts or ends at a node is charged half by each of the two edges meeting there, so it is paid once and costs the same in both directions.
- **Fill crossings** — the design runs between raw-DEM elevations at the end nodes, and fill may dip `fill_tol` (0.25 m) below ground.
- **Alternatives** — taken from **plateaus** (`alt_method = "plateau"`, the default): stretches of road that are in both full search trees, so cheapest from both ends. A route through a plateau is optimal over any window shorter than the plateau, which rules out detours to a via point and back without any extra search. Plateaus must cost at least `alt_plateau` × the best cost. The usual rules then apply: at most `eps_alt` above the best, at least `D_alt` from chosen routes, at most half shared, no self-overlap.
  - `alt_method = "via"` keeps the brief's via-cell rule with an overlap prefilter and a local-optimality test, which runs one A\* search per candidate (within `alt_window` × the best cost either side of the via point, the path may cost at most (1 + `alt_local_tol`) times the cheapest path between the window ends). It finds similar routes, but in a timed Red Mountain run those searches were half the run time.
  - At most `alt_max_tries` candidates are examined.
- **Nodata cells** are slope-excluded (code 1). The flood-fill connectivity check uses 4-connectivity, matching the search's rule that a road cannot pass diagonally between two excluded cells.
- **pysheds 0.5 on NumPy 2.4** — `np.in1d` is aliased to `np.isin` in `roadsearch/flow.py`.

## Known simplifications

- Road elevation follows the ground except at hairpins, bends and fill crossings. There are no vertical curves and no through-cuts.
- Section geometry assumes locally planar ground and ignores plan curvature.
- Soil depth is a function of slope only.
- Stacked-leg conflicts between separate route segments are detected and reported, not resolved. Within a hairpin they are resolved by walls when `C_wall` is finite.
- Culvert, wall and unit costs are placeholders.
- **Kinks between straight moves.** Binned moves (below) can change direction by up to `kink_max_deg` at a node. The design would smooth these with short curves, which are not modeled. Turn primitives still start and end on exact lattice headings, so turns climb more slowly than straights.
- **Optimality is on the lattice, for the surrogate cost.** The surrogate uses the section-cost table and the crossing-fill price λ. The true cost with spoil reuse is evaluated afterward and ranks the saved routes.

## Data and references

- **DEMs:** `data/topo.tif` and `data/topo_redmtn.tif`, from the U.S. Geological Survey (public domain).
- **US 550 centerline:** `data/us550_red_mountain.geojson`, from the Colorado Department of Transportation's Scenic Byways layer (San Juan Skyway), clipped to the Red Mountain DEM.
- Stückelberger, J., Heinimann, H. R., and Chung, W. (2007). Improved road network design models with the consideration of various link patterns and road design elements. *Canadian Journal of Forest Research* 37: 2281–2298. doi:10.1139/X07-036. This is the source of the binned link patterns and of the Swiss retaining-wall cost ratio.
- Grigolato, S., Mologni, O., and Cavalli, R. (2017). GIS applications in forest operations and road network planning: an overview over the last two decades. *Croatian Journal of Forest Engineering* 38(2).
