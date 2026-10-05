# Brief: least-construction-cost road alignment search over a DEM

## 1. Goal

Build a command-line tool that, given a DEM and two endpoints, finds low-construction-cost road alignments through mountainous terrain, and saves the best route, several genuinely different alternatives, and the search fields needed to visualize why.

The objective is **construction cost**, not length. Length enters only through a per-meter base cost.

Before writing any search code:

1. Inspect the DEM and report CRS, resolution, extent, nodata, vertical units and elevation range. If the CRS is geographic, reproject to the local UTM zone. All work is in a projected metric CRS.
2. If the endpoints in section 2 are still placeholders, stop and ask.
3. Propose a working resolution (section 5.5) and confirm it before the first full run.

Check current documentation for every library call and CLI flag you use (rasterio, numba, geopandas, pyarrow, flow-routing library). Do not rely on remembered signatures.

## 2. Inputs

| Input | Value |
|---|---|
| DEM | `<path to GeoTIFF>` |
| Start | `<x, y in DEM CRS, or lon/lat>` |
| Goal | `<x, y in DEM CRS, or lon/lat>` |
| Manual exclusion zones (optional) | `<path to polygon shapefile>`, passed as `--exclude`, repeatable |

### Parameters

All values below are placeholder defaults chosen by the brief's author, not by the user. Put them in a `params.toml`, echo them into every output, and make every one overridable from the CLI. Slopes are tangents (rise over run) unless stated.

| Symbol | Meaning | Default |
|---|---|---|
| `w` | Road width (running surface) | 5.0 m |
| `g_max` | Maximum road grade | 0.10 |
| `R_min` | Minimum open-road centerline radius | 20 m |
| `R_hp` | Hairpin centerline radius | 10 m |
| `g_hp` | Maximum grade through a hairpin | 0.04 |
| `w_hp` | Road width through a hairpin (widened) | 7.0 m |
| `d0` | Soil depth on level ground (vertical) | 2.0 m |
| `S0` | e-folding slope for soil thinning | 0.30 |
| `c_soil` | Cut backslope in soil | 1.0 (1H:1V) |
| `c_rock` | Cut backslope in rock | 4.0 (0.25H:1V) |
| `f_soil` | Fill slope (repose) for soil spoil | 0.67 |
| `f_rock` | Fill slope (repose) for rock spoil | 0.80 |
| `k_soil` | Bank-to-sidecast volume factor, soil | 1.0 |
| `k_rock` | Bank-to-sidecast volume factor, rock | 1.3 |
| `b_min` | Narrowest usable fill bench; below this, full bench | 0.5 m |
| `mask_margin` | Fraction of `c_rock` at which ground is excluded | 0.9 |
| `C_base` | Base cost per meter (grading, surfacing, ditching) | 10 |
| `C_soil` | Soil excavation per bank m³ | 1 |
| `C_rock` | Rock excavation per bank m³ | 6 |
| `C_borrow` | Imported crossing fill per m³ | 4 |
| `C_haul_down`, `C_haul_up` | Spoil haul per m³·m, loaded downhill / uphill | 0.01 / 0.02 |
| `haul_max` | Maximum economic haul distance along route | 500 m |
| `A_min` | Contributing area defining a channel | 2 ha |
| `culvert_cost` | Step function of contributing area | 50 / 150 / 400 for <10 / <50 / ≥50 ha |
| `H_fill_max` | Maximum crossing fill height | 8 m |
| `L_cross_max` | Maximum fill-crossing length | 60 m |
| `N_theta` | Heading bins | 16 |
| `K_alt` | Number of alternatives to save | 5 |
| `eps_alt` | Alternatives must cost within this fraction of the best | 0.25 |
| `D_alt` | Minimum separation between alternatives' via points | 100 m |

Costs are relative units (soil excavation = 1 per m³). Only ratios affect the route. `C_base` is the main knob trading length against earthwork.

## 3. Terrain preprocessing

Produce and save each of these as a raster.

1. **Smoothed DEM.** Gaussian smooth with sigma about equal to `w`. Gradients for section geometry come from this surface, so the section sees slope at the scale of the road footprint rather than per-cell noise. Keep the raw DEM for elevations.
2. **Gradient** (`dz/dx`, `dz/dy`) and **slope magnitude** `S`.
3. **Soil depth.** `d = d0 * exp(-S / S0)`, and `d = 0` wherever `S >= c_soil` (soil cannot stand steeper than its own cut slope). Depth is vertical.
4. **Flow accumulation** and contributing area, using an established library. Channel cells are those with contributing area at least `A_min`.
5. **Exclusion mask.** A road limited to grade `g_max` on ground of slope `S` has sidehill slope at least `sqrt(S² - g_max²)`. Exclude cells where that value is at least `mask_margin * c_rock`. Then dilate the mask by the largest cut catch distance that can occur at the threshold slope, so a road prism cannot overhang excluded ground.
6. **Manual exclusion zones.** If one or more polygon files are supplied:
   - Read with geopandas (accept shapefile and any other OGR polygon format). Fail with a clear message if the file has no CRS. Reproject to the working CRS. Repair invalid geometries and ignore non-polygon features with a warning.
   - Buffer by `excl_buffer` (default 0 m, a parameter), then rasterize onto the working grid with all-touched cells included, so a polygon never leaks through at cell corners.
   - Do not apply the catch-distance dilation from step 5 to manual zones. The user drew the boundary they meant; `excl_buffer` is the only adjustment.
   - Combine with the slope mask into a single coded raster: 0 = open, 1 = slope-excluded, 2 = manually excluded, 3 = both. The search treats any non-zero value as excluded.
   - Manual zones are hard constraints everywhere: ordinary edges, hairpin footprints, and fill-crossing segments (a crossing may not span an excluded zone).
   - If the start or goal falls inside a manual zone, stop and report which one and which polygon.
   - Record each file's path, feature count and total excluded area in `params.json`.
7. **Connectivity check.** Flood fill on the unmasked cells from the start. If the goal is unreachable, stop and report, with a raster of the reachable region. Also report whether the goal is reachable with manual zones ignored, so the user can tell whether their polygons or the terrain caused the disconnection.

## 4. Cross-section model (half-bench, sidecast)

At any point on the route with unit tangent `t` and unit normal `n`:

- Road grade `g = grad(z) · t`
- Sidehill slope `s = |grad(z) · n|`

The road is built half-bench: the hinge sits at ground level, the uphill part of the width `a` is cut, and the downhill part `b = w - a` is formed from the sidecast spoil. Road elevation therefore equals ground elevation along the centerline, and road grade equals terrain grade along the path. There is no separate vertical profile on ordinary sections.

Work in the section plane with `x` horizontal from the hinge and `y` vertical.

**Cut (uphill, ground is `y = s*x`, rock surface is `y = s*x - d`):**

- If `a <= d/s`, the cut is all soil: area `= 0.5 * a² * s * c_soil / (c_soil - s)`.
- Otherwise the bench floor enters rock at `x = d/s`. The backslope rises from `(a, 0)` at `c_rock` to the rock surface at `x1 = (c_rock*a - d) / (c_rock - s)`, then at `c_soil` through the soil to ground at `x2 = x1 + d / (c_soil - s)`.
- Build the cut polygon and the rock sub-polygon explicitly and take areas by the shoelace formula. Soil area is total minus rock. Do not hand-derive a combined closed form.
- `x2` is the cut catch distance. Record it.

**Fill (downhill, ground is `y = -s*x`):**

- Fill slope `f` is the volume-weighted blend of `f_soil` and `f_rock` according to the spoil composition for the current `a`.
- Area `= 0.5 * b² * s * f / (f - s)`, valid for `s < f`.
- Fill toe horizontal distance from the hinge is `f*b / (f - s)`. Record it.

**Balance:** solve for `a` in `[0, w]` by bisection such that `k_soil*A_soil + k_rock*A_rock = A_fill(w - a)`.

**Full bench:** if `s >= f`, or the balanced `b < b_min`, set `a = w`. All cut becomes spoil, with volume `k_soil*A_soil + k_rock*A_rock` per meter. This spoil is free to sidecast but is also available as crossing fill (section 8).

**Flat ground:** as `s` goes to 0, areas go to 0 and cost per meter goes to `C_base`.

**Infeasible:** `s >= mask_margin * c_rock`.

**Cost per meter:** `C_base + C_soil*A_soil + C_rock*A_rock`. Sidecast fill placement is free.

Implement the section as a pure function of `(s, d, w, params)` returning areas, widths, catch distances, section type and cost. It must be unit-tested in isolation before any search code depends on it (section 11).

## 5. Search

### 5.1 State and graph

- State is `(i, j, h)`: grid cell and one of `N_theta` heading bins.
- Use a state lattice. For each heading, precompute a set of motion primitives (straight, and left/right arcs at a few curvatures up to `1/R_min`) whose end states land exactly on lattice nodes and lattice headings. Every primitive must satisfy the curvature bound.
- Construction cost is independent of direction of travel, so the cost of a primitive equals the cost of its reverse with headings rotated by pi. The implementation must preserve this symmetry exactly; section 9 depends on it.

### 5.2 Edge evaluation

Sample each primitive at spacing no more than half a cell. At each sample compute `g`, `s`, `d` and the section. An edge is infeasible if any sample is masked, has `|g| > g_max`, or has an infeasible section. Edge cost is the integral of cost per meter along the primitive. Store per-edge volumes by material and spoil so they can be reported without recomputation.

### 5.3 Heuristic

For A*: `h = C_base * max(euclidean distance, |dz| / g_max)`. This is admissible because cost per meter is at least `C_base` and climbing `dz` needs at least `|dz|/g_max` of road.

### 5.4 Which search to run

The main run is two full Dijkstra searches, forward from the start and backward from the goal, because the alternatives in section 9 need both fields. A* with the heuristic above is for fast single-route re-runs (parameter sweeps, shadow-price iterations).

### 5.5 Size and performance

- States number `rows * cols * N_theta`. Choose a working resolution that keeps this under roughly 2e7, and confirm the choice with the user. Resample the DEM if needed.
- Write the search kernel in Numba (or Rust via PyO3 if Numba proves inadequate). Evaluate edges lazily inside the search loop. Do not materialize the full edge list.
- Tie-breaking in the priority queue must be deterministic.

## 6. Hairpins

A hairpin is a discrete engineered feature, not a tight curve. It is a macro-action in the graph.

- **Geometry.** Enter at a lattice node with heading `theta`, follow an arc of radius `R_hp` through 180 degrees to either side, exit at the node offset `2*R_hp` laterally with heading `theta + pi`. Snap `2*R_hp` to the lattice.
- **Design surface.** The hinge-at-ground assumption does not hold here. Interpolate design elevation linearly along the arc from ground elevation at entry to ground elevation at exit. The hairpin is infeasible if the resulting grade exceeds `g_hp`.
- **Earthwork.** Over the swept footprint (arc centerline buffered by `w_hp/2`), integrate design minus ground. Split cut into soil and rock using the soil depth raster. Add backslope and fill-slope volumes around the footprint perimeter using the catch geometry from section 4. Cut is priced by material. Fill is taken from the hairpin's own cut first, with any deficit priced at `C_borrow` and any surplus recorded as spoil.
- **Siting.** Precompute feasible hairpins for every node, heading and side. Reject any whose footprint touches the exclusion mask. Save the candidate set as a point layer with cost, so it can be visualized.
- **Symmetry.** A hairpin and its reverse must have identical cost.

**Stacked-leg conflicts.** Legs entering and leaving a hairpin, and long parallel traverses on one slope, can have the upper leg's fill toe overlap the lower leg's cut catch. Per-edge pricing cannot see this. In version 1:

- After a route is found, test every pair of non-adjacent segments. If their horizontal separation is less than the upper segment's fill toe distance plus the lower segment's cut catch distance, record a conflict.
- Write conflicts to the features output and count them in the route summary.
- Do not attempt to resolve them in the search. Report them. A penalty-and-re-search pass is a possible version 2.

## 7. Drainage crossings

Two mechanisms.

1. **At-grade crossing.** An ordinary edge that passes through a channel cell incurs `culvert_cost` for that cell's contributing area.
2. **Fill crossing.** A macro-action: a straight segment between two lattice nodes on opposite sides of a channel, with the same heading at both ends, length at most `L_cross_max`. Design elevation is linear between the endpoint ground elevations, with grade at most `g_max`. Fill height `h = design - ground` must be non-negative along the segment (allow a small tolerance) and at most `H_fill_max`. Embankment section area is `w*h + h²/f_rock`. Cost is fill volume times the current fill price `lambda_fill` (section 8), plus `culvert_cost`, plus `C_base` per meter.

Generate fill-crossing actions only where the segment actually crosses a channel cell.

## 8. Spoil reuse (two-pass mass haul)

Full-bench spoil and hairpin surplus can supply crossing fill. This makes cost non-local, so separate the search surrogate from the true cost.

- **True cost evaluator.** Given any complete route, list spoil sources and fill sinks by station. Solve the small transport problem assigning spoil to sinks, with haul cost per m³·m depending on whether the loaded haul runs downhill or uphill, and no assignment beyond `haul_max`. Unmet fill is bought at `C_borrow`. Unused spoil is sidecast at no cost. Use a linear program or min-cost flow; the problem is tiny. Total route cost is section costs plus hairpins plus culverts plus this result.
- **Iteration.** Search first with `lambda_fill = C_borrow`. Evaluate the route's true cost. Set `lambda_fill` to the realized mean cost per m³ of crossing fill on that route, and search again. Stop when the route no longer changes materially or after 5 iterations.
- **Reporting.** Every saved route, including alternatives, is costed and ranked by the true evaluator. The search surrogate cost is reported alongside for comparison, never as the headline number.

## 9. Alternatives and saved search fields

Let `G_f(i,j,h)` be cost-to-come from the start and `G_b(i,j,h)` be cost-to-come from the goal in the backward search.

- **Corridor field.** `F(i,j) = min over h of [ G_f(i,j,h) + G_b(i,j,h + pi) ]`. Each cell holds the cost of the cheapest route through it. Save as a raster. Distinct corridors appear as separate valleys.
- **Alternatives.** The best route is the global minimum of `F`. For each further alternative, choose the cell with the lowest `F` that is at least `D_alt` from every route already chosen and has `F <= (1 + eps_alt) * F_min`. Reconstruct the route through that cell from the forward and backward back-pointers. Discard a candidate that shares more than half its length with an existing route. Stop at `K_alt` routes or when no candidate qualifies.
- **Explored set.** For A* runs, save the closed set as a raster of minimum cost-to-come over headings.
- **Parameter sweeps.** Every run gets a `run_id`. Outputs from multiple runs accumulate in the same output directory and are distinguished by `run_id`, so routes under different parameters can be overlaid.

## 10. Outputs

All vectors as GeoParquet, all rasters as GeoTIFF, all in the DEM's projected CRS.

- `routes.parquet`: one row per segment. `run_id`, `route_id`, `rank`, `seq`, 3D line geometry, `station_start`, `length`, `grade`, `sidehill_slope`, `terrain_slope`, `soil_depth`, `section_type` (flat, half_bench, full_bench, hairpin, fill_crossing), `cut_width`, `fill_width`, `cut_catch`, `fill_toe`, `cut_soil_m3`, `cut_rock_m3`, `fill_m3`, `spoil_m3`, `cost_base`, `cost_earthwork`, `cost_total`.
- `route_summary.parquet`: one row per route. Length, total climb, volumes by material, hairpin count, crossing count, conflict count, surrogate cost, true cost, cost breakdown.
- `features.parquet`: points for hairpins used, hairpin candidates, crossings (with type, contributing area, fill volume, fill source), and stacked-leg conflicts.
- Rasters: smoothed DEM, slope, soil depth, coded exclusion mask (slope, manual, both), reachable region, contributing area, corridor field, forward and backward cost-to-come (minimum over headings), explored set.
- `params.json` per run, with every parameter and the DEM metadata.
- A quick-look PNG per run: hillshade, exclusion mask, corridor field, and the ranked routes. This is for sanity checking only; the user will do their own visualization from the files above.

## 11. Validation

Write these as pytest tests. All must pass before running on the real DEM.

1. **Section, single material.** With `d` very large, cut area matches `0.5*a²*s*c_soil/(c_soil - s)`. With `d = 0`, the same with `c_rock`.
2. **Section balance.** On every half-bench result, `k_soil*A_soil + k_rock*A_rock` equals fill area to tolerance.
3. **Section limits.** Cost per meter tends to `C_base` as `s` tends to 0. Section becomes full bench as `s` approaches `f`. Cost is monotonic in `s`.
4. **Flat plane.** Route is straight; cost equals `C_base` times length.
5. **Inclined plane steeper than `g_max`.** Route climbs at about `g_max`. With the goal directly upslope of the start and limited lateral room, the route uses hairpins. Total cost matches a hand calculation.
6. **Symmetry.** Start-to-goal and goal-to-start give the same cost and the same route reversed.
7. **A* against Dijkstra.** Identical cost on several random synthetic terrains.
8. **Mask.** A synthetic cliff band is never entered. A fully enclosing band triggers the connectivity failure with a clear message. A manual polygon placed across the optimal route forces a detour that never touches it, including by hairpin footprint or fill crossing; a polygon in a different CRS from the DEM gives the same result after reprojection; an endpoint inside a polygon is rejected.
9. **Corridor field.** The minimum of `F` equals the Dijkstra optimum.
10. **Determinism.** Two identical runs produce byte-identical route output.

## 12. Order of work

1. DEM inspection and report. Confirm working resolution.
2. Section function and its tests.
3. Preprocessing rasters and connectivity check.
4. Lattice primitives and plain search, without hairpins or crossings. Tests 4, 6, 7, 8.
5. Hairpins. Test 5.
6. Crossings and the true cost evaluator with the two-pass iteration.
7. Bidirectional fields, corridor raster, alternatives. Test 9.
8. Outputs, quick-look, run on the real DEM.

Report after steps 1, 4 and 8, with the quick-look image at 4 and 8.

## 13. Known simplifications to state in the final report

- Road elevation follows the ground except at hairpins and fill crossings. No vertical curves, no through-cuts across ridges.
- Section geometry assumes locally planar ground at footprint scale and ignores curvature of the road in plan.
- Soil depth is a function of slope only.
- Stacked-leg conflicts are detected and reported, not resolved.
- Culvert costs and all unit costs are placeholders until the user supplies real ratios.
- Global optimality holds on the lattice for the surrogate cost. The true cost with spoil reuse is evaluated afterward and may rank routes differently.
