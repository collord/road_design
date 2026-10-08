# Road search in Rust and in the browser: design

Status: agreed 2026-10-07. Phases 1 and 2 done; phase 3's first version done (`--clip tin`).

## Goal

Run the whole road search in a browser tab: a DEM goes in, and corridors, routes, costs and
profiles come out. Results stream while the search converges, so a user can watch corridors form
and routes improve. The Python package remains as a thin command-line wrapper over the same Rust
crates, for batch runs, sweeps and the test suite.

Two problems with the current method drive this, beyond speed:

- **The coarse pass has blind spots.** It searches the endpoint box plus a margin on a 30 m lattice
  and keeps cells within 5% of its best cost. On Stibnite this hid the valley route entirely
  (outside the box), and the 30 m costs differed from the fine costs by more than the 5% band.
- **Results depend strongly on resolution in steep terrain.** The 15 m Stibnite route cost about 10%
  more than the 12 m route, even when both are costed on the same terrain: the coarser lattice
  could not follow the benches and saddles the finer one used. The fine search should run as fine
  as memory allows, which means memory must follow the corridor, not the run.

## Method

### Coarse stage: a triangulated graph instead of the 30 m lattice

1. **Triangulate the DEM** to a stated maximum vertical error with greedy insertion (Delatin).
   Measured on the Stibnite DEM (12.7 million cells at 9.16 m): 70,000 vertices at 10 m tolerance in
   1.9 s, 178,000 at 5 m in 3.2 s, 566,000 at 2 m in 6.8 s (native code). The triangulation keeps
   ridges, valley floors and slope breaks and merges uniform slopes.
2. **Add road-relevant nodes**: channel nodes from flow accumulation, saddles (critical points of the
   DEM), and Steiner points along triangle edges, so that a path can cross a face at any grade.
   Re-triangulate with streams as constraint lines (`spade`).
3. **Cost the graph edges on the full DEM**, not on the simplified surface: length, climb, sidehill
   earthwork from the section model, culverts at channel crossings, operating costs, and a switchback
   proxy for edges steeper than g_max (charged as if stretched to |dz| / g_max, plus an allowance per
   implied hairpin). Construction and operating parts are kept separately, so re-weighting is cheap.
4. **Search the whole DEM.** There is no window. Two exact searches (from the start and from the goal)
   take milliseconds on a graph this size and give the corridor field: the cost of the best route
   forced through each node.
5. **Distinct corridors**:
   - saddle candidates, ranked by the corridor field at each saddle (free once both searches are done);
   - a penalty loop: each kept route is discouraged where its position was arbitrary (the corridor field
     is flat across it) and left open where it passed a constriction (the field rises on both sides);
   - each candidate is kept if it is within a stretch bound of the best and dissimilar from those kept.
   The cutoff for keeping a corridor is generous: fine-searching one corridor too many is cheap, and
   dropping the true best is not.
6. **Sampling, both optional.**
   - *Node sampling* (`tin_node_samples`): random nodes added to the terrain-informed ones, the
     graph rebuilt and searched again. How often each corridor wins when the nodes move, and new
     corridors the fixed nodes missed, which join the fine search's region. In the browser, node
     batches arriving one after another are the convergence to watch.
   - *Cost sampling* (`cost_samples`): the uncertain unit costs (the cost unit, operating weight,
     local ground) perturbed and the coarse stage re-costed: how often each corridor still wins. A
     sensitivity check, not odds: the spreads are judgments, and it inherits the coarse model's
     biases (on Stibnite the graph preferred a ridge corridor in 94% of cost samples; the fine search
     then found the valley cheaper). Implemented on the graph, where costs stay split into
     construction and operating parts; the lattice coarse pass would need its tables split too.

### Fine stage: the heading lattice, per corridor, in chunks

The lattice search is unchanged in what it computes (states are cell and heading; motion primitives,
hairpins, bends and fill crossings; the road cost model). What changes is how much of it is in memory.

- **Exact, gate by gate.** Gates are short cuts across the corridor at constrictions, where routes
  cannot zigzag. Each chunk is a multi-source search seeded with the arrival costs at its entry gate
  (all cells and headings of the gate). Only one chunk is in memory, plus the gates' cost vectors.
  The route is recovered by re-running each chunk with its entry and exit known (Hirschberg's
  divide-and-conquer): about twice the time, memory set by the chunk. Chunks overlap by a hairpin's
  reach (about 100 m).
- **Approximate and parallel, for interactive drafts.** Via points with positions and headings from
  the coarse route; each leg is solved independently and streamed as it finishes. Kinks at the vias
  are removed by re-optimizing windows that span each via, until nothing changes (optimal over any
  window of that size).
- **Alternatives come from the coarse stage**, one corridor each, so the fine stage needs only the best
  route per corridor. The two full whole-corridor searches (today's largest memory cost) are no longer
  needed. The plateau method remains for the CLI's full searches.
- **Resolution check**: every result records its resolution, and the CLI can rerun a corridor finer.

A 5 km by 1.5 km chunk at 10 m is about 75,000 cells and 2.4 million states, about 100 MB.

### Streaming

The engine advances in bounded steps (`step(budget)`) and emits events:

- stage start and end, with counters (states settled, nodes added, rows built);
- progress: current best cost and route, the search frontier, corridor probabilities;
- resource samples (memory) and warnings.

The CLI writes them as JSON lines (`events.jsonl` in the run folder); the readable log and the timing
table are generated from them. In the browser a Web Worker posts the same events to the UI as
transferable arrays, at about 10 to 30 per second.

In the browser, quick candidates come first: weighted A* (f = g + w h, w about 1.5 to 2, so cost at
most w times the optimum) with the corridor-field-graded penalty loop, replaced by exact routes as the
full searches finish.

## Engineering

### Crates

| crate | contents | status |
|---|---|---|
| `lattice-search` | generic sparse heading-lattice Dijkstra and A* (traits for edge cost and estimate) | done |
| `road-cost` | the road cost model as an `EdgeCost`; hairpin and bend macro designs and tables (`macros`) | done (lazy per-cell costs: phase 4) |
| `terrain` | smoothing, gradients, slope mask (exact distance transform), Priority-Flood flow accumulation, connected regions | done (saddles: phase 3) |
| `tin-graph` | triangulation and road-relevant nodes, edges costed on the DEM, corridor field, distinct corridors | first version done (Monte Carlo, turning on edges, field-graded penalties: to do) |
| `roadsearch-core` | Python bindings (PyO3) | done; grows with each phase |
| `roadsearch-wasm` | browser bindings (wasm-bindgen), the step and event API | phase 4 |

### Third-party crates (checked 2026-10-07)

All pure Rust, so they build for WebAssembly.

| need | crate |
|---|---|
| heightfield triangulation with max error | `delatin` 0.2 (young, single maintainer, no dependencies: pin it, fork it if needed) |
| constrained Delaunay with incremental insertion | `spade` 2.15 |
| spatial index | `kiddo` or `rstar` |
| GeoTIFF input, including cloud-optimized files over HTTP ranges | `geotiff-reader` 0.8 |
| browser bindings and threads | `wasm-bindgen`, `js-sys`, `web-sys`; `wasm-bindgen-rayon` |

Quadric-error decimation (`meshopt-rs`) is available if a non-heightfield mesh is ever needed. Avoid
`meshopt`, which wraps C++.

### Browser constraints

- **Memory**: WebAssembly's 32-bit memory has a 4 GB ceiling; about 1 to 2 GB is realistic in a tab.
  The fine stage must work in chunks and compute macro costs lazily.
- **Threads** need cross-origin isolation headers, which GitHub Pages cannot set; a service-worker
  workaround exists. Everything must also work single-threaded.
- **Input**: GeoTIFFs over HTTP, read by tile. **Output**: GeoJSON or GeoParquet downloads, with the
  browser's private file system for caching runs.

### Verification

- The numba and Python code stays as the reference during the port; every Rust piece gets parity
  tests against it (bit for bit where the arithmetic allows).
- Benchmark problems track cost, time and peak memory from commit to commit.
- For the first projects, chunked fine searches are compared against a single full search, and the
  triangulated coarse stage against the lattice coarse pass, on Stibnite and Red Mountain.

## Phases

1. **Macro tables in Rust** (done): hairpin and bend costs built in parallel, same feasibility as
   numba and float32 costs equal but for last-bit differences, about 3x faster. Event stream
   (`events.jsonl`), progress and cancellation from the Rust search, `roadsearch status` (done).
   Computing macro costs lazily per cell moves to phase 4, where the chunked fine search needs it.
2. **Preprocessing in Rust** (done): smoothing, gradients, slope masks and connected regions are
   identical to the Python path; flow accumulation (Priority-Flood with an epsilon rise, D8) agrees
   on 99.96% of channel cells on Stibnite and is about 7x faster than pysheds. Runs record
   `terrain_engine` in their inputs (so in their ids) when Rust preprocessed them.
3. **Triangulated coarse graph** (first version done: `--clip tin`). Nodes: Delatin vertices of the
   smoothed DEM, channel nodes, saddles, a background grid; edges to the k nearest neighbors (a
   roadmap rather than triangle edges with Steiner points), costed on the DEM with the switchback
   proxy. Corridors: the best route, saddle candidates, a penalty loop; buffered into the fine
   search's region. On Stibnite (10.5%) it covers the whole DEM in 12 s (154,000 nodes, 1.4 million
   edges) and found the valley route the lattice coarse pass had excluded; the fine search confirmed
   it at 14.99 M against 17.04 M for the best ridge route at the same resolution, in 6 minutes all
   told. Since then (`tin-2`): turning costs on the graph (a search over directed edges), penalties
   graded by the corridor field, node and cost sampling. On Red Mountain (20 m radius, 9%, k = 1,
   turning) the graph stage found the same best and second routes as the lattice coarse pass, to
   the unit, and better third and fourth ones, in 7 minutes all told (the lattice run, 100).
4. **Browser**: the WebAssembly crate, a Web Worker, the viewer in TypeScript (Vite, still a static
   site), streamed convergence, weighted-A* candidates and exact refinement, chunked fine search.
   Done so far: the viewer (GeoTIFFs in any CRS, hillshade, base maps, the coarse graph and its
   corridors as they are found, progress and stopping). The fine stage is ported (`road-fine`:
   lattice, section, problem, fill-price iteration, self-overlap avoidance, segment evaluation and
   true cost with mass haul), at parity with the Python (same edges and costs on three terrains),
   and runs per corridor in the browser with hairpin and bend costs computed lazily, as the search
   reaches them (the same values as the tables). On Red Mountain at 5 m a corridor costs 263,468
   in the browser against 275,504 for the command line's run; it takes about 6 minutes in one
   thread, two thirds of it in the first search's macro designs (about 13 microseconds each).
   Since then: up to 8 helper workers compute the designs (all of them, about 42 million on Red
   Mountain at 5 m) into the search's cache, and each search's route is drawn as it is found:
   about 2 minutes in all, the first route at about 80 to 110 s. A long profile and a GeoJSON
   download follow, and a corridor needing more than 1.5 GB is refused.
   Tried and dropped: weighted-A* drafts. The climb heuristic underestimates the cost about
   threefold here, so w = 3 still settled half the states; a draft confined near the coarse line
   (150 m) cost four times the best, as the fine route leaves the coarse line. A tighter
   admissible heuristic (a relaxed cell-graph search pricing earthwork from the slope) would cut
   the states every search settles; it has to stay admissible to keep routes exact.
   Next: that heuristic, and long corridors in pieces (the gates above).
5. **Browser input and output**, and a memory budget for a whole run in a tab.
