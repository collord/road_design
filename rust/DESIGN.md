# Road search in Rust and in the browser: design

Status: agreed 2026-10-07. Phase 1 in progress.

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
6. **Uncertainty by Monte Carlo**: sample the uncertain unit costs (the cost unit, soil depth, rock
   fraction, fuel) and re-cost the graph for each sample. Report how often each corridor wins. Nodes
   used by most samples are constrictions, which also places the fine stage's gates.

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
| `road-cost` | the road cost model as an `EdgeCost`; hairpin and bend macro costs (to be ported: built in parallel or lazily, cached per cell) | phase 1 |
| `terrain` | smoothing, gradients, slope mask, flow accumulation, channels, saddles | phase 2 |
| `tin-graph` | triangulation, extra nodes, graph costing, corridor field, distinct corridors, Monte Carlo | phase 3 |
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

1. **Macro tables in Rust**: hairpin and bend costs, built in parallel for full searches or lazily per
   cell for A* and chunks; parity with numba. Plus the event stream and progress reporting.
2. **Preprocessing in Rust**: smoothing, gradients, slope mask, flow accumulation, channels.
3. **Triangulated coarse graph**: triangulation, extra nodes, costing, corridor field, distinct
   corridors, Monte Carlo; compared with the lattice coarse pass.
4. **Browser**: the WebAssembly crate, a Web Worker, the viewer in TypeScript (Vite, still a static
   site), streamed convergence, weighted-A* candidates and exact refinement, chunked fine search.
5. **Browser input and output**, and a memory budget for a whole run in a tab.
