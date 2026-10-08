# Rust core

Three crates in one workspace, each usable on its own:

| crate | what it is | depends on |
|---|---|---|
| `lattice-search` | Dijkstra and A* over (row, col, heading) states on a sparse grid. Edge costs and the A* estimate are traits (`EdgeCost`, `Heuristic`); only the cells of a `CellIndex` (any mask, e.g. a corridor) hold states. `search_with_progress` reports every N settled states and can be stopped. No knowledge of roads. | nothing |
| `road-cost` | The road cost model of `roadsearch/search.py` (earthwork from the section table, grade limits on the profile, culverts, fill crossings, hairpin and bend tables, truck travel, climb and turning), as an `EdgeCost`, plus the climb-aware `ClimbHeuristic`. `macros`: the hairpin and bend designs and their tables (`hairpin.py`, `bend.py`), built on all cores with the `parallel` feature (rayon, on by default; off for WebAssembly). | `lattice-search` |
| `roadsearch-core` | Python bindings (PyO3, numpy). `search` and `edge_costs` take the same lattice and terrain tuples as the numba kernel (`SearchProblem.lat_tuple()`, `terr_tuple()`, via `roadsearch.search.rust_args`); `search` takes an optional progress callback. `hairpin_table` and `bend_table` build the macro tables. | both |

Search results (`roadsearch.search.SearchResult`) and the hairpin and bend tables hold only the
cells that are searched (the open cells of the corridor), with a lookup from grid cells to rows,
so memory follows the corridor, not its bounding box.

The Rust kernel gives the numba kernel's costs and parents bit for bit, and the same macro
tables (feasibility identical; float32 costs equal but for last-bit differences in sines and
cosines) (`tests/test_rust_engine.py`). Searches are about ten times faster, table builds about
three times. See `DESIGN.md` for where this is going.
`roadsearch` uses it whenever `roadsearch_core` is installed; set
`ROADSEARCH_ENGINE=numba` to use the numba kernel instead.

## Build

Needs a Rust toolchain (`rustup`) and `maturin`:

```
pip install maturin
cd rust/roadsearch-core
maturin build --release
pip install ../target/wheels/roadsearch_core-*.whl
```

or in one step, `pip install ./rust/roadsearch-core`.

Rust tests: `cargo test` in `rust/`.
