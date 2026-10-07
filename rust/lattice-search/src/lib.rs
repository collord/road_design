//! Shortest paths over a heading lattice on a sparse grid.
//!
//! A *state* is a grid cell and a heading bin, `(row, col, h)` with `h` in
//! `0..n_theta`. An *action* moves from a state with heading `h0` to the cell
//! `(row + dr, col + dc)` with heading `h1`. What an action costs from a given
//! cell is up to the caller ([`EdgeCost`]), as is the lower bound used by A*
//! ([`Heuristic`]).
//!
//! Only the cells of a [`CellIndex`] (an arbitrary mask, typically a corridor)
//! hold states, so memory follows the searched area, not its bounding box.
//! Compact ids keep row-major order, so ties are broken exactly as a dense
//! `(row * ncols + col) * n_theta + h` numbering would break them.
//!
//! ```
//! use lattice_search::{Action, CellIndex, EdgeCost, Lattice, search, SearchSpec, ZeroHeuristic};
//!
//! // Four headings (E, N, W, S), one step forward in each.
//! let moves = [(0, 1), (-1, 0), (0, -1), (1, 0)];
//! let actions: Vec<Action> = (0..4u16)
//!     .map(|h| Action { h0: h, h1: h, dr: moves[h as usize].0, dc: moves[h as usize].1 })
//!     .collect();
//! let lat = Lattice::new(4, actions);
//! let index = CellIndex::full(3, 3);
//! struct Unit;
//! impl EdgeCost for Unit {
//!     fn cost(&self, _r: i32, _c: i32, _a: usize) -> f64 { 1.0 }
//! }
//! let src = index.state(0, 0, 0, 4).unwrap();
//! let res = search(&lat, &index, &Unit, &ZeroHeuristic, &SearchSpec::dijkstra(&[src]));
//! assert_eq!(res.g[index.state(0, 2, 0, 4).unwrap() as usize], 2.0);
//! ```

mod heap;
mod index;
mod lattice;
mod search;

pub use heap::MinHeap;
pub use index::CellIndex;
pub use lattice::{Action, Lattice};
pub use search::{search, EdgeCost, Heuristic, SearchResult, SearchSpec, ZeroHeuristic, NO_PARENT};
