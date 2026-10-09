//! The fine road search: a port of the Python pipeline around the lattice search
//! (`roadsearch/lattice.py`, `section.py`, `search.make_problem`, `pipeline.fill_price_iteration`
//! and `avoid_self_overlap`, `route.evaluate_route`, `masshaul.true_cost`), so the whole fine
//! stage runs without Python: in the browser (WebAssembly) and, for parity tests, from Python.
//!
//! Kept operation for operation with the Python, so a route and its costs agree with it.

// `!(x < y)` keeps the Python's NaN semantics; shapes and kernels take many scalars, as there.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::too_many_arguments, clippy::type_complexity)]

pub mod lattice;
pub mod masshaul;
pub mod network;
pub mod params;
pub mod pipeline;
pub mod problem;
pub mod route;
pub mod section;
