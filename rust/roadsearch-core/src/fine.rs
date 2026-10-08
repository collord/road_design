//! Python bindings for road_fine (parity tests against the Python pipeline). Parameters come
//! as JSON (Params.to_dict() with infinities as null).

use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use road_fine::params::Params;
use road_fine::pipeline::{find_route, RouteResult};
use road_fine::problem::{FineInputs, Problem};
use road_fine::section::{cost_table, section, SectionParams, TABLE_NG, TABLE_NS};

fn params(json: &str) -> PyResult<Params> {
    Params::from_json(json).map_err(PyValueError::new_err)
}

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// section.section_arrays: (n, 11) outputs of the section for sidehill slopes s and soil depths d.
#[pyfunction]
fn fine_section<'py>(py: Python<'py>, s: PyReadonlyArray1<'py, f64>, d: PyReadonlyArray1<'py, f64>, params_json: &str)
                     -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = params(params_json)?;
    let sp = SectionParams::new(&p);
    let (s, d) = (s.as_slice()?, d.as_slice()?);
    let mut out = Vec::with_capacity(s.len() * 11);
    for i in 0..s.len() {
        let x = section(s[i], d[i], &sp);
        out.extend([x.kind as f64, x.a, x.b, x.a_soil, x.a_rock, x.a_fill, x.spoil, x.cut_catch, x.fill_toe, x.f, x.cost]);
    }
    out.into_pyarray(py).reshape([s.len(), 11])
}

/// search.build_cost_table: (table, ds, dg).
#[pyfunction]
fn fine_cost_table<'py>(py: Python<'py>, params_json: &str) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, f64)> {
    let p = params(params_json)?;
    let (t, ds, dg) = cost_table(&p);
    Ok((t.into_pyarray(py).reshape([TABLE_NS, TABLE_NG])?, ds, dg))
}

fn inputs(z_raw: PyReadonlyArray2<f64>, zs: PyReadonlyArray2<f64>, gx: PyReadonlyArray2<f64>, gy: PyReadonlyArray2<f64>,
          area: PyReadonlyArray2<f64>, open: PyReadonlyArray2<u8>, res: f64) -> PyResult<FineInputs> {
    let (nr, nc) = (z_raw.shape()[0], z_raw.shape()[1]);
    Ok(FineInputs {
        nr, nc, res,
        z_raw: z_raw.as_slice()?.to_vec(),
        zs: zs.as_slice()?.to_vec(),
        gx: gx.as_slice()?.to_vec(),
        gy: gy.as_slice()?.to_vec(),
        area: area.as_slice()?.to_vec(),
        open: open.as_slice()?.to_vec(),
    })
}

/// search.make_problem's arrays: {soil, channel, chdist, hp, bend, macro_lookup, table, ds, dg}.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fine_problem<'py>(py: Python<'py>, z_raw: PyReadonlyArray2<'py, f64>, zs: PyReadonlyArray2<'py, f64>,
                     gx: PyReadonlyArray2<'py, f64>, gy: PyReadonlyArray2<'py, f64>, area: PyReadonlyArray2<'py, f64>,
                     open: PyReadonlyArray2<'py, u8>, res: f64, params_json: &str) -> PyResult<Bound<'py, PyDict>> {
    let inp = inputs(z_raw, zs, gx, gy, area, open, res)?;
    let (nr, nc) = (inp.nr, inp.nc);
    let p = params(params_json)?;
    let prob = py.detach(|| Problem::build(inp, p)).map_err(err)?;
    let nh = prob.n_theta();
    let n = prob.table_cells.len();
    let d = PyDict::new(py);
    d.set_item("soil", prob.soil.clone().into_pyarray(py).reshape([nr, nc])?)?;
    d.set_item("channel", prob.channel.clone().into_pyarray(py).reshape([nr, nc])?)?;
    d.set_item("chdist", prob.chdist.clone().into_pyarray(py).reshape([nr, nc])?)?;
    d.set_item("hp", prob.hp.clone().into_pyarray(py).reshape([n, nh, 2])?)?;
    d.set_item("bend", prob.bend.clone().into_pyarray(py).reshape([n, nh, prob.n_bend_slots])?)?;
    d.set_item("macro_lookup", prob.macro_lookup.clone().into_pyarray(py).reshape([nr, nc])?)?;
    d.set_item("table", prob.table.clone().into_pyarray(py).reshape([TABLE_NS, TABLE_NG])?)?;
    d.set_item("ds", prob.ds)?;
    d.set_item("dg", prob.dg)?;
    Ok(d)
}

fn route_dict<'py>(py: Python<'py>, prob: &Problem, r: &RouteResult, start: (i64, i64)) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let edges: Vec<(i64, i64, usize)> = r.edges.clone();
    d.set_item("edges", edges)?;
    d.set_item("lam_fill", r.lam_fill)?;
    let c = &r.cost;
    let cost = PyDict::new(py);
    for (k, v) in [("total", c.total), ("surrogate", c.surrogate), ("base", c.base), ("excavation", c.excavation),
                   ("culverts", c.culverts), ("crossing_fill", c.crossing_fill), ("operating", c.operating),
                   ("fill_m3", c.mass.fill_m3), ("borrow_m3", c.mass.borrow_m3), ("haul_cost", c.mass.haul_cost)] {
        cost.set_item(k, v)?;
    }
    d.set_item("cost", cost)?;
    let segs = PyList::empty(py);
    for s in &r.segments {
        let x = PyDict::new(py);
        x.set_item("seq", s.seq)?;
        x.set_item("r", s.r)?;
        x.set_item("c", s.c)?;
        x.set_item("action", s.action)?;
        x.set_item("section_type", s.kind)?;
        for (k, v) in [("length", s.length), ("grade", s.grade), ("cut_soil_m3", s.cut_soil_m3),
                       ("cut_rock_m3", s.cut_rock_m3), ("fill_m3", s.fill_m3), ("spoil_m3", s.spoil_m3),
                       ("borrow_m3", s.borrow_m3), ("cost_base", s.cost_base), ("cost_earthwork", s.cost_earthwork),
                       ("cost_surrogate", s.cost_surrogate), ("cost_culvert", s.cost_culvert),
                       ("vertical_travel", s.vertical_travel), ("turning", s.turning), ("cost_operating", s.cost_operating)] {
            x.set_item(k, v)?;
        }
        x.set_item("pts", s.pts.clone())?;
        x.set_item("z", s.z.clone())?;
        segs.append(x)?;
    }
    d.set_item("segments", segs)?;
    d.set_item("path", prob.path_points(&r.edges, start).map_err(err)?)?;
    Ok(d)
}

/// The fine stage on one grid (pipeline.fill_price_iteration then avoid_self_overlap):
/// {best: {edges, lam_fill, cost, segments, path} or None, history, rounds, blocked, completed}.
#[pyfunction]
#[pyo3(signature = (z_raw, zs, gx, gy, area, open, res, params_json, start, goal, lazy=false))]
#[allow(clippy::too_many_arguments)]
fn fine_route<'py>(py: Python<'py>, z_raw: PyReadonlyArray2<'py, f64>, zs: PyReadonlyArray2<'py, f64>,
                   gx: PyReadonlyArray2<'py, f64>, gy: PyReadonlyArray2<'py, f64>, area: PyReadonlyArray2<'py, f64>,
                   open: PyReadonlyArray2<'py, u8>, res: f64, params_json: &str, start: (i64, i64), goal: (i64, i64),
                   lazy: bool) -> PyResult<Bound<'py, PyDict>> {
    let inp = inputs(z_raw, zs, gx, gy, area, open, res)?;
    let (nr, nc) = (inp.nr, inp.nc);
    let p = params(params_json)?;
    let out = py.detach(|| -> Result<_, String> {
        let mut prob = if lazy { Problem::new(inp, p, true)? } else { Problem::build(inp, p)? };
        let r = find_route(&mut prob, start, goal, &mut |_, _| true)?;
        Ok((prob, r))
    });
    let (prob, (it, rounds, blocked)) = out.map_err(err)?;
    let d = PyDict::new(py);
    match &it.best {
        Some(b) => d.set_item("best", route_dict(py, &prob, b, start)?)?,
        None => d.set_item("best", py.None())?,
    }
    let hist = PyList::empty(py);
    for h in &it.history {
        let x = PyDict::new(py);
        x.set_item("iteration", h.iteration)?;
        x.set_item("lam_fill", h.lam_fill)?;
        x.set_item("surrogate", h.surrogate)?;
        x.set_item("true", h.true_cost)?;
        x.set_item("n_crossings", h.n_crossings)?;
        x.set_item("shared_with_previous", h.shared_with_previous)?;
        hist.append(x)?;
    }
    d.set_item("history", hist)?;
    let rl = PyList::empty(py);
    for r in &rounds {
        let x = PyDict::new(py);
        x.set_item("round", r.round)?;
        x.set_item("overlapping_pairs", r.overlapping_pairs)?;
        x.set_item("seq_range", vec![r.seq_range.0, r.seq_range.1])?;
        x.set_item("blocked_cells", r.blocked_cells)?;
        x.set_item("cost_before", r.cost_before)?;
        x.set_item("cost_after", r.cost_after)?;
        x.set_item("result", r.result)?;
        rl.append(x)?;
    }
    d.set_item("rounds", rl)?;
    d.set_item("blocked", blocked.into_pyarray(py).reshape([nr, nc])?)?;
    d.set_item("final_lam", it.final_lam)?;
    d.set_item("completed", it.completed)?;
    Ok(d)
}

/// route.self_overlaps: (i, j, separation m) for polylines of (row, col) points.
#[pyfunction]
fn fine_self_overlaps(polylines: Vec<Vec<(f64, f64)>>, res: f64, width: f64) -> Vec<(usize, usize, f64)> {
    road_fine::route::self_overlaps(&polylines, res, width, 1.3)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fine_self_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(fine_section, m)?)?;
    m.add_function(wrap_pyfunction!(fine_cost_table, m)?)?;
    m.add_function(wrap_pyfunction!(fine_problem, m)?)?;
    m.add_function(wrap_pyfunction!(fine_route, m)?)?;
    Ok(())
}
