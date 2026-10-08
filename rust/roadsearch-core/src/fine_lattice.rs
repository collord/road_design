//! Python bindings for road_fine::lattice (parity tests against roadsearch/lattice.py).

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use road_fine::lattice::{build_bends, lattice_arrays, BendGeometry};

fn value_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Row-major values of a 2D array of any layout, and its column count.
fn rows<T: numpy::Element + Copy>(t: &Bound<'_, PyTuple>, i: usize, name: &str) -> PyResult<(Vec<T>, usize)> {
    let a: PyReadonlyArray2<T> = t
        .get_item(i)?
        .extract()
        .map_err(|e| PyValueError::new_err(format!("bends item {i} ({name}): {}", PyErr::from(e))))?;
    Ok((a.as_array().iter().copied().collect(), a.shape()[1]))
}

/// The bend tuple (X, Y, h1, L1, L2, dth, R) of lattice.BendGeometry, each array (n_theta, slots).
fn bends_from(t: &Bound<'_, PyTuple>) -> PyResult<BendGeometry> {
    let (x, ns) = rows::<i64>(t, 0, "X")?;
    let bg = BendGeometry {
        ns,
        x,
        y: rows(t, 1, "Y")?.0,
        h1: rows(t, 2, "h1")?.0,
        l1: rows(t, 3, "L1")?.0,
        l2: rows(t, 4, "L2")?.0,
        dth: rows(t, 5, "dth")?.0,
        r: t.get_item(6)?.extract()?,
    };
    let n = bg.x.len();
    if [bg.y.len(), bg.h1.len(), bg.l1.len(), bg.l2.len(), bg.dth.len()].iter().any(|&m| m != n) {
        return Err(PyValueError::new_err("bend arrays must share one shape"));
    }
    Ok(bg)
}

/// lattice.lattice_arrays as the tuple of SearchProblem.lat_tuple(): (h0, h1, dr, dc, length, rev,
/// s0, ns, sdr, sdc, tx, ty, by_heading, kind, slot, turn). hp_slots is (n_theta, n_slots); bends
/// is (X, Y, h1, L1, L2, dth, R) as lattice.BendGeometry.
#[pyfunction]
#[pyo3(signature = (n_theta, r_min_cells, hp_slots, cross_max_cells, bends, move_max_cells, kink_max_deg))]
#[allow(clippy::too_many_arguments)]
fn fine_lattice_arrays<'py>(
    py: Python<'py>,
    n_theta: usize,
    r_min_cells: f64,
    hp_slots: Option<PyReadonlyArray2<'py, i64>>,
    cross_max_cells: f64,
    bends: Option<Bound<'py, PyTuple>>,
    move_max_cells: i64,
    kink_max_deg: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let slots: Option<Vec<i64>> = match &hp_slots {
        Some(a) if a.shape()[0] != n_theta => {
            return Err(PyValueError::new_err(format!("hp_slots must have {n_theta} rows")));
        }
        Some(a) => Some(a.as_array().iter().copied().collect()),
        None => None,
    };
    let bends = bends.as_ref().map(bends_from).transpose()?;
    let l = lattice_arrays(
        n_theta,
        r_min_cells,
        slots.as_deref(),
        cross_max_cells,
        bends.as_ref(),
        move_max_cells,
        kink_max_deg,
    )
    .map_err(value_err)?;
    let by_heading = l.by_heading.into_pyarray(py).reshape([n_theta, l.max_out])?;
    let items: Vec<Bound<'py, PyAny>> = vec![
        l.h0.into_pyarray(py).into_any(),
        l.h1.into_pyarray(py).into_any(),
        l.dr.into_pyarray(py).into_any(),
        l.dc.into_pyarray(py).into_any(),
        l.length.into_pyarray(py).into_any(),
        l.rev.into_pyarray(py).into_any(),
        l.s0.into_pyarray(py).into_any(),
        l.ns.into_pyarray(py).into_any(),
        l.sdr.into_pyarray(py).into_any(),
        l.sdc.into_pyarray(py).into_any(),
        l.tx.into_pyarray(py).into_any(),
        l.ty.into_pyarray(py).into_any(),
        by_heading.into_any(),
        l.kind.into_pyarray(py).into_any(),
        l.slot.into_pyarray(py).into_any(),
        l.turn.into_pyarray(py).into_any(),
    ];
    PyTuple::new(py, items)
}

/// lattice.build_bends as (X, Y, h1, L1, L2, dth, R), the arrays (n_theta, slots).
#[pyfunction]
fn fine_build_bends<'py>(py: Python<'py>, n_theta: usize, r_cells: f64) -> PyResult<Bound<'py, PyTuple>> {
    let b = build_bends(n_theta, r_cells).map_err(value_err)?;
    let ns = b.ns;
    let items: Vec<Bound<'py, PyAny>> = vec![
        b.x.into_pyarray(py).reshape([n_theta, ns])?.into_any(),
        b.y.into_pyarray(py).reshape([n_theta, ns])?.into_any(),
        b.h1.into_pyarray(py).reshape([n_theta, ns])?.into_any(),
        b.l1.into_pyarray(py).reshape([n_theta, ns])?.into_any(),
        b.l2.into_pyarray(py).reshape([n_theta, ns])?.into_any(),
        b.dth.into_pyarray(py).reshape([n_theta, ns])?.into_any(),
        b.r.into_pyobject(py)?.into_any(),
    ];
    PyTuple::new(py, items)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fine_lattice_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(fine_build_bends, m)?)?;
    Ok(())
}
