//! Python bindings for `lattice-search` and `road-cost`.
//!
//! `search` and `edge_costs` take the same lattice and terrain tuples as the numba
//! kernel in `roadsearch/search.py` (`SearchProblem.lat_tuple()`, `terr_tuple()`),
//! so either engine can serve the same problem. Large arrays are borrowed, not
//! copied, and the GIL is released while searching.

use lattice_search::{search as lsearch, Action, CellIndex, Lattice, SearchSpec, ZeroHeuristic};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use road_cost::{Actions, ClimbHeuristic, CostParams, Grid, MacroTable, RoadCost, SectionTable, Terrain};

fn item<'py, T: FromPyObjectOwned<'py>>(t: &Bound<'py, PyTuple>, i: usize, name: &str) -> PyResult<T> {
    t.get_item(i)?
        .extract::<T>()
        .map_err(|e| {
            let e: PyErr = e.into();
            PyValueError::new_err(format!("item {i} ({name}): {e}"))
        })
}

fn vec_i64(t: &Bound<'_, PyTuple>, i: usize, name: &str) -> PyResult<Vec<i64>> {
    let a: PyReadonlyArray1<i64> = item(t, i, name)?;
    Ok(a.as_slice()?.to_vec())
}

fn vec_f64(t: &Bound<'_, PyTuple>, i: usize, name: &str) -> PyResult<Vec<f64>> {
    let a: PyReadonlyArray1<f64> = item(t, i, name)?;
    Ok(a.as_slice()?.to_vec())
}

/// The lattice tuple (h0, h1, dr, dc, length, rev, s0, ns, sdr, sdc, tx, ty, by_heading, kind, slot, turn).
fn lattice_from(n_theta: usize, lat: &Bound<'_, PyTuple>) -> PyResult<(Lattice, Actions)> {
    let usz = |v: Vec<i64>| v.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    let h0 = vec_i64(lat, 0, "h0")?;
    let h1 = vec_i64(lat, 1, "h1")?;
    let dr = vec_i64(lat, 2, "dr")?;
    let dc = vec_i64(lat, 3, "dc")?;
    let bh: PyReadonlyArray2<i64> = item(lat, 12, "by_heading")?;
    let bh = bh.as_array();
    let by_heading = (0..n_theta)
        .map(|h| bh.row(h).iter().take_while(|&&a| a >= 0).map(|&a| a as u32).collect())
        .collect();
    let actions: Vec<Action> = (0..h0.len())
        .map(|i| Action { h0: h0[i] as u16, h1: h1[i] as u16, dr: dr[i] as i32, dc: dc[i] as i32 })
        .collect();
    let kind = vec_i64(lat, 13, "kind")?.into_iter().map(|k| k as u8).collect();
    let acts = Actions {
        h0: usz(h0),
        dr,
        dc,
        length: vec_f64(lat, 4, "length")?,
        rev: usz(vec_i64(lat, 5, "rev")?),
        s0: usz(vec_i64(lat, 6, "s0")?),
        ns: usz(vec_i64(lat, 7, "ns")?),
        sdr: vec_f64(lat, 8, "sdr")?,
        sdc: vec_f64(lat, 9, "sdc")?,
        tx: vec_f64(lat, 10, "tx")?,
        ty: vec_f64(lat, 11, "ty")?,
        kind,
        slot: usz(vec_i64(lat, 14, "slot")?),
        turn: vec_f64(lat, 15, "turn")?,
    };
    Ok((Lattice::with_order(n_theta, actions, by_heading), acts))
}

/// Borrowed views of the terrain tuple, alive while `guards` is.
struct TerrArrays<'py> {
    open: PyReadonlyArray2<'py, u8>,
    gx: PyReadonlyArray2<'py, f64>,
    gy: PyReadonlyArray2<'py, f64>,
    zs: PyReadonlyArray2<'py, f64>,
    hp: PyReadonlyArray3<'py, f32>,
    table: PyReadonlyArray2<'py, f64>,
    z_raw: PyReadonlyArray2<'py, f64>,
    channel: PyReadonlyArray2<'py, u8>,
    area: PyReadonlyArray2<'py, f64>,
    chdist: PyReadonlyArray2<'py, f64>,
    bend: PyReadonlyArray3<'py, f32>,
    macro_lookup: PyReadonlyArray2<'py, i32>,
    params: CostParams,
    ds: f64,
    dg: f64,
}

/// The terrain tuple (open_, gx, gy, zs, hp, table, ds, dg, res, g_max, C_base, z_raw, channel,
/// area, chdist, cbreaks, ccosts, lam, w, f_rock, H_max, tol, bend, C_travel, C_vert, C_turn, macro_lookup);
/// hp and bend are (cells, headings, slots), their rows given by macro_lookup (nr, nc).
fn terrain_from<'py>(terr: &Bound<'py, PyTuple>) -> PyResult<TerrArrays<'py>> {
    Ok(TerrArrays {
        open: item(terr, 0, "open_")?,
        gx: item(terr, 1, "gx")?,
        gy: item(terr, 2, "gy")?,
        zs: item(terr, 3, "zs")?,
        hp: item(terr, 4, "hp")?,
        table: item(terr, 5, "table")?,
        ds: item(terr, 6, "ds")?,
        dg: item(terr, 7, "dg")?,
        z_raw: item(terr, 11, "z_raw")?,
        channel: item(terr, 12, "channel")?,
        area: item(terr, 13, "area")?,
        chdist: item(terr, 14, "chdist")?,
        bend: item(terr, 22, "bend")?,
        macro_lookup: item(terr, 26, "macro_lookup")?,
        params: CostParams {
            res: item(terr, 8, "res")?,
            g_max: item(terr, 9, "g_max")?,
            c_base: item(terr, 10, "C_base")?,
            culvert_breaks: vec_f64(terr, 15, "cbreaks")?,
            culvert_costs: vec_f64(terr, 16, "ccosts")?,
            lam_fill: item(terr, 17, "lam")?,
            w: item(terr, 18, "w")?,
            f_rock: item(terr, 19, "f_rock")?,
            h_fill_max: item(terr, 20, "H_max")?,
            fill_tol: item(terr, 21, "tol")?,
            c_travel: item(terr, 23, "C_travel")?,
            c_vertical: item(terr, 24, "C_vert")?,
            c_turn: item(terr, 25, "C_turn")?,
        },
    })
}

fn shape3(a: &PyReadonlyArray3<f32>) -> [usize; 3] {
    let s = a.shape();
    [s[0], s[1], s[2]]
}

impl<'py> TerrArrays<'py> {
    fn views(&self) -> PyResult<(usize, usize, Terrain<'_>, SectionTable<'_>, MacroTable<'_>, MacroTable<'_>)> {
        let (nr, nc) = (self.open.shape()[0], self.open.shape()[1]);
        for (name, s) in [
            ("gx", self.gx.shape()),
            ("gy", self.gy.shape()),
            ("zs", self.zs.shape()),
            ("z_raw", self.z_raw.shape()),
            ("channel", self.channel.shape()),
            ("area", self.area.shape()),
            ("chdist", self.chdist.shape()),
            ("macro_lookup", self.macro_lookup.shape()),
        ] {
            if s != [nr, nc] {
                return Err(PyValueError::new_err(format!("{name} has shape {s:?}, open_ has ({nr}, {nc})")));
            }
        }
        let terrain = Terrain {
            open: Grid::new(self.open.as_slice()?, nr, nc),
            gx: Grid::new(self.gx.as_slice()?, nr, nc),
            gy: Grid::new(self.gy.as_slice()?, nr, nc),
            zs: Grid::new(self.zs.as_slice()?, nr, nc),
            z_raw: Grid::new(self.z_raw.as_slice()?, nr, nc),
            channel: Grid::new(self.channel.as_slice()?, nr, nc),
            area: Grid::new(self.area.as_slice()?, nr, nc),
            chdist: Grid::new(self.chdist.as_slice()?, nr, nc),
        };
        let ts = self.table.shape();
        let section = SectionTable { data: self.table.as_slice()?, ns: ts[0], ng: ts[1], ds: self.ds, dg: self.dg };
        let look = Grid::new(self.macro_lookup.as_slice()?, nr, nc);
        let hp = MacroTable::new(self.hp.as_slice()?, shape3(&self.hp), look);
        let bend = MacroTable::new(self.bend.as_slice()?, shape3(&self.bend), look);
        Ok((nr, nc, terrain, section, hp, bend))
    }
}

type SearchOut<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<u8>>,
    u64,
    i64,
);

/// Dijkstra (astar=False) or A* (astar=True, stops at the goal cell) over the open cells.
///
/// sources: dense state ids, (row * ncols + col) * n_theta + h, starting at cost 0.
/// Returns (cells, G, parent, closed, n_expanded, goal_state): `cells` are the flat grid
/// indices of the searched cells, and G, parent and closed hold n_theta values per cell in
/// that order; goal_state is a dense state id, or -1.
#[pyfunction]
#[pyo3(signature = (n_theta, lat, terr, sources, goal_r, goal_c, astar, max_cost))]
#[allow(clippy::too_many_arguments)]
fn search<'py>(
    py: Python<'py>,
    n_theta: usize,
    lat: &Bound<'py, PyTuple>,
    terr: &Bound<'py, PyTuple>,
    sources: PyReadonlyArray1<'py, i64>,
    goal_r: i64,
    goal_c: i64,
    astar: bool,
    max_cost: f64,
) -> PyResult<SearchOut<'py>> {
    let (lattice, actions) = lattice_from(n_theta, lat)?;
    let ta = terrain_from(terr)?;
    let (nr, nc, terrain, section, hp, bend) = ta.views()?;
    let sources = sources.as_slice()?.to_vec();
    let nh = n_theta as i64;
    // States live on open cells; sources and the goal cell are always included.
    let mut mask: Vec<bool> = terrain.open.data.iter().map(|&o| o != 0).collect();
    for &s in &sources {
        let cell = s / nh;
        if s < 0 || cell >= (nr * nc) as i64 {
            return Err(PyValueError::new_err(format!("source state {s} is outside the grid")));
        }
        mask[cell as usize] = true;
    }
    let goal = if goal_r >= 0 { Some((goal_r as i32, goal_c as i32)) } else { None };
    if let Some((r, c)) = goal {
        if r as usize >= nr || c as usize >= nc {
            return Err(PyValueError::new_err("goal cell is outside the grid"));
        }
        mask[r as usize * nc + c as usize] = true;
    }
    if astar && goal.is_none() {
        return Err(PyValueError::new_err("A* needs a goal"));
    }
    let params = &ta.params;
    let model = RoadCost { actions: &actions, terrain, section, hairpins: hp, bends: bend, params };
    let res = py.detach(|| {
        let index = CellIndex::from_mask(nr, nc, &mask);
        let src: Vec<u64> = sources
            .iter()
            .map(|&s| index.get((s / nh) as i64 / nc as i64, (s / nh) % nc as i64).unwrap() as u64 * nh as u64 + (s % nh) as u64)
            .collect();
        let spec = SearchSpec { sources: &src, goal, stop_at_goal: astar, max_cost };
        let out = if astar {
            let (gr, gc) = goal.unwrap();
            let heur = ClimbHeuristic {
                goal: (gr, gc),
                goal_z: terrain.zs.data[gr as usize * nc + gc as usize],
                zs: terrain.zs,
                res: params.res,
                g_max: params.g_max,
                c_len: params.c_base + params.c_travel,
                c_vertical: params.c_vertical,
            };
            lsearch(&lattice, &index, &model, &heur, &spec)
        } else {
            lsearch(&lattice, &index, &model, &ZeroHeuristic, &spec)
        };
        let goal_state = out.goal_state.map_or(-1, |s| {
            let cell = index.cells()[(s / nh as u64) as usize] as i64;
            cell * nh + (s % nh as u64) as i64
        });
        let cells: Vec<i64> = index.cells().iter().map(|&k| k as i64).collect();
        (cells, out.g, out.parent, out.closed, out.n_expanded, goal_state)
    });
    let (cells, g, parent, closed, n_exp, goal_state) = res;
    Ok((cells.into_pyarray(py), g.into_pyarray(py), parent.into_pyarray(py), closed.into_pyarray(py), n_exp, goal_state))
}

/// Cost of many edges (rows[i], cols[i], actions[i]), as search.edge_cost gives them.
#[pyfunction]
fn edge_costs<'py>(
    py: Python<'py>,
    n_theta: usize,
    lat: &Bound<'py, PyTuple>,
    terr: &Bound<'py, PyTuple>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    actions: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (_, acts) = lattice_from(n_theta, lat)?;
    let ta = terrain_from(terr)?;
    let (_, _, terrain, section, hp, bend) = ta.views()?;
    let model = RoadCost { actions: &acts, terrain, section, hairpins: hp, bends: bend, params: &ta.params };
    let (r, c, a) = (rows.as_slice()?, cols.as_slice()?, actions.as_slice()?);
    if r.len() != c.len() || r.len() != a.len() {
        return Err(PyValueError::new_err("rows, cols and actions must have the same length"));
    }
    let out: Vec<f64> = py.detach(|| (0..r.len()).map(|i| model.edge_cost(r[i], c[i], a[i] as usize)).collect());
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn roadsearch_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(edge_costs, m)?)?;
    Ok(())
}
