//! Python bindings for `lattice-search` and `road-cost`.
//!
//! `search` and `edge_costs` take the same lattice and terrain tuples as the numba
//! kernel in `roadsearch/search.py` (`SearchProblem.lat_tuple()`, `terr_tuple()`),
//! so either engine can serve the same problem. Large arrays are borrowed, not
//! copied, and the GIL is released while searching.

use lattice_search::{search_with_progress, Action, CellIndex, Lattice, Progress, SearchSpec, ZeroHeuristic};
use numpy::{
    IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use road_cost::macros::{self, BendParams, BendShapes, HairpinParams, Headings, MacroTerrain, N_SLOTS};
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
    bool,
);

/// Dijkstra (astar=False) or A* (astar=True, stops at the goal cell) over the open cells.
///
/// sources: dense state ids, (row * ncols + col) * n_theta + h, starting at cost 0.
/// Returns (cells, G, parent, closed, n_expanded, goal_state, completed): `cells` are the flat
/// grid indices of the searched cells, and G, parent and closed hold n_theta values per cell in
/// that order; goal_state is a dense state id, or -1.
///
/// progress: called as progress(n_expanded, key, queue_len) every `every` settled states, with the
/// GIL held. If it returns False the search stops (completed is False); if it raises, the search
/// stops and the exception propagates (a pending KeyboardInterrupt lands here).
#[pyfunction]
#[pyo3(signature = (n_theta, lat, terr, sources, goal_r, goal_c, astar, max_cost, progress=None, every=1 << 20))]
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
    progress: Option<Py<PyAny>>,
    every: u64,
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
    let mut error: Option<PyErr> = None;
    let res = py.detach(|| {
        let mut report = |p: &Progress| -> bool {
            let Some(cb) = progress.as_ref() else { return true };
            Python::attach(|py| match cb.call1(py, (p.n_expanded, p.key, p.queue_len)) {
                Ok(v) => v.bind(py).is_truthy().unwrap_or(true),
                Err(e) => {
                    error = Some(e);
                    false
                }
            })
        };
        let every = if progress.is_some() { every.max(1) } else { u64::MAX };
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
            search_with_progress(&lattice, &index, &model, &heur, &spec, every, &mut report)
        } else {
            search_with_progress(&lattice, &index, &model, &ZeroHeuristic, &spec, every, &mut report)
        };
        let (out, completed) = out;
        let goal_state = out.goal_state.map_or(-1, |s| {
            let cell = index.cells()[(s / nh as u64) as usize] as i64;
            cell * nh + (s % nh as u64) as i64
        });
        let cells: Vec<i64> = index.cells().iter().map(|&k| k as i64).collect();
        (cells, out.g, out.parent, out.closed, out.n_expanded, goal_state, completed)
    });
    if let Some(e) = error {
        return Err(e);
    }
    let (cells, g, parent, closed, n_exp, goal_state, completed) = res;
    Ok((cells.into_pyarray(py), g.into_pyarray(py), parent.into_pyarray(py), closed.into_pyarray(py), n_exp, goal_state,
        completed))
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

/// Borrowed terrain for macro tables: (open_, zs, gx, gy, soil, z_raw) and drainage
/// (channel, area_ha, culvert breaks, culvert costs).
struct MacroArrays<'py> {
    open: PyReadonlyArray2<'py, u8>,
    zs: PyReadonlyArray2<'py, f64>,
    gx: PyReadonlyArray2<'py, f64>,
    gy: PyReadonlyArray2<'py, f64>,
    soil: PyReadonlyArray2<'py, f64>,
    zr: PyReadonlyArray2<'py, f64>,
    channel: PyReadonlyArray2<'py, u8>,
    area: PyReadonlyArray2<'py, f64>,
    breaks: PyReadonlyArray1<'py, f64>,
    costs: PyReadonlyArray1<'py, f64>,
}

impl<'py> MacroArrays<'py> {
    fn from_tuples(terrain: &Bound<'py, PyTuple>, drainage: &Bound<'py, PyTuple>) -> PyResult<Self> {
        Ok(MacroArrays {
            open: item(terrain, 0, "open_")?,
            zs: item(terrain, 1, "zs")?,
            gx: item(terrain, 2, "gx")?,
            gy: item(terrain, 3, "gy")?,
            soil: item(terrain, 4, "soil")?,
            zr: item(terrain, 5, "z_raw")?,
            channel: item(drainage, 0, "channel")?,
            area: item(drainage, 1, "area")?,
            breaks: item(drainage, 2, "culvert breaks")?,
            costs: item(drainage, 3, "culvert costs")?,
        })
    }

    fn view(&self) -> PyResult<MacroTerrain<'_>> {
        let (nr, nc) = (self.zr.shape()[0], self.zr.shape()[1]);
        for (name, s) in [
            ("open_", self.open.shape()),
            ("zs", self.zs.shape()),
            ("gx", self.gx.shape()),
            ("gy", self.gy.shape()),
            ("soil", self.soil.shape()),
            ("channel", self.channel.shape()),
            ("area", self.area.shape()),
        ] {
            if s != [nr, nc] {
                return Err(PyValueError::new_err(format!("{name} has shape {s:?}, z_raw has ({nr}, {nc})")));
            }
        }
        Ok(MacroTerrain {
            open: Grid::new(self.open.as_slice()?, nr, nc),
            zs: Grid::new(self.zs.as_slice()?, nr, nc),
            gx: Grid::new(self.gx.as_slice()?, nr, nc),
            gy: Grid::new(self.gy.as_slice()?, nr, nc),
            soil: Grid::new(self.soil.as_slice()?, nr, nc),
            zr: Grid::new(self.zr.as_slice()?, nr, nc),
            channel: Grid::new(self.channel.as_slice()?, nr, nc),
            area: Grid::new(self.area.as_slice()?, nr, nc),
            culvert_breaks: self.breaks.as_slice()?,
            culvert_costs: self.costs.as_slice()?,
        })
    }
}

fn floats<const N: usize>(t: &Bound<'_, PyTuple>, what: &str) -> PyResult<[f64; N]> {
    if t.len() != N {
        return Err(PyValueError::new_err(format!("{what} needs {N} values, got {}", t.len())));
    }
    let mut out = [0.0; N];
    for (i, v) in out.iter_mut().enumerate() {
        *v = item(t, i, what)?;
    }
    Ok(out)
}

/// Left-hairpin costs per (cell, heading, slot), float32, inf where infeasible (as hairpin.hairpin_table).
/// terrain: (open_, zs, gx, gy, soil, z_raw); drainage: (channel, area_ha, breaks, costs);
/// params: hairpin.HairpinParams as a tuple; slots: (n_theta, 2) leg spacings in lattice steps.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn hairpin_table<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, i64>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
    slots: PyReadonlyArray2<'py, i64>,
    res: f64,
    terrain: &Bound<'py, PyTuple>,
    drainage: &Bound<'py, PyTuple>,
    params: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a = MacroArrays::from_tuples(terrain, drainage)?;
    let t = a.view()?;
    let p = floats::<21>(params, "hairpin params")?;
    let hp = HairpinParams {
        w_hp: p[0], g_hp: p[1], g_max: p[2], leg_max: p[3], c_soil: p[4], c_rock: p[5], f_soil: p[6],
        f_rock: p[7], k_soil: p[8], k_rock: p[9], cost_base: p[10], cost_soil: p[11], cost_rock: p[12],
        cost_borrow: p[13], catch_max: p[14], delta: p[15], r_hp: p[16], rho: p[17], cost_wall: p[18],
        c_travel: p[19], c_vertical: p[20],
    };
    let nh = vx.len();
    if slots.shape() != [nh, N_SLOTS] {
        return Err(PyValueError::new_err(format!("slots must be ({nh}, {N_SLOTS})")));
    }
    let (cells, vx, vy, slots) = (cells.as_slice()?, vx.as_slice()?, vy.as_slice()?, slots.as_slice()?);
    let heads = Headings { vx, vy };
    let table = py.detach(|| macros::hairpin_table(cells, &heads, slots, res, &t, &hp));
    Ok(table.into_pyarray(py).reshape([cells.len(), nh, N_SLOTS])?)
}

/// Left-bend costs per (cell, heading, slot), float32, inf where infeasible (as bend.bend_table).
/// shapes: (X, Y, L1, L2, dth), each (n_theta, slots), from lattice.BendGeometry; radius in cells;
/// params: bend.BendParams as a tuple.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bend_table<'py>(
    py: Python<'py>,
    cells: PyReadonlyArray1<'py, i64>,
    vx: PyReadonlyArray1<'py, f64>,
    vy: PyReadonlyArray1<'py, f64>,
    shapes: &Bound<'py, PyTuple>,
    radius: f64,
    res: f64,
    terrain: &Bound<'py, PyTuple>,
    drainage: &Bound<'py, PyTuple>,
    params: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let a = MacroArrays::from_tuples(terrain, drainage)?;
    let t = a.view()?;
    let p = floats::<16>(params, "bend params")?;
    let bp = BendParams {
        w: p[0], g_max: p[1], c_soil: p[2], c_rock: p[3], f_soil: p[4], f_rock: p[5], k_soil: p[6], k_rock: p[7],
        cost_base: p[8], cost_soil: p[9], cost_rock: p[10], cost_borrow: p[11], catch_max: p[12], delta: p[13],
        c_travel: p[14], c_vertical: p[15],
    };
    let bx: PyReadonlyArray2<i64> = item(shapes, 0, "X")?;
    let by: PyReadonlyArray2<i64> = item(shapes, 1, "Y")?;
    let l1: PyReadonlyArray2<f64> = item(shapes, 2, "L1")?;
    let l2: PyReadonlyArray2<f64> = item(shapes, 3, "L2")?;
    let dth: PyReadonlyArray2<f64> = item(shapes, 4, "dth")?;
    let nh = vx.len();
    let ns = bx.shape()[1];
    for (name, s) in [("X", bx.shape()), ("Y", by.shape()), ("L1", l1.shape()), ("L2", l2.shape()), ("dth", dth.shape())] {
        if s != [nh, ns] {
            return Err(PyValueError::new_err(format!("{name} must be ({nh}, {ns})")));
        }
    }
    let shapes = BendShapes {
        ns,
        x: bx.as_slice()?,
        y: by.as_slice()?,
        l1: l1.as_slice()?,
        l2: l2.as_slice()?,
        dth: dth.as_slice()?,
        radius,
    };
    let (cells, vx, vy) = (cells.as_slice()?, vx.as_slice()?, vy.as_slice()?);
    let heads = Headings { vx, vy };
    let table = py.detach(|| macros::bend_table(cells, &heads, &shapes, res, &t, &bp));
    Ok(table.into_pyarray(py).reshape([cells.len(), nh, ns])?)
}

#[pymodule]
fn roadsearch_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(edge_costs, m)?)?;
    m.add_function(wrap_pyfunction!(hairpin_table, m)?)?;
    m.add_function(wrap_pyfunction!(bend_table, m)?)?;
    Ok(())
}
