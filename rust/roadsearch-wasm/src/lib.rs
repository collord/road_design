//! Browser bindings (wasm-bindgen) for the road search, run in a Web Worker.
//!
//! Free functions: UTM conversions, resampling a GeoTIFF window onto a UTM grid, hillshade.
//! `Session` holds one area of interest (a UTM grid of the DEM) and runs the pipeline a step at a
//! time, so the page can show each layer as it is ready: `prepare` (smoothing, gradients, slope
//! mask, flow accumulation), then `triangulate`, then `corridors` (graph, two-way search,
//! distinct corridors). Large results are typed arrays; small ones are JS objects.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use road_cost::{Grid, SectionTable};

#[wasm_bindgen]
pub fn utm_zone(lon: f64) -> u32 {
    terrain::utm::zone_for(lon)
}

/// [easting, northing] of a longitude and latitude (degrees).
#[wasm_bindgen]
pub fn to_utm(lon: f64, lat: f64, zone: u32, north: bool) -> Vec<f64> {
    let (e, n) = terrain::utm::to_utm(lon, lat, zone, north);
    vec![e, n]
}

/// [longitude, latitude] of a UTM position.
#[wasm_bindgen]
pub fn from_utm(e: f64, n: f64, zone: u32, north: bool) -> Vec<f64> {
    let (lon, lat) = terrain::utm::from_utm(e, n, zone, north);
    vec![lon, lat]
}

/// Resample a source raster (row-major sw x sh, NaN for nodata) onto a w x h target grid, given
/// the source pixel position (column, row; pixel centers at integers) of every `step`-th target
/// cell center on an lw x lh lattice; k x k sub-samples per cell when downsampling.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn resample(
    src: &[f32], sw: usize, sh: usize, lat_col: &[f32], lat_row: &[f32], lw: usize, lh: usize, step: usize, w: usize,
    h: usize, k: usize,
) -> Vec<f32> {
    terrain::resample_lattice(src, sw, sh, lat_col, lat_row, lw, lh, step, w, h, k)
}

/// Shaded relief, 0..255 (0 where there is no data).
#[wasm_bindgen]
pub fn hillshade(z: &[f32], w: usize, h: usize, res: f64) -> Vec<u8> {
    let z: Vec<f64> = z.iter().map(|&v| v as f64).collect();
    terrain::hillshade(&z, h, w, res, 315.0, 45.0, 1.0)
}

#[derive(Serialize)]
struct RouteOut {
    rows: Vec<u32>,
    cols: Vec<u32>,
    cost: f64,
    construction: f64,
    operating: f64,
    length: f64,
    hairpins: u32,
    turning: f64,
    how: String,
}

/// One area of interest: the DEM on a UTM grid and what has been computed on it.
#[wasm_bindgen]
pub struct Session {
    nr: usize,
    nc: usize,
    res: f64,
    z: Vec<f64>,
    zs: Vec<f64>,
    gx: Vec<f64>,
    gy: Vec<f64>,
    area: Vec<f64>,
    open: Vec<u8>,
    node_rows: Vec<u32>,
    node_cols: Vec<u32>,
    node_kinds: Vec<u8>,
    field: Vec<f32>,
    edge_pairs: Vec<u32>,
    edge_cost_per_m: Vec<f32>,
    routes: Vec<RouteOut>,
    tri_points: Vec<u32>,
    tri_triangles: Vec<u32>,
}

#[wasm_bindgen]
impl Session {
    /// The DEM (row-major w x h, meters, NaN for nodata) on a grid of `res` m cells.
    #[wasm_bindgen(constructor)]
    pub fn new(z: &[f32], w: usize, h: usize, res: f64) -> Session {
        Session {
            nr: h,
            nc: w,
            res,
            z: z.iter().map(|&v| v as f64).collect(),
            zs: Vec::new(),
            gx: Vec::new(),
            gy: Vec::new(),
            area: Vec::new(),
            open: Vec::new(),
            node_rows: Vec::new(),
            node_cols: Vec::new(),
            node_kinds: Vec::new(),
            field: Vec::new(),
            edge_pairs: Vec::new(),
            edge_cost_per_m: Vec::new(),
            routes: Vec::new(),
            tri_points: Vec::new(),
            tri_triangles: Vec::new(),
        }
    }

    /// Smoothing (sigma in cells), gradients, the slope mask (ground too steep for any road at
    /// g_max: sidehill at least s_mask, dilated by `radius` m) and flow accumulation.
    pub fn prepare(&mut self, sigma_cells: f64, g_max: f64, s_mask: f64, radius: f64) {
        let (nr, nc, res) = (self.nr, self.nc, self.res);
        self.zs = terrain::smooth(&self.z, nr, nc, sigma_cells);
        let (gx, gy) = terrain::gradient(&self.zs, nr, nc, res, -res);
        let slope: Vec<f64> = gx.iter().zip(&gy).map(|(a, b)| a.hypot(*b)).collect();
        let (_raw, dilated) = terrain::slope_mask(&slope, nr, nc, g_max, s_mask, radius, res, res);
        self.open = dilated.iter().map(|&m| u8::from(!m)).collect();
        self.gx = gx;
        self.gy = gy;
        self.area = terrain::contributing_area(&self.z, nr, nc, res, res, res * res / 1e4);
    }

    /// Exclude cells (1 = keep out), e.g. avoid zones drawn on the map.
    pub fn exclude(&mut self, mask: &[u8]) {
        for (o, &m) in self.open.iter_mut().zip(mask) {
            if m != 0 {
                *o = 0;
            }
        }
    }

    /// 1 where a road may go.
    pub fn open_mask(&self) -> Vec<u8> {
        self.open.clone()
    }

    /// Contributing area, ha (NaN at nodata).
    pub fn contributing_area(&self) -> Vec<f32> {
        self.area.iter().map(|&v| v as f32).collect()
    }

    /// Flat indices of the saddles of the smoothed DEM on open ground.
    pub fn saddles(&self) -> Vec<u32> {
        let zs = Grid::new(&self.zs, self.nr, self.nc);
        tin_graph::saddles(&zs)
            .into_iter()
            .filter(|&(r, c)| self.open[r * self.nc + c] != 0)
            .map(|(r, c)| (r * self.nc + c) as u32)
            .collect()
    }

    /// Triangulate the smoothed DEM to `tolerance` m (vertical); returns the number of triangles.
    /// Then tri_points() and tri_triangles().
    pub fn triangulate(&mut self, tolerance: f64) -> Result<usize, JsValue> {
        let zmin = self.zs.iter().copied().filter(|v| !v.is_nan()).fold(f64::INFINITY, f64::min);
        let h: Vec<f64> = self.zs.iter().map(|&v| if v.is_nan() { zmin } else { v }).collect();
        let (pts, tris) = delatin::triangulate(&h, (self.nc, self.nr), delatin::Error(tolerance))
            .map_err(|e| JsValue::from_str(&format!("triangulation failed: {e:?}")))?;
        self.tri_points = pts.iter().flat_map(|&(x, y)| [x as u32, y as u32]).collect();
        self.tri_triangles = tris.iter().flat_map(|&(a, b, c)| [a as u32, b as u32, c as u32]).collect();
        Ok(tris.len())
    }

    /// [col, row, ...] of the triangulation's vertices.
    pub fn tri_points(&self) -> Vec<u32> {
        self.tri_points.clone()
    }

    /// [a, b, c, ...] vertex indices of its triangles.
    pub fn tri_triangles(&self) -> Vec<u32> {
        self.tri_triangles.clone()
    }

    /// The coarse graph and its corridors between two cells. `edge`: [g_max, C_travel, C_vertical,
    /// A_min (ha), hairpin allowance, C_turn]; `tin`: [tolerance m, k, edge_max cells, block cells,
    /// channel node ha, alternatives, penalty, stretch, max_shared]; `section`: cost per meter over
    /// (sidehill, grade), ns x ng with steps ds, dg. Returns the routes; the graph and corridor
    /// field are then available from the getters.
    #[allow(clippy::too_many_arguments)]
    pub fn corridors(
        &mut self, start_r: usize, start_c: usize, goal_r: usize, goal_c: usize, edge: &[f64], breaks: &[f64],
        costs: &[f64], tin: &[f64], section: &[f64], ns: usize, ng: usize, ds: f64, dg: f64,
    ) -> Result<JsValue, JsValue> {
        if self.zs.is_empty() {
            return Err(JsValue::from_str("prepare() first"));
        }
        let (nr, nc) = (self.nr, self.nc);
        if start_r >= nr || start_c >= nc || goal_r >= nr || goal_c >= nc {
            return Err(JsValue::from_str("start or goal outside the area"));
        }
        if edge.len() < 6 || tin.len() < 9 || section.len() != ns * ng {
            return Err(JsValue::from_str("bad parameter arrays"));
        }
        let t = tin_graph::Rasters {
            z: Grid::new(&self.z, nr, nc),
            zs: Grid::new(&self.zs, nr, nc),
            gx: Grid::new(&self.gx, nr, nc),
            gy: Grid::new(&self.gy, nr, nc),
            open: Grid::new(&self.open, nr, nc),
            area: Grid::new(&self.area, nr, nc),
            section: SectionTable { data: section, ns, ng, ds, dg },
            res: self.res,
        };
        let e = tin_graph::EdgeParams {
            g_max: edge[0], c_travel: edge[1], c_vertical: edge[2], a_min: edge[3], hairpin_cost: edge[4],
            culvert_breaks: breaks.to_vec(), culvert_costs: costs.to_vec(),
        };
        let c_turn = edge[5];
        let p = tin_graph::TinParams {
            tolerance: tin[0], k: tin[1] as usize, edge_max: tin[2], block: tin[3] as usize, channel_node_area: tin[4],
            n_alt: tin[5] as usize, penalty: tin[6], stretch: tin[7], max_shared: tin[8],
        };
        let nodes = tin_graph::build_nodes(&t, &p, (start_r, start_c), (goal_r, goal_c));
        let g = tin_graph::build_graph(&t, &e, &p, nodes);
        let c = tin_graph::corridors(&g, &p, c_turn, 0, 1);
        self.node_rows = g.nodes.iter().map(|n| n.r as u32).collect();
        self.node_cols = g.nodes.iter().map(|n| n.c as u32).collect();
        self.node_kinds = g.nodes.iter().map(|n| n.kind).collect();
        self.field = c.field.iter().map(|&v| v as f32).collect();
        self.edge_pairs = g.edges.iter().flat_map(|&(u, v)| [u, v]).collect();
        self.edge_cost_per_m = g.costs.iter().map(|k| (k.total / k.length.max(1e-9)) as f32).collect();
        self.routes = c
            .routes
            .iter()
            .map(|r| RouteOut {
                rows: r.coords.iter().map(|x| x.0 as u32).collect(),
                cols: r.coords.iter().map(|x| x.1 as u32).collect(),
                cost: r.cost,
                construction: r.construction,
                operating: r.operating,
                length: r.length,
                hairpins: r.hairpins,
                turning: r.turning,
                how: r.how.to_string(),
            })
            .collect();
        serde_wasm_bindgen::to_value(&self.routes).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn node_rows(&self) -> Vec<u32> {
        self.node_rows.clone()
    }

    pub fn node_cols(&self) -> Vec<u32> {
        self.node_cols.clone()
    }

    /// 0 triangulation, 1 channel, 2 saddle, 3 background, 4 endpoint.
    pub fn node_kinds(&self) -> Vec<u8> {
        self.node_kinds.clone()
    }

    /// Cost of the best route through each node (inf where none).
    pub fn field(&self) -> Vec<f32> {
        self.field.clone()
    }

    /// [u, v, ...] node pairs of the graph's edges.
    pub fn edge_pairs(&self) -> Vec<u32> {
        self.edge_pairs.clone()
    }

    /// Each edge's cost per meter of road.
    pub fn edge_cost_per_m(&self) -> Vec<f32> {
        self.edge_cost_per_m.clone()
    }
}
