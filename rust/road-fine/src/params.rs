//! The parameters the fine search uses (a subset of roadsearch/params.py's `Params`), read from
//! the same JSON the Python writes (`Params.to_dict()`, infinities as null).

use serde::{Deserialize, Deserializer};

/// null (Python's inf written as JSON) reads as infinity.
fn inf_if_null<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::INFINITY))
}

#[derive(Clone, Debug, Deserialize)]
pub struct Params {
    pub w: f64,
    pub g_max: f64,
    #[serde(rename = "R_min")]
    pub r_min: f64,
    #[serde(rename = "R_hp")]
    pub r_hp: f64,
    pub g_hp: f64,
    pub w_hp: f64,
    pub hp_leg_max: f64,
    pub hp_gap_min: f64,
    /// Hairpin sites every this many R_hp (hairpin.py site_step); 0 or absent: every cell.
    #[serde(default)]
    pub hp_site_spacing: f64,
    /// Reversing switchbacks instead of loop hairpins (temporary pad roads): the length (m) of
    /// the level landing at each, where a truck pulls in and reverses onto the next leg. 0 or
    /// absent: loop hairpins.
    #[serde(default)]
    pub switchback_landing: f64,
    /// Each switchback costs this many times its landing (reversing's time, wear and risk).
    #[serde(default = "one")]
    pub switchback_weight: f64,
    #[serde(rename = "C_wall", deserialize_with = "inf_if_null")]
    pub c_wall: f64,
    pub d0: f64,
    #[serde(rename = "S0")]
    pub s0: f64,
    /// Roughness (m) over which soil thins by e (roughness.py); 0 or absent: ignored.
    #[serde(default)]
    pub rough_r0: f64,
    pub c_soil: f64,
    pub c_rock: f64,
    pub f_soil: f64,
    pub f_rock: f64,
    pub k_soil: f64,
    pub k_rock: f64,
    pub b_min: f64,
    pub mask_margin: f64,
    pub catch_max: f64,
    #[serde(rename = "C_base")]
    pub c_base: f64,
    #[serde(rename = "C_soil")]
    pub c_soil_cost: f64,
    #[serde(rename = "C_rock")]
    pub c_rock_cost: f64,
    #[serde(rename = "C_borrow")]
    pub c_borrow: f64,
    #[serde(rename = "C_haul_down")]
    pub c_haul_down: f64,
    #[serde(rename = "C_haul_up")]
    pub c_haul_up: f64,
    pub haul_max: f64,
    #[serde(rename = "C_travel")]
    pub c_travel: f64,
    #[serde(rename = "C_vertical")]
    pub c_vertical: f64,
    #[serde(rename = "C_turn")]
    pub c_turn: f64,
    #[serde(rename = "A_min")]
    pub a_min: f64,
    pub culvert_area_breaks: Vec<f64>,
    pub culvert_costs: Vec<f64>,
    #[serde(rename = "H_fill_max")]
    pub h_fill_max: f64,
    #[serde(rename = "L_cross_max")]
    pub l_cross_max: f64,
    pub fill_tol: f64,
    #[serde(rename = "N_theta")]
    pub n_theta: usize,
    pub move_max_cells: i64,
    pub kink_max_deg: f64,
    pub smooth_sigma: f64,
    pub s_flat: f64,
}

impl Params {
    /// Parameters from JSON (the keys of `Params.to_dict()`; others are ignored).
    pub fn from_json(text: &str) -> Result<Params, String> {
        serde_json::from_str(text).map_err(|e| format!("parameters: {e}"))
    }

    /// Smoothing sigma in meters (smooth_sigma, or w when that is 0).
    pub fn sigma_m(&self) -> f64 {
        if self.smooth_sigma > 0.0 { self.smooth_sigma } else { self.w }
    }

    /// Sidehill slope at and above which ground is infeasible.
    pub fn s_mask(&self) -> f64 {
        self.mask_margin * self.c_rock
    }

    /// Slope added to the soil-thinning slope per meter of roughness (S0 / rough_r0); 0: off.
    pub fn rough_k(&self) -> f64 {
        if self.rough_r0 > 0.0 { self.s0 / self.rough_r0 } else { 0.0 }
    }

    pub fn culvert_cost(&self, area_ha: f64) -> f64 {
        road_cost::culvert_cost(area_ha, &self.culvert_area_breaks, &self.culvert_costs)
    }
}

fn one() -> f64 {
    1.0
}
