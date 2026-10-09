"""Parameter schema, loading from params.toml, and CLI overrides."""

from __future__ import annotations

import argparse
import dataclasses
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.toml"


@dataclass(frozen=True)
class Params:
    """All run parameters. Defaults live in params.toml, not here."""

    w: float
    g_max: float
    R_min: float
    R_hp: float
    g_hp: float
    w_hp: float
    hp_leg_max: float
    hp_gap_min: float
    C_wall: float
    d0: float
    S0: float
    rough_r0: float
    c_soil: float
    c_rock: float
    f_soil: float
    f_rock: float
    k_soil: float
    k_rock: float
    b_min: float
    mask_margin: float
    catch_max: float
    C_base: float
    C_soil: float
    C_rock: float
    C_borrow: float
    C_haul_down: float
    C_haul_up: float
    haul_max: float
    C_travel: float
    C_vertical: float
    C_turn: float
    A_min: float
    culvert_area_breaks: tuple[float, ...]
    culvert_costs: tuple[float, ...]
    H_fill_max: float
    L_cross_max: float
    fill_tol: float
    N_theta: int
    move_max_cells: int
    kink_max_deg: float
    K_alt: int
    eps_alt: float
    D_alt: float
    alt_window: float
    alt_local_tol: float
    alt_max_tries: int
    alt_method: str
    alt_plateau: float
    smooth_sigma: float
    excl_buffer: float
    s_flat: float
    z_units: str
    max_states: float
    max_memory_gb: float
    clip_slack: float
    clip_buffer: float
    clip_coarse_margin: float
    tin_res: float
    tin_margin: float
    tin_max_cells: float
    tin_tolerance: float
    tin_k: float
    tin_edge_max: float
    tin_block: float
    tin_channel_ha: float
    tin_alternatives: float
    tin_penalty: float
    tin_stretch: float
    tin_max_shared: float
    tin_separation: float
    tin_buffer: float
    tin_hairpin_cost: float
    cost_samples: float
    cost_sigma: float
    cost_sigma_edge: float
    tin_node_samples: float
    tin_node_density: float

    def __post_init__(self) -> None:
        if len(self.culvert_costs) != len(self.culvert_area_breaks) + 1:
            raise ValueError("culvert_costs needs exactly one more entry than culvert_area_breaks")
        if list(self.culvert_area_breaks) != sorted(self.culvert_area_breaks):
            raise ValueError("culvert_area_breaks must be ascending")
        if not 0.0 < self.mask_margin < 1.0:
            raise ValueError("mask_margin must be in (0, 1)")
        if self.catch_max <= max(self.w, self.w_hp):
            raise ValueError("catch_max must exceed the road width")
        if not self.c_rock > self.c_soil > 0.0:
            raise ValueError("expected c_rock > c_soil > 0")
        if self.z_units not in Z_UNIT_SCALE:
            raise ValueError(f"z_units must be one of {sorted(Z_UNIT_SCALE)}")
        if self.N_theta not in (8, 16, 32):
            raise ValueError("N_theta must be 8, 16 or 32")
        if self.alt_method not in ("plateau", "via"):
            raise ValueError("alt_method must be 'plateau' or 'via'")

    @property
    def sigma_m(self) -> float:
        """Smoothing sigma in meters (smooth_sigma, or w when that is 0)."""
        return self.smooth_sigma if self.smooth_sigma > 0 else self.w

    @property
    def s_mask(self) -> float:
        """Sidehill slope at and above which ground is infeasible."""
        return self.mask_margin * self.c_rock

    def culvert_cost(self, area_ha: float) -> float:
        for brk, cost in zip(self.culvert_area_breaks, self.culvert_costs):
            if area_ha < brk:
                return cost
        return self.culvert_costs[-1]

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


Z_UNIT_SCALE = {"m": 1.0, "ft": 0.3048, "ftUS": 1200.0 / 3937.0}

_LIST_FIELDS = {f.name for f in fields(Params) if str(f.type).startswith("tuple")}


def _coerce(name: str, value: Any) -> Any:
    ftype = {f.name: str(f.type) for f in fields(Params)}[name]
    if name in _LIST_FIELDS:
        if isinstance(value, str):
            value = [v for v in value.split(",") if v.strip()]
        return tuple(float(v) for v in value)
    if ftype == "int":
        fv = float(value)
        if fv != int(fv):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        return int(fv)
    if ftype == "float":
        return float(value)
    return str(value)


# Parameters that were renamed: old name -> new name (runs record the names of their day).
RENAMED = {"tin_mc_samples": "cost_samples", "tin_mc_sigma": "cost_sigma", "tin_mc_sigma_edge": "cost_sigma_edge",
           "tin_cost_samples": "cost_samples", "tin_cost_sigma": "cost_sigma",
           "tin_cost_sigma_edge": "cost_sigma_edge"}


def _renamed(d: dict) -> dict:
    return {RENAMED.get(k, k): v for k, v in d.items()}


def load_params(
    params_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    defaults_file: str | Path = DEFAULT_PARAMS_PATH,
) -> Params:
    """Defaults from params.toml, then a partial TOML file, then explicit overrides."""
    with open(defaults_file, "rb") as fh:
        raw = tomllib.load(fh)
    known = {f.name for f in fields(Params)}
    raw = _renamed(raw)
    missing = known - raw.keys()
    if missing:
        raise ValueError(f"{defaults_file} is missing parameters: {sorted(missing)}")

    layers = []
    if params_file is not None:
        with open(params_file, "rb") as fh:
            layers.append((str(params_file), tomllib.load(fh)))
    if overrides:
        layers.append(("overrides", overrides))
    for source, layer in [(str(defaults_file), raw)] + layers:
        layer = _renamed(layer)
        unknown = layer.keys() - known
        if unknown:
            raise ValueError(f"unknown parameters in {source}: {sorted(unknown)}")
        raw = {**raw, **{k: v for k, v in layer.items() if v is not None}}
    return Params(**{k: _coerce(k, raw[k]) for k in sorted(known)})


def add_param_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --params FILE and one --<name> override flag per parameter."""
    parser.add_argument("--params", type=Path, default=None,
                        help="partial TOML file overriding params.toml")
    grp = parser.add_argument_group("parameter overrides (defaults in params.toml)")
    for f in fields(Params):
        hint = "comma-separated list" if f.name in _LIST_FIELDS else str(f.type)
        grp.add_argument(f"--{f.name}", dest=f"param_{f.name}", default=None,
                         metavar="V", help=hint)


def params_from_args(args: argparse.Namespace) -> Params:
    overrides = {f.name: getattr(args, f"param_{f.name}") for f in fields(Params)}
    return load_params(args.params, {k: v for k, v in overrides.items() if v is not None})
