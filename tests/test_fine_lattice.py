"""The Rust lattice (road_fine::lattice) against roadsearch.lattice: same actions, bit for bit."""

import numpy as np
import pytest

roadsearch_core = pytest.importorskip("roadsearch_core")

from roadsearch.hairpin import hairpin_slots  # noqa: E402
from roadsearch.lattice import build_bends, lattice_arrays  # noqa: E402
from roadsearch.params import load_params  # noqa: E402

NAMES = ("h0", "h1", "dr", "dc", "length", "rev", "s0", "ns", "sdr", "sdc", "tx", "ty", "by_heading", "kind",
         "slot", "turn")
BEND_NAMES = ("X", "Y", "h1", "L1", "L2", "dth")


def bends_tuple(b):
    return (b.X, b.Y, b.h1, b.L1, b.L2, b.dth, b.R)


def lat_tuple(L):
    return tuple(getattr(L, n) for n in NAMES)


def assert_same(name, py, rs):
    py, rs = np.asarray(py), np.asarray(rs)
    assert py.shape == rs.shape, f"{name}: shape {py.shape} != {rs.shape}"
    if np.issubdtype(py.dtype, np.integer):
        assert rs.dtype == np.int64, f"{name}: dtype {rs.dtype}"
        np.testing.assert_array_equal(rs, py, err_msg=name)
    else:
        # Bit for bit: the Rust follows the Python operation for operation on the same libm.
        diff = float(np.max(np.abs(rs - py))) if py.size else 0.0
        assert diff == 0.0, f"{name}: max difference {diff}"


@pytest.mark.parametrize("n_theta", [8, 16, 32])
@pytest.mark.parametrize("res", [10.0, 15.0, 5.0])
def test_build_bends_match(n_theta, res):
    p = load_params()
    py = bends_tuple(build_bends(n_theta, p.R_min / res))
    rs = roadsearch_core.fine_build_bends(n_theta, p.R_min / res)
    for name, a, b in zip(BEND_NAMES, py, rs):
        assert_same(name, a, b)
    assert rs[6] == py[6]


@pytest.mark.parametrize("n_theta", [16, 32])
@pytest.mark.parametrize("res", [10.0, 15.0])
@pytest.mark.parametrize("move_max_cells", [7, 0])
@pytest.mark.parametrize("macros", [True, False])
def test_lattice_arrays_match(n_theta, res, move_max_cells, macros):
    p = load_params(overrides={"N_theta": n_theta})
    r_min = p.R_min / res
    slots = hairpin_slots(p, res) if macros else None
    cross = p.L_cross_max / res if macros else 0.0
    bends = build_bends(n_theta, r_min) if macros else None
    py = lattice_arrays(n_theta, r_min, slots, cross, bends, move_max_cells, p.kink_max_deg)
    rs = roadsearch_core.fine_lattice_arrays(n_theta, r_min, slots, cross,
                                             None if bends is None else bends_tuple(bends),
                                             move_max_cells, p.kink_max_deg)
    assert len(rs) == len(NAMES)
    for name, a, b in zip(NAMES, lat_tuple(py), rs):
        assert_same(name, a, b)


def test_lattice_arrays_with_rust_bends():
    """The Rust bends feed the Rust lattice to the same arrays as the Python bends."""
    p = load_params()
    res = 10.0
    r_min = p.R_min / res
    slots = hairpin_slots(p, res)
    py = lattice_arrays(p.N_theta, r_min, slots, p.L_cross_max / res, build_bends(p.N_theta, r_min),
                        p.move_max_cells, p.kink_max_deg)
    rs = roadsearch_core.fine_lattice_arrays(p.N_theta, r_min, slots, p.L_cross_max / res,
                                             roadsearch_core.fine_build_bends(p.N_theta, r_min),
                                             p.move_max_cells, p.kink_max_deg)
    for name, a, b in zip(NAMES, lat_tuple(py), rs):
        assert_same(name, a, b)


def test_bad_n_theta():
    with pytest.raises(ValueError):
        roadsearch_core.fine_lattice_arrays(12, 2.0, None, 0.0, None, 0, 10.0)
    with pytest.raises(ValueError):
        roadsearch_core.fine_build_bends(12, 2.0)
