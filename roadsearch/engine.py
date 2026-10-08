"""Which kernels run: the Rust core (the roadsearch_core package built from rust/) when it is
installed, else the numba kernels in this package. Both give the same results; set
ROADSEARCH_ENGINE=numba (or rust) to choose."""

import os


def _default() -> str:
    try:
        import roadsearch_core  # noqa: F401
        return "rust"
    except ImportError:
        return "numba"


ENGINE = os.environ.get("ROADSEARCH_ENGINE") or _default()


# Recorded in a run's inputs (so in its id) when the Rust core preprocessed its terrain: its flow
# accumulation fills depressions and resolves flats differently from pysheds, so channels, and
# with them culverts and fill crossings, can differ slightly. Bump it when Rust preprocessing
# changes results.
TERRAIN_ENGINE_TAG = "rust-1"


def terrain_tag(engine: str | None = None) -> str | None:
    """The terrain engine to record in a run's inputs: None for the Python path (ids as before)."""
    return TERRAIN_ENGINE_TAG if use_rust(engine) else None


def engine_of_tag(tag: str | None) -> str:
    """The engine that made a run, from the tag in its inputs."""
    return "rust" if tag else "numba"


def use_rust(engine: str | None = None) -> bool:
    engine = engine or ENGINE
    if engine not in ("rust", "numba"):
        raise ValueError(f"unknown engine {engine!r} (numba or rust)")
    return engine == "rust"
