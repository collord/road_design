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


def use_rust(engine: str | None = None) -> bool:
    engine = engine or ENGINE
    if engine not in ("rust", "numba"):
        raise ValueError(f"unknown engine {engine!r} (numba or rust)")
    return engine == "rust"
