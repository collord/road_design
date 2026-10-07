"""Least-construction-cost road alignment search over a DEM."""

import os

# Nothing here does heavy linear algebra, but OpenBLAS reserves scratch buffers for every
# thread when numpy loads: about 0.4 GB on a 14-core machine, and pysheds' import adds more.
# One thread keeps importing the package to about 0.2 GB instead of 1 GB. This only takes
# effect if numpy has not been imported yet; set the variables yourself to override.
# (OMP_NUM_THREADS is left alone: numba's parallel loops may use OpenMP.)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
