"""Thread-pool pinning — MUST be imported before numpy / scipy / torch.

The pipeline mixes GNU OpenMP (libgomp, used by numpy/scipy/geopandas) with
Intel OpenMP / MKL (libiomp5, used by PyTorch). When both thread pools are
initialised in one process they fight over resources and trigger random native
segfaults / general-protection-faults (see crash_report_20260613.md).

Pinning every BLAS/OpenMP backend to a single thread removes the contention.
These variables only take effect if set *before* the native libraries load, so
this module sets them at import time and every stage runner imports it first.
"""
from __future__ import annotations

import os

_THREAD_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}

for _k, _v in _THREAD_VARS.items():
    os.environ.setdefault(_k, _v)
