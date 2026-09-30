#!/usr/bin/env python3
"""Crash-isolated runner for the ten-area four-direction experiment."""
from __future__ import annotations

import os
import subprocess
import sys


def safe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return environment


def main() -> None:
    modules = (
        "scripts.image_graph_compact_archetypes.09_infer_multi_area_four_directions",
        "scripts.image_graph_compact_archetypes.10_make_multi_area_four_direction_figures",
    )
    for module in modules:
        command = [sys.executable, "-m", module]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=safe_environment())


if __name__ == "__main__":
    main()
