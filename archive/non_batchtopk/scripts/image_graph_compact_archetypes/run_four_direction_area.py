#!/usr/bin/env python3
"""Crash-isolated runner for four-direction MAE completion and visualization."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CITY = "UnitedStates/NewYorkCity"
DEFAULT_PANOID = "BFlilbUSNYggO3YcnRY3kg"
DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/four_direction_area"
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--panoid", default=DEFAULT_PANOID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    commands = [
        [
            sys.executable, "-m",
            "scripts.image_graph_compact_archetypes.07_infer_four_direction_mae",
            "--city", args.city, "--panoid", args.panoid,
            "--output", str(args.output),
        ],
        [
            sys.executable, "-m",
            "scripts.image_graph_compact_archetypes.08_make_four_direction_figure",
            "--output", str(args.output),
        ],
    ]
    for command in commands:
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=safe_environment())


if __name__ == "__main__":
    main()
