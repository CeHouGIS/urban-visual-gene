#!/usr/bin/env python3
"""Crash-isolated runner for overlap-fused panorama inference and figures."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc/mae/model_best.pt"
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


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=safe_environment())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tokens", type=Path, default=DEFAULT_OUTPUT / "dino_patch_tokens.npy"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    run(
        [
            sys.executable, "-m",
            "scripts.image_graph_compact_archetypes.12_infer_overlap_panorama",
            "--output", str(args.output), "--tokens", str(args.tokens),
            "--checkpoint", str(args.checkpoint), "--batch-size", str(args.batch_size),
        ]
    )
    run(
        [
            sys.executable, "-m",
            "scripts.image_graph_compact_archetypes.13_build_overlap_panorama_graphs",
            "--output", str(args.output),
        ]
    )
    run(
        [
            sys.executable, "-m",
            "scripts.image_graph_compact_archetypes.11_build_stitched_area_results",
            "--data", str(args.output),
        ]
    )


if __name__ == "__main__":
    main()
