"""Run selection, DINOv3 extraction, and shared Top-K SAE in isolated stages."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(module: str, extra: list[str]) -> None:
    command = [sys.executable, "-m", module, *extra]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-stage", choices=("select", "extract", "train"), default="select")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    stages = ("select", "extract", "train")
    start = stages.index(args.start_stage)
    if start <= 0:
        run("scripts.multicity.select_panos", [])
    if start <= 1:
        run(
            "scripts.multicity.extract_dinov3_features",
            ["--batch-size", str(args.batch_size), "--workers", str(args.workers)],
        )
    if start <= 2:
        run("scripts.multicity.train_topk_sae", ["--epochs", str(args.epochs)])


if __name__ == "__main__":
    main()
