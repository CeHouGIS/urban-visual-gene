#!/usr/bin/env python3
"""Safe subprocess orchestrator for the full rectangular-panorama baseline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.multicity.config import CITIES


DEFAULT_OUTPUT_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "rectangular_panorama_frozen_mae_n30"
)


def safe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })
    return environment


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(command), flush=True)
    with log_path.open("a") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, check=True, env=safe_environment(), stdout=log, stderr=subprocess.STDOUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--city", action="append", choices=CITIES)
    parser.add_argument("--skip-manifests", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if set(affinity).intersection({8, 9}):
        raise RuntimeError("run orchestrator with taskset -c 0-7,10-15")
    cities = args.city or list(CITIES)
    logs = args.output_root / "logs"
    if not args.skip_manifests:
        command = [sys.executable, "-m", "scripts.multicity.build_rectangular_panorama_manifests", "--output-root", str(args.output_root)]
        for city in cities:
            command.extend(["--city", city])
        run(command, logs / "build_manifests.log")
    for city in cities:
        command = [
            sys.executable, "-m", "scripts.multicity.infer_rectangular_panorama_frozen_mae",
            "--output-root", str(args.output_root), "--city", city,
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        run(command, logs / f"infer_{city.replace('/', '__')}.log")
    payload = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cities": cities,
        "limit": args.limit,
        "cpu_affinity": affinity,
    }
    (args.output_root / "run_complete.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
