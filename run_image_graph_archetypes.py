#!/usr/bin/env python3
"""Crash-conscious subprocess orchestrator for the graph archetype pipeline."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.image_graph_archetypes.utils import (
    FIGURE_ROOT,
    OUTPUT_ROOT,
    assert_safe_affinity,
    save_json,
)


STAGES = Path("scripts/image_graph_archetypes")


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, check=True, env=environment)


def stage_command(script: str, *arguments: str) -> list[str]:
    module = "scripts.image_graph_archetypes." + Path(script).stem
    return [sys.executable, "-m", module, *arguments]


def run_feature_stage(output: Path, figure: Path, limit: int) -> None:
    arguments = ["--output", str(output), "--qa-figure", str(figure), "--qa-samples", "20"]
    if limit:
        arguments.extend(("--limit", str(limit)))
    run(stage_command("01_build_graph_features.py", *arguments))


def run_complete_pipeline(
    output: Path,
    figure_root: Path,
    limit: int,
    sample_per_city: int,
) -> None:
    run_feature_stage(output, figure_root / "Fig_Graph_QA.png", limit)
    run(
        stage_command(
            "02_train_graph_clustering.py",
            "--output", str(output),
            "--sample-per-city", str(sample_per_city),
        )
    )
    run(stage_command("03_assign_all_images.py", "--output", str(output)))
    run(stage_command("04_build_prototypes.py", "--output", str(output)))
    run(
        stage_command(
            "05_make_figures.py",
            "--output", str(output),
            "--figure-root", str(figure_root),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke500", "smoke5000", "full", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()
    assert_safe_affinity()
    started = datetime.now(timezone.utc)
    completed_modes = []
    if args.mode in ("smoke500", "all"):
        smoke_output = args.output / "smoke_500"
        run_feature_stage(
            smoke_output,
            smoke_output / "Fig_Graph_QA.png",
            500,
        )
        completed_modes.append("smoke500")
    if args.mode in ("smoke5000", "all"):
        smoke_output = args.output / "smoke_5000"
        run_complete_pipeline(
            smoke_output,
            smoke_output / "figures",
            limit=5000,
            sample_per_city=2000,
        )
        completed_modes.append("smoke5000")
    if args.mode in ("full", "all"):
        run_complete_pipeline(
            args.output,
            args.figure_root,
            limit=0,
            sample_per_city=2000,
        )
        completed_modes.append("full")
    finished = datetime.now(timezone.utc)
    report = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "completed_modes": completed_modes,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "forbidden_cpus_excluded": True,
        "subprocess_stage_isolation": True,
    }
    save_json(report, args.output / "run_report.json")
    print(report, flush=True)


if __name__ == "__main__":
    main()
