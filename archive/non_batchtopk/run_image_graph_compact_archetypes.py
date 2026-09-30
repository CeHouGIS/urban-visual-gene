#!/usr/bin/env python3
"""Crash-conscious runner for the compact image-graph experiment."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.image_graph_archetypes.utils import assert_safe_affinity, save_json
from scripts.image_graph_compact_archetypes.utils import FIGURE_ROOT, OUTPUT_ROOT, SOURCE_ROOT


PACKAGE = "scripts.image_graph_compact_archetypes"


def run(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", f"{PACKAGE}.{module}", *arguments]
    print("\n$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, check=True, env=environment)


def feature_stage(source: Path, output: Path, figure: Path, limit: int, sample_per_city: int) -> None:
    arguments = [
        "--source", str(source), "--output", str(output),
        "--qa-figure", str(figure), "--sample-per-city", str(sample_per_city),
    ]
    if limit:
        arguments.extend(("--limit", str(limit)))
    run("01_build_compact_features", *arguments)


def complete_pipeline(
    source: Path, output: Path, figure_root: Path, limit: int, sample_per_city: int,
) -> None:
    feature_stage(
        source, output, figure_root / "Fig_Compact_Graph_QA.png",
        limit, sample_per_city,
    )
    run("02_train_assign", "--output", str(output))
    run("03_build_prototypes", "--source", str(source), "--output", str(output))
    run("05_evaluate_clusters", "--output", str(output))
    run("04_make_figures", "--output", str(output), "--figure-root", str(figure_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke500", "smoke5000", "full", "all"), default="all")
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()
    assert_safe_affinity()
    started = datetime.now(timezone.utc)
    completed: list[str] = []
    if args.mode in ("smoke500", "all"):
        output = args.output / "smoke_500"
        feature_stage(args.source, output, output / "Fig_Compact_Graph_QA.png", 500, 500)
        completed.append("smoke500")
    if args.mode in ("smoke5000", "all"):
        output = args.output / "smoke_5000"
        complete_pipeline(args.source, output, output / "figures", 5000, 2000)
        completed.append("smoke5000")
    if args.mode in ("full", "all"):
        complete_pipeline(args.source, args.output, args.figure_root, 0, 2000)
        completed.append("full")
    finished = datetime.now(timezone.utc)
    report = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "completed_modes": completed,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "forbidden_cpus_excluded": True,
        "subprocess_stage_isolation": True,
    }
    if args.mode == "smoke500":
        report_path = args.output / "smoke_500" / "run_report.json"
    elif args.mode == "smoke5000":
        report_path = args.output / "smoke_5000" / "run_report.json"
    else:
        report_path = args.output / "run_report.json"
    save_json(report, report_path)
    print(report, flush=True)


if __name__ == "__main__":
    main()
