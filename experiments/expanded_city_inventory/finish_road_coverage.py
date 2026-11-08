"""Finish a detached road-atlas build and publish the complete webpage."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from render_google_coverage import CITY_MAP, OUTPUT as COVERAGE_OUTPUT
from render_google_road_coverage import OUTPUT as ROAD_OUTPUT


SCRIPT_DIR = Path(__file__).resolve().parent


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def complete_report(expected: int) -> bool:
    report = ROAD_OUTPUT / "city_road_coverage.csv"
    if not report.is_file():
        return False
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return len(rows) == expected and all((ROAD_OUTPUT / row["image"]).is_file() for row in rows)


def publish(site_worktree: Path) -> None:
    site_folder = site_worktree / "street-view-coverage"
    if not (site_worktree / ".git").exists() or not site_folder.is_dir():
        raise RuntimeError(f"Missing gh-pages worktree: {site_worktree}")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=site_worktree, text=True).strip()
    if branch != "gh-pages":
        raise RuntimeError(f"Expected gh-pages branch, found {branch}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "street-view-coverage"],
        cwd=site_worktree, text=True)
    if dirty:
        raise RuntimeError("The published coverage page has uncommitted changes")
    subprocess.run(["git", "pull", "--ff-only", "origin", "gh-pages"],
                   cwd=site_worktree, check=True)
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "render_google_coverage.py"),
        "--output", str(COVERAGE_OUTPUT),
        "--road-report", str(ROAD_OUTPUT / "city_road_coverage.csv"),
        "--refresh-site-only",
    ], check=True)
    for name in ("index.html", "city_road_coverage.csv"):
        shutil.copy2(COVERAGE_OUTPUT / name, site_folder / name)
    destination = site_folder / "road-maps"
    destination.mkdir(exist_ok=True)
    for source in (COVERAGE_OUTPUT / "road-maps").glob("*.png"):
        shutil.copy2(source, destination / source.name)
    subprocess.run(["git", "add", "street-view-coverage/index.html",
                    "street-view-coverage/city_road_coverage.csv",
                    "street-view-coverage/road-maps"], cwd=site_worktree, check=True)
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only", "--",
                                      "street-view-coverage"],
                                     cwd=site_worktree, text=True)
    if staged:
        subprocess.run(["git", "commit", "-m", "Complete 300-city segmented road coverage atlas",
                        "--", "street-view-coverage"],
                       cwd=site_worktree, check=True)
        subprocess.run(["git", "push", "origin", "gh-pages"],
                       cwd=site_worktree, check=True)
    print("Complete road atlas published.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-pid", required=True, type=int)
    parser.add_argument("--site-worktree", required=True, type=Path)
    args = parser.parse_args()
    expected = len(set(json.loads(CITY_MAP.read_text()).values()))
    while live(args.runner_pid):
        time.sleep(30)
    for attempt in range(2):
        if complete_report(expected):
            break
        print(f"Road atlas incomplete; retrying missing cities ({attempt + 1}/2).", flush=True)
        subprocess.run([sys.executable, str(SCRIPT_DIR / "render_google_road_coverage.py"),
                        "--workers", "3"], check=False)
    if not complete_report(expected):
        raise RuntimeError("Road atlas still incomplete after retries; local results retained")
    publish(args.site_worktree)


if __name__ == "__main__":
    main()
