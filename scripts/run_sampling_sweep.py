"""P0-2 sampling-sensitivity sweep (subprocess-isolated, like run_experiment).

For each pano sample size N, runs the full pipeline for a city into its own
output dir and records: coverage, boundary stats, and unit counts under BOTH
the full-graph and covered-only segmentation. Produces outputs/sweep/summary.csv.

  python -m scripts.run_sampling_sweep --city Vienna --sizes 500 2000 5000

Vienna has all street-view images locally, so it can scale; Hong Kong only has
the 500-pano sample copied locally.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def _run(module: str, *cli: str) -> None:
    env = {**os.environ, **THREAD_ENV}
    cmd = [sys.executable, "-u", "-m", module, *cli]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"{module} failed ({r.returncode})")


def _report(out: Path, stage: str) -> dict:
    p = out / "stage_reports" / f"{stage}_report.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run_one(city: str, n: int, K: int, epochs: int) -> dict:
    out = ROOT / "outputs/sweep" / f"{city}_N{n}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {city}  N={n} =====", flush=True)
    _run("scripts.run_stage1", "--city", city, "--max-panos", str(n), "--out", str(out))
    _run("scripts.run_stage2", "--city", city, "--out", str(out))
    _run("scripts.run_stage3", "--out", str(out))
    _run("scripts.run_stage45", "--out", str(out), "--K", str(K), "--epochs", str(epochs))

    _run("scripts.run_stage6", "--out", str(out))
    r6_full = _report(out, "stage6")
    _run("scripts.run_stage6", "--out", str(out), "--covered-only")
    r6_cov = _report(out, "stage6")

    r2, r3, r5 = _report(out, "stage2"), _report(out, "stage3"), _report(out, "stage5")
    return {
        "city": city, "n_panos": n,
        "lcc_ratio": r2.get("lcc_ratio"),
        "n_road_nodes": r2.get("n_road_nodes"),
        "covered_nodes": r3.get("n_nodes_with_panos"),
        "coverage_ratio": r3.get("coverage_ratio"),
        "median_active": r5.get("median_active_basis"),
        "full_tau": r6_full.get("tau"),
        "full_frac_pos_edges": r6_full.get("frac_positive_edges"),
        "full_units": r6_full.get("n_units_after_filter"),
        "covered_tau": r6_cov.get("tau"),
        "covered_frac_pos_edges": r6_cov.get("frac_positive_edges"),
        "covered_units": r6_cov.get("n_units_after_filter"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Vienna", choices=["Vienna", "HongKong"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 5000])
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    rows = [run_one(args.city, n, args.K, args.epochs) for n in args.sizes]
    df = pd.DataFrame(rows)
    sweep_dir = ROOT / "outputs/sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(sweep_dir / "summary.csv", index=False)
    print("\n===== SWEEP SUMMARY =====")
    print(df.to_string(index=False))
    print(f"\nsaved {sweep_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
