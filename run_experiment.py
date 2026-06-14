"""End-to-end MRLU pipeline orchestrator.

Each stage runs in its OWN subprocess. This is deliberate: PyTorch (Intel
OpenMP/MKL, libiomp5) and the geo stack (numpy/scipy/geopandas on GNU OpenMP,
libgomp) corrupt each other when initialised in the same process, causing
random native segfaults (see crash_report_20260613.md). Process isolation is
the durable fix — this orchestrator itself imports nothing heavy.

Stage → process mapping:
  1 run_stage1   torch   DINOv2 feature extraction
  2 run_stage2   geo     map-match + road graph
  3 run_stage3   geo     Z_road context features
  4-5 run_stage45 torch  basis training + activation inference
  6 run_stage6   geo     MRLU extraction

Usage:
  python run_experiment.py --city both --max-panos 500 --K 32 --epochs 50
  python run_experiment.py --city Vienna --skip-stage1   # reuse features
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = {
    "Vienna":   "outputs/Austria/Vienna",
    "HongKong": "outputs/China/HongKong",
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def _run(label: str, module: str, *cli: str) -> None:
    """Run one stage as an isolated subprocess; abort the pipeline on failure."""
    env = {**os.environ, **THREAD_ENV}
    cmd = [sys.executable, "-u", "-m", module, *cli]
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=ROOT, env=env)
    if res.returncode != 0:
        raise SystemExit(f"[{label}] failed with exit code {res.returncode}")


def run_city(city: str, max_panos, model, K, batch_size, epochs,
             skip_stage1, skip_stage2):
    out = ROOT / OUT_DIR[city]
    mp = ["--max-panos", str(max_panos)] if max_panos else []

    if skip_stage1 and (out / "pano_features.parquet").exists():
        print(f"\n[{city}] Stage 1 — skipping (features exist)")
    else:
        _run(f"{city} Stage 1 (torch)", "scripts.run_stage1",
             "--city", city, "--model", model, "--batch-size", str(batch_size), *mp)

    if skip_stage2 and (out / "road_graph_edges.parquet").exists():
        print(f"\n[{city}] Stage 2 — skipping (road graph exists)")
    else:
        _run(f"{city} Stage 2 (geo)", "scripts.run_stage2", "--city", city)

    _run(f"{city} Stage 3 (geo)",   "scripts.run_stage3", "--out", str(out))
    _run(f"{city} Stage 4-5 (torch)", "scripts.run_stage45",
         "--out", str(out), "--K", str(K), "--epochs", str(epochs))
    _run(f"{city} Stage 6 (geo)",   "scripts.run_stage6", "--out", str(out))
    print(f"\n[{city}] pipeline complete → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong", "both"])
    ap.add_argument("--max-panos", type=int, default=None)
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--skip-stage1", action="store_true",
                    help="Reuse existing pano_features.parquet")
    ap.add_argument("--skip-stage2", action="store_true",
                    help="Reuse existing road graph")
    args = ap.parse_args()

    cities = ["Vienna", "HongKong"] if args.city == "both" else [args.city]
    for c in cities:
        run_city(c, args.max_panos, args.model, args.K, args.batch_size,
                 args.epochs, args.skip_stage1, args.skip_stage2)
