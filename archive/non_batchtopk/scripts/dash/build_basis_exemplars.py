#!/usr/bin/env python3
"""Exemplar street-view panos per visual basis (for the local image-browser page).

For each city and each basis (under both the K=32 and K=128 SAE), pick the road
nodes with the strongest activation of that basis and map each to its nearest
pano -> a gallery of "what this visual type looks like".

Writes dashboard/data/basis_exemplars.json:
  {city: {"32": {basisIdx: [{id,a,lon,lat}, ...]}, "128": {...}}}
  OMP_NUM_THREADS=1 python -m scripts.dash.build_basis_exemplars [--per-basis 24]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
CITIES = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}


def exemplars_for(A, lon, lat, ptree, pxy, pid, per_basis, scan):
    """For each basis column, top-`scan` nodes by activation -> distinct nearest panos."""
    K = A.shape[1]
    out = {}
    for k in range(K):
        col = A[:, k]
        if col.max() <= 0:
            continue
        top = np.argpartition(col, -min(scan, len(col)))[-min(scan, len(col)):]
        top = top[np.argsort(col[top])[::-1]]            # strongest first
        seen, items = set(), []
        for ni in top:
            d, pi = ptree.query([lon[ni], lat[ni]])      # nearest pano (deg space ok, local)
            pi = int(pi)
            if pid[pi] in seen:
                continue
            seen.add(pid[pi])
            items.append({"id": pid[pi], "a": round(float(col[ni]), 3),
                          "lon": round(float(pxy[pi, 0]), 5), "lat": round(float(pxy[pi, 1]), 5)})
            if len(items) >= per_basis:
                break
        if items:
            out[str(k)] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-basis", type=int, default=24)
    ap.add_argument("--scan", type=int, default=400, help="top-N nodes scanned per basis")
    args = ap.parse_args()

    data = {}
    for c, run in CITIES.items():
        a32 = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"),
                              columns=["lat", "lon"] + [f"a_{i:03d}" for i in range(32)])
        lon = a32["lon"].to_numpy(np.float64); lat = a32["lat"].to_numpy(np.float64)
        A32 = a32[[f"a_{i:03d}" for i in range(32)]].to_numpy(np.float32)
        A128 = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_k128.parquet"),
                               columns=[f"a_{i:03d}" for i in range(128)]).to_numpy(np.float32)
        pxy = np.fromfile(os.path.join(OUT, f"panos_{c}_xy.bin"), np.float32).reshape(-1, 2)
        pid = open(os.path.join(OUT, f"panos_{c}_ids.txt")).read().split("\n")
        ptree = cKDTree(pxy)
        data[c] = {
            "32":  exemplars_for(A32,  lon, lat, ptree, pxy, pid, args.per_basis, args.scan),
            "128": exemplars_for(A128, lon, lat, ptree, pxy, pid, args.per_basis, args.scan),
        }
        print(f"[exem] {c}: 32-bases={len(data[c]['32'])} 128-bases={len(data[c]['128'])}", flush=True)

    json.dump(data, open(os.path.join(OUT, "basis_exemplars.json"), "w"), separators=(",", ":"))
    sz = os.path.getsize(os.path.join(OUT, "basis_exemplars.json"))
    print(f"[exem] done -> basis_exemplars.json ({sz/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
