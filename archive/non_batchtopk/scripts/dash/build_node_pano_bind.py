#!/usr/bin/env python3
"""For each road node, find its nearest street-view pano (from the FULL on-disk
library) and the distance, so the dashboard can draw node<->pano binding lines
and flag nodes with no pano within 100 m.

Output: dashboard/data/<city>_bind.bin  Float32 [panoLon, panoLat, distM] per node
(same node order as <city>_pos.bin).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
RUN = {"Vienna": "outputs/dedup_bld/Vienna", "HongKong": "outputs/dedup_bld/HongKong"}


def main():
    for city, run in RUN.items():
        nodes = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation_joint.parquet"),
                                columns=["lon", "lat"])
        nlon = nodes["lon"].to_numpy(np.float64); nlat = nodes["lat"].to_numpy(np.float64)
        pxy = np.fromfile(os.path.join(OUT, f"panos_{city}_xy.bin"), np.float32).reshape(-1, 2).astype(np.float64)
        lat0 = float(nlat.mean()); m_per_deg = 111320.0; kx = m_per_deg * np.cos(np.radians(lat0))
        # project to local meters for a metric nearest-neighbour query
        ptree = cKDTree(np.column_stack([pxy[:, 0] * kx, pxy[:, 1] * m_per_deg]))
        nq = np.column_stack([nlon * kx, nlat * m_per_deg])
        dist, idx = ptree.query(nq, k=1)
        out = np.empty(len(nlon) * 3, np.float32)
        out[0::3] = pxy[idx, 0]; out[1::3] = pxy[idx, 1]; out[2::3] = dist
        out.tofile(os.path.join(OUT, f"{city}_bind.bin"))
        far = int((dist > 100).sum())
        print(f"[bind] {city}: {len(nlon):,} nodes | nearest-pano median {np.median(dist):.0f}m "
              f"| >100m (no streetview): {far:,} ({far/len(nlon)*100:.1f}%)")


if __name__ == "__main__":
    main()
