#!/usr/bin/env python3
"""Export FULL pano dots (no subsampling) + node<->nearest-pano binding lines.

For each city writes to dashboard/data/:
  panos_<city>_xy.bin   Float32 [lon,lat,...]   ALL download=1 panos
  panos_<city>_ids.txt  pano ids (same order)
  <city>_bind.bin       Float32 [panoLon,panoLat,distM] per road node
                        (same node order as <city>_pos.bin; the map draws a line
                        node->bound pano for nodes within 100 m when both the
                        node-cloud and pano layers are on)

Memory-light: only reads lon/lat/id columns (not the 4096-d embeddings), so it
does NOT redo UMAP/units — run it after build_core_dashboard.
  OMP_NUM_THREADS=1 python -m scripts.dash.build_panos_bind
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
os.makedirs(OUT, exist_ok=True)
CITIES = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}


def main():
    for city, run in CITIES.items():
        # ---- full pano dots ----
        pdf = pd.read_parquet(os.path.join(ROOT, run, "pano_features.parquet"),
                              columns=["pano_id", "lon", "lat"])
        n = len(pdf)
        xy = np.empty(n * 2, np.float32)
        xy[0::2] = pdf["lon"].to_numpy(np.float32)
        xy[1::2] = pdf["lat"].to_numpy(np.float32)
        xy.tofile(os.path.join(OUT, f"panos_{city}_xy.bin"))
        with open(os.path.join(OUT, f"panos_{city}_ids.txt"), "w") as fh:
            fh.write("\n".join(pdf["pano_id"].astype(str).tolist()))

        # ---- node <-> nearest pano bind (node order = pos.bin = activation order) ----
        nodes = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"),
                                columns=["lon", "lat"])
        nlon = nodes["lon"].to_numpy(np.float64); nlat = nodes["lat"].to_numpy(np.float64)
        plon = pdf["lon"].to_numpy(np.float64); plat = pdf["lat"].to_numpy(np.float64)
        lat0 = float(nlat.mean()); mpd = 111320.0; kx = mpd * np.cos(np.radians(lat0))
        ptree = cKDTree(np.column_stack([plon * kx, plat * mpd]))
        dist, idx = ptree.query(np.column_stack([nlon * kx, nlat * mpd]), k=1)
        out = np.empty(len(nlon) * 3, np.float32)
        out[0::3] = plon[idx]; out[1::3] = plat[idx]; out[2::3] = dist
        out.tofile(os.path.join(OUT, f"{city}_bind.bin"))
        far = int((dist > 100).sum())
        print(f"[panos+bind] {city}: {n:,} pano dots | {len(nlon):,} nodes, "
              f"nearest-pano median {np.median(dist):.0f}m, >100m {far/len(nlon)*100:.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
