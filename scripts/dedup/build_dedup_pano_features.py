#!/usr/bin/env python3
"""Build pano_features.parquet for the dedup re-run, from the kept (deduped) pano set
+ precomputed DINOv2 features (no GPU). Optional spatial grid-thinning to ~max panos.

    /opt/conda/bin/python3 scripts/build_dedup_pano_features.py <City> <out_dir> [max]
Output: <out_dir>/pano_features.parquet  (pano_id, city, lat, lon, heading, pano_embedding)
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"
FEAT = ROOT / "outputs" / "full_feats"


def grid_thin(lon, lat, target, seed=0):
    n = len(lon)
    if n <= target:
        return np.arange(n)
    g = max(2, int(np.sqrt(target)))
    slon, slat = lon.max() - lon.min() + 1e-9, lat.max() - lat.min() + 1e-9
    while True:
        cx = np.clip(((lon - lon.min()) / slon * g).astype(int), 0, g - 1)
        cy = np.clip(((lat - lat.min()) / slat * g).astype(int), 0, g - 1)
        _, idx = np.unique(cx * g + cy, return_index=True)   # one pano per occupied cell
        if len(idx) >= target or g > 3000:
            rng = np.random.default_rng(seed)
            return np.sort(rng.choice(idx, size=min(target, len(idx)), replace=False))
        g = int(g * 1.3) + 1


def main():
    city, out = sys.argv[1], Path(sys.argv[2])
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    kept_ids = (DATA / f"panos_{city}_ids.txt").read_text().split("\n")
    kept_xy = np.fromfile(DATA / f"panos_{city}_xy.bin", np.float32).reshape(-1, 2)
    full_ids = (DATA / f"panos_{city}_ids.full.txt").read_text().split("\n")
    F = np.load(FEAT / f"{city}_feats.f16.npy")                 # (Nfull, 3072) float16
    pos = {p: i for i, p in enumerate(full_ids)}
    rows = np.array([pos[p] for p in kept_ids])                 # kept -> full index

    sel = grid_thin(kept_xy[:, 0].astype(float), kept_xy[:, 1].astype(float), mx) if mx else np.arange(len(kept_ids))
    emb = F[rows[sel]].astype(np.float32)
    df = pd.DataFrame({
        "pano_id": [kept_ids[i] for i in sel],
        "city": city,
        "lat": kept_xy[sel, 1].astype(np.float64),
        "lon": kept_xy[sel, 0].astype(np.float64),
        "heading": 0.0,
        "pano_embedding": list(emb),
    })
    df.to_parquet(out / "pano_features.parquet", index=False)
    print(f"[dedup-feat] {city}: kept {len(kept_ids):,} -> using {len(sel):,} panos "
          f"-> {out}/pano_features.parquet")


if __name__ == "__main__":
    main()
