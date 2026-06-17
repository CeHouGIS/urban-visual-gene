#!/usr/bin/env python3
"""Precompute within-15m pano pairs sorted by DINOv2 cosine, for the similar-image
browser. Light: reads only the feature npy + coords (no images).
Output: dashboard/data/sim_pairs_<City>.json  {ids:[...], pairs:[[i,j,cos,dist],...]}"""
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"
FEAT = ROOT / "outputs" / "full_feats"


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "Vienna"
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    idf = DATA / (f"panos_{city}_ids.full.txt" if (DATA / f"panos_{city}_ids.full.txt").exists()
                  else f"panos_{city}_ids.txt")
    xyf = DATA / (f"panos_{city}_xy.full.bin" if (DATA / f"panos_{city}_xy.full.bin").exists()
                  else f"panos_{city}_xy.bin")
    ids = idf.read_text().split("\n")
    xy = np.fromfile(xyf, np.float32).reshape(-1, 2).astype(np.float64)
    F = np.load(FEAT / f"{city}_feats.f16.npy").astype(np.float32)
    n = np.linalg.norm(F, axis=1); v = n > 1e-6; F[v] /= n[v, None]
    # per-heading sub-vectors (split 3072 -> 4x768, normalise each) for per-image cosine
    F4 = F.reshape(-1, 4, 768).copy()
    hn = np.linalg.norm(F4, axis=2, keepdims=True); F4 /= (hn + 1e-9)

    lat0 = xy[:, 1].mean(); kx = 111320 * np.cos(np.radians(lat0))
    tree = cKDTree(np.column_stack([xy[:, 0] * kx, xy[:, 1] * 111320]))
    pr = tree.query_pairs(R, output_type="ndarray")
    cos = np.einsum("ij,ij->i", F[pr[:, 0]], F[pr[:, 1]])
    cosh = np.einsum("phk,phk->ph", F4[pr[:, 0]], F4[pr[:, 1]])   # (P,4): per-heading cos
    dm = np.hypot((xy[pr[:, 0], 0] - xy[pr[:, 1], 0]) * kx,
                  (xy[pr[:, 0], 1] - xy[pr[:, 1], 1]) * 111320)
    ok = np.isfinite(cos)
    pr, cos, dm, cosh = pr[ok], cos[ok], dm[ok], cosh[ok]
    order = np.argsort(-cos)                       # high similarity first
    pairs = [[int(pr[k, 0]), int(pr[k, 1]), round(float(cos[k]), 3), round(float(dm[k]), 1)]
             + [round(float(x), 2) for x in cosh[k]]            # c0,c90,c180,c270
             for k in order]
    json.dump({"city": city, "ids": ids, "pairs": pairs},
              open(DATA / f"sim_pairs_{city}.json", "w"), separators=(",", ":"))
    print(f"[sim] {city}: {len(pairs):,} pairs (cos {cos.min():.2f}..{cos.max():.2f}) "
          f"-> sim_pairs_{city}.json ({os.path.getsize(DATA / f'sim_pairs_{city}.json')/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
