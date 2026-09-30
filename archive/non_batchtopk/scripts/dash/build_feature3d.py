#!/usr/bin/env python3
"""3D feature-space assets: a global random subsample (~80k nodes) embedded with
a JOINT 3D UMAP, for the dashboard's 3D feature view.

Per city writes to dashboard/data/ (parallel arrays, same sampled point order):
  <city>_fxyz.bin  Float32 [x,y,z,...]   joint 3D UMAP coords (centred, ~[-500,500])
  <city>_fgeo.bin  Float32 [lon,lat,...] geo position (for the linked map)
  <city>_frgb.bin  Uint8   [r,g,b,...]   joint-PCA colour (same as map)
  <city>_fattr.bin Uint8   [dom,act,entq,recq,...]  same 4-byte layout as *_attr.bin
  feature3d_meta.json   per-city sampled counts + total

Subsample is global-random (proportional to each city's node count) so it stays
light to render; 3D UMAP on ~80k x 32 activations is fast.
  OMP_NUM_THREADS=1 python -m scripts.dash.build_feature3d [--n 80000]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
CITIES = {
    "HongKong":  "outputs/China/HongKong",
    "Singapore": "outputs/Singapore/Singapore",
    "Amsterdam": "outputs/Netherlands/Amsterdam",
    "CapeTown":  "outputs/SouthAfrica/CapeTown",
}
K = 32
ACT = [f"a_{i:03d}" for i in range(K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80000, help="total sampled points")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    # ---- load activations + per-city node counts ----
    acts, counts = {}, {}
    for c, run in CITIES.items():
        A = pd.read_parquet(os.path.join(ROOT, run, "road_basis_activation.parquet"),
                            columns=ACT).to_numpy(np.float32)
        acts[c] = A; counts[c] = len(A)
        print(f"[f3d] {c}: {len(A):,} nodes", flush=True)
    total = sum(counts.values())

    # ---- per-city proportional random sample indices ----
    sample_idx = {}
    for c in CITIES:
        n_c = int(round(args.n * counts[c] / total))
        n_c = min(n_c, counts[c])
        idx = np.sort(rng.choice(counts[c], n_c, replace=False))
        sample_idx[c] = idx
        print(f"[f3d] {c}: sampled {n_c:,}", flush=True)

    # ---- pool sampled activations -> joint 3D UMAP ----
    import umap
    stack, order = [], []
    for c in CITIES:
        stack.append(acts[c][sample_idx[c]]); order.append((c, len(sample_idx[c])))
    allA = np.vstack(stack)
    mu = allA.mean(0); Xc = allA - mu
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    w, V = np.linalg.eigh(cov)
    init3 = (Xc @ V[:, w.argsort()[::-1][:3]]).astype(np.float32)
    init3 = (init3 - init3.mean(0)) / (init3.std(0) + 1e-9) * 10.0
    print(f"[f3d] fitting 3D UMAP on {len(allA):,} pts…", flush=True)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine",
                        n_components=3, init=init3, verbose=True)
    emb = reducer.fit_transform(allA).astype(np.float32)
    # centre + scale to ~[-500,500] per axis for the OrbitView
    emb = emb - emb.mean(0)
    emb = emb / (np.abs(emb).max() + 1e-9) * 500.0

    # ---- split back per city + write f* files ----
    meta = {"total": int(len(allA)), "cities": {}}
    off = 0
    for c, n in order:
        e = emb[off:off + n]; off += n
        idx = sample_idx[c]
        e.astype(np.float32).tofile(os.path.join(OUT, f"{c}_fxyz.bin"))
        # geo / rgb / attr sampled from the already-built full map assets
        pos = np.fromfile(os.path.join(OUT, f"{c}_pos.bin"), np.float32).reshape(-1, 2)
        rgb = np.fromfile(os.path.join(OUT, f"{c}_rgb.bin"), np.uint8).reshape(-1, 3)
        attr = np.fromfile(os.path.join(OUT, f"{c}_attr.bin"), np.uint8).reshape(-1, 4)
        pos[idx].astype(np.float32).tofile(os.path.join(OUT, f"{c}_fgeo.bin"))
        rgb[idx].tofile(os.path.join(OUT, f"{c}_frgb.bin"))
        attr[idx].tofile(os.path.join(OUT, f"{c}_fattr.bin"))
        meta["cities"][c] = int(n)
        print(f"[f3d] {c}: wrote {n:,} feature points", flush=True)

    json.dump(meta, open(os.path.join(OUT, "feature3d_meta.json"), "w"))
    print(f"[f3d] done — total {len(allA):,} 3D feature points", flush=True)


if __name__ == "__main__":
    main()
