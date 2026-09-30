#!/usr/bin/env python3
"""Two-stage pano dedup, applied in sequence:
  Stage A (NEW): within 5 m AND max per-direction cosine > 0.6  -> keep NEWEST (tie: medoid)
  Stage B (OLD): within 15 m AND pano cosine >= tau(0.90)        -> keep MEDOID
Writes the final kept set to the dashboard pano assets (full set kept as *.full.*).

    /opt/conda/bin/python3 scripts/dedup_chain.py <City> [tau_old=0.90]
"""
import os, sys, shutil, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = ROOT / "dashboard" / "data"
FEAT = ROOT / "outputs" / "full_feats"


def load_dates(city, ids):
    d = {}
    if city == "Vienna":
        df = pd.read_parquet(ROOT / "data/SVIs/GSV/metadata/Austria/Vienna/panoids.parquet",
                             columns=["panoid", "year", "month"])
        d = {p: (int(y) * 12 + int(m)) for p, y, m in zip(df.panoid, df.year.fillna(0), df.month.fillna(0))}
    else:
        con = sqlite3.connect(ROOT / "data/SVIs/GSV/metadata/China/HongKong/meta/China_HongKong.db")
        for p, y, m in con.execute("SELECT panoid,year,month FROM gsv"):
            d[p] = (int(y or 0) * 12 + int(m or 0))
    return np.array([d.get(i, 0) for i in ids], np.int32)


def medoid(idxs, F):
    g = np.asarray(idxs); mu = F[g].mean(0); mu /= (np.linalg.norm(mu) + 1e-9)
    return int(g[np.argmax(F[g] @ mu)])


def main():
    city = sys.argv[1]
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    base_ids = DATA / (f"panos_{city}_ids.full.txt" if (DATA / f"panos_{city}_ids.full.txt").exists()
                       else f"panos_{city}_ids.txt")
    base_xy = DATA / (f"panos_{city}_xy.full.bin" if (DATA / f"panos_{city}_xy.full.bin").exists()
                      else f"panos_{city}_xy.bin")
    ids = base_ids.read_text().split("\n")
    xy = np.fromfile(base_xy, np.float32).reshape(-1, 2).astype(np.float64)
    F = np.load(FEAT / f"{city}_feats.f16.npy").astype(np.float32)
    N = len(ids); assert len(xy) == N == len(F)
    nrm = np.linalg.norm(F, axis=1); valid = nrm > 1e-6; F[valid] /= nrm[valid, None]
    F4 = F.reshape(N, 4, 768).copy(); hn = np.linalg.norm(F4, axis=2, keepdims=True); F4 /= (hn + 1e-9)
    dates = load_dates(city, ids)
    lat0 = xy[:, 1].mean(); kx = 111320 * np.cos(np.radians(lat0))
    M = np.column_stack([xy[:, 0] * kx, xy[:, 1] * 111320])      # metres

    # ---------- Stage A: NEW rule (5 m, any-direction cos > 0.6, keep newest) ----------
    tree = cKDTree(M); prA = tree.query_pairs(5.0, output_type="ndarray")
    if len(prA):
        cmax = np.einsum("phk,phk->ph", F4[prA[:, 0]], F4[prA[:, 1]]).max(1)
        eA = prA[(cmax > 0.6)]
    else:
        eA = np.empty((0, 2), int)
    parent = list(range(N))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in eA:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb: parent[ra] = rb
    comp = {}
    for i in range(N): comp.setdefault(find(i), []).append(i)
    keepA = []
    for g in comp.values():
        if len(g) == 1: keepA.append(g[0]); continue
        dmax = max(dates[j] for j in g)
        newest = [j for j in g if dates[j] == dmax]
        keepA.append(newest[0] if len(newest) == 1 else medoid(newest, F))
    keepA = np.array(sorted(keepA))
    print(f"[chain] {city} StageA(new 5m,>0.6,newest): {N:,} -> {len(keepA):,} "
          f"({(1-len(keepA)/N)*100:.1f}% dropped)")

    # ---------- Stage B: OLD rule (15 m, pano cos >= tau, keep medoid) on Stage-A survivors ----
    sub = keepA; subM = M[sub]; subtree = cKDTree(subM)
    used = np.zeros(len(sub), bool); keepB = []
    for li in range(len(sub)):
        if used[li]: continue
        gi = sub[li]
        if not valid[gi]: used[li] = True; keepB.append(gi); continue
        nb = subtree.query_ball_point(subM[li], 15.0)
        grp = [lj for lj in nb if not used[lj] and valid[sub[lj]] and F[gi] @ F[sub[lj]] >= tau]
        if li not in grp: grp.append(li)
        for lj in grp: used[lj] = True
        keepB.append(medoid([sub[lj] for lj in grp], F))
    keepB = np.array(sorted(set(keepB)))
    print(f"[chain] {city} StageB(old 15m,>={tau},medoid): {len(keepA):,} -> {len(keepB):,} "
          f"({(1-len(keepB)/len(keepA))*100:.1f}% dropped) | final {len(keepB):,}/{N:,} "
          f"({(1-len(keepB)/N)*100:.1f}% total)")

    # ---------- write final assets (back up full once) ----------
    if not (DATA / f"panos_{city}_xy.full.bin").exists():
        shutil.copy(DATA / f"panos_{city}_xy.bin", DATA / f"panos_{city}_xy.full.bin")
        shutil.copy(DATA / f"panos_{city}_ids.txt", DATA / f"panos_{city}_ids.full.txt")
    xy[keepB].astype(np.float32).tofile(DATA / f"panos_{city}_xy.bin")
    open(DATA / f"panos_{city}_ids.txt", "w").write("\n".join(ids[k] for k in keepB))
    print(f"[chain] {city}: wrote {len(keepB):,} kept panos")


if __name__ == "__main__":
    main()
