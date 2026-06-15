"""Validate the quality filter: does excluding bad images remove the glare basis?

For each basis, computes the mean magenta colour-cast (rg_mean) of its
top-activated panos. A glare-artifact basis shows a high value; a clean model
should have none. Compares two run dirs.

  python -m scripts.validate_glare --dirs outputs/sweep/Vienna_N2000 \
      outputs/sweep/Vienna_N2000_clean --labels baseline clean --city Vienna
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.image_quality import _load, image_features


def glare_per_basis(d, city, topn=20):
    a = pd.read_parquet(f"{d}/road_basis_activation.parquet")
    panos = pd.read_parquet(f"{d}/road_matched_panos.parquet")
    ctx = pd.read_parquet(f"{d}/road_context_features.parquet")
    cov = set(ctx[ctx.n_panos > 0].road_node_id)
    a = a[a.road_node_id.isin(cov)].reset_index(drop=True)
    ac = sorted([c for c in a.columns if c.startswith("a_")])
    A = a[ac].values
    tree = cKDTree(panos[["lat", "lon"]].values)
    _, nn = tree.query(a[["lat", "lon"]].values, k=1)
    node_pano = panos["pano_id"].values[nn]
    dom = A.argmax(1)
    rg = {}
    for k in range(A.shape[1]):
        idx = np.where(dom == k)[0]
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-A[idx, k])][:topn]
        vals = [image_features(_load(city, pid, 0))["rg_mean"]
                for pid in pd.unique(node_pano[idx]) if _load(city, pid, 0) is not None]
        if vals:
            rg[k] = float(np.mean(vals))
    return rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--city", default="Vienna")
    args = ap.parse_args()

    rows = []
    for d, lab in zip(args.dirs, args.labels):
        g = glare_per_basis(d, args.city)
        top = sorted(g.items(), key=lambda x: -x[1])[:1][0]
        rows.append({"run": lab, "glare_basis": top[0],
                     "max_basis_magenta_rg": round(top[1], 4)})
        print(f"{lab}: most-magenta basis b{top[0]} = {top[1]:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/sweep/glare_validation.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
