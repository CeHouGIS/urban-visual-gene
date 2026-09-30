"""Spatial organization of visual bases on the road network (H2).

Tests whether basis activations are organized ALONG the road network rather than
randomly in geographic space:

  S(A,G) = mean_{(i,j) in E} ||a_i - a_j||_2          (graph smoothness)
  vs a permutation null (shuffle node->activation) -> z-score (expect z << 0)
  graph Moran's I on the 1st PCA component of A      (expect I >> 0)
  S on a Euclidean kNN graph of equal degree         (road should be smoother)

  python -m scripts.analysis.spatial_organization --dirs outputs/sweep/Vienna_N2000 \
      outputs/sweep/HongKong_N2000 --labels Vienna HongKong --perm 200

Output: outputs/sweep/spatial_organization.csv
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.analysis.baseline_common import load_city


def _edge_idx(node_ids, edges):
    nid2idx = {n: i for i, n in enumerate(node_ids)}
    si = np.array([nid2idx.get(s, -1) for s in edges["src_node_id"]])
    di = np.array([nid2idx.get(d, -1) for d in edges["dst_node_id"]])
    ok = (si >= 0) & (di >= 0)
    return si[ok], di[ok]


def smoothness(A, si, di):
    return float(np.linalg.norm(A[si] - A[di], axis=1).mean())


def morans_I(x, si, di):
    """Graph Moran's I of scalar field x over undirected edges (si,di)."""
    xc = x - x.mean()
    num = np.sum(xc[si] * xc[di]) * 2          # both directions
    den = np.sum(xc ** 2)
    W = 2 * len(si)
    return float((len(x) / W) * (num / (den + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--knn", type=int, default=4, help="Euclidean comparison degree")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    rows = []
    for d, lab in zip(args.dirs, args.labels):
        node_ids, Z, A, lat, lon, edges, n_panos = load_city(d, covered_only=True)
        si, di = _edge_idx(node_ids, edges)

        S = smoothness(A, si, di)
        perm = np.array([smoothness(A[rng.permutation(len(A))], si, di)
                         for _ in range(args.perm)])
        z = (S - perm.mean()) / (perm.std() + 1e-12)

        # Euclidean kNN graph of comparable degree (deg ~ 2|E|/N)
        deg = max(args.knn, int(round(2 * len(si) / len(A))))
        tree = cKDTree(np.column_stack([lat, lon]))
        _, nn = tree.query(np.column_stack([lat, lon]), k=deg + 1)
        esi = np.repeat(np.arange(len(A)), deg)
        edi = nn[:, 1:].ravel()
        S_eucl = smoothness(A, esi, edi)

        # Moran's I on 1st PCA component of A
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1, random_state=42).fit_transform(A)[:, 0]
        I_road = morans_I(pc1, si, di)
        I_eucl = morans_I(pc1, esi, edi)

        rows.append(dict(city=lab, n_covered=len(A),
                         S_road=round(S, 4), S_perm_mean=round(perm.mean(), 4),
                         S_z=round(z, 1),
                         S_euclidean=round(S_eucl, 4),
                         road_vs_eucl=round(S / (S_eucl + 1e-9), 3),
                         morans_I_road=round(I_road, 3),
                         morans_I_eucl=round(I_eucl, 3)))
        print(f"{lab}: S_road={S:.3f} vs perm {perm.mean():.3f} (z={z:.0f}); "
              f"S_eucl={S_eucl:.3f}; Moran I road={I_road:.2f} eucl={I_eucl:.2f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/sweep/spatial_organization.csv", index=False)
    print("\n", df.to_string(index=False))
    print("\nReading: S_road << S_perm (z<<0) → activations are smooth ALONG the "
          "road graph, not random. Moran I road > Moran I eucl and S_road < "
          "S_euclidean → the ROAD network explains visual continuity better than "
          "plain geographic proximity.")
    print("saved outputs/sweep/spatial_organization.csv")


if __name__ == "__main__":
    main()
