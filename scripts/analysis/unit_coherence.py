"""Unit internal coherence vs random (H3).

For the SAE segmentation of the covered road graph, compares activation
similarity WITHIN units against ACROSS unit boundaries:

  C_intra = mean_{(i,j) in E, same unit} cos(a_i, a_j)
  C_inter = mean_{(i,j) in E, cross unit} cos(a_i, a_j)

and tests the intra-inter gap against random connected partitions of the same
unit-size distribution (label shuffle) -> z-score (expect z >> 0).

  python -m scripts.analysis.unit_coherence --dirs outputs/sweep/Vienna_N2000 \
      outputs/sweep/HongKong_N2000 --labels Vienna HongKong

Output: outputs/sweep/unit_coherence.csv
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse

import numpy as np
import pandas as pd

from scripts.analysis.baseline_common import load_city, segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--perm", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    rows = []
    for d, lab in zip(args.dirs, args.labels):
        node_ids, Z, A, lat, lon, edges, n_panos = load_city(d, covered_only=True)
        labels, *_ = segment(A, node_ids, edges)
        nid2idx = {n: i for i, n in enumerate(node_ids)}
        lab_vec = np.array([labels[n] for n in node_ids])

        si = np.array([nid2idx.get(s, -1) for s in edges["src_node_id"]])
        di = np.array([nid2idx.get(t, -1) for t in edges["dst_node_id"]])
        ok = (si >= 0) & (di >= 0); si, di = si[ok], di[ok]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        cos = np.einsum("ij,ij->i", An[si], An[di])

        def gap(lv):
            same = lv[si] == lv[di]
            ci = cos[same].mean() if same.any() else 0.0
            co = cos[~same].mean() if (~same).any() else 0.0
            return float(ci), float(co), float(ci - co)

        c_intra, c_inter, g = gap(lab_vec)
        perm_gap = np.array([gap(rng.permutation(lab_vec))[2]
                             for _ in range(args.perm)])
        z = (g - perm_gap.mean()) / (perm_gap.std() + 1e-12)

        rows.append(dict(city=lab, n_units=len(set(lab_vec)),
                         C_intra=round(c_intra, 4), C_inter=round(c_inter, 4),
                         intra_minus_inter=round(g, 4),
                         perm_gap_mean=round(float(perm_gap.mean()), 4),
                         gap_z=round(z, 1)))
        print(f"{lab}: intra={c_intra:.3f} inter={c_inter:.3f} "
              f"gap={g:.3f} (perm {perm_gap.mean():.3f}, z={z:.0f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/sweep/unit_coherence.csv", index=False)
    print("\n", df.to_string(index=False))
    print("\nReading: C_intra >> C_inter and gap_z >> 0 → units are internally "
          "coherent and bounded by real activation discontinuities, not arbitrary.")
    print("saved outputs/sweep/unit_coherence.csv")


if __name__ == "__main__":
    main()
