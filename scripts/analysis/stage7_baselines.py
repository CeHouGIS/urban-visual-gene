"""Stage 7 — baseline comparison (RQ2): is the sparse visual basis necessary?

Segments the (covered) road graph with several methods under an identical
protocol and scores them in the common DINOv2 space. Methods:
  sae      our learned sparse basis activation
  dino     raw DINOv2 Z_road, segmented directly
  pca      K-dim PCA of Z_road
  kmeans   K visual clusters (one-hot)
  spatial  pure geometry (lat/lon clusters) — non-visual control
  random   random directions — negative control (lower bound)
  shuffled real features, shuffled to wrong nodes — negative control

  python -m scripts.analysis.stage7_baselines --out outputs/sweep/Vienna_N2000

Outputs: <out>/baselines/comparison.csv  and  <out>/baselines/labels_<method>.parquet
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import pandas as pd

from scripts.analysis.baseline_common import (build_features, evaluate, load_city, segment)

METHODS = ["sae", "dino", "pca", "kmeans", "spatial", "random", "shuffled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="city output dir with stage 3/5 results")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--boundary-quantile", type=float, default=0.90)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--full-graph", action="store_true",
                    help="segment full interpolated graph (default: covered-only)")
    args = ap.parse_args()
    out = Path(args.out)
    bdir = out / "baselines"; bdir.mkdir(parents=True, exist_ok=True)

    node_ids, Z, A, lat, lon, edges, n_panos = load_city(
        str(out), covered_only=not args.full_graph)
    print(f"loaded {len(node_ids)} nodes, {len(edges)} edges, "
          f"D={Z.shape[1]}, K={A.shape[1]}", flush=True)

    rows = []
    for m in args.methods:
        F = build_features(m, Z, A, lat, lon, K=args.K)
        labels, n_b, n_e, tau = segment(F, node_ids, edges,
                                        boundary_quantile=args.boundary_quantile)
        metrics = evaluate(labels, node_ids, Z, edges)
        row = {"method": m, "n_boundary": n_b, "n_edges": n_e,
               "tau": round(tau, 5), **metrics}
        rows.append(row)
        pd.DataFrame({"road_node_id": list(labels), "unit": list(labels.values())}) \
            .to_parquet(bdir / f"labels_{m}.parquet", index=False)
        print(f"  {m:9s} units={metrics['n_units']:5d} "
              f"within_var={metrics['within_var']:.4f} "
              f"within_var_z={metrics['within_var_z']:+.2f} "
              f"boundary_contrast={metrics['boundary_contrast']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(bdir / "comparison.csv", index=False)
    print("\n=== comparison (eval in raw DINOv2 space) ===")
    print(df.to_string(index=False))
    print(f"\nsaved {bdir/'comparison.csv'}")
    print("\nReading: lower within_var (more negative within_var_z) and higher "
          "boundary_contrast = visually more coherent units. 'sae' should beat "
          "'random'/'shuffled'/'spatial'; compare vs 'dino'/'pca'/'kmeans' to "
          "judge whether the sparse basis adds value.")


if __name__ == "__main__":
    main()
