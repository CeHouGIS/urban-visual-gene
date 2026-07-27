"""Isolated Stage 3 runner (NO torch) — build Z_road from saved Stage 2.

torch 2.1 corrupts numpy/scipy fancy-indexing once imported, so Stage 3
(which uses scipy.spatial.cKDTree + numpy fancy indexing) must run in a
process that never imports torch.

  python -m scripts.pipeline.run_stage3 --out outputs/China/HongKong
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (sets thread limits before numpy/scipy)

import argparse
from pathlib import Path

import pandas as pd

from scripts.core.io_utils import save_report
from scripts.pipeline.stage3_build_road_context_features import build_road_context_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--search-radius-m", type=float, default=100.0)
    ap.add_argument("--kernel-sigma-m", type=float, default=30.0)
    ap.add_argument("--context-radius-m", type=float, default=100.0)
    args = ap.parse_args()
    out = Path(args.out)

    matched = pd.read_parquet(out / "road_matched_panos.parquet")
    nodes = pd.read_parquet(out / "road_graph_nodes.parquet")
    edges = pd.read_parquet(out / "road_graph_edges.parquet")

    ctx_df, r3 = build_road_context_features(
        matched, nodes, edges,
        search_radius_m=args.search_radius_m,
        kernel_sigma_m=args.kernel_sigma_m,
        context_radius_m=args.context_radius_m,
    )
    # row_group_size caps each row group's list-column at <2^31 elements so the
    # 3072-float road_context_embedding stays readable for very large road
    # networks (e.g. CapeTown ~774k nodes x 3072 > int32 list-offset limit).
    ctx_df.to_parquet(out / "road_context_features.parquet", index=False,
                      row_group_size=200000)
    save_report(out / "stage_reports/stage3_report.json", r3)
    print(f"Stage 3 ✓ {r3['n_road_nodes']} nodes, "
          f"coverage={r3['coverage_ratio']*100:.1f}%, D={r3['embedding_dim']}")


if __name__ == "__main__":
    main()
