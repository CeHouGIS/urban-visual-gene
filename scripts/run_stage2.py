"""Isolated Stage 2 runner (geopandas/scipy, NO torch) — map-match + road graph.

  python -m scripts.run_stage2 --city HongKong
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (sets thread limits before numpy/scipy)

import argparse

import pandas as pd

from scripts.cities import load_roads, out_dir
from scripts.io_utils import save_report
from scripts.stage2_map_match_road import map_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["Vienna", "HongKong"])
    ap.add_argument("--max-dist-m", type=float, default=30.0)
    ap.add_argument("--node-spacing-m", type=float, default=25.0)
    args = ap.parse_args()

    out = out_dir(args.city)
    (out / "stage_reports").mkdir(parents=True, exist_ok=True)

    feat_df = pd.read_parquet(out / "pano_features.parquet")
    roads = load_roads(args.city)

    # Clip the road network to the pano bounding box + 500 m buffer so we only
    # build graph nodes around the sampled panos.
    buf = 500.0 / 111_000.0
    lat0, lat1 = feat_df["lat"].min() - buf, feat_df["lat"].max() + buf
    lon0, lon1 = feat_df["lon"].min() - buf, feat_df["lon"].max() + buf
    roads = roads.cx[lon0:lon1, lat0:lat1].copy()
    print(f"  road segments in coverage area: {len(roads)}")

    matched, nodes, edges, r2 = map_match(
        feat_df, roads,
        max_dist_m=args.max_dist_m, node_spacing_m=args.node_spacing_m,
    )
    matched.to_parquet(out / "road_matched_panos.parquet", index=False)
    nodes.to_parquet(out / "road_graph_nodes.parquet", index=False)
    edges.to_parquet(out / "road_graph_edges.parquet", index=False)
    save_report(out / "stage_reports/stage2_report.json", r2)
    print(f"Stage 2 ✓ matched={r2['n_panos_matched']}/{r2['n_panos_total']}, "
          f"nodes={r2['n_road_nodes']}, lcc={r2['lcc_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()
