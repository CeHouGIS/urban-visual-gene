#!/usr/bin/env python3
"""Building-aware Stage 3: aggregate panos onto road nodes but drop pano->node
contributions whose sight-line crosses a building footprint.

    python -m scripts.run_stage3_bld --out outputs/dedup_bld/Vienna --buildings data/buildings/Vienna/buildings.parquet
"""
import scripts._env  # noqa: F401
import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
from scripts.pipeline.stage3_build_road_context_features import build_road_context_features
from scripts.core.io_utils import save_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--search-radius-m", type=float, default=100.0)
    ap.add_argument("--kernel-sigma-m", type=float, default=30.0)
    args = ap.parse_args()
    out = Path(args.out)

    matched = pd.read_parquet(out / "road_matched_panos.parquet")
    nodes = pd.read_parquet(out / "road_graph_nodes.parquet")
    edges = pd.read_parquet(out / "road_graph_edges.parquet")
    geoms = gpd.read_parquet(args.buildings).geometry.values
    print(f"  panos={len(matched)} nodes={len(nodes)} buildings={len(geoms)} (building-aware)", flush=True)

    ctx_df, r3 = build_road_context_features(
        matched, nodes, edges,
        search_radius_m=args.search_radius_m, kernel_sigma_m=args.kernel_sigma_m,
        buildings_geoms=geoms,
    )
    ctx_df.to_parquet(out / "road_context_features.parquet", index=False)
    save_report(out / "stage_reports/stage3_report.json", r3)
    print(f"Stage 3(bld) ✓ {r3['n_road_nodes']} nodes, coverage={r3['coverage_ratio']*100:.1f}%, "
          f"occluded pairs={r3['n_occluded_pano_node_pairs']:,}", flush=True)


if __name__ == "__main__":
    main()
