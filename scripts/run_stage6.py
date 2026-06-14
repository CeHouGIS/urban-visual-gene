"""Isolated Stage 6 runner — extract MRLU from saved Stage 3/5 outputs.

Stage 6 (networkx + KDTree on the full road graph) is run in its own process
because chaining it after the other heavy native stages intermittently
segfaults. Reads the saved parquets, attaches n_panos from Z_road (so the
min_panos filter reflects real coverage), writes the unit artefacts.

  python -m scripts.run_stage6 --out outputs/Austria/Vienna
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (sets thread limits before numpy/scipy)

import argparse
from pathlib import Path

import pandas as pd

from scripts.io_utils import save_report
from scripts.stage6_extract_road_units import extract_road_units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="city output dir")
    ap.add_argument("--boundary-quantile", type=float, default=0.90)
    ap.add_argument("--min-road-nodes", type=int, default=3)
    ap.add_argument("--min-road-length-m", type=float, default=50.0)
    ap.add_argument("--min-panos", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)

    act = pd.read_parquet(out / "road_basis_activation.parquet")
    nodes = pd.read_parquet(out / "road_graph_nodes.parquet")
    edges = pd.read_parquet(out / "road_graph_edges.parquet")
    ctx = pd.read_parquet(out / "road_context_features.parquet")

    npano = ctx.drop_duplicates("road_node_id").set_index("road_node_id")["n_panos"]
    nodes = nodes.copy()
    nodes["n_panos"] = nodes["road_node_id"].map(npano).fillna(0).astype(int)

    b_gdf, u_gdf, stats, r6 = extract_road_units(
        act, nodes, edges,
        boundary_quantile=args.boundary_quantile,
        min_road_nodes=args.min_road_nodes,
        min_road_length_m=args.min_road_length_m,
        min_panos=args.min_panos,
    )
    if len(b_gdf) > 0:
        b_gdf.to_file(out / "road_activation_boundaries.geojson", driver="GeoJSON")
    if len(u_gdf) > 0:
        u_gdf.to_file(out / "minimum_road_landscape_units.geojson", driver="GeoJSON")
    stats.to_csv(out / "unit_statistics.csv", index=False)
    save_report(out / "stage_reports/stage6_report.json", r6)
    print(f"Stage 6 ✓ units={r6['n_units_after_filter']} "
          f"boundaries={r6['n_boundary_edges']} tau={r6['tau']}")


if __name__ == "__main__":
    main()
