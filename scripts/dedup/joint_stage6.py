#!/usr/bin/env python3
"""Re-extract Stage-6 units from the JOINT-basis activations (isolated geo process).
Mirrors the formal config of these runs: covered_only, boundary_quantile=0.90.
Writes *_joint variants next to the per-city originals.
"""
import scripts._env  # noqa: F401
from pathlib import Path
import pandas as pd
from scripts.pipeline.stage6_extract_road_units import extract_road_units

CITIES = ({"Vienna": "outputs/dedup_bld/Vienna", "HongKong": "outputs/dedup_bld/HongKong"}
          if __import__("os").environ.get("USE_DEDUP_BLD")
          else {"Vienna": "outputs/sweep/Vienna_N5000", "HongKong": "outputs/sweep/HongKong_N2000"})


def main():
    for city, run in CITIES.items():
        d = Path(run)
        act = pd.read_parquet(d / "road_basis_activation_joint.parquet")
        nodes = pd.read_parquet(d / "road_graph_nodes.parquet")
        edges = pd.read_parquet(d / "road_graph_edges.parquet")
        ctx = pd.read_parquet(d / "road_context_features.parquet",
                              columns=["road_node_id", "n_panos"])
        cd = ctx.drop_duplicates("road_node_id").set_index("road_node_id")
        nodes = nodes.copy()
        nodes["n_panos"] = nodes["road_node_id"].map(cd["n_panos"]).fillna(0).astype(int)

        b_gdf, u_gdf, stats, r6 = extract_road_units(
            act, nodes, edges, boundary_quantile=0.90, covered_only=True)
        if len(b_gdf):
            b_gdf.to_file(d / "road_activation_boundaries_joint.geojson", driver="GeoJSON")
        if len(u_gdf):
            u_gdf.to_file(d / "minimum_road_landscape_units_joint.geojson", driver="GeoJSON")
        stats.to_csv(d / "unit_statistics_joint.csv", index=False)
        print(f"[joint-s6] {city}: units={r6['n_units_after_filter']} "
              f"boundaries={r6['n_boundary_edges']} tau={r6['tau']:.5f}")


if __name__ == "__main__":
    main()
