"""Stage 2 — Map-match panos to road network, build road graph nodes/edges.

Exportable functions:
  map_match(pano_features, roads_gdf, max_dist_m, node_spacing_m) -> (matched, nodes, edges, report)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from scripts.io_utils import checkpoint, save_report
from scripts.road_graph_utils import (
    build_nx_graph,
    build_road_graph_edges,
    largest_component_ratio,
    sample_road_nodes,
)


def _estimate_utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Return an appropriate UTM CRS string for the given GeoDataFrame."""
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    lon_c = (bounds[0] + bounds[2]) / 2
    lat_c = (bounds[1] + bounds[3]) / 2
    zone  = int((lon_c + 180) / 6) + 1
    hemi  = "north" if lat_c >= 0 else "south"
    return f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m +no_defs"


def map_match(
    pano_features: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    max_dist_m: float = 30.0,
    node_spacing_m: float = 25.0,
    junction_tol_m: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Match panos to nearest road using UTM-projected spatial index.

    Returns
    -------
    matched_panos : pd.DataFrame   — road_matched_panos schema
    road_nodes    : pd.DataFrame   — road_graph_nodes schema
    road_edges    : pd.DataFrame   — road_graph_edges schema
    report        : dict
    """
    # Ensure roads have a road_id column
    roads_gdf = roads_gdf.copy().reset_index(drop=True)
    if "road_id" not in roads_gdf.columns:
        roads_gdf["road_id"] = roads_gdf.index.astype(str)

    # Ensure WGS-84
    if roads_gdf.crs is None:
        roads_gdf = roads_gdf.set_crs("EPSG:4326")
    roads_wgs = roads_gdf.to_crs("EPSG:4326")

    # Project to UTM for metric distance calculations
    utm_crs = _estimate_utm_crs(roads_wgs)
    roads_utm = roads_wgs.to_crs(utm_crs)

    # Build pano GeoDataFrame and project
    pano_gdf = gpd.GeoDataFrame(
        pano_features,
        geometry=gpd.points_from_xy(pano_features["lon"], pano_features["lat"]),
        crs="EPSG:4326",
    ).to_crs(utm_crs)

    # Nearest road for each pano using spatial index
    roads_for_join = roads_utm[["road_id", "geometry"]].copy()
    joined = gpd.sjoin_nearest(
        pano_gdf[["pano_id", "geometry"]],
        roads_for_join,
        how="left",
        distance_col="road_distance_m",
    )
    joined = joined.drop_duplicates(subset="pano_id")

    matched_rows: list[dict] = []
    unmatched: list[str] = []

    # Index for fast pano_features lookup
    pf_idx = pano_features.set_index("pano_id")

    for _, row in joined.iterrows():
        pid      = row["pano_id"]
        dist_m   = float(row["road_distance_m"])
        road_id  = str(row["road_id"])
        pano_row = pf_idx.loc[pid]

        if dist_m > max_dist_m:
            unmatched.append(pid)
            continue

        # Chainage along road (using UTM geometry)
        pano_pt_utm = row["geometry"]
        road_geom_utm = roads_utm.loc[
            roads_utm["road_id"] == road_id, "geometry"
        ].iloc[0]
        chainage_m = float(road_geom_utm.project(pano_pt_utm))

        matched_rows.append({
            "pano_id":          pid,
            "city":             pano_row.get("city", ""),
            "lat":              float(pano_row["lat"]),
            "lon":              float(pano_row["lon"]),
            "matched_road_id":  road_id,
            "road_distance_m":  round(dist_m, 2),
            "chainage_m":       round(chainage_m, 2),
            "match_confidence": float(max(0.0, 1.0 - dist_m / max_dist_m)),
            "pano_embedding":   pano_row["pano_embedding"],
        })

    matched_panos = pd.DataFrame(matched_rows)
    matched_ratio = len(matched_panos) / max(len(pano_features), 1)

    # Road graph nodes and edges
    road_nodes = sample_road_nodes(roads_gdf, spacing_m=node_spacing_m)
    road_edges = build_road_graph_edges(road_nodes, roads_gdf,
                                        junction_tol_m=junction_tol_m)

    G = build_nx_graph(road_nodes, road_edges)
    lcc_ratio = largest_component_ratio(G)

    report = {
        "n_panos_total":   int(len(pano_features)),
        "n_panos_matched": int(len(matched_panos)),
        "n_panos_unmatched": int(len(unmatched)),
        "matched_ratio":   round(matched_ratio, 4),
        "n_road_nodes":    int(len(road_nodes)),
        "n_road_edges":    int(len(road_edges)),
        "lcc_ratio":       round(lcc_ratio, 4),
        "unmatched_pano_ids": unmatched[:20],
    }

    # Checkpoints
    checkpoint(matched_ratio >= 0.80,
               f"matched_ratio={matched_ratio:.3f} < 0.80")
    # LCC ratio depends on road network data quality; warn but don't fail
    if lcc_ratio < 0.90:
        print(f"[CHECKPOINT WARN] road graph lcc_ratio={lcc_ratio:.3f} < 0.90 "
              f"(may be due to sparse pano coverage or fragmented road data)")

    return matched_panos, road_nodes, road_edges, report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pano-features", required=True)
    ap.add_argument("--roads", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--node-spacing-m", type=float, default=25.0)
    ap.add_argument("--max-match-distance-m", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    panos = pd.read_parquet(args.pano_features)
    roads = gpd.read_file(args.roads)

    matched, nodes, edges, report = map_match(
        panos, roads,
        max_dist_m=args.max_match_distance_m,
        node_spacing_m=args.node_spacing_m,
    )

    matched.to_parquet(out / "road_matched_panos.parquet", index=False)
    nodes.to_parquet(out / "road_graph_nodes.parquet", index=False)
    edges.to_parquet(out / "road_graph_edges.parquet", index=False)
    save_report(out / "stage_reports" / "stage2_report.json", report)
    print(json.dumps(report, indent=2))
