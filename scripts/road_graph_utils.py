"""Road graph construction utilities."""
from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString


def sample_road_nodes(roads_gdf: gpd.GeoDataFrame, spacing_m: float = 25.0) -> pd.DataFrame:
    """Sample nodes at fixed intervals along each road geometry.

    Returns DataFrame: road_node_id, road_id, lat, lon, chainage_m
    """
    # 1 degree latitude ≈ 111 000 m; use as rough scale for projected length
    DEG_TO_M = 111_000.0

    records = []
    for _, row in roads_gdf.iterrows():
        road_id = str(row.get("road_id", row.get("LS_id", row.name)))
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Sample points with pure-numpy linear interpolation along the vertex
        # polyline instead of shapely's geom.interpolate(): the GEOS call
        # segfaults on certain real-world geometries (observed on HK OSM data),
        # and numpy is also faster. Only handle LineString here; skip others.
        try:
            xy = np.asarray(geom.coords, dtype=float)
        except (NotImplementedError, TypeError):
            continue              # e.g. MultiLineString — no flat .coords
        if len(xy) < 2:
            continue
        seg_deg = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
        cum_deg = np.concatenate([[0.0], np.cumsum(seg_deg)])  # along-line, deg
        length_deg = float(cum_deg[-1])
        if length_deg <= 0:
            continue
        length_m = length_deg * DEG_TO_M
        n_points = max(2, int(length_m / spacing_m) + 1)
        targets_deg = np.linspace(0.0, length_deg, n_points)
        lons = np.interp(targets_deg, cum_deg, xy[:, 0])
        lats = np.interp(targets_deg, cum_deg, xy[:, 1])
        chainages_m = targets_deg * DEG_TO_M
        seen_pts: set[tuple] = set()
        for seq_idx in range(n_points):
            lat_v = float(lats[seq_idx])
            lon_v = float(lons[seq_idx])
            pt_key = (round(lat_v, 7), round(lon_v, 7))
            if pt_key in seen_pts:
                continue          # skip duplicate points on degenerate geometries
            seen_pts.add(pt_key)
            nid = f"{road_id}__{seq_idx:04d}"
            records.append({
                "road_node_id": nid,
                "road_id":      road_id,
                "lat":          lat_v,
                "lon":          lon_v,
                "chainage_m":   float(chainages_m[seq_idx]),
            })
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["road_node_id", "road_id", "lat", "lon", "chainage_m"]
    )


def build_road_graph_edges(
    road_nodes: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    junction_tol_m: float = 10.0,
) -> pd.DataFrame:
    """Build same_road_next and intersection_connect edges (bidirectional).

    Returns DataFrame: src_node_id, dst_node_id, edge_type, network_distance_m, edge_confidence
    """
    edges: list[dict] = []

    # ── same_road_next ────────────────────────────────────────────────────────
    for road_id, grp in road_nodes.groupby("road_id"):
        grp_sorted = grp.sort_values("chainage_m").reset_index(drop=True)
        for i in range(len(grp_sorted) - 1):
            a = grp_sorted.iloc[i]
            b = grp_sorted.iloc[i + 1]
            dist = float(b["chainage_m"] - a["chainage_m"])
            for src, dst in [(a["road_node_id"], b["road_node_id"]),
                             (b["road_node_id"], a["road_node_id"])]:
                edges.append({
                    "src_node_id": src, "dst_node_id": dst,
                    "edge_type": "same_road_next",
                    "network_distance_m": dist,
                    "edge_confidence": 1.0,
                })

    # ── intersection_connect ─────────────────────────────────────────────────
    # collect first/last node per road
    endpoint_nodes: list[dict] = []
    for road_id, grp in road_nodes.groupby("road_id"):
        grp_sorted = grp.sort_values("chainage_m")
        for idx in [0, -1]:
            ep = grp_sorted.iloc[idx]
            endpoint_nodes.append({
                "nid": ep["road_node_id"],
                "lat": ep["lat"],
                "lon": ep["lon"],
                "road_id": road_id,
            })

    # grid-snap endpoints by tolerance
    tol_deg = junction_tol_m / 111_000.0
    grid: dict[tuple, list[dict]] = defaultdict(list)
    for ep in endpoint_nodes:
        key = (round(ep["lat"] / tol_deg), round(ep["lon"] / tol_deg))
        grid[key].append(ep)

    seen: set[tuple] = set()
    for group in grid.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["road_id"] == b["road_id"]:
                    continue
                key = tuple(sorted([a["nid"], b["nid"]]))
                if key in seen:
                    continue
                seen.add(key)
                dist = _haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
                for src, dst in [(a["nid"], b["nid"]), (b["nid"], a["nid"])]:
                    edges.append({
                        "src_node_id": src, "dst_node_id": dst,
                        "edge_type": "intersection_connect",
                        "network_distance_m": dist,
                        "edge_confidence": 0.9,
                    })

    return pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["src_node_id", "dst_node_id", "edge_type",
                 "network_distance_m", "edge_confidence"]
    )


def build_nx_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(nodes["road_node_id"])
    for _, row in edges.iterrows():
        G.add_edge(
            row["src_node_id"], row["dst_node_id"],
            edge_type=row["edge_type"],
            network_distance_m=row["network_distance_m"],
        )
    return G


def largest_component_ratio(G: nx.Graph) -> float:
    if len(G) == 0:
        return 0.0
    lcc = max(nx.connected_components(G), key=len)
    return len(lcc) / len(G)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
