"""Synthetic fixtures for the MRLU pipeline tests.

Synthetic city design:
  - 6 roads, 2 roads per region → 3 landscape regions
  - 10 panos per road, 4 headings each → pano_embedding = concat[f0,f90,f180,f270] (4D)
  - 10 road nodes per road sampled at 25m intervals
  - Region-internal Z_road cosine sim > 0.85, cross-region < 0.50
  - Cross-region road graph edges included for boundary detection tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

# ── Constants ────────────────────────────────────────────────────────────────
N_ROADS              = 6
N_PANOS_PER_ROAD     = 10
N_ROAD_NODES_PER_ROAD = 10
N_REGIONS            = 3
D_SINGLE             = 64          # single-heading embedding dim
D                    = D_SINGLE * 4  # concat of 4 headings = 256
K                    = 16          # basis count for tests
ROAD_SPACING_M       = 25.0        # metres between road nodes
PANO_SPACING_DEG     = 0.0005      # ~55m between panos along road
RNG_SEED             = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_region_centers(rng: np.random.Generator) -> np.ndarray:
    """3 well-separated unit vectors in R^D."""
    centers = rng.standard_normal((N_REGIONS, D)).astype(np.float32)
    # orthogonalise roughly via Gram-Schmidt so regions are clearly separated
    for i in range(1, N_REGIONS):
        for j in range(i):
            centers[i] -= (centers[i] @ centers[j]) * centers[j]
        centers[i] /= np.linalg.norm(centers[i])
    centers[0] /= np.linalg.norm(centers[0])
    return centers


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_pano_features():
    """
    Returns dict with:
      panos      : pd.DataFrame  (pano_id, city, lat, lon, road_id, pano_embedding, feature_model)
      region_labels : np.ndarray  (N_PANOS_TOTAL,)
      region_centers: np.ndarray  (N_REGIONS, D)
      D          : int
      K          : int
    """
    rng = np.random.default_rng(RNG_SEED)
    region_centers = _make_region_centers(rng)

    rows = []
    region_labels = []

    for road_idx in range(N_ROADS):
        region = road_idx // 2          # roads 0-1 → region 0, 2-3 → 1, 4-5 → 2
        base_lat = 48.20 + road_idx * 0.005
        base_lon = 16.37

        for j in range(N_PANOS_PER_ROAD):
            pid = f"pano_{road_idx:02d}_{j:02d}"
            lat = base_lat + j * PANO_SPACING_DEG
            lon = base_lon

            # 4-heading concat embedding: each heading has small noise around region center
            heading_feats = []
            for _ in range(4):
                noise = rng.standard_normal(D_SINGLE).astype(np.float32) * 0.12
                # use region center for the D_SINGLE slice corresponding to this heading
                center_slice = region_centers[region][_ * D_SINGLE: (_ + 1) * D_SINGLE]
                f = center_slice + noise
                heading_feats.append(f)

            emb = np.concatenate(heading_feats).astype(np.float32)  # (4D,)
            emb /= np.linalg.norm(emb)                               # L2 norm

            rows.append({
                "pano_id":       pid,
                "city":          "SyntheticCity",
                "lat":           lat,
                "lon":           lon,
                "road_id":       f"road_{road_idx:02d}",
                "pano_embedding": emb,
                "feature_model": "synthetic",
            })
            region_labels.append(region)

    return {
        "panos":          pd.DataFrame(rows),
        "region_labels":  np.array(region_labels),
        "region_centers": region_centers,
        "D":              D,
        "K":              K,
    }


@pytest.fixture(scope="session")
def synthetic_road_network():
    """
    Returns GeoDataFrame with 6 road LineStrings.
    Each road is a straight N-S line ~500m long (10 nodes × 25m + buffer).
    Cross-region connectors (road_01 end → road_02 start) included as separate entries.
    """
    records = []
    for road_idx in range(N_ROADS):
        base_lat = 48.20 + road_idx * 0.005
        lon = 16.37
        # road runs 0 → ~0.0045° lat (~500m)
        start = (base_lat, lon)
        end   = (base_lat + N_ROAD_NODES_PER_ROAD * PANO_SPACING_DEG, lon)
        geom  = LineString([(lon, start[0]), (lon, end[0])])
        records.append({
            "road_id":   f"road_{road_idx:02d}",
            "LS_id":     f"road_{road_idx:02d}",
            "road_class": "residential",
            "length_m":  500.0,
            "geometry":  geom,
        })

    # cross-region connectors (horizontal segments linking region boundaries)
    for r_src, r_dst in [(1, 2), (3, 4)]:
        lat_src = 48.20 + r_src * 0.005 + N_ROAD_NODES_PER_ROAD * PANO_SPACING_DEG
        lat_dst = 48.20 + r_dst * 0.005
        lon_src = 16.37
        lon_dst = 16.37
        geom = LineString([(lon_src, lat_src), (lon_dst, lat_dst)])
        records.append({
            "road_id":   f"connector_{r_src}_{r_dst}",
            "LS_id":     f"connector_{r_src}_{r_dst}",
            "road_class": "connector",
            "length_m":  50.0,
            "geometry":  geom,
        })

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


@pytest.fixture(scope="session")
def synthetic_road_graph_nodes(synthetic_road_network):
    """
    Road nodes sampled every 25m along each road.
    Returns pd.DataFrame with road_node_id, road_id, lat, lon, chainage_m.
    """
    from scripts.core.road_graph_utils import sample_road_nodes
    return sample_road_nodes(synthetic_road_network, spacing_m=ROAD_SPACING_M)


@pytest.fixture(scope="session")
def synthetic_road_graph_edges(synthetic_road_network, synthetic_road_graph_nodes):
    """
    same_road_next + intersection_connect edges.
    Returns pd.DataFrame with src_node_id, dst_node_id, edge_type, network_distance_m.
    """
    from scripts.core.road_graph_utils import build_road_graph_edges
    return build_road_graph_edges(synthetic_road_graph_nodes, synthetic_road_network)


@pytest.fixture(scope="session")
def synthetic_road_matched_panos(synthetic_pano_features, synthetic_road_network):
    """
    Each pano matched to its road (known by construction).
    Returns pd.DataFrame matching road_matched_panos schema.
    """
    panos = synthetic_pano_features["panos"]
    rows = []
    for _, p in panos.iterrows():
        road_id = p["road_id"]
        # chainage = pano index along road * PANO_SPACING_DEG * 111000m/deg
        j = int(p["pano_id"].split("_")[2])
        chainage = j * PANO_SPACING_DEG * 111_000
        rows.append({
            "pano_id":        p["pano_id"],
            "city":           p["city"],
            "lat":            p["lat"],
            "lon":            p["lon"],
            "matched_road_id": road_id,
            "road_distance_m": 0.0,
            "chainage_m":     chainage,
            "match_confidence": 1.0,
            "pano_embedding": p["pano_embedding"],
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_road_context_features(synthetic_pano_features, synthetic_road_graph_nodes):
    """
    Z_road per road node: weighted average of pano embeddings from the same region.
    Region-internal cosine sim > 0.85, cross-region < 0.50 (by construction).
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    region_centers = synthetic_pano_features["region_centers"]
    nodes = synthetic_road_graph_nodes.copy()

    embeddings = []
    for _, node in nodes.iterrows():
        road_idx = int(node["road_id"].replace("road_", "").split("_")[0]) \
            if node["road_id"].startswith("road_") else -1
        if road_idx >= 0:
            region = road_idx // 2
            noise = rng.standard_normal(D).astype(np.float32) * 0.08
            emb = region_centers[region] + noise
        else:
            # connector nodes: average of adjacent regions
            emb = rng.standard_normal(D).astype(np.float32)
        emb = emb.astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings.append(emb)

    nodes["road_context_embedding"] = embeddings
    nodes["n_panos"] = N_PANOS_PER_ROAD
    nodes["total_weight"] = 5.0
    nodes["context_radius_m"] = 100.0
    nodes["aggregation_method"] = "multi_road_decay"
    nodes["city"] = "SyntheticCity"
    return nodes
