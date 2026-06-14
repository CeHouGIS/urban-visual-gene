"""Stage 6 — Extract Minimum Road Landscape Units (MRLU).

Exportable:
  extract_road_units(activation_df, road_nodes, road_edges,
                     boundary_quantile, min_road_nodes, min_road_length_m,
                     min_panos) -> (boundaries_gdf, units_gdf, stats_df, report)
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import entr
from shapely.geometry import LineString, MultiLineString

from scripts.io_utils import checkpoint, save_report
from scripts.road_graph_utils import build_nx_graph


def extract_road_units(
    activation_df: pd.DataFrame,
    road_nodes: pd.DataFrame,
    road_edges: pd.DataFrame,
    boundary_quantile: float = 0.90,
    min_road_nodes: int = 3,
    min_road_length_m: float = 50.0,
    min_panos: int = 3,
    return_unit_activations: bool = False,
    covered_only: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, dict]:
    """
    Returns
    -------
    boundaries_gdf : GeoDataFrame — road_activation_boundaries schema
    units_gdf      : GeoDataFrame — minimum_road_landscape_units schema
    stats_df       : DataFrame    — unit_statistics schema
    report         : dict
    """
    # ── Covered-subgraph mode ────────────────────────────────────────────────
    # Restrict segmentation to road nodes with DIRECT pano coverage (n_panos>0)
    # and edges between two such nodes. The full graph is ~95% interpolated
    # (zero-coverage nodes copy their nearest covered node), so segmenting it
    # mixes real visual signal with interpolation artefacts. covered_only keeps
    # only observed nodes, so boundaries reflect actual street-view evidence.
    if covered_only and "n_panos" in road_nodes.columns:
        cov_ids = set(road_nodes.loc[road_nodes["n_panos"] > 0, "road_node_id"])
        road_nodes = road_nodes[road_nodes["road_node_id"].isin(cov_ids)].reset_index(drop=True)
        road_edges = road_edges[road_edges["src_node_id"].isin(cov_ids)
                                & road_edges["dst_node_id"].isin(cov_ids)].reset_index(drop=True)
        activation_df = activation_df[activation_df["road_node_id"].isin(cov_ids)].reset_index(drop=True)

    # ── Extract A matrix (dedup road_node_id to be safe) ─────────────────────
    activation_df = activation_df.drop_duplicates(subset="road_node_id") \
                                  .reset_index(drop=True)
    a_cols = sorted([c for c in activation_df.columns if c.startswith("a_")])
    K = len(a_cols)
    nid2idx = {n: i for i, n in enumerate(activation_df["road_node_id"])}
    A = activation_df[a_cols].values.astype(np.float32)   # (M, K)

    # ── Compute edge activation distances ────────────────────────────────────
    # Iterate over plain Python tuples (itertuples-free zip on the relevant
    # columns) instead of road_edges.iterrows()/boolean-masking the frame later
    # — the latter scans 500K rows per boundary edge and can hit numpy axis
    # errors on awkward in-memory frames. We also cache edge metadata here so
    # the boundary loop never re-queries road_edges.
    boundary_rows: list[dict] = []
    edge_dist: dict[tuple, float] = {}
    edge_meta: dict[tuple, tuple] = {}   # (src,dst) -> (network_distance_m, edge_type)

    _src = road_edges["src_node_id"].tolist()
    _dst = road_edges["dst_node_id"].tolist()
    _ndm = road_edges["network_distance_m"].tolist()
    _et  = road_edges["edge_type"].tolist()
    for src_id, dst_id, ndm, et in zip(_src, _dst, _ndm, _et):
        edge_meta[(src_id, dst_id)] = (float(ndm), str(et))
        si = nid2idx.get(src_id)
        di = nid2idx.get(dst_id)
        if si is None or di is None:
            continue
        a_i = A[si]
        a_j = A[di]
        norm_i = np.linalg.norm(a_i) + 1e-9
        norm_j = np.linalg.norm(a_j) + 1e-9
        cos_sim = float(np.dot(a_i / norm_i, a_j / norm_j))
        d_ij    = float(1.0 - cos_sim)
        edge_dist[(src_id, dst_id)] = d_ij

    if not edge_dist:
        empty_gdf = gpd.GeoDataFrame()
        return empty_gdf, empty_gdf, pd.DataFrame(), {"error": "no edges with activation"}

    # Boundary threshold τ_c. Compute the quantile over the NON-ZERO edge
    # distances: when many adjacent nodes share identical activations (d_ij≈0),
    # the quantile over the full distribution degenerates to 0 and "any nonzero
    # difference" becomes a boundary. Restricting to positive distances yields a
    # meaningful threshold; fall back to the full distribution only if needed.
    dist_vals = np.array(list(edge_dist.values()))
    _eps = 1e-6
    _pos = dist_vals[dist_vals > _eps]
    if _pos.size:
        tau = float(np.quantile(_pos, boundary_quantile))
    else:
        tau = float(np.quantile(dist_vals, boundary_quantile))

    # ── Build boundary GeoDataFrame ───────────────────────────────────────────
    node_latlon = {row["road_node_id"]: {"lat": row["lat"], "lon": row["lon"]}
                   for _, row in activation_df.iterrows()}

    bid = 0
    for (src, dst), d in edge_dist.items():
        if d <= tau:
            continue
        s_ll = node_latlon.get(src)
        d_ll = node_latlon.get(dst)
        if s_ll and d_ll:
            geom = LineString([(s_ll["lon"], s_ll["lat"]),
                               (d_ll["lon"], d_ll["lat"])])
        else:
            geom = None
        dist_m, etype = edge_meta.get((src, dst), (0.0, ""))
        boundary_rows.append({
            "boundary_id":        f"be_{bid:06d}",
            "src_node_id":        src,
            "dst_node_id":        dst,
            "activation_distance": round(d, 5),
            "boundary_score":      round((d - tau) / (dist_vals.max() - tau + 1e-9), 4),
            "edge_type":           etype,
            "network_distance_m":  dist_m,
            "geometry":            geom,
        })
        bid += 1

    boundaries_gdf = gpd.GeoDataFrame(boundary_rows, geometry="geometry",
                                       crs="EPSG:4326") \
        if boundary_rows else gpd.GeoDataFrame()

    # ── Remove boundary edges, find connected components ─────────────────────
    boundary_pairs = {(r["src_node_id"], r["dst_node_id"]) for r in boundary_rows}
    boundary_pairs |= {(r["dst_node_id"], r["src_node_id"]) for r in boundary_rows}

    G = build_nx_graph(road_nodes, road_edges)
    for src, dst in boundary_pairs:
        if G.has_edge(src, dst):
            G.remove_edge(src, dst)

    # ── Characterise each component ───────────────────────────────────────────
    n_panos_map = {}
    if "n_panos" in road_nodes.columns:
        n_panos_map = road_nodes.set_index("road_node_id")["n_panos"].to_dict()

    # Reuse the metadata cached above instead of re-scanning road_edges.
    edge_len_map: dict[tuple, float] = {k: v[0] for k, v in edge_meta.items()}

    unit_records: list[dict] = []
    unit_geoms:   list       = []
    unit_mean_acts: list     = []   # full K-dim mean activation per unit

    for comp_idx, component in enumerate(nx.connected_components(G)):
        comp_list = list(component)
        n_rn = len(comp_list)

        # road_length_m
        road_len = sum(
            edge_len_map.get((u, v), 0.0)
            for u in comp_list for v in G.neighbors(u)
            if v in component
        ) / 2.0   # each edge counted twice

        n_panos = sum(n_panos_map.get(n, 0) for n in comp_list)

        # Filter
        if n_rn < min_road_nodes or road_len < min_road_length_m:
            continue
        if n_panos < min_panos:
            continue

        # Activation stats
        idxs = [nid2idx[n] for n in comp_list if n in nid2idx]
        if not idxs:
            continue
        a_unit = A[idxs]              # (n_rn, K)
        mean_a = a_unit.mean(axis=0)  # (K,)
        within_var  = float(np.mean(np.var(a_unit, axis=0)))
        dominant_k  = int(mean_a.argmax())
        a_norm = mean_a / (mean_a.sum() + 1e-9)
        act_ent = float(entr(a_norm).sum())

        # Boundary contrast: mean activation distance of boundary edges incident to unit
        inc_dists = [
            d for (s, t), d in edge_dist.items()
            if (s in component) != (t in component)
        ]
        mean_boundary_contrast = float(np.mean(inc_dists)) if inc_dists else 0.0

        # unit_confidence
        unit_confidence = float(
            1.0 / (1.0 + np.exp(-(mean_boundary_contrast - within_var)))
        )

        # Geometry: MultiLineString from internal edges
        segs: list = []
        for u in comp_list:
            ll_u = node_latlon.get(u)
            if not ll_u:
                continue
            for v in G.neighbors(u):
                if v in component and v > u:
                    ll_v = node_latlon.get(v)
                    if ll_v:
                        segs.append(LineString([
                            (ll_u["lon"], ll_u["lat"]),
                            (ll_v["lon"], ll_v["lat"]),
                        ]))

        geom = MultiLineString(segs) if segs else None

        lats = [node_latlon[n]["lat"] for n in comp_list if n in node_latlon]
        lons = [node_latlon[n]["lon"] for n in comp_list if n in node_latlon]

        top_k = np.argsort(-mean_a)[:5]
        unit_id = f"unit_{comp_idx:06d}"
        unit_records.append({
            "unit_id":                    unit_id,
            "city":                       activation_df["city"].iloc[0] if "city" in activation_df.columns else "",
            "n_road_nodes":               n_rn,
            "n_panos":                    int(n_panos),
            "road_length_m":              round(road_len, 1),
            "dominant_basis_id":          dominant_k,
            "activation_entropy":         round(act_ent, 4),
            "mean_within_activation_variance": round(within_var, 5),
            "mean_boundary_contrast":     round(mean_boundary_contrast, 5),
            "unit_confidence":            round(unit_confidence, 4),
            "centroid_lat":               round(float(np.mean(lats)), 6) if lats else 0.0,
            "centroid_lon":               round(float(np.mean(lons)), 6) if lons else 0.0,
            "top_basis_ids":              "|".join(str(i) for i in top_k),
            "top_basis_weights":          "|".join(f"{mean_a[i]:.4f}" for i in top_k),
            "stability_score":            None,
        })
        unit_geoms.append(geom)
        unit_mean_acts.append(mean_a.astype(np.float32))

    stats_df = pd.DataFrame(unit_records)

    gdf_cols = ["unit_id", "city", "n_road_nodes", "n_panos", "road_length_m",
                "dominant_basis_id", "activation_entropy",
                "mean_within_activation_variance", "mean_boundary_contrast",
                "unit_confidence"]
    if len(unit_records) > 0:
        units_gdf = gpd.GeoDataFrame(stats_df[gdf_cols], geometry=unit_geoms,
                                      crs="EPSG:4326")
    else:
        units_gdf = gpd.GeoDataFrame()

    # ── Checkpoints ───────────────────────────────────────────────────────────
    n_units = len(unit_records)
    checkpoint(n_units >= 1, f"no road landscape units extracted after filtering")

    # Boundary fraction is now relative to the POSITIVE-distance edges (τ_c is
    # their quantile). Edges with d_ij≈0 can never be boundaries, so the overall
    # ratio is expected to be ~ (1-q)·frac_positive — report it, don't hard-fail.
    boundary_ratio = len(boundary_rows) / max(len(edge_dist), 1)
    frac_pos = float((dist_vals > _eps).mean())
    expected = (1 - boundary_quantile) * frac_pos
    if abs(boundary_ratio - expected) > 0.10:
        print(f"[CHECKPOINT WARN] boundary ratio {boundary_ratio:.3f} vs "
              f"expected ~{expected:.3f} (q={boundary_quantile}, "
              f"frac_positive_edges={frac_pos:.3f})")

    report = {
        "n_edges_total":     len(edge_dist),
        "n_boundary_edges":  len(boundary_rows),
        "boundary_ratio":    round(boundary_ratio, 4),
        "frac_positive_edges": round(frac_pos, 4),
        "tau":               round(tau, 5),
        "n_components_raw":  nx.number_connected_components(G),
        "n_units_after_filter": n_units,
        "boundary_quantile": boundary_quantile,
        "covered_only": covered_only,
    }

    if return_unit_activations:
        acts = (np.vstack(unit_mean_acts) if unit_mean_acts
                else np.empty((0, K), dtype=np.float32))
        return boundaries_gdf, units_gdf, stats_df, report, acts
    return boundaries_gdf, units_gdf, stats_df, report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--activation",  required=True)
    ap.add_argument("--road-nodes",  required=True)
    ap.add_argument("--road-edges",  required=True)
    ap.add_argument("--output-dir",  required=True)
    ap.add_argument("--boundary-quantile", type=float, default=0.90)
    ap.add_argument("--min-road-nodes",    type=int,   default=3)
    ap.add_argument("--min-road-length-m", type=float, default=50.0)
    ap.add_argument("--min-panos",         type=int,   default=3)
    args = ap.parse_args()

    act  = pd.read_parquet(args.activation)
    nds  = pd.read_parquet(args.road_nodes)
    eds  = pd.read_parquet(args.road_edges)

    b_gdf, u_gdf, stats, report = extract_road_units(
        act, nds, eds,
        boundary_quantile=args.boundary_quantile,
        min_road_nodes=args.min_road_nodes,
        min_road_length_m=args.min_road_length_m,
        min_panos=args.min_panos,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(b_gdf) > 0:
        b_gdf.to_file(out / "road_activation_boundaries.geojson", driver="GeoJSON")
    if len(u_gdf) > 0:
        u_gdf.to_file(out / "minimum_road_landscape_units.geojson", driver="GeoJSON")
    stats.to_csv(out / "unit_statistics.csv", index=False)
    save_report(out / "stage_reports" / "stage6_report.json", report)
    print(json.dumps(report, indent=2))
