"""Stage 3 — Build road-network-organised landscape features Z_road.

Core logic: each pano contributes to ALL road segments within search_radius_m
with Gaussian distance-decay weight (not normalised across roads).
Each road node aggregates contributions from panos within context_radius_m
along its road.

Exportable:
  build_road_context_features(matched_panos, road_nodes, road_edges,
                               search_radius_m, kernel_sigma_m,
                               context_radius_m) -> (context_df, report)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scripts.core.io_utils import assert_l2_normed, checkpoint, save_report


def _gaussian_weight(dist_m: float | np.ndarray, sigma_m: float) -> np.ndarray:
    d = np.asarray(dist_m, dtype=np.float64)
    return np.exp(-(d ** 2) / (2 * sigma_m ** 2))


def _graph_diffuse(z0, covered, road_nodes, road_edges, n_iter=60):
    """Harmonic interpolation along the ROAD graph via label propagation.

    Covered nodes are anchors (fixed to z0); uncovered nodes iterate to the
    degree-normalised average of their graph neighbours. In the limit this is
    the harmonic (Laplacian) solution — a smooth blend ALONG roads between
    observations, instead of a hard Euclidean nearest-neighbour copy. Nodes in a
    component with no covered anchor stay at 0 (and get ~0 confidence).
    """
    import scipy.sparse as sp
    n = len(road_nodes)
    nid2idx = {nid: i for i, nid in enumerate(road_nodes["road_node_id"])}
    si = road_edges["src_node_id"].map(nid2idx).to_numpy()
    di = road_edges["dst_node_id"].map(nid2idx).to_numpy()
    ok = ~(pd.isna(si) | pd.isna(di))
    si, di = si[ok].astype(np.int64), di[ok].astype(np.int64)
    # symmetric adjacency
    rows = np.concatenate([si, di]); cols = np.concatenate([di, si])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
    Dinv = sp.diags(inv_deg)
    P = Dinv @ A                                   # row-normalised transition

    # Process embedding dimensions in chunks so the dense intermediate stays
    # small (full 3072-D × ~10^5 nodes would be multi-GB and unstable here).
    z = z0.astype(np.float32).copy()
    cov = covered.copy()
    z0_cov = z0[cov]
    D = z.shape[1]
    chunk = 128
    for c0 in range(0, D, chunk):
        zc = z[:, c0:c0 + chunk].copy()
        anc = z0_cov[:, c0:c0 + chunk]
        for _ in range(n_iter):
            zc = P @ zc
            zc[cov] = anc
        z[:, c0:c0 + chunk] = zc
    return z


def build_road_context_features(
    matched_panos: pd.DataFrame,
    road_nodes: pd.DataFrame,
    road_edges: pd.DataFrame,
    search_radius_m: float = 100.0,
    kernel_sigma_m: float = 30.0,
    context_radius_m: float = 100.0,
    method: str = "multi_road_decay",
    interp: str = "diffusion",
    confidence_lambda_m: float = 150.0,
    buildings_geoms=None,
) -> Tuple[pd.DataFrame, dict]:
    """Aggregate pano embeddings onto road nodes with distance-decay weighting.

    Each pano contributes to every road node within `search_radius_m`
    (independently — no normalisation across roads).

    Returns
    -------
    context_features : pd.DataFrame  — road_context_features schema
    report           : dict
    """
    DEG_TO_M = 111_000.0
    search_radius_deg = search_radius_m / DEG_TO_M

    pano_xy = matched_panos[["lon", "lat"]].values.astype(np.float64)
    node_xy = road_nodes[["lon", "lat"]].values.astype(np.float64)

    emb_list = matched_panos["pano_embedding"].values
    D = len(emb_list[0])
    pano_emb = np.stack(emb_list).astype(np.float32)   # (N_panos, D)

    agg_emb   = np.zeros((len(road_nodes), D), dtype=np.float64)
    agg_w     = np.zeros(len(road_nodes), dtype=np.float64)
    n_panos   = np.zeros(len(road_nodes), dtype=np.int32)

    # Iterate over PANOS (typically a few hundred–thousand), not road NODES
    # (~10^5–10^6). Each pano contributes a Gaussian-weighted copy of its
    # embedding to every nearby node — mathematically identical to the
    # node-centric form, but it issues ~N_panos KD-tree queries instead of
    # ~N_nodes, which is both far faster and far less likely to trip the
    # environment's intermittent native numpy/scipy corruption on big graphs.
    # Optional building-occlusion: a pano only contributes to a node if the
    # straight sight-line between them does NOT cross a building footprint.
    btree = None
    if buildings_geoms is not None:
        import shapely
        btree = shapely.STRtree(buildings_geoms)
    n_occluded = 0

    node_tree = cKDTree(node_xy)
    for pi in range(len(pano_xy)):
        px, py = pano_xy[pi]
        node_idxs = node_tree.query_ball_point([px, py], r=search_radius_deg)
        if not node_idxs:
            continue
        node_idxs = np.asarray(node_idxs, dtype=np.int64)
        dlat = (node_xy[node_idxs, 1] - py) * DEG_TO_M
        dlon = (node_xy[node_idxs, 0] - px) * DEG_TO_M * np.cos(np.radians(py))
        dists_m = np.sqrt(dlat ** 2 + dlon ** 2)

        within = dists_m <= search_radius_m
        node_idxs = node_idxs[within]
        dists_m   = dists_m[within]
        if len(node_idxs) == 0:
            continue

        if btree is not None:
            import shapely
            segs = shapely.linestrings(np.stack([
                np.broadcast_to([px, py], (len(node_idxs), 2)),
                node_xy[node_idxs]], axis=1))
            hit = btree.query(segs, predicate="intersects")   # (2, K): [seg_idx, bld_idx]
            if hit.size:
                keep = np.ones(len(node_idxs), bool)
                keep[np.unique(hit[0])] = False
                n_occluded += int((~keep).sum())
                node_idxs = node_idxs[keep]; dists_m = dists_m[keep]
                if len(node_idxs) == 0:
                    continue

        weights = _gaussian_weight(dists_m, kernel_sigma_m)   # (k,)
        # add this pano's weighted embedding to each nearby node (NOT
        # normalised across roads — every road independently gets full weight)
        agg_emb[node_idxs] += weights[:, None] * pano_emb[pi][None, :]
        agg_w[node_idxs]   += weights
        n_panos[node_idxs] += 1

    # Per-node weighted-mean embedding (covered nodes only); 0 elsewhere.
    covered_mask = agg_w > 0
    denom = np.where(covered_mask, agg_w, 1.0)[:, None]
    z0 = (agg_emb / denom).astype(np.float32)

    # Coverage confidence: decays with Euclidean distance (m) to the nearest
    # OBSERVED node. Covered -> 1; far interpolated -> ~0.
    if covered_mask.any():
        ctree = cKDTree(node_xy[covered_mask])
        d_deg, _ = ctree.query(node_xy, k=1)
        coverage_confidence = np.exp(-(d_deg * DEG_TO_M) / confidence_lambda_m)
    else:
        coverage_confidence = np.zeros(len(road_nodes))

    # Fill uncovered nodes by ROAD-graph diffusion (harmonic) instead of a hard
    # Euclidean nearest-copy. Falls back to nearest-copy on any failure / when
    # diffusion is disabled. (No uncovered nodes -> z_road is just z0.)
    z_road = z0.copy()
    use_diff = interp == "diffusion"
    if covered_mask.any() and (~covered_mask).any():
        if use_diff:
            try:
                z_road = _graph_diffuse(z0, covered_mask, road_nodes, road_edges)
            except Exception as e:  # pragma: no cover
                print(f"[stage3] diffusion failed ({e}); nearest-copy fallback")
                use_diff = False
        # nearest-covered copy for: the nearest method, OR any node still a zero
        # vector after diffusion (component with no observed anchor)
        zero_mask = (np.linalg.norm(z_road, axis=1) < 1e-9) if use_diff else (~covered_mask)
        if zero_mask.any():
            ct = cKDTree(node_xy[covered_mask])
            _, nn = ct.query(node_xy[zero_mask], k=1)
            z_road[zero_mask] = z0[covered_mask][nn]

    # L2-normalise
    row_norms = np.linalg.norm(z_road, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    z_road = (z_road / row_norms).astype(np.float32)

    assert_l2_normed(z_road, name="road_context_embedding")

    context_df = road_nodes.copy()
    context_df["road_context_embedding"] = list(z_road)
    context_df["n_panos"]            = n_panos
    context_df["total_weight"]       = agg_w.round(4)
    context_df["coverage_confidence"] = coverage_confidence.round(4)
    context_df["context_radius_m"]   = context_radius_m
    context_df["aggregation_method"] = method
    if "city" not in context_df.columns:
        context_df["city"] = matched_panos["city"].iloc[0] \
            if "city" in matched_panos.columns else ""

    # Checkpoint: covered nodes
    covered = int((n_panos > 0).sum())
    total   = len(road_nodes)
    report  = {
        "n_road_nodes":        total,
        "n_nodes_with_panos":  covered,
        "n_nodes_interpolated": int((~covered_mask).sum()),
        "coverage_ratio":      round(covered / max(total, 1), 4),
        "embedding_dim":       D,
        "search_radius_m":     search_radius_m,
        "kernel_sigma_m":      kernel_sigma_m,
        "method":              method,
        "interp":              interp,
        "mean_coverage_confidence": round(float(coverage_confidence.mean()), 4),
        "n_occluded_pano_node_pairs": int(n_occluded),
        "building_aware": buildings_geoms is not None,
    }

    # Direct coverage will be low for sparse pano sets; interpolation fills the rest.
    if covered / max(total, 1) < 0.10:
        print(f"[CHECKPOINT WARN] only {covered}/{total} road nodes "
              f"have direct pano coverage ({covered/max(total,1)*100:.1f}%)")

    return context_df, report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--matched-panos", required=True)
    ap.add_argument("--road-nodes", required=True)
    ap.add_argument("--road-edges", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="multi_road_decay")
    ap.add_argument("--search-radius-m", type=float, default=100.0)
    ap.add_argument("--kernel-sigma-m", type=float, default=30.0)
    ap.add_argument("--context-radius-m", type=float, default=100.0)
    args = ap.parse_args()

    matched = pd.read_parquet(args.matched_panos)
    nodes   = pd.read_parquet(args.road_nodes)
    edges   = pd.read_parquet(args.road_edges)

    ctx, report = build_road_context_features(
        matched, nodes, edges,
        search_radius_m=args.search_radius_m,
        kernel_sigma_m=args.kernel_sigma_m,
        context_radius_m=args.context_radius_m,
        method=args.method,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx.to_parquet(out, index=False)
    save_report(out.parent / "stage_reports" / "stage3_report.json", report)
    print(json.dumps(report, indent=2))
