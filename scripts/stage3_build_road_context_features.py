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

from scripts.io_utils import assert_l2_normed, checkpoint, save_report


def _gaussian_weight(dist_m: float | np.ndarray, sigma_m: float) -> np.ndarray:
    d = np.asarray(dist_m, dtype=np.float64)
    return np.exp(-(d ** 2) / (2 * sigma_m ** 2))


def build_road_context_features(
    matched_panos: pd.DataFrame,
    road_nodes: pd.DataFrame,
    road_edges: pd.DataFrame,
    search_radius_m: float = 100.0,
    kernel_sigma_m: float = 30.0,
    context_radius_m: float = 100.0,
    method: str = "multi_road_decay",
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

        weights = _gaussian_weight(dists_m, kernel_sigma_m)   # (k,)
        # add this pano's weighted embedding to each nearby node (NOT
        # normalised across roads — every road independently gets full weight)
        agg_emb[node_idxs] += weights[:, None] * pano_emb[pi][None, :]
        agg_w[node_idxs]   += weights
        n_panos[node_idxs] += 1

    # Nodes with zero weight: interpolate from nearest node that has data
    zero_mask = agg_w == 0
    if zero_mask.any() and (~zero_mask).any():
        have_tree = cKDTree(node_xy[~zero_mask])
        _, nn_idxs = have_tree.query(node_xy[zero_mask], k=1)
        have_indices = np.where(~zero_mask)[0]
        agg_emb[zero_mask] = agg_emb[have_indices[nn_idxs]]
        agg_w[zero_mask]   = agg_w[have_indices[nn_idxs]]

    # Normalise each node embedding by its own weight sum, then L2 norm
    denom = np.where(agg_w > 0, agg_w, 1.0)[:, None]
    z_road = (agg_emb / denom).astype(np.float32)
    row_norms = np.linalg.norm(z_road, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    z_road = z_road / row_norms

    assert_l2_normed(z_road, name="road_context_embedding")

    context_df = road_nodes.copy()
    context_df["road_context_embedding"] = list(z_road)
    context_df["n_panos"]            = n_panos
    context_df["total_weight"]       = agg_w.round(4)
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
        "n_nodes_interpolated": int(zero_mask.sum()),
        "coverage_ratio":      round(covered / max(total, 1), 4),
        "embedding_dim":       D,
        "search_radius_m":     search_radius_m,
        "kernel_sigma_m":      kernel_sigma_m,
        "method":              method,
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
