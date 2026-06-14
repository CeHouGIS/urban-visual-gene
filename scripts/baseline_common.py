"""Shared building blocks for the baseline comparison (RQ2).

All methods produce a per-node FEATURE matrix, are segmented by the SAME rule
(edge cosine distance > positive-quantile threshold → remove → connected
components), and are evaluated in the SAME common space (raw DINOv2 Z_road), so
differences reflect the representation, not the protocol.

torch-free (numpy / scipy / sklearn / networkx).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.io_utils import stack_embeddings


# ── data loading ────────────────────────────────────────────────────────────
def load_city(out: str, covered_only: bool = True):
    """Return (node_ids, Z_road[M,D], A[M,K], lat, lon, edges_df, n_panos)."""
    from pathlib import Path
    out = Path(out)
    ctx = pd.read_parquet(out / "road_context_features.parquet")
    act = pd.read_parquet(out / "road_basis_activation.parquet")
    edges = pd.read_parquet(out / "road_graph_edges.parquet")

    ctx = ctx.drop_duplicates("road_node_id").reset_index(drop=True)
    if covered_only:
        ctx = ctx[ctx["n_panos"] > 0].reset_index(drop=True)
    keep = set(ctx["road_node_id"])
    act = act.drop_duplicates("road_node_id")
    act = act[act["road_node_id"].isin(keep)].set_index("road_node_id").reindex(ctx["road_node_id"])
    edges = edges[edges["src_node_id"].isin(keep) & edges["dst_node_id"].isin(keep)].reset_index(drop=True)

    node_ids = ctx["road_node_id"].tolist()
    Z = stack_embeddings(ctx["road_context_embedding"].values)
    a_cols = sorted([c for c in act.columns if c.startswith("a_")])
    A = act[a_cols].values.astype(np.float32)
    return (node_ids, Z, A, ctx["lat"].values, ctx["lon"].values, edges,
            ctx["n_panos"].values)


# ── feature builders (one per method) ───────────────────────────────────────
def build_features(method: str, Z, A, lat, lon, K: int = 32, seed: int = 42):
    rng = np.random.default_rng(seed)
    if method == "sae":          # our method: learned sparse activation
        return A
    if method == "dino":         # raw DINOv2 context feature, segment directly
        return Z
    if method == "pca":          # linear K-dim projection of DINO
        from sklearn.decomposition import PCA
        return PCA(n_components=min(K, Z.shape[1]), random_state=seed).fit_transform(Z)
    if method == "kmeans":       # visual clusters → node = its cluster centroid
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(Z)
        return km.cluster_centers_[km.labels_].astype(np.float32)
    if method == "spatial":      # pure geometry control: standardised lat/lon
        xy = np.column_stack([lat, lon]).astype(np.float64)
        xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-9)
        return xy.astype(np.float32)
    if method == "random":       # negative control: random directions
        return rng.standard_normal((Z.shape[0], K)).astype(np.float32)
    if method == "shuffled":     # negative control: real features, wrong nodes
        return Z[rng.permutation(Z.shape[0])]
    raise ValueError(f"unknown method {method}")


# ── unified segmentation ────────────────────────────────────────────────────
def segment(features, node_ids, edges, boundary_quantile: float = 0.90):
    """Edge cosine distance > positive-quantile τ → boundary; remove → components.

    Returns node_id -> integer unit label, plus (n_boundary, n_edges, tau).
    """
    import networkx as nx
    nid2idx = {n: i for i, n in enumerate(node_ids)}
    F = np.asarray(features, dtype=np.float32)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)

    src = edges["src_node_id"].to_numpy()
    dst = edges["dst_node_id"].to_numpy()
    si = np.array([nid2idx.get(s, -1) for s in src])
    di = np.array([nid2idx.get(d, -1) for d in dst])
    ok = (si >= 0) & (di >= 0)
    si, di = si[ok], di[ok]
    src, dst = src[ok], dst[ok]
    d_ij = 1.0 - np.einsum("ij,ij->i", F[si], F[di])   # cosine distance

    pos = d_ij[d_ij > 1e-6]
    tau = float(np.quantile(pos, boundary_quantile)) if pos.size else 0.0

    G = nx.Graph()
    G.add_nodes_from(node_ids)
    keep_edge = d_ij <= tau
    G.add_edges_from(zip(src[keep_edge], dst[keep_edge]))

    labels = {}
    for lab, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            labels[n] = lab
    n_boundary = int((~keep_edge).sum())
    return labels, n_boundary, len(d_ij), tau


# ── evaluation in the common DINO space ─────────────────────────────────────
def evaluate(labels, node_ids, Z, edges,
             min_unit_nodes: int = 3, n_perm: int = 200, seed: int = 42):
    """Return dict of segmentation-quality metrics measured on raw Z_road.

    within_var       : mean within-unit feature variance (lower = better)
    boundary_contrast: mean Z cosine distance across boundary vs within edges
    *_z              : z-score vs random connected partitions of same #labels
    """
    rng = np.random.default_rng(seed)
    nid2idx = {n: i for i, n in enumerate(node_ids)}
    Zc = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    lab = np.array([labels.get(n, -1) for n in node_ids])

    def within_var(lab_vec):
        tot, w = 0.0, 0
        for u in np.unique(lab_vec):
            idx = np.where(lab_vec == u)[0]
            if len(idx) < min_unit_nodes:
                continue
            tot += np.var(Z[idx], axis=0).mean() * len(idx)
            w += len(idx)
        return tot / max(w, 1)

    # boundary contrast on the (filtered) edge list
    src = edges["src_node_id"].to_numpy(); dst = edges["dst_node_id"].to_numpy()
    si = np.array([nid2idx.get(s, -1) for s in src])
    di = np.array([nid2idx.get(d, -1) for d in dst])
    ok = (si >= 0) & (di >= 0); si, di = si[ok], di[ok]
    d_edge = 1.0 - np.einsum("ij,ij->i", Zc[si], Zc[di])
    cross = lab[si] != lab[di]
    b_contrast = (float(d_edge[cross].mean()) if cross.any() else 0.0) \
        - (float(d_edge[~cross].mean()) if (~cross).any() else 0.0)

    wv = within_var(lab)
    n_units = len(set(l for l in labels.values()))
    # permutation null: shuffle labels (same multiset) → random partition
    perm_wv = np.array([within_var(rng.permutation(lab)) for _ in range(n_perm)])
    wv_z = float((wv - perm_wv.mean()) / (perm_wv.std() + 1e-9))

    return {
        "n_units": n_units,
        "within_var": round(wv, 6),
        "within_var_z": round(wv_z, 3),       # negative = better than random
        "boundary_contrast": round(b_contrast, 6),
    }
