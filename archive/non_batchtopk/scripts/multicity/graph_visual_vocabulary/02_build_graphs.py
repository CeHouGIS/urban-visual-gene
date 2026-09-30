#!/usr/bin/env python3
"""Build three locally scaled kNN graphs and their equal-weight union."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy import sparse

from .utils import DEFAULT_CONFIG, assert_safe_affinity, cosine_rows, ensure_layout, load_config, save_json, should_skip


def local_knn_graph(
    similarity: np.ndarray, k: int, epsilon: float, valid: np.ndarray | None = None
) -> sparse.csr_matrix:
    n = len(similarity); valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    work = similarity.copy(); np.fill_diagonal(work, -np.inf); work[:, ~valid] = -np.inf; work[~valid, :] = -np.inf
    neighbours = np.argpartition(-work, k - 1, axis=1)[:, :k]
    distances = 1 - similarity
    sigma = np.take_along_axis(distances, neighbours, axis=1).max(axis=1)
    mask = np.zeros((n, n), dtype=bool)
    mask[np.arange(n)[valid, None], neighbours[valid]] = True
    mask |= mask.T; np.fill_diagonal(mask, False)
    left, right = np.where(np.triu(mask, 1))
    weights = np.exp(-(distances[left, right] ** 2) / (sigma[left] * sigma[right] + epsilon))
    positive_mean = weights[weights > 0].mean()
    weights /= positive_mean
    matrix = sparse.coo_matrix((np.r_[weights, weights], (np.r_[left, right], np.r_[right, left])), shape=(n, n))
    return matrix.tocsr()


def diagnostics(matrix: sparse.csr_matrix) -> dict:
    graph = sparse.csgraph.connected_components(matrix, directed=False)
    degrees = np.diff(matrix.indptr)
    n = matrix.shape[0]; edges = matrix.nnz // 2
    return {"nodes": n, "edges": edges, "density": 2 * edges / (n * (n - 1)), "connected_components": int(graph[0]), "degree_min": int(degrees.min()), "degree_mean": float(degrees.mean()), "degree_median": float(np.median(degrees)), "degree_max": int(degrees.max()), "mean_nonzero_weight": float(matrix.data.mean())}


def build_graphs(config: dict, force: bool = False) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    cache = config["paths"]["cache_root"]; data = config["paths"]["paper_data_root"]
    target = data / "graph/visual_graph_edges.csv"
    if should_skip([target, data / "graph/visual_graph.graphml"], force): return {"status": "existing"}
    views = cache / "views"; graph_dir = cache / "graph"
    spatial = np.load(views / "spatial_profile_14x56.u64.npy").astype(np.float64)
    spatial /= np.maximum(spatial.sum(axis=1, keepdims=True), 1)
    vectors = {"E": np.load(views / "encoder_directions.f32.npy"), "D": np.load(views / "decoder_directions.f32.npy"), "P": spatial}
    k = int(config["graph"]["k_neighbors"]); epsilon = float(config["graph"]["epsilon"])
    similarities = {}; matrices = {}; reports = {}
    for key, values in vectors.items():
        similarities[key] = cosine_rows(values).astype(np.float32)
        valid = np.linalg.norm(values, axis=1) > 0
        matrices[key] = local_knn_graph(similarities[key], k, epsilon, valid)
        np.save(graph_dir / f"similarity_{key}.f32.npy", similarities[key])
        sparse.save_npz(graph_dir / f"graph_{key}.npz", matrices[key])
        reports[key] = diagnostics(matrices[key])
    weights = config["graph"]["fusion_weights"]
    fused = matrices["E"] * float(weights["encoder"]) + matrices["D"] * float(weights["decoder"]) + matrices["P"] * float(weights["spatial"])
    fused.eliminate_zeros(); sparse.save_npz(graph_dir / "visual_graph_fused.npz", fused)
    left, right = fused.nonzero(); keep = left < right; left, right = left[keep], right[keep]
    frame = pd.DataFrame({"source": left, "target": right})
    for key in ("E", "D", "P"):
        frame[f"weight_{key}"] = np.asarray(matrices[key][left, right]).ravel()
    frame["weight_fused"] = np.asarray(fused[left, right]).ravel()
    frame["view_support"] = (frame[["weight_E", "weight_D", "weight_P"]] > 0).sum(axis=1)
    frame.to_csv(target, index=False)
    node_frame = pd.DataFrame({"node_id": np.arange(512), "dimension": [f"D{x:03d}" for x in range(512)], "weighted_degree": np.asarray(fused.sum(axis=1)).ravel(), "degree": np.diff(fused.indptr)})
    node_frame.to_csv(data / "graph/visual_graph_nodes.csv", index=False)
    graph = ig.Graph(n=512, edges=list(zip(left.tolist(), right.tolist())), directed=False)
    graph.vs["name"] = node_frame.dimension.tolist(); graph.vs["weighted_degree"] = node_frame.weighted_degree.tolist()
    for column in ["weight_E", "weight_D", "weight_P", "weight_fused", "view_support"]: graph.es[column] = frame[column].tolist()
    graph.write_graphml(str(data / "graph/visual_graph.graphml"))
    reports["fused"] = diagnostics(fused); save_json(data / "graph/graph_diagnostics.json", reports)
    return reports


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--force",action="store_true"); args=parser.parse_args(); print(build_graphs(load_config(args.config),args.force))


if __name__ == "__main__": main()
