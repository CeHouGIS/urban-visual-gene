#!/usr/bin/env python3
"""Build interpretable V1-space prototypes for compact-encoding clusters."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from scripts.image_graph_archetypes.utils import (
    ADJACENCIES,
    DESCRIPTOR_DIM,
    EDGE_COUNT,
    NODES,
    SEED,
    assert_safe_affinity,
    edge_definition,
    save_csv,
    save_json,
)
from scripts.image_graph_compact_archetypes.utils import COMPACT_DIM, OUTPUT_ROOT, SOURCE_ROOT


def build(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    original = np.load(args.source / "image_graph_features.npy", mmap_mode="r")
    compact = np.load(args.output / "image_graph_compact_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(args.output / "image_graph_metadata.parquet")
    labels = pd.read_parquet(args.output / "image_graph_archetype_labels.parquet")
    n = len(metadata)
    if original.shape[0] < n or original.shape[1] != DESCRIPTOR_DIM:
        raise ValueError("original graph feature contract mismatch")
    if compact.shape != (n, COMPACT_DIM) or len(labels) != n:
        raise ValueError("compact feature/metadata/label contract mismatch")
    if not np.array_equal(labels["image_id"].to_numpy(), metadata["image_id"].to_numpy()):
        raise ValueError("labels are not aligned with metadata")
    cluster_index = labels["archetype_id"].str[1:].astype(int).to_numpy() - 1
    counts = np.bincount(cluster_index, minlength=8)
    original_sums = np.zeros((8, DESCRIPTOR_DIM), dtype=np.float64)
    compact_sums = np.zeros((8, COMPACT_DIM), dtype=np.float64)
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        block_labels = cluster_index[start:stop]
        original_block = np.asarray(original[start:stop], dtype=np.float32)
        compact_block = np.asarray(compact[start:stop], dtype=np.float32)
        for cluster in np.unique(block_labels):
            selected = block_labels == cluster
            original_sums[cluster] += original_block[selected].sum(axis=0, dtype=np.float64)
            compact_sums[cluster] += compact_block[selected].sum(axis=0, dtype=np.float64)
        if start == 0 or stop == n or stop % 50000 < args.batch_size:
            print(f"prototype accumulation {stop:,}/{n:,}", flush=True)
    prototypes = original_sums / counts[:, None]
    compact_prototypes = (compact_sums / counts[:, None]).astype(np.float32)
    np.save(args.output / "prototype_compact_features.npy", compact_prototypes)
    if not np.allclose(prototypes[:, :NODES].sum(axis=1), 1, atol=1e-6):
        raise ValueError("prototype node areas do not sum to one")

    left, right, _ = edge_definition()
    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    summary_rows: list[dict] = []
    for cluster in range(8):
        archetype = f"A{cluster + 1:02d}"
        node_values = prototypes[cluster, :NODES]
        edge_values = prototypes[cluster, NODES:]
        node_order = np.argsort(-node_values)
        edge_order = np.argsort(-edge_values)
        node_rank = np.empty(NODES, dtype=np.int16)
        edge_rank = np.empty(EDGE_COUNT, dtype=np.int16)
        node_rank[node_order] = np.arange(1, NODES + 1)
        edge_rank[edge_order] = np.arange(1, EDGE_COUNT + 1)
        for node in range(NODES):
            node_rows.append({
                "archetype_id": archetype,
                "node_id": f"F{node:03d}",
                "node_index": node,
                "mean_node_area": float(node_values[node]),
                "within_archetype_rank": int(node_rank[node]),
            })
        for edge in range(EDGE_COUNT):
            edge_rows.append({
                "archetype_id": archetype,
                "edge_id": edge,
                "F_i": f"F{left[edge]:03d}",
                "F_j": f"F{right[edge]:03d}",
                "node_i": int(left[edge]),
                "node_j": int(right[edge]),
                "mean_edge_weight": float(edge_values[edge]),
                "mean_boundary_contacts": float(edge_values[edge] * ADJACENCIES),
                "within_archetype_rank": int(edge_rank[edge]),
            })
        summary_rows.append({
            "archetype_id": archetype,
            "image_count": int(counts[cluster]),
            "image_share": float(counts[cluster] / n),
            **{f"top_node_{rank + 1}": f"F{node_order[rank]:03d}" for rank in range(3)},
            **{
                f"top_edge_{rank + 1}": f"F{left[edge_order[rank]]:03d}-F{right[edge_order[rank]]:03d}"
                for rank in range(3)
            },
        })
    node_frame = pd.DataFrame(node_rows)
    edge_frame = pd.DataFrame(edge_rows)
    summary = pd.DataFrame(summary_rows)
    save_csv(node_frame, args.output / "prototype_node_weights.csv")
    save_csv(edge_frame, args.output / "prototype_edges.csv")
    save_csv(summary, args.output / "cluster_summary.csv")

    mean_edges = np.load(args.output / "global_mean_edge_weights.npy")
    graph = nx.Graph()
    graph.add_nodes_from(range(NODES))
    for edge, weight in enumerate(mean_edges):
        if weight > 0:
            graph.add_edge(int(left[edge]), int(right[edge]), weight=float(weight))
    layout_graph = nx.maximum_spanning_tree(graph, weight="weight")
    layout_graph.add_nodes_from(range(NODES))
    for edge in np.argsort(-mean_edges)[:256]:
        if mean_edges[edge] > 0:
            layout_graph.add_edge(int(left[edge]), int(right[edge]), weight=float(mean_edges[edge]))
    positions = nx.spring_layout(
        layout_graph, weight="weight", seed=args.seed, iterations=400, k=0.38
    )
    save_json(
        {f"F{node:03d}": [float(xy[0]), float(xy[1])] for node, xy in positions.items()},
        args.output / "global_node_positions.json",
    )

    representative_rows: list[dict] = []
    for cluster in range(8):
        archetype = f"A{cluster + 1:02d}"
        candidates = labels.index[labels["archetype_id"] == archetype].to_numpy(np.int64)
        candidate_distance = labels.loc[candidates, "distance_to_centroid"].to_numpy()
        ordered = candidates[np.argsort(candidate_distance)]
        seen: set[str] = set()
        selected: list[int] = []
        for index in ordered:
            panorama = str(metadata.iloc[index]["panorama_id"])
            if panorama in seen:
                continue
            seen.add(panorama)
            selected.append(int(index))
            if len(selected) == args.representatives:
                break
        if len(selected) < min(args.representatives, len(candidates)):
            raise ValueError(f"could not select representatives for {archetype}")
        for rank, index in enumerate(selected, 1):
            row = metadata.iloc[index]
            representative_rows.append({
                "archetype_id": archetype,
                "rank": rank,
                "image_id": int(row["image_id"]),
                "panorama_id": row["panorama_id"],
                "city": row["city"],
                "image_path": row["image_path"],
                "jpg_offset": int(row["jpg_offset"]),
                "jpg_size": int(row["jpg_size"]),
                "distance_to_centroid": float(labels.iloc[index]["distance_to_centroid"]),
            })
    representatives = pd.DataFrame(representative_rows)
    save_csv(representatives, args.output / "prototype_representative_images.csv")

    city_counts = pd.crosstab(metadata["city"], labels["archetype_id"]).reindex(
        columns=[f"A{x:02d}" for x in range(1, 9)], fill_value=0
    )
    city_rows: list[dict] = []
    for city, row in city_counts.iterrows():
        total = int(row.sum())
        for archetype in city_counts.columns:
            city_rows.append({
                "city": city,
                "archetype_id": archetype,
                "image_count": int(row[archetype]),
                "city_image_count": total,
                "city_prevalence": float(row[archetype] / total),
            })
    save_csv(pd.DataFrame(city_rows), args.output / "city_archetype_prevalence.csv")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": int(n),
        "archetypes": 8,
        "prototype_definition": "mean original 2080D graph descriptor for compact-encoding assignments",
        "representatives_per_archetype": int(args.representatives),
        "representative_panoramas_unique_within_archetype": True,
        "cities": int(metadata["city"].nunique()),
        "fixed_layout_nodes": int(layout_graph.number_of_nodes()),
        "fixed_layout_edges": int(layout_graph.number_of_edges()),
    }
    save_json(report, args.output / "prototype_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--representatives", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
