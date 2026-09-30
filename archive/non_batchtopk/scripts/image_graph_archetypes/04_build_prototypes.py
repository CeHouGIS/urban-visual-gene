#!/usr/bin/env python3
"""Build mean prototype graphs, representatives, summaries, and city shares."""
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
    OUTPUT_ROOT,
    SEED,
    assert_safe_affinity,
    edge_definition,
    save_csv,
    save_json,
)


def build(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    features = np.load(args.output / "image_graph_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(args.output / "image_graph_metadata.parquet")
    labels = pd.read_parquet(args.output / "image_graph_archetype_labels.parquet")
    training = pd.read_csv(args.output / "training_sample_ids.csv")
    if len(features) != len(metadata) or len(labels) != len(metadata):
        raise ValueError("feature/metadata/label length mismatch")
    if not np.array_equal(labels["image_id"], metadata["image_id"]):
        raise ValueError("labels are not aligned to graph feature rows")
    archetype_index = labels["archetype_id"].str[1:].astype(int).to_numpy() - 1
    if archetype_index.min() != 0 or archetype_index.max() != 7:
        raise ValueError("expected A01--A08 labels")
    counts = np.bincount(archetype_index, minlength=8)
    sums = np.zeros((8, DESCRIPTOR_DIM), dtype=np.float64)
    for start in range(0, len(features), args.batch_size):
        stop = min(start + args.batch_size, len(features))
        block = np.asarray(features[start:stop], dtype=np.float32)
        block_labels = archetype_index[start:stop]
        for cluster in np.unique(block_labels):
            sums[cluster] += block[block_labels == cluster].sum(axis=0, dtype=np.float64)
        if start == 0 or stop == len(features) or start % 50000 == 0:
            print(f"prototype accumulation {stop:,}/{len(features):,}", flush=True)
    prototypes = sums / counts[:, None]
    if not np.allclose(prototypes[:, :NODES].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("prototype node areas do not sum to one")
    left, right, _ = edge_definition()

    node_rows = []
    edge_rows = []
    summary_rows = []
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
            node_rows.append(
                {
                    "archetype_id": archetype,
                    "node_id": f"F{node:03d}",
                    "node_index": node,
                    "mean_node_area": float(node_values[node]),
                    "within_archetype_rank": int(node_rank[node]),
                }
            )
        for edge in range(EDGE_COUNT):
            edge_rows.append(
                {
                    "archetype_id": archetype,
                    "edge_id": edge,
                    "F_i": f"F{left[edge]:03d}",
                    "F_j": f"F{right[edge]:03d}",
                    "node_i": int(left[edge]),
                    "node_j": int(right[edge]),
                    "mean_edge_weight": float(edge_values[edge]),
                    "mean_boundary_contacts": float(edge_values[edge] * ADJACENCIES),
                    "within_archetype_rank": int(edge_rank[edge]),
                }
            )
        top_nodes = [f"F{x:03d}" for x in node_order[:3]]
        top_edges = [f"F{left[x]:03d}-F{right[x]:03d}" for x in edge_order[:3]]
        summary_rows.append(
            {
                "archetype_id": archetype,
                "image_count": int(counts[cluster]),
                "image_share": float(counts[cluster] / len(features)),
                "top_node_1": top_nodes[0],
                "top_node_2": top_nodes[1],
                "top_node_3": top_nodes[2],
                "top_edge_1": top_edges[0],
                "top_edge_2": top_edges[1],
                "top_edge_3": top_edges[2],
            }
        )
    node_frame = pd.DataFrame(node_rows)
    edge_frame = pd.DataFrame(edge_rows)
    summary = pd.DataFrame(summary_rows)
    save_csv(node_frame, args.output / "prototype_node_weights.csv")
    save_csv(edge_frame, args.output / "prototype_edges.csv")
    save_csv(summary, args.output / "cluster_summary.csv")

    # Global mean adjacency from the exact clustering training sample, followed
    # by one and only one spring-layout fit shared by all eight graph panels.
    training_indices = training["feature_row"].to_numpy(np.int64)
    global_edges = np.zeros(EDGE_COUNT, dtype=np.float64)
    for start in range(0, len(training_indices), args.batch_size):
        selected = training_indices[start : start + args.batch_size]
        global_edges += np.asarray(features[selected, NODES:], dtype=np.float32).sum(
            axis=0, dtype=np.float64
        )
    global_edges /= len(training_indices)
    global_graph = nx.Graph()
    global_graph.add_nodes_from(range(NODES))
    for edge, weight in enumerate(global_edges):
        if weight > 0:
            global_graph.add_edge(int(left[edge]), int(right[edge]), weight=float(weight))
    # A nearly complete mean graph collapses spring-layout coordinates into an
    # unreadable ball.  Use its maximum-weight spanning backbone plus the 256
    # strongest mean-adjacency edges only for coordinate estimation.  All
    # prototype values and displayed top edges still come from the unsparsified
    # mean graph descriptors.
    layout_graph = nx.maximum_spanning_tree(global_graph, weight="weight")
    layout_graph.add_nodes_from(range(NODES))
    for edge in np.argsort(-global_edges)[:256]:
        if global_edges[edge] > 0:
            layout_graph.add_edge(
                int(left[edge]), int(right[edge]), weight=float(global_edges[edge])
            )
    positions = nx.spring_layout(
        layout_graph, weight="weight", seed=args.seed, iterations=400, k=0.38
    )
    position_payload = {
        f"F{node:03d}": [float(positions[node][0]), float(positions[node][1])]
        for node in range(NODES)
    }
    save_json(position_payload, args.output / "global_node_positions.json")

    representative_rows = []
    for cluster in range(8):
        archetype = f"A{cluster + 1:02d}"
        candidates = labels.index[labels["archetype_id"] == archetype].to_numpy(np.int64)
        candidate_distances = labels.loc[candidates, "distance_to_centroid"].to_numpy()
        order = candidates[np.argsort(candidate_distances)]
        seen: set[str] = set()
        selected: list[int] = []
        for index in order:
            panorama = str(metadata.iloc[index]["panorama_id"])
            if panorama in seen:
                continue
            seen.add(panorama)
            selected.append(int(index))
            if len(selected) == args.representatives:
                break
        if len(selected) < args.representatives:
            raise ValueError(f"{archetype} has only {len(selected)} unique representatives")
        for rank, index in enumerate(selected, 1):
            source = metadata.iloc[index]
            representative_rows.append(
                {
                    "archetype_id": archetype,
                    "rank": rank,
                    "image_id": int(source["image_id"]),
                    "panorama_id": source["panorama_id"],
                    "city": source["city"],
                    "image_path": source["image_path"],
                    "jpg_offset": int(source["jpg_offset"]),
                    "jpg_size": int(source["jpg_size"]),
                    "distance_to_centroid": float(labels.iloc[index]["distance_to_centroid"]),
                }
            )
    representatives = pd.DataFrame(representative_rows)
    save_csv(representatives, args.output / "prototype_representative_images.csv")

    city_counts = pd.crosstab(metadata["city"], labels["archetype_id"]).reindex(
        columns=[f"A{x:02d}" for x in range(1, 9)], fill_value=0
    )
    city_rows = []
    for city, row in city_counts.iterrows():
        total = int(row.sum())
        for archetype in city_counts.columns:
            city_rows.append(
                {
                    "city": city,
                    "archetype_id": archetype,
                    "image_count": int(row[archetype]),
                    "city_image_count": total,
                    "city_prevalence": float(row[archetype] / total),
                }
            )
    save_csv(pd.DataFrame(city_rows), args.output / "city_archetype_prevalence.csv")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": int(len(features)),
        "archetypes": 8,
        "prototype_definition": "mean original 2080D graph descriptor over assigned images",
        "prototype_node_rows": int(len(node_frame)),
        "prototype_edge_rows": int(len(edge_frame)),
        "fixed_layout_training_images": int(len(training_indices)),
        "fixed_layout_nodes": int(global_graph.number_of_nodes()),
        "global_mean_positive_edges": int(global_graph.number_of_edges()),
        "fixed_layout_edges": int(layout_graph.number_of_edges()),
        "fixed_layout_sparsification": "maximum-weight spanning tree plus top 256 global mean-adjacency edges",
        "representatives_per_archetype": int(args.representatives),
        "representative_panoramas_unique_within_archetype": True,
        "cities": int(metadata["city"].nunique()),
    }
    save_json(report, args.output / "prototype_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--representatives", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
