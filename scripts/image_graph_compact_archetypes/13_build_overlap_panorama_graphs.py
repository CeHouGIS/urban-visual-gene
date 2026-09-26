#!/usr/bin/env python3
"""Build circular 14x56 F maps and graphs from overlap-fused activations."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.image_graph_archetypes.utils import (
    DESCRIPTOR_DIM,
    EDGE_COUNT,
    NODES,
    assert_safe_affinity,
    edge_definition,
    load_fine_labels,
)


GRID = 14
PANORAMA_WIDTH = 56
PANORAMA_PATCHES = GRID * PANORAMA_WIDTH
PANORAMA_ADJACENCIES = GRID * PANORAMA_WIDTH + (GRID - 1) * PANORAMA_WIDTH
DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)


def seam_agreement(maps: np.ndarray) -> float:
    seams = ((13, 14), (27, 28), (41, 42), (55, 0))
    return float(
        np.mean([maps[:, :, left] == maps[:, :, right] for left, right in seams])
    )


def internal_agreement(maps: np.ndarray) -> float:
    seam_columns = {13, 27, 41, 55}
    return float(
        np.mean(
            [
                maps[:, :, column] == maps[:, :, (column + 1) % PANORAMA_WIDTH]
                for column in range(PANORAMA_WIDTH)
                if column not in seam_columns
            ]
        )
    )


def panorama_graph_descriptors(
    maps: np.ndarray, edge_lookup: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(maps, dtype=np.int64)
    if labels.ndim != 3 or labels.shape[1:] != (GRID, PANORAMA_WIDTH):
        raise ValueError(f"expected N x {GRID} x {PANORAMA_WIDTH}, got {labels.shape}")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= NODES:
        raise ValueError("invalid panorama F category ID")
    count = len(labels)
    rows = np.repeat(np.arange(count, dtype=np.int64), PANORAMA_PATCHES)
    node_keys = rows * NODES + labels.reshape(-1)
    node_counts = np.bincount(
        node_keys, minlength=count * NODES
    ).reshape(count, NODES)

    # Horizontal adjacency is circular: the fourth direction reconnects to the
    # first direction. Vertical adjacency is ordinary 4-neighbour adjacency.
    first = np.concatenate(
        (labels.reshape(count, -1), labels[:, :-1, :].reshape(count, -1)), axis=1
    )
    second = np.concatenate(
        (
            np.roll(labels, -1, axis=2).reshape(count, -1),
            labels[:, 1:, :].reshape(count, -1),
        ),
        axis=1,
    )
    if first.shape[1] != PANORAMA_ADJACENCIES:
        raise ValueError("circular panorama must contain 1,512 adjacency pairs")
    different = first != second
    low, high = np.minimum(first, second), np.maximum(first, second)
    edge_ids = edge_lookup[low, high]
    valid_rows, valid_columns = np.nonzero(different)
    valid_edges = edge_ids[valid_rows, valid_columns]
    if len(valid_edges) and valid_edges.min() < 0:
        raise ValueError("invalid circular edge lookup")
    edge_keys = valid_rows.astype(np.int64) * EDGE_COUNT + valid_edges.astype(np.int64)
    edge_counts = np.bincount(
        edge_keys, minlength=count * EDGE_COUNT
    ).reshape(count, EDGE_COUNT)

    output = np.empty((count, DESCRIPTOR_DIM), dtype=np.float32)
    output[:, :NODES] = node_counts / PANORAMA_PATCHES
    output[:, NODES:] = edge_counts / PANORAMA_ADJACENCIES
    return output, different.sum(axis=1).astype(np.int16)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    winners = np.load(args.output / "panorama_top1_dimensions.npy")
    if winners.ndim != 3 or winners.shape[1:] != (GRID, PANORAMA_WIDTH):
        raise ValueError(f"unexpected panorama winner shape {winners.shape}")
    fine_labels = load_fine_labels()
    maps = fine_labels[winners.astype(np.int64)].astype(np.uint8)
    _, _, lookup = edge_definition()
    graph, cross_boundaries = panorama_graph_descriptors(maps, lookup)
    independent = np.load(args.output / "f_category_maps.npy")
    independent = independent.reshape(-1, 4, GRID, GRID).transpose(0, 2, 1, 3)
    independent = independent.reshape(-1, GRID, PANORAMA_WIDTH)
    if independent.shape != maps.shape:
        raise ValueError("independent and fused F maps are not aligned")

    node_error = np.abs(graph[:, :NODES].sum(axis=1) - 1.0)
    reconstructed = graph[:, NODES:].sum(axis=1) * PANORAMA_ADJACENCIES
    if node_error.max(initial=0) >= 1e-6:
        raise ValueError(f"panorama node sum error {node_error.max()}")
    if not np.allclose(reconstructed, cross_boundaries, atol=1e-4):
        raise ValueError("panorama edge weights do not reconstruct contacts")
    if not np.isfinite(graph).all():
        raise ValueError("panorama graph contains NaN or Inf")

    np.save(args.output / "panorama_f_category_maps.npy", maps)
    np.save(args.output / "panorama_graph_features_2080d.npy", graph)
    np.save(args.output / "panorama_cross_category_boundaries.npy", cross_boundaries)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "areas": int(len(maps)),
        "winner_shape": list(winners.shape),
        "f_category_map_shape": list(maps.shape),
        "graph_shape": list(graph.shape),
        "patches_per_panorama": PANORAMA_PATCHES,
        "adjacency_pairs_per_panorama": PANORAMA_ADJACENCIES,
        "horizontal_adjacency": "circular, including column 55 to column 0",
        "independent_f_seam_agreement": seam_agreement(independent),
        "fused_f_seam_agreement": seam_agreement(maps),
        "independent_f_internal_agreement": internal_agreement(independent),
        "fused_f_internal_agreement": internal_agreement(maps),
        "maximum_node_sum_error": float(node_error.max(initial=0)),
        "minimum_cross_category_boundaries": int(cross_boundaries.min()),
        "maximum_cross_category_boundaries": int(cross_boundaries.max()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "overlap_panorama_graph_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
