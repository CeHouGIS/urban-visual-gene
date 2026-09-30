#!/usr/bin/env python3
"""Build a dense, composition-aware alternative to the raw Graph descriptor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.image_graph_archetypes.utils import EDGE_COUNT, NODES, OUTPUT_ROOT


def run(args: argparse.Namespace) -> None:
    source = np.load(args.output / "image_graph_features.npy", mmap_mode="r")
    if source.shape[1] != NODES + EDGE_COUNT:
        raise ValueError(f"unexpected descriptor width: {source.shape[1]}")

    # The scalar preserves how much boundary contact exists after normalizing
    # the edge composition within each image.
    edge_sums = np.empty(len(source), dtype=np.float32)
    chunk = 8192
    for start in range(0, len(source), chunk):
        stop = min(start + chunk, len(source))
        edge_sums[start:stop] = np.asarray(source[start:stop, NODES:], dtype=np.float32).sum(axis=1)
    log_density = np.log1p(edge_sums / max(float(np.median(edge_sums)), 1e-8))
    density_mean = float(log_density.mean())
    density_std = float(log_density.std()) or 1.0

    output_path = args.output / args.filename
    dense = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32,
        shape=(len(source), NODES + EDGE_COUNT + 1),
    )
    alpha = float(args.alpha)
    node_prior = alpha / NODES
    edge_prior = alpha / EDGE_COUNT
    for start in range(0, len(source), chunk):
        stop = min(start + chunk, len(source))
        block = np.asarray(source[start:stop], dtype=np.float32)
        nodes = (block[:, :NODES] + node_prior) / (1.0 + alpha)
        edges = block[:, NODES:]
        edge_total = edges.sum(axis=1, keepdims=True)
        edge_comp = (edges + edge_prior) / (edge_total + alpha)
        dense[start:stop, :NODES] = np.sqrt(nodes)
        dense[start:stop, NODES:NODES + EDGE_COUNT] = np.sqrt(edge_comp)
        dense[start:stop, -1] = (log_density[start:stop] - density_mean) / density_std
    dense.flush()
    metadata = {
        "source": "image_graph_features.npy",
        "shape": [int(x) for x in dense.shape],
        "transform": "sqrt of Dirichlet-smoothed node composition and per-image edge composition, plus standardized log edge density",
        "alpha": alpha,
        "node_dimensions": NODES,
        "edge_dimensions": EDGE_COUNT,
        "density_dimension": 1,
        "density_median_raw": float(np.median(edge_sums)),
    }
    (args.output / args.metadata).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--filename", default="image_graph_features_smoothed.npy")
    parser.add_argument("--metadata", default="image_graph_features_smoothed.json")
    parser.add_argument("--alpha", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
