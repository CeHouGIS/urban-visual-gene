#!/usr/bin/env python3
"""Build N x 2080 node-area plus 4-neighbour edge graph descriptors."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap

from scripts.image_graph_archetypes.utils import (
    ADJACENCIES,
    DESCRIPTOR_DIM,
    EDGE_COUNT,
    FIGURE_ROOT,
    MAIN_HIERARCHY,
    METADATA_PATH,
    NODES,
    OUTPUT_ROOT,
    PATCHES,
    SEED,
    TOP_PATH,
    assert_safe_affinity,
    edge_definition,
    edge_index_frame,
    graph_descriptors,
    load_fine_labels,
    read_metadata,
    read_original,
    save_csv,
    save_json,
    save_parquet,
    validate_descriptor_block,
)


def scan_features(features: np.ndarray, batch_size: int) -> dict[str, float | int]:
    maximum_node_error = 0.0
    nan_count = 0
    inf_count = 0
    negative_count = 0
    non_integer_edge_contact = 0
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        block = np.asarray(features[start:stop])
        maximum_node_error = max(
            maximum_node_error,
            float(np.abs(block[:, :NODES].sum(axis=1) - 1.0).max(initial=0.0)),
        )
        nan_count += int(np.isnan(block).sum())
        inf_count += int(np.isinf(block).sum())
        negative_count += int((block < 0).sum())
        contacts = block[:, NODES:].sum(axis=1) * ADJACENCIES
        non_integer_edge_contact += int((np.abs(contacts - np.rint(contacts)) > 1e-4).sum())
    if maximum_node_error >= 1e-6:
        raise ValueError(f"full-file node sum QA failed: {maximum_node_error}")
    if nan_count or inf_count or negative_count or non_integer_edge_contact:
        raise ValueError(
            "full-file graph QA failed: "
            f"nan={nan_count}, inf={inf_count}, negative={negative_count}, "
            f"non_integer_edge_contacts={non_integer_edge_contact}"
        )
    return {
        "maximum_node_sum_error": maximum_node_error,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "negative_count": negative_count,
        "non_integer_edge_contact_images": non_integer_edge_contact,
    }


def make_qa_figure(
    metadata,
    top: np.ndarray,
    fine_labels: np.ndarray,
    features: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    output: Path,
    samples: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(features), size=min(samples, len(features)), replace=False))
    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2.65 * rows), squeeze=False)
    cmap = plt.get_cmap("turbo", NODES)
    for plot_row, image_index in enumerate(selected):
        row = metadata.iloc[int(image_index)]
        original = read_original(row)
        labels = fine_labels[np.asarray(top[image_index], dtype=np.int64)].reshape(14, 14)
        vector = np.asarray(features[image_index])
        nodes = vector[:NODES]
        edges = vector[NODES:]
        axes[plot_row, 0].imshow(original)
        axes[plot_row, 0].set_title(
            f"image {int(row['image_id'])} | {str(row['city']).split('/')[-1]}", fontsize=9
        )
        axes[plot_row, 0].axis("off")
        axes[plot_row, 1].imshow(labels, cmap=cmap, vmin=-0.5, vmax=NODES - 0.5, interpolation="nearest")
        axes[plot_row, 1].set_title("14×14 frozen F map", fontsize=9)
        axes[plot_row, 1].set_xticks([])
        axes[plot_row, 1].set_yticks([])
        top_nodes = np.argsort(-nodes)[:6]
        top_edges = np.argsort(-edges)[:6]
        node_lines = [f"F{x:03d}: {nodes[x]:.1%}" for x in top_nodes if nodes[x] > 0]
        edge_lines = [
            f"F{edge_left[x]:03d}–F{edge_right[x]:03d}: "
            f"{edges[x]:.4f} ({int(round(edges[x] * ADJACENCIES))})"
            for x in top_edges if edges[x] > 0
        ]
        contacts = int(round(edges.sum() * ADJACENCIES))
        text = (
            "Top nodes (area)\n" + "\n".join(node_lines)
            + "\n\nTop edges (weight; contacts)\n" + "\n".join(edge_lines)
            + f"\n\nnode sum = {nodes.sum():.8f}"
            + f"\nall adjacency pairs = {ADJACENCIES}"
            + f"\ncross-category pairs = {contacts}"
        )
        axes[plot_row, 2].text(0.0, 1.0, text, va="top", family="monospace", fontsize=8)
        axes[plot_row, 2].axis("off")
    fig.suptitle(
        "Image graph QA: original image, F-category map, node areas and adjacency edges",
        fontsize=14,
        y=1.0,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected.astype(int).tolist()


def build(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    args.output.mkdir(parents=True, exist_ok=True)
    top = np.load(args.top_path, mmap_mode="r")
    images = min(args.limit, len(top)) if args.limit else len(top)
    metadata = read_metadata(args.metadata_path, limit=images)
    if len(metadata) != images:
        raise ValueError("metadata and winner-map length mismatch")
    fine_labels = load_fine_labels(args.main_hierarchy)
    edge_left, edge_right, edge_lookup = edge_definition()
    edge_frame = edge_index_frame()
    if len(edge_frame) != EDGE_COUNT:
        raise ValueError("edge index must contain exactly 2016 rows")
    save_csv(edge_frame, args.output / "edge_index.csv")
    save_parquet(metadata, args.output / "image_graph_metadata.parquet")

    final_path = args.output / "image_graph_features.npy"
    partial_path = args.output / "image_graph_features.partial.npy"
    progress_path = args.output / "build_progress.json"
    next_image = 0
    if final_path.is_file():
        existing = np.load(final_path, mmap_mode="r")
        if existing.shape != (images, DESCRIPTOR_DIM) or existing.dtype != np.float32:
            raise ValueError(f"existing graph feature file has wrong contract: {existing.shape}, {existing.dtype}")
        feature_map = existing
        next_image = images
        print(f"reusing completed graph features: {final_path}", flush=True)
    else:
        if partial_path.is_file() and progress_path.is_file():
            feature_map = open_memmap(partial_path, mode="r+")
            progress = json.loads(progress_path.read_text())
            next_image = int(progress["next_image"])
            if feature_map.shape != (images, DESCRIPTOR_DIM):
                raise ValueError("partial feature file shape does not match requested run")
            print(f"resuming graph construction at image {next_image:,}", flush=True)
        else:
            feature_map = open_memmap(
                partial_path, mode="w+", dtype=np.float32,
                shape=(images, DESCRIPTOR_DIM),
            )
        for start in range(next_image, images, args.batch_size):
            stop = min(start + args.batch_size, images)
            block, cross = graph_descriptors(top[start:stop], fine_labels, edge_lookup)
            validate_descriptor_block(block, cross)
            feature_map[start:stop] = block
            feature_map.flush()
            save_json(
                {
                    "images": images,
                    "descriptor_dim": DESCRIPTOR_DIM,
                    "next_image": stop,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                progress_path,
            )
            if start == 0 or stop == images or start % 50000 == 0:
                print(f"graph features {stop:,}/{images:,}", flush=True)
        del feature_map
        os.replace(partial_path, final_path)
        progress_path.unlink(missing_ok=True)
        feature_map = np.load(final_path, mmap_mode="r")

    full_qa = scan_features(feature_map, args.batch_size)
    selected = make_qa_figure(
        metadata, top, fine_labels, feature_map, edge_left, edge_right,
        args.qa_figure, args.qa_samples, args.seed,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": images,
        "nodes": NODES,
        "possible_undirected_edges": EDGE_COUNT,
        "descriptor_dim": DESCRIPTOR_DIM,
        "dtype": "float32",
        "patch_grid": [14, 14],
        "patches_per_image": PATCHES,
        "adjacency": "4-neighbour only",
        "adjacency_pairs_per_image": ADJACENCIES,
        "self_loops": False,
        "edge_normalization": "cross-category contact count / 364",
        "edge_order": "lexicographic upper triangle F000-F001 ... F062-F063",
        "invalid_f_ids": 0,
        "qa_random_image_ids": selected,
        **full_qa,
    }
    save_json(report, args.output / "graph_feature_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-path", type=Path, default=TOP_PATH)
    parser.add_argument("--main-hierarchy", type=Path, default=MAIN_HIERARCHY)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--qa-figure", type=Path, default=FIGURE_ROOT / "Fig_Graph_QA.png")
    parser.add_argument("--qa-samples", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
