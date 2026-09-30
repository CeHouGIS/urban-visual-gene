#!/usr/bin/env python3
"""Assign every image graph to K=8 and relabel clusters by prevalence."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.image_graph_archetypes.utils import (
    DESCRIPTOR_DIM,
    OUTPUT_ROOT,
    assert_safe_affinity,
    save_csv,
    save_json,
    save_parquet,
)


def assign(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    features = np.load(args.output / "image_graph_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(args.output / "image_graph_metadata.parquet")
    pca = joblib.load(args.output / "pca_model.joblib")
    kmeans = joblib.load(args.output / "kmeans_K8.joblib")
    if features.shape != (len(metadata), DESCRIPTOR_DIM):
        raise ValueError("feature/metadata contract mismatch")
    raw_labels = np.empty(len(features), dtype=np.int8)
    distances = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), args.batch_size):
        stop = min(start + args.batch_size, len(features))
        block = np.asarray(features[start:stop], dtype=np.float32)
        projected = pca.transform(block)
        all_distances = kmeans.transform(projected).astype(np.float32)
        predicted = all_distances.argmin(axis=1)
        raw_labels[start:stop] = predicted
        distances[start:stop] = all_distances[np.arange(len(block)), predicted]
        if start == 0 or stop == len(features) or start % 50000 == 0:
            print(f"assigned {stop:,}/{len(features):,}", flush=True)
    if not np.isfinite(distances).all():
        raise ValueError("non-finite centroid distance")
    raw_counts = np.bincount(raw_labels, minlength=8)
    if (raw_counts == 0).any():
        raise ValueError(f"empty full-data cluster: {raw_counts.tolist()}")
    prevalence_order = np.argsort(-raw_counts)
    raw_to_rank = np.empty(8, dtype=np.int8)
    raw_to_rank[prevalence_order] = np.arange(8, dtype=np.int8)
    ranked = raw_to_rank[raw_labels]
    mapping = pd.DataFrame(
        {
            "raw_cluster_id": prevalence_order,
            "archetype_index": np.arange(8, dtype=np.int8),
            "archetype_id": [f"A{x + 1:02d}" for x in range(8)],
            "image_count": raw_counts[prevalence_order],
            "image_share": raw_counts[prevalence_order] / len(features),
        }
    )
    save_csv(mapping, args.output / "cluster_id_mapping.csv")
    labels = metadata[
        ["image_id", "panorama_id", "city", "image_path"]
    ].copy()
    labels["raw_cluster_id"] = raw_labels
    labels["archetype_id"] = [f"A{x + 1:02d}" for x in ranked]
    labels["distance_to_centroid"] = distances
    save_parquet(labels, args.output / "image_graph_archetype_labels.parquet")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": int(len(features)),
        "k": 8,
        "raw_cluster_counts": raw_counts.tolist(),
        "prevalence_order_raw_cluster_ids": prevalence_order.tolist(),
        "archetype_counts": mapping["image_count"].astype(int).tolist(),
        "archetype_shares": mapping["image_share"].astype(float).tolist(),
        "distance_min": float(distances.min()),
        "distance_median": float(np.median(distances)),
        "distance_max": float(distances.max()),
    }
    save_json(report, args.output / "assignment_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    assign(parse_args())
