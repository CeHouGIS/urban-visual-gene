#!/usr/bin/env python3
"""Fit PCA/K=8 on the compact encoding and assign every image."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from scripts.image_graph_archetypes.utils import (
    SEED,
    assert_safe_affinity,
    dump_model,
    fit_variance_pca,
    save_csv,
    save_json,
    save_parquet,
)
from scripts.image_graph_compact_archetypes.utils import COMPACT_DIM, NODES, OUTPUT_ROOT


def train_and_assign(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    features = np.load(args.output / "image_graph_compact_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(args.output / "image_graph_metadata.parquet")
    training_frame = pd.read_csv(args.output / "training_sample_ids.csv")
    indices = training_frame["feature_row"].to_numpy(np.int64)
    if features.shape != (len(metadata), COMPACT_DIM):
        raise ValueError("compact feature/metadata contract mismatch")
    training = np.asarray(features[indices], dtype=np.float32)
    if not np.isfinite(training).all():
        raise ValueError("training features contain NaN or Inf")

    pca = fit_variance_pca(
        training, target=args.variance_target, seed=args.seed,
        initial_components=min(200, COMPACT_DIM - 1),
    )
    projected = pca.transform(training)
    dump_model(pca, args.output / "pca_model.joblib")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_summary = pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )
    save_csv(pca_summary, args.output / "pca_summary.csv")

    model = MiniBatchKMeans(
        n_clusters=8, batch_size=4096, n_init=20, max_iter=300,
        random_state=args.seed,
    )
    training_labels = model.fit_predict(projected)
    dump_model(model, args.output / "kmeans_K8.joblib")
    training_counts = np.bincount(training_labels, minlength=8)
    if (training_counts == 0).any():
        raise ValueError(f"empty training cluster: {training_counts.tolist()}")

    # Loading energy is descriptive only; no post-PCA block rescaling is used.
    loading_energy = np.sum(pca.components_ * pca.components_, axis=0)
    energy_total = float(loading_energy.sum())
    block_loading_share = {
        "node_pca_loading_energy_share": float(loading_energy[:NODES].sum() / energy_total),
        "boundary_density_pca_loading_energy_share": float(loading_energy[NODES] / energy_total),
        "spectral_topology_pca_loading_energy_share": float(loading_energy[NODES + 1 :].sum() / energy_total),
    }
    del training, projected
    gc.collect()

    raw_labels = np.empty(len(features), dtype=np.int8)
    distances = np.empty(len(features), dtype=np.float32)
    second_distances = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), args.batch_size):
        stop = min(start + args.batch_size, len(features))
        block = np.asarray(features[start:stop], dtype=np.float32)
        transformed = pca.transform(block)
        all_distances = model.transform(transformed).astype(np.float32)
        closest_two = np.partition(all_distances, 1, axis=1)[:, :2]
        predicted = all_distances.argmin(axis=1)
        raw_labels[start:stop] = predicted
        distances[start:stop] = all_distances[np.arange(len(block)), predicted]
        second_distances[start:stop] = closest_two.max(axis=1)
        if start == 0 or stop == len(features) or stop % 50000 < args.batch_size:
            print(f"assigned {stop:,}/{len(features):,}", flush=True)
    raw_counts = np.bincount(raw_labels, minlength=8)
    if (raw_counts == 0).any():
        raise ValueError(f"empty full-data cluster: {raw_counts.tolist()}")
    order = np.argsort(-raw_counts)
    raw_to_rank = np.empty(8, dtype=np.int8)
    raw_to_rank[order] = np.arange(8, dtype=np.int8)
    ranked = raw_to_rank[raw_labels]
    mapping = pd.DataFrame(
        {
            "raw_cluster_id": order,
            "archetype_index": np.arange(8, dtype=np.int8),
            "archetype_id": [f"A{x:02d}" for x in range(1, 9)],
            "image_count": raw_counts[order],
            "image_share": raw_counts[order] / len(features),
        }
    )
    save_csv(mapping, args.output / "cluster_id_mapping.csv")
    labels = metadata[["image_id", "panorama_id", "city", "image_path"]].copy()
    labels["raw_cluster_id"] = raw_labels
    labels["archetype_id"] = [f"A{x + 1:02d}" for x in ranked]
    labels["distance_to_centroid"] = distances
    labels["second_centroid_distance"] = second_distances
    labels["distance_margin"] = second_distances - distances
    save_parquet(labels, args.output / "image_graph_archetype_labels.parquet")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": int(len(features)),
        "training_images": int(len(indices)),
        "cities": int(metadata["city"].nunique()),
        "input_dimensions": COMPACT_DIM,
        "standard_scaler": False,
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(cumulative[-1]),
        "pca_variance_target": float(args.variance_target),
        "algorithm": "MiniBatchKMeans",
        "k": 8,
        "batch_size": 4096,
        "n_init": 20,
        "max_iter": 300,
        "random_seed": int(args.seed),
        "training_cluster_counts_raw": training_counts.tolist(),
        "raw_cluster_counts": raw_counts.tolist(),
        "archetype_counts": raw_counts[order].tolist(),
        "archetype_shares": (raw_counts[order] / len(features)).tolist(),
        "inertia": float(model.inertia_),
        "distance_median": float(np.median(distances)),
        "distance_margin_median": float(np.median(second_distances - distances)),
        **block_loading_share,
    }
    save_json(report, args.output / "training_assignment_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--variance-target", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_assign(parse_args())
