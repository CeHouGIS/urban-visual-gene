#!/usr/bin/env python3
"""Fit direct PCA and fixed-K MiniBatchKMeans on balanced graph descriptors."""
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
    DESCRIPTOR_DIM,
    OUTPUT_ROOT,
    SEED,
    assert_safe_affinity,
    balanced_training_indices,
    dump_model,
    fit_variance_pca,
    save_csv,
    save_json,
)


def train(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    features = np.load(args.output / "image_graph_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(args.output / "image_graph_metadata.parquet")
    if features.shape != (len(metadata), DESCRIPTOR_DIM):
        raise ValueError("feature/metadata contract mismatch")
    indices = balanced_training_indices(metadata, args.sample_per_city, args.seed)
    sample = metadata.iloc[indices][
        ["image_id", "panorama_id", "city", "image_path"]
    ].copy()
    sample.insert(0, "feature_row", indices)
    if sample.duplicated(["city", "panorama_id"]).any():
        raise ValueError("training sample contains duplicate panoramas within a city")
    save_csv(sample, args.output / "training_sample_ids.csv")
    print(
        f"training sample: {len(indices):,} images, "
        f"{sample['city'].nunique()} cities", flush=True,
    )

    training = np.asarray(features[indices], dtype=np.float32)
    if not np.isfinite(training).all():
        raise ValueError("training graph features contain NaN or Inf")
    pca = fit_variance_pca(training, target=args.variance_target, seed=args.seed)
    projected = pca.transform(training)
    del training
    gc.collect()
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
        n_clusters=8,
        batch_size=4096,
        n_init=20,
        max_iter=300,
        random_state=args.seed,
    )
    training_labels = model.fit_predict(projected)
    dump_model(model, args.output / "kmeans_K8.joblib")
    train_counts = np.bincount(training_labels, minlength=8)
    if (train_counts == 0).any():
        raise ValueError(f"empty training cluster: {train_counts.tolist()}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_images": int(len(indices)),
        "cities": int(sample["city"].nunique()),
        "requested_sample_per_city": int(args.sample_per_city),
        "panorama_deduplicated_within_city": True,
        "random_seed": int(args.seed),
        "input_dimensions": DESCRIPTOR_DIM,
        "standard_scaler": False,
        "hellinger_transform": False,
        "pca_components": pca.n_components_,
        "pca_explained_variance": float(cumulative[-1]),
        "pca_variance_target": float(args.variance_target),
        "algorithm": "MiniBatchKMeans",
        "k": 8,
        "batch_size": 4096,
        "n_init": 20,
        "max_iter": 300,
        "training_cluster_counts_raw": train_counts.tolist(),
        "inertia": float(model.inertia_),
    }
    save_json(report, args.output / "training_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sample-per-city", type=int, default=2000)
    parser.add_argument("--variance-target", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
