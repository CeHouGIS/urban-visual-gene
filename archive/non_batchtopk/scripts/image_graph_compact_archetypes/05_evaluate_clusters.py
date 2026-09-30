#!/usr/bin/env python3
"""Compute one descriptive fixed-sample clustering diagnostic."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from scripts.image_graph_archetypes.utils import SEED, assert_safe_affinity, save_json
from scripts.image_graph_compact_archetypes.utils import OUTPUT_ROOT


def evaluate(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    features = np.load(args.output / "image_graph_compact_features.npy", mmap_mode="r")
    labels = pd.read_parquet(
        args.output / "image_graph_archetype_labels.parquet",
        columns=["raw_cluster_id", "archetype_id"],
    )
    pca = joblib.load(args.output / "pca_model.joblib")
    rng = np.random.default_rng(args.seed)
    n = min(args.sample_size, len(features))
    indices = np.sort(rng.choice(len(features), n, replace=False))
    projected = pca.transform(np.asarray(features[indices], dtype=np.float32))
    raw = labels.iloc[indices]["raw_cluster_id"].to_numpy()
    values = silhouette_samples(projected, raw, metric="euclidean")
    sampled_archetypes = labels.iloc[indices]["archetype_id"].to_numpy()
    by_archetype = {}
    for archetype in sorted(np.unique(sampled_archetypes)):
        selected = sampled_archetypes == archetype
        by_archetype[str(archetype)] = {
            "n": int(selected.sum()),
            "mean": float(values[selected].mean()),
            "negative_share": float((values[selected] < 0).mean()),
        }
    report = {
        "sample_size": int(n),
        "random_seed": int(args.seed),
        "space": "retained PCA coordinates",
        "mean_silhouette": float(values.mean()),
        "median_silhouette": float(np.median(values)),
        "negative_silhouette_share": float((values < 0).mean()),
        "by_archetype": by_archetype,
        "interpretation": "descriptive diagnostic only; not used for K selection",
    }
    save_json(report, args.output / "cluster_quality_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
