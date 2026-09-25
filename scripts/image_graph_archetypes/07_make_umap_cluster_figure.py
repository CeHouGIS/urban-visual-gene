#!/usr/bin/env python3
"""Visualize raw Graph descriptors with UMAP."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from scripts.image_graph_archetypes.utils import OUTPUT_ROOT, assert_safe_affinity, save_csv


def run(args: argparse.Namespace) -> None:
    assert_safe_affinity()
    output = args.output
    features = np.load(output / args.features, mmap_mode="r")
    metadata = pd.read_parquet(output / "image_graph_metadata.parquet")
    if len(features) != len(metadata):
        raise ValueError("feature and metadata lengths do not match")

    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(features), size=min(args.sample_size, len(features)), replace=False)
    selected.sort()
    raw = np.asarray(features[selected], dtype=np.float32)
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        n_components=2,
        verbose=True,
    )
    embedding = reducer.fit_transform(raw)
    coords = metadata.iloc[selected][["image_id", "city"]].copy()
    coords["UMAP1"] = embedding[:, 0]
    coords["UMAP2"] = embedding[:, 1]
    save_csv(coords, output / "umap_coordinates_sample.csv")

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.scatter(coords.UMAP1, coords.UMAP2, s=5, alpha=0.30,
               color="#277da1", linewidths=0, rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Graph descriptors in UMAP space")
    ax.spines[["top", "right"]].set_visible(False)
    descriptor_label = "smoothed composition descriptors" if "smoothed" in args.features else "raw Graph descriptors"
    fig.suptitle(
        f"Image Graph Descriptors: UMAP (n={len(features):,}; "
        f"plot sample={len(coords):,})\n"
        f"{descriptor_label}, {features.shape[1]}D; n_neighbors={args.n_neighbors}, "
        f"min_dist={args.min_dist}, metric={args.metric}", fontsize=13,
    )
    args.figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_root / args.figure_name, dpi=220, bbox_inches="tight")
    fig.savefig(args.figure_root / args.figure_name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=Path("paper/figures/main"))
    parser.add_argument("--features", default="image_graph_features.npy")
    parser.add_argument("--figure-name", default="Fig_Graph_Archetype_UMAP.png")
    parser.add_argument("--sample-size", type=int, default=24000)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
