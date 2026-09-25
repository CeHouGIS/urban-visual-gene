#!/usr/bin/env python3
"""Visualize the fitted PCA space and the final full-data archetype labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.image_graph_archetypes.utils import OUTPUT_ROOT, assert_safe_affinity, save_csv


def run(args: argparse.Namespace) -> None:
    assert_safe_affinity()
    output = args.output
    features = np.load(output / "image_graph_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(output / "image_graph_metadata.parquet")
    labels = pd.read_parquet(output / "image_graph_archetype_labels.parquet")
    pca = joblib.load(output / "pca_model.joblib")
    kmeans = joblib.load(output / "kmeans_K8.joblib")
    mapping = pd.read_csv(output / "cluster_id_mapping.csv")
    if len(features) != len(metadata) or len(labels) != len(features):
        raise ValueError("feature, metadata, and labels lengths do not match")

    # Keep the plot readable while retaining equal representation from every archetype.
    rng = np.random.default_rng(args.seed)
    selected = []
    for archetype in sorted(labels["archetype_id"].unique()):
        ids = labels.index[labels["archetype_id"].eq(archetype)].to_numpy()
        take = min(args.per_archetype, len(ids))
        selected.append(rng.choice(ids, size=take, replace=False))
    selected = np.concatenate(selected)
    selected.sort()
    projected = pca.transform(np.asarray(features[selected], dtype=np.float32))[:, :2]
    coords = labels.iloc[selected][
        ["image_id", "city", "archetype_id", "raw_cluster_id", "distance_to_centroid"]
    ].copy()
    coords["PC1"] = projected[:, 0]
    coords["PC2"] = projected[:, 1]
    save_csv(coords, output / "pca_coordinates_sample.csv")

    prevalence = mapping.sort_values("archetype_index")
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 8))
    color_by_arch = {row.archetype_id: colors[int(row.archetype_index)] for row in prevalence.itertuples()}
    fig = plt.figure(figsize=(13, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(2.15, 1.0, 1.0))
    ax = fig.add_subplot(grid[0, 0])
    for archetype in prevalence["archetype_id"]:
        block = coords[coords["archetype_id"].eq(archetype)]
        ax.scatter(
            block["PC1"], block["PC2"], s=5, alpha=0.28,
            color=color_by_arch[archetype], label=archetype, linewidths=0,
            rasterized=True,
        )
    centers = np.asarray(kmeans.cluster_centers_)[:, :2]
    raw_to_arch = dict(zip(mapping["raw_cluster_id"], mapping["archetype_id"]))
    for raw_id, center in enumerate(centers):
        ax.scatter(center[0], center[1], marker="X", s=90, edgecolor="black",
                   linewidth=0.8, color=color_by_arch[raw_to_arch[raw_id]], zorder=5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Graph archetypes in PCA space")
    ax.legend(title="Archetype", ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    summary = pd.read_csv(output / "pca_summary.csv")
    n_show = len(summary)
    ax.plot(summary["component"].iloc[:n_show], summary["cumulative_explained_variance"].iloc[:n_show] * 100,
            color="#222222", marker="o", markersize=3, linewidth=1.4)
    ax.axhline(90, color="#b23a48", linestyle="--", linewidth=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("PCA variance")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 2])
    ax.bar(prevalence["archetype_id"], prevalence["image_count"],
           color=[color_by_arch[x] for x in prevalence["archetype_id"]])
    ax.set_xlabel("Archetype")
    ax.set_ylabel("Images")
    ax.set_title("Full-data cluster sizes")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Image Graph Archetypes: PCA and K=8 clustering (n={len(features):,}; "
        f"plot sample={len(coords):,})", fontsize=14,
    )
    figure_root = args.figure_root
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "Fig_Graph_Archetype_PCA_Clusters.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_root / "Fig_Graph_Archetype_PCA_Clusters.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=Path("paper/figures/main"))
    parser.add_argument("--per-archetype", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
