#!/usr/bin/env python3
"""Visualize Graph archetypes with UMAP on raw graph descriptors."""
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
    features = np.load(output / "image_graph_features.npy", mmap_mode="r")
    metadata = pd.read_parquet(output / "image_graph_metadata.parquet")
    labels = pd.read_parquet(output / "image_graph_archetype_labels.parquet")
    mapping = pd.read_csv(output / "cluster_id_mapping.csv")
    if len(features) != len(metadata) or len(labels) != len(features):
        raise ValueError("feature, metadata, and labels lengths do not match")

    rng = np.random.default_rng(args.seed)
    selected = []
    for archetype in sorted(labels["archetype_id"].unique()):
        ids = labels.index[labels["archetype_id"].eq(archetype)].to_numpy()
        selected.append(rng.choice(ids, size=min(args.per_archetype, len(ids)), replace=False))
    selected = np.concatenate(selected)
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
    coords = labels.iloc[selected][
        ["image_id", "city", "archetype_id", "raw_cluster_id", "distance_to_centroid"]
    ].copy()
    coords["UMAP1"] = embedding[:, 0]
    coords["UMAP2"] = embedding[:, 1]
    save_csv(coords, output / "umap_coordinates_sample.csv")

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 8))
    color_by_arch = {row.archetype_id: colors[int(row.archetype_index)] for row in mapping.itertuples()}
    fig, (ax, size_ax) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": (2.2, 1)}, constrained_layout=True)
    for archetype in mapping.sort_values("archetype_index")["archetype_id"]:
        block = coords[coords["archetype_id"].eq(archetype)]
        ax.scatter(block.UMAP1, block.UMAP2, s=5, alpha=0.30,
                   color=color_by_arch[archetype], label=archetype,
                   linewidths=0, rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Graph archetypes in UMAP space")
    ax.legend(title="Archetype", ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    summary = mapping.sort_values("archetype_index")
    size_ax.bar(summary.archetype_id, summary.image_count,
                color=[color_by_arch[x] for x in summary.archetype_id])
    size_ax.set_xlabel("Archetype")
    size_ax.set_ylabel("Images")
    size_ax.set_title("Full-data cluster sizes")
    size_ax.tick_params(axis="x", rotation=45)
    size_ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Image Graph Archetypes: UMAP and K=8 clustering (n={len(features):,}; "
        f"plot sample={len(coords):,})\n"
        f"raw {features.shape[1]}D descriptors; n_neighbors={args.n_neighbors}, "
        f"min_dist={args.min_dist}, metric={args.metric}", fontsize=13,
    )
    args.figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_root / "Fig_Graph_Archetype_UMAP_Clusters.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.figure_root / "Fig_Graph_Archetype_UMAP_Clusters.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=Path("paper/figures/main"))
    parser.add_argument("--per-archetype", type=int, default=3000)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
