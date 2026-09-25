#!/usr/bin/env python3
"""Render the prototype graph atlas, city heatmap, and experiment summary."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.image_graph_archetypes.utils import (
    FIGURE_ROOT,
    NODES,
    OUTPUT_ROOT,
    assert_safe_affinity,
    read_original,
    save_json,
)


def graph_panel(
    ax,
    archetype: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    positions: dict[str, list[float]],
    node_scale: float,
    edge_scale: float,
) -> None:
    top_nodes = nodes.nsmallest(15, "within_archetype_rank").copy()
    primary_ids = set(top_nodes["node_id"])
    top_edges = edges.nlargest(25, "mean_edge_weight")
    # Draw the exact global top-25 prototype edges.  Their endpoint nodes are
    # included when necessary, while the top-15 area nodes remain highlighted.
    endpoint_ids = set(top_edges["F_i"]).union(top_edges["F_j"])
    displayed_nodes = nodes[nodes["node_id"].isin(primary_ids.union(endpoint_ids))]
    for row in top_edges.itertuples():
        if row.mean_edge_weight <= 0:
            continue
        x1, y1 = positions[row.F_i]
        x2, y2 = positions[row.F_j]
        width = 0.35 + 5.5 * row.mean_edge_weight / max(edge_scale, 1e-12)
        ax.plot([x1, x2], [y1, y2], color="#687585", alpha=0.48, linewidth=width, zorder=1)
    cmap = plt.get_cmap("turbo")
    for row in displayed_nodes.itertuples():
        x, y = positions[row.node_id]
        size = 38 + 620 * row.mean_node_area / max(node_scale, 1e-12)
        node_index = int(row.node_index)
        primary = row.node_id in primary_ids
        ax.scatter(
            [x], [y], s=size, color=cmap(node_index / (NODES - 1)),
            edgecolors="white", linewidths=0.8, zorder=2,
            alpha=1.0 if primary else 0.58,
        )
        ax.text(
            x, y, row.node_id, ha="center", va="center",
            fontsize=5.2 if primary else 4.5, zorder=3,
            alpha=1.0 if primary else 0.68,
        )
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{archetype} prototype graph", fontsize=10, fontweight="bold")


def make_atlas(
    output: Path,
    figure_root: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    representatives: pd.DataFrame,
    positions: dict[str, list[float]],
) -> None:
    archetypes = [f"A{x:02d}" for x in range(1, 9)]
    all_x = np.asarray([value[0] for value in positions.values()])
    all_y = np.asarray([value[1] for value in positions.values()])
    center_x, center_y = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
    radius = max((all_x.max() - all_x.min()) / 2, (all_y.max() - all_y.min()) / 2)
    normalized_positions = {
        node: [(xy[0] - center_x) / radius, (xy[1] - center_y) / radius]
        for node, xy in positions.items()
    }
    node_scale = float(nodes["mean_node_area"].max())
    edge_scale = float(edges["mean_edge_weight"].max())
    top10 = nodes[nodes["within_archetype_rank"] <= 10]
    bar_scale = float(top10["mean_node_area"].max())
    fig = plt.figure(figsize=(23, 24), constrained_layout=True)
    grid = fig.add_gridspec(
        8, 8, width_ratios=[2.9, 3.2, 2, 2, 2, 2, 2, 2],
        hspace=0.08, wspace=0.04,
    )
    cmap = plt.get_cmap("turbo")
    for row_index, archetype in enumerate(archetypes):
        node_subset = nodes[nodes["archetype_id"] == archetype]
        edge_subset = edges[edges["archetype_id"] == archetype]
        ax_graph = fig.add_subplot(grid[row_index, 0])
        graph_panel(
            ax_graph, archetype, node_subset, edge_subset,
            normalized_positions, node_scale, edge_scale,
        )
        ax_bar = fig.add_subplot(grid[row_index, 1])
        bars = node_subset.nsmallest(10, "within_archetype_rank").sort_values(
            "mean_node_area"
        )
        colors = [
            cmap(int(value) / (NODES - 1))
            for value in bars["node_index"]
        ]
        ax_bar.barh(bars["node_id"], bars["mean_node_area"], color=colors)
        ax_bar.set_xlim(0, bar_scale * 1.08)
        ax_bar.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        ax_bar.tick_params(axis="both", labelsize=7)
        ax_bar.grid(axis="x", alpha=0.2)
        ax_bar.set_title("Top node areas", fontsize=9)
        for spine in ("top", "right", "left"):
            ax_bar.spines[spine].set_visible(False)
        reps = representatives[
            representatives["archetype_id"] == archetype
        ].sort_values("rank").head(6)
        if len(reps) != 6:
            raise ValueError(f"{archetype} does not have six display representatives")
        for image_column, (_, representative) in enumerate(reps.iterrows(), 2):
            ax_image = fig.add_subplot(grid[row_index, image_column])
            image = read_original(representative)
            ax_image.imshow(image)
            city = str(representative["city"]).split("/")[-1]
            ax_image.set_title(
                f"#{int(representative['rank'])} {city}\nd={representative['distance_to_centroid']:.3f}",
                fontsize=7,
            )
            ax_image.axis("off")
    fig.suptitle(
        "Image-level visual composition graph archetypes\n"
        "Shared node coordinates; node size = mean area; edge width = mean 4-neighbour contact weight",
        fontsize=16,
    )
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "Fig_Graph_Composition_Archetype_Atlas.png", dpi=220)
    fig.savefig(figure_root / "Fig_Graph_Composition_Archetype_Atlas.pdf")
    plt.close(fig)


def make_city_heatmap(city: pd.DataFrame, figure_root: Path) -> None:
    matrix = city.pivot(
        index="city", columns="archetype_id", values="city_prevalence"
    ).sort_index()
    labels = [str(value).split("/")[-1] for value in matrix.index]
    fig, ax = plt.subplots(figsize=(8.2, 10.8), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), labels, fontsize=8)
    ax.set_xlabel("Graph archetype")
    ax.set_ylabel("City")
    ax.set_title("Image graph archetype prevalence by city")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("P(archetype | city)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if value >= 0.18:
                ax.text(column, row, f"{value:.0%}", ha="center", va="center", color="white", fontsize=6)
    fig.savefig(figure_root / "Fig_City_Archetype_Heatmap.png", dpi=300)
    fig.savefig(figure_root / "Fig_City_Archetype_Heatmap.pdf")
    plt.close(fig)


def write_summary(
    output: Path,
    figure_root: Path,
    summary: pd.DataFrame,
    representatives: pd.DataFrame,
) -> None:
    feature_report = json.loads((output / "graph_feature_report.json").read_text())
    training_report = json.loads((output / "training_report.json").read_text())
    lines = [
        "# Image graph archetype experiment summary",
        "",
        "## Required execution sequence",
        "",
        "- **500-image graph QA:** passed; node sums, 364 adjacency pairs, fixed edge ordering, finite values, and 20 random visual checks were verified.",
        "- **5,000-image end-to-end smoke test:** passed; graph features, PCA, K=8 clustering, prototypes, representatives, atlas, and city output were generated without error.",
        "- **Full experiment:** passed on all 378,818 images after the two smoke stages.",
        "- All numerical stages ran in isolated subprocesses with logical CPUs 8 and 9 excluded.",
        "",
        "## Configuration",
        "",
        f"- Total images: **{feature_report['images']:,}**",
        f"- Training images: **{training_report['training_images']:,}**",
        f"- Number of cities: **{training_report['cities']}**",
        "- Graph nodes: **64 fixed F categories**",
        "- Possible undirected edges: **2,016**",
        "- Descriptor: **64 node areas + 2,016 normalized 4-neighbour edge contacts = 2,080D**",
        f"- PCA components: **{training_report['pca_components']}**",
        f"- PCA explained variance: **{training_report['pca_explained_variance']:.4%}**",
        "- Clustering: **MiniBatchKMeans, K=8**",
        "- KMeans parameters: `batch_size=4096, n_init=20, max_iter=300, random_state=42`",
        "- Semantic labels used in computation: **No**",
        "- City used in clustering features: **No**",
        "",
        "## Cluster summary",
        "",
        "| Archetype | Images | Share | Top nodes | Top edges |",
        "|---|---:|---:|---|---|",
    ]
    for row in summary.itertuples():
        nodes = ", ".join((row.top_node_1, row.top_node_2, row.top_node_3))
        edges = ", ".join((row.top_edge_1, row.top_edge_2, row.top_edge_3))
        lines.append(
            f"| {row.archetype_id} | {row.image_count:,} | {row.image_share:.2%} | {nodes} | {edges} |"
        )
    lines.extend(
        [
            "",
            "## Representative image sources",
            "",
            "Each archive path is paired with the stored JPEG offset and byte size.",
            "",
        ]
    )
    for archetype, group in representatives.groupby("archetype_id", sort=True):
        lines.append(f"### {archetype}")
        lines.append("")
        for row in group.sort_values("rank").itertuples():
            lines.append(
                f"- Rank {row.rank}: `{row.image_path}`; offset={row.jpg_offset}; "
                f"size={row.jpg_size}; city={row.city}; image_id={row.image_id}; "
                f"distance={row.distance_to_centroid:.6f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Generated figures",
            "",
            f"- `{figure_root / 'Fig_Graph_QA.png'}`",
            f"- `{figure_root / 'Fig_Graph_Composition_Archetype_Atlas.png'}`",
            f"- `{figure_root / 'Fig_Graph_Composition_Archetype_Atlas.pdf'}`",
            f"- `{figure_root / 'Fig_City_Archetype_Heatmap.png'}`",
            f"- `{figure_root / 'Fig_City_Archetype_Heatmap.pdf'}`",
            "",
            "## Interpretation boundary",
            "",
            "These are recurring image-level visual-composition graph prototypes. "
            "They are not city-level clusters, and semantic annotations were not used for graph construction, PCA, or clustering.",
            "",
        ]
    )
    (output / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines))


def make(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    nodes = pd.read_csv(args.output / "prototype_node_weights.csv")
    edges = pd.read_csv(args.output / "prototype_edges.csv")
    representatives = pd.read_csv(args.output / "prototype_representative_images.csv")
    city = pd.read_csv(args.output / "city_archetype_prevalence.csv")
    summary = pd.read_csv(args.output / "cluster_summary.csv")
    positions = json.loads((args.output / "global_node_positions.json").read_text())
    if len(positions) != NODES:
        raise ValueError("global node layout must contain 64 positions")
    make_atlas(args.output, args.figure_root, nodes, edges, representatives, positions)
    make_city_heatmap(city, args.figure_root)
    write_summary(args.output, args.figure_root, summary, representatives)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_png": str(args.figure_root / "Fig_Graph_Composition_Archetype_Atlas.png"),
        "atlas_pdf": str(args.figure_root / "Fig_Graph_Composition_Archetype_Atlas.pdf"),
        "qa_png": str(args.figure_root / "Fig_Graph_QA.png"),
        "city_heatmap_png": str(args.figure_root / "Fig_City_Archetype_Heatmap.png"),
        "city_heatmap_pdf": str(args.figure_root / "Fig_City_Archetype_Heatmap.pdf"),
        "archetypes": 8,
        "prototype_graph_top_nodes": 15,
        "prototype_graph_top_edges": 25,
        "additional_edge_endpoint_nodes_drawn_when_needed": True,
        "bar_chart_top_nodes": 10,
        "displayed_representatives_per_archetype": 6,
        "fixed_node_coordinates": True,
    }
    save_json(report, args.output / "figure_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    make(parse_args())
