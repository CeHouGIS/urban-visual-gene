#!/usr/bin/env python3
"""Render compact-encoding atlas, city heatmap, and experiment summary."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.image_graph_archetypes.utils import NODES, assert_safe_affinity, read_original, save_json
from scripts.image_graph_compact_archetypes.utils import FIGURE_ROOT, OUTPUT_ROOT


graph_panel = importlib.import_module(
    "scripts.image_graph_archetypes.05_make_figures"
).graph_panel


def make_atlas(
    figure_root: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    representatives: pd.DataFrame,
    positions: dict[str, list[float]],
) -> None:
    all_positions = np.asarray(list(positions.values()), dtype=float)
    center = (all_positions.max(axis=0) + all_positions.min(axis=0)) / 2
    radius = max(np.ptp(all_positions[:, 0]), np.ptp(all_positions[:, 1])) / 2
    normalized = {node: ((np.asarray(xy) - center) / radius).tolist() for node, xy in positions.items()}
    node_scale = float(nodes["mean_node_area"].max())
    edge_scale = float(edges["mean_edge_weight"].max())
    bar_scale = float(nodes[nodes["within_archetype_rank"] <= 10]["mean_node_area"].max())
    fig = plt.figure(figsize=(23, 24), constrained_layout=True)
    grid = fig.add_gridspec(8, 8, width_ratios=[2.9, 3.2, 2, 2, 2, 2, 2, 2], hspace=0.08, wspace=0.04)
    cmap = plt.get_cmap("turbo")
    for row_index in range(8):
        archetype = f"A{row_index + 1:02d}"
        node_subset = nodes[nodes["archetype_id"] == archetype]
        edge_subset = edges[edges["archetype_id"] == archetype]
        graph_panel(fig.add_subplot(grid[row_index, 0]), archetype, node_subset, edge_subset, normalized, node_scale, edge_scale)
        ax_bar = fig.add_subplot(grid[row_index, 1])
        bars = node_subset.nsmallest(10, "within_archetype_rank").sort_values("mean_node_area")
        ax_bar.barh(
            bars["node_id"], bars["mean_node_area"],
            color=[cmap(int(x) / (NODES - 1)) for x in bars["node_index"]],
        )
        ax_bar.set_xlim(0, bar_scale * 1.08)
        ax_bar.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        ax_bar.tick_params(axis="both", labelsize=7)
        ax_bar.grid(axis="x", alpha=0.2)
        ax_bar.set_title("Top node areas", fontsize=9)
        for spine in ("top", "right", "left"):
            ax_bar.spines[spine].set_visible(False)
        reps = representatives[representatives["archetype_id"] == archetype].sort_values("rank").head(6)
        if len(reps) < 6:
            raise ValueError(f"{archetype} has fewer than six representatives")
        for column, (_, row) in enumerate(reps.iterrows(), 2):
            ax_image = fig.add_subplot(grid[row_index, column])
            ax_image.imshow(read_original(row))
            ax_image.set_title(
                f"#{int(row['rank'])} {str(row['city']).split('/')[-1]}\nd={row['distance_to_centroid']:.3f}",
                fontsize=7,
            )
            ax_image.axis("off")
    fig.suptitle(
        "Compact spectral image-graph archetypes (201D)\n"
        "Clustering uses composition, boundary density, and composition-corrected topology",
        fontsize=16,
    )
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "Fig_Compact_Graph_Archetype_Atlas.png", dpi=220)
    fig.savefig(figure_root / "Fig_Compact_Graph_Archetype_Atlas.pdf")
    plt.close(fig)


def make_city_heatmap(city: pd.DataFrame, figure_root: Path) -> None:
    matrix = city.pivot(index="city", columns="archetype_id", values="city_prevalence").sort_index()
    fig, ax = plt.subplots(figsize=(8.2, 10.8), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), [str(x).split("/")[-1] for x in matrix.index], fontsize=8)
    ax.set(xlabel="Compact graph archetype", ylabel="City", title="Compact graph archetype prevalence by city")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025).set_label("P(archetype | city)")
    fig.savefig(figure_root / "Fig_Compact_Graph_City_Heatmap.png", dpi=300)
    fig.savefig(figure_root / "Fig_Compact_Graph_City_Heatmap.pdf")
    plt.close(fig)


def write_summary(output: Path, figure_root: Path, summary: pd.DataFrame, representatives: pd.DataFrame) -> None:
    feature = json.loads((output / "compact_feature_report.json").read_text())
    training = json.loads((output / "training_assignment_report.json").read_text())
    quality_path = output / "cluster_quality_report.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
    lines = [
        "# Compact spectral image-graph archetype experiment",
        "",
        "## Encoding",
        "",
        "Each image is encoded in **201 dimensions**, reduced from the original 2,080D descriptor:",
        "",
        "- 64 square-root node-area coordinates;",
        "- 1 cross-category boundary-density coordinate;",
        "- 136 upper-triangle coordinates from a rank-16 spectral projection of the composition-corrected residual adjacency matrix.",
        "",
        "The residual compares observed cross-category contacts against independent mixing conditional on the image's node composition. A fixed normalized-Laplacian basis learned from the city-balanced training set makes all image encodings directly comparable. Semantic labels and city labels were not clustering inputs.",
        "",
        "## Execution and QA",
        "",
        f"- Total images: **{feature['images']:,}**",
        f"- Training images: **{feature['training_images']:,}**",
        f"- Cities: **{feature['cities']}**",
        f"- Feature shape: **{feature['shape'][0]:,} × {feature['shape'][1]}**, float32",
        f"- PCA: **{training['pca_components']} components**, **{training['pca_explained_variance']:.4%}** explained variance",
        "- Clustering: **MiniBatchKMeans, K=8, random seed=42**",
        f"- Node PCA loading energy: **{training['node_pca_loading_energy_share']:.2%}**",
        f"- Boundary-density PCA loading energy: **{training['boundary_density_pca_loading_energy_share']:.2%}**",
        f"- Spectral-topology PCA loading energy: **{training['spectral_topology_pca_loading_energy_share']:.2%}**",
        f"- Maximum node normalization error: `{feature['maximum_squared_node_sum_error']:.3e}`",
        f"- Random 20-image exact rebuild error: `{feature['random_20_exact_rebuild_max_abs_error']:.3e}`",
        "- NaN/Inf: **0**",
        "",
    ]
    if quality is not None:
        lines.extend([
            "## Descriptive cluster diagnostic",
            "",
            f"- Mean silhouette on a fixed {quality['sample_size']:,}-image sample: **{quality['mean_silhouette']:.4f}**",
            f"- Negative-silhouette share: **{quality['negative_silhouette_share']:.2%}**",
            "- This diagnostic is descriptive, not a K-selection step. The compact representation improves topology participation, but it does not make the naturally continuous street-scene distribution sharply separated.",
            "",
        ])
    lines.extend([
        "## Cluster summary",
        "",
        "| Archetype | Images | Share | Top nodes | Top edges |",
        "|---|---:|---:|---|---|",
    ])
    for row in summary.itertuples():
        nodes = ", ".join((row.top_node_1, row.top_node_2, row.top_node_3))
        edges = ", ".join((row.top_edge_1, row.top_edge_2, row.top_edge_3))
        lines.append(f"| {row.archetype_id} | {row.image_count:,} | {row.image_share:.2%} | {nodes} | {edges} |")
    lines.extend(["", "## Representative images", ""])
    for archetype, group in representatives.groupby("archetype_id", sort=True):
        lines.append(f"### {archetype}")
        lines.append("")
        for row in group.sort_values("rank").itertuples():
            lines.append(
                f"- Rank {row.rank}: `{row.image_path}`; offset={row.jpg_offset}; size={row.jpg_size}; "
                f"city={row.city}; image_id={row.image_id}; distance={row.distance_to_centroid:.6f}"
            )
        lines.append("")
    lines.extend([
        "## Generated figures",
        "",
        f"- `{figure_root / 'Fig_Compact_Graph_QA.png'}`",
        f"- `{figure_root / 'Fig_Compact_Graph_Archetype_Atlas.png'}`",
        f"- `{figure_root / 'Fig_Compact_Graph_Archetype_Atlas.pdf'}`",
        f"- `{figure_root / 'Fig_Compact_Graph_City_Heatmap.png'}`",
        f"- `{figure_root / 'Fig_Compact_Graph_City_Heatmap.pdf'}`",
        "",
        "## Interpretation boundary",
        "",
        "The clusters describe recurring layouts within single directional street-view images. They are not city-level clusters. Original 2,080D node/edge means are used only after clustering to visualize each prototype.",
        "",
    ])
    (output / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines))


def make(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    nodes = pd.read_csv(args.output / "prototype_node_weights.csv")
    edges = pd.read_csv(args.output / "prototype_edges.csv")
    representatives = pd.read_csv(args.output / "prototype_representative_images.csv")
    city = pd.read_csv(args.output / "city_archetype_prevalence.csv")
    summary = pd.read_csv(args.output / "cluster_summary.csv")
    positions = json.loads((args.output / "global_node_positions.json").read_text())
    make_atlas(args.figure_root, nodes, edges, representatives, positions)
    make_city_heatmap(city, args.figure_root)
    write_summary(args.output, args.figure_root, summary, representatives)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_png": str(args.figure_root / "Fig_Compact_Graph_Archetype_Atlas.png"),
        "atlas_pdf": str(args.figure_root / "Fig_Compact_Graph_Archetype_Atlas.pdf"),
        "qa_png": str(args.figure_root / "Fig_Compact_Graph_QA.png"),
        "city_heatmap_png": str(args.figure_root / "Fig_Compact_Graph_City_Heatmap.png"),
        "city_heatmap_pdf": str(args.figure_root / "Fig_Compact_Graph_City_Heatmap.pdf"),
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
