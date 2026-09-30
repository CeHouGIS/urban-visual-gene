#!/usr/bin/env python3
"""Visualize ten image graphs and their complete 512D activation profiles."""
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
from PIL import Image

from scripts.image_graph_archetypes.utils import (
    ADJACENCIES,
    EDGE_COUNT,
    NODES,
    assert_safe_affinity,
    edge_definition,
    load_fine_labels,
    read_original,
    save_csv,
    save_json,
)
from scripts.image_graph_compact_archetypes.utils import OUTPUT_ROOT, SOURCE_ROOT


DEFAULT_TOP = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc/"
    "mae/hierarchy_32_64/top1_dimensions.npy"
)
DEFAULT_RAW_ACTIVATIONS = Path(
    "outputs/analysis/all_city_umap_activation/image_activations.npy"
)
DEFAULT_STANDARDIZED_ACTIVATIONS = Path(
    "outputs/analysis/all_city_umap_activation/standardized_activations.npy"
)
DEFAULT_FIGURE_ROOT = Path(
    "paper/figures/supplementary/compact_graph_examples"
)
MIN_WINNER_PATCHES = 4


def select_examples(representatives: pd.DataFrame) -> pd.DataFrame:
    """Select A01--A08 rank 1, plus rank 2 from A01 and A02."""
    requested = [(f"A{x:02d}", 1) for x in range(1, 9)] + [("A01", 2), ("A02", 2)]
    selected = []
    for sample_index, (archetype, rank) in enumerate(requested, 1):
        match = representatives[
            (representatives["archetype_id"] == archetype)
            & (representatives["rank"] == rank)
        ]
        if len(match) != 1:
            raise ValueError(f"expected one representative for {archetype} rank {rank}")
        row = match.iloc[0].copy()
        row["sample_id"] = f"S{sample_index:02d}"
        row["selection_reason"] = f"{archetype} centroid representative rank {rank}"
        selected.append(row)
    output = pd.DataFrame(selected).reset_index(drop=True)
    if output["panorama_id"].duplicated().any():
        raise ValueError("ten example panoramas must be unique")
    return output


def normalize_positions(payload: dict[str, list[float]]) -> dict[str, np.ndarray]:
    values = np.asarray(list(payload.values()), dtype=np.float64)
    center = (values.max(axis=0) + values.min(axis=0)) / 2
    radius = max(np.ptp(values[:, 0]), np.ptp(values[:, 1])) / 2
    return {
        node: (np.asarray(position, dtype=np.float64) - center) / max(radius, 1e-12)
        for node, position in payload.items()
    }


def image_graph_panel(
    ax,
    vector: np.ndarray,
    positions: dict[str, np.ndarray],
    node_scale: float,
    contact_scale: float,
    adjacencies: int = ADJACENCIES,
) -> None:
    left, right, _ = edge_definition()
    node_values = vector[:NODES]
    edge_values = vector[NODES:]
    contacts = edge_values * adjacencies
    primary_nodes = set(np.argsort(-node_values)[:12].tolist())
    positive_edges = np.flatnonzero(contacts > 0)
    edge_order = positive_edges[np.argsort(-contacts[positive_edges])[:18]]
    displayed_nodes = primary_nodes.union(left[edge_order].tolist()).union(
        right[edge_order].tolist()
    )
    # Preserve the shared global layout geometry, but crop/rescale the visible
    # subgraph so sparse image graphs do not occupy a tiny corner of the panel.
    visible_ids = sorted(displayed_nodes)
    visible_xy = np.asarray(
        [positions[f"F{node:03d}"] for node in visible_ids], dtype=np.float64
    )
    visible_center = (visible_xy.max(axis=0) + visible_xy.min(axis=0)) / 2
    visible_radius = max(np.ptp(visible_xy[:, 0]), np.ptp(visible_xy[:, 1])) / 2
    panel_positions = {
        f"F{node:03d}": (positions[f"F{node:03d}"] - visible_center)
        / max(visible_radius, 1e-12)
        for node in visible_ids
    }
    for edge in edge_order:
        first, second = int(left[edge]), int(right[edge])
        first_xy = panel_positions[f"F{first:03d}"]
        second_xy = panel_positions[f"F{second:03d}"]
        width = 0.35 + 5.0 * contacts[edge] / max(contact_scale, 1e-12)
        ax.plot(
            [first_xy[0], second_xy[0]], [first_xy[1], second_xy[1]],
            color="#52606d", alpha=0.55, linewidth=width, zorder=1,
        )
    cmap = plt.get_cmap("turbo")
    for node in sorted(displayed_nodes):
        node_id = f"F{node:03d}"
        x, y = panel_positions[node_id]
        primary = node in primary_nodes
        size = 35 + 690 * node_values[node] / max(node_scale, 1e-12)
        ax.scatter(
            x, y, s=size, color=cmap(node / (NODES - 1)),
            edgecolors="white", linewidths=0.7, zorder=2,
            alpha=1.0 if primary else 0.55,
        )
        ax.text(
            x, y, node_id, ha="center", va="center",
            fontsize=4.8 if primary else 4.0,
            alpha=1.0 if primary else 0.65, zorder=3,
        )
    active_nodes = int(np.count_nonzero(node_values))
    cross_contacts = int(round(contacts.sum()))
    ax.text(
        0.02, 0.02,
        f"active F: {active_nodes}\ncross contacts: {cross_contacts}/{adjacencies}",
        transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
    )
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")


def activation_panel(ax, z_scores: np.ndarray, label_top: bool = True) -> None:
    dimensions = np.arange(512)
    clipped = np.clip(z_scores, -3.5, 4.0)
    colors = np.where(z_scores >= 2, "#d1495b", np.where(z_scores > 0, "#3973ac", "#c5c9ce"))
    ax.bar(dimensions, clipped, width=1.0, color=colors, linewidth=0)
    ax.axhline(0, color="#333333", linewidth=0.45)
    ax.axhline(2, color="#d1495b", linewidth=0.7, linestyle="--")
    top = np.argsort(-z_scores)[:6]
    if label_top:
        for dimension in top:
            ax.text(
                dimension, min(float(z_scores[dimension]), 3.85) + 0.08,
                f"D{dimension:03d}", rotation=90, ha="center", va="bottom", fontsize=4.4,
            )
    ax.set_xlim(-1, 512)
    ax.set_ylim(-3.55, 4.35)
    ax.set_xticks([0, 128, 256, 384, 511])
    ax.tick_params(labelsize=5, length=2)
    ax.set_ylabel("z", fontsize=6)
    ax.grid(axis="y", alpha=0.18, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def winner_panel(
    ax, winner_counts: np.ndarray, minimum_patches: int = MIN_WINNER_PATCHES
) -> None:
    dimensions = np.arange(512)
    # Suppress isolated 1--3 patch winners in the visualization. The complete
    # unthresholded 512D count vector remains available in the exported CSV.
    positive = winner_counts >= minimum_patches
    ax.bar(
        dimensions[positive], winner_counts[positive], width=1.4,
        color="#e9a23b", linewidth=0,
    )
    displayed = np.where(positive, winner_counts, 0)
    top = np.argsort(-displayed)[:5]
    for dimension in top:
        if displayed[dimension] > 0:
            ax.text(
                dimension, winner_counts[dimension] + 0.6, f"D{dimension:03d}",
                rotation=90, ha="center", va="bottom", fontsize=4.4,
            )
    ax.set_xlim(-1, 512)
    ax.set_ylim(0, max(5, float(displayed.max(initial=0)) * 1.28))
    ax.set_xticks([0, 128, 256, 384, 511])
    ax.tick_params(labelsize=5, length=2)
    ax.set_ylabel("patches", fontsize=6)
    ax.grid(axis="y", alpha=0.18, linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def activation_overlay(
    original: Image.Image, f_map: np.ndarray, alpha: float = 0.42
) -> np.ndarray:
    """Blend the categorical 14×14 F-winner activation map over the photo."""
    rgb = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    colours = plt.get_cmap("turbo")(f_map / (NODES - 1))[..., :3]
    colour_image = Image.fromarray(
        np.uint8(np.clip(colours, 0, 1) * 255), mode="RGB"
    ).resize(original.size, resample=Image.Resampling.NEAREST)
    overlay = np.asarray(colour_image, dtype=np.float32) / 255.0
    blended = (1.0 - alpha) * rgb + alpha * overlay
    return np.uint8(np.clip(blended, 0, 1) * 255)


def make_atlas(
    selected: pd.DataFrame,
    graph_vectors: np.ndarray,
    f_maps: np.ndarray,
    z_scores: np.ndarray,
    winner_counts: np.ndarray,
    positions: dict[str, np.ndarray],
    output: Path,
) -> None:
    node_scale = float(graph_vectors[:, :NODES].max())
    contact_scale = float((graph_vectors[:, NODES:] * ADJACENCIES).max())
    fig = plt.figure(figsize=(19, 27), constrained_layout=True)
    grid = fig.add_gridspec(
        10, 4, width_ratios=[2.65, 1.45, 2.35, 4.7],
        hspace=0.12, wspace=0.06,
    )
    cmap = plt.get_cmap("turbo", NODES)
    for row_index, row in selected.iterrows():
        ax_original = fig.add_subplot(grid[row_index, 0])
        original = read_original(row)
        ax_original.imshow(activation_overlay(original, f_maps[row_index]))
        city = str(row["city"]).split("/")[-1]
        ax_original.set_title(
            f"{row['sample_id']} · {row['archetype_id']} · {city}\n"
            f"image {int(row['image_id'])}", fontsize=7, fontweight="bold",
        )
        ax_original.axis("off")

        ax_map = fig.add_subplot(grid[row_index, 1])
        ax_map.imshow(
            f_maps[row_index], cmap=cmap, vmin=-0.5, vmax=NODES - 0.5,
            interpolation="nearest",
        )
        ax_map.set_title("14×14 F map", fontsize=7)
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        ax_graph = fig.add_subplot(grid[row_index, 2])
        image_graph_panel(
            ax_graph, graph_vectors[row_index], positions, node_scale, contact_scale
        )
        ax_graph.set_title("Image graph", fontsize=7)

        ax_winner = fig.add_subplot(grid[row_index, 3])
        winner_panel(ax_winner, winner_counts[row_index])
        ax_winner.set_title(
            f"512D spatial support (shown when ≥{MIN_WINNER_PATCHES}/196 patches)",
            fontsize=7,
        )
    fig.suptitle(
        "Ten image-level graph examples with activation overlaid on the photographs\n"
        "Overlay colour = patch-winning F category; node size = F area; edge width = 4-neighbour contacts",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_overlay_atlas(
    selected: pd.DataFrame,
    f_maps: np.ndarray,
    graph_vectors: np.ndarray,
    output: Path,
) -> None:
    """Render a photo-focused 2×5 atlas of categorical activation overlays."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.4), constrained_layout=True)
    for index, (ax, (_, row)) in enumerate(zip(axes.flat, selected.iterrows())):
        original = read_original(row)
        ax.imshow(activation_overlay(original, f_maps[index], alpha=0.45))
        areas = graph_vectors[index, :NODES]
        top = np.argsort(-areas)[:4]
        top_text = " · ".join(f"F{x:03d} {areas[x]:.0%}" for x in top)
        city = str(row["city"]).split("/")[-1]
        ax.set_title(
            f"{row['sample_id']} · {row['archetype_id']} · {city}\n{top_text}",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(
        "Feature-MAE activation overlays for ten graph-archetype representatives\n"
        "Each translucent colour is the frozen F category winning that 14×14 patch",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_activation_heatmap(
    selected: pd.DataFrame, z_scores: np.ndarray, output: Path
) -> None:
    labels = [
        f"{row.sample_id}  {row.archetype_id}  {str(row.city).split('/')[-1]}"
        for row in selected.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(18, 5.5), constrained_layout=True)
    image = ax.imshow(
        np.clip(z_scores, -3, 3), cmap="coolwarm", vmin=-3, vmax=3,
        aspect="auto", interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(labels)), labels)
    ticks = np.arange(0, 512, 32)
    ax.set_xticks(ticks, [f"D{x:03d}" for x in ticks], rotation=90)
    ax.set_xlabel("Feature-MAE dimension")
    ax.set_title("Complete 10 × 512 standardized activation matrix")
    for row_index, values in enumerate(z_scores):
        top = np.argsort(-values)[:5]
        ax.scatter(top, np.full(5, row_index), marker="s", s=8, facecolor="none", edgecolor="black", linewidth=0.45)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    colorbar.set_label("Global standardized activation (z)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_individual_figures(
    selected: pd.DataFrame,
    graph_vectors: np.ndarray,
    f_maps: np.ndarray,
    z_scores: np.ndarray,
    winner_counts: np.ndarray,
    positions: dict[str, np.ndarray],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    node_scale = float(graph_vectors[:, :NODES].max())
    contact_scale = float((graph_vectors[:, NODES:] * ADJACENCIES).max())
    cmap = plt.get_cmap("turbo", NODES)
    for index, row in selected.iterrows():
        fig = plt.figure(figsize=(15.5, 8.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.95])
        ax_original = fig.add_subplot(grid[0, 0])
        ax_original.imshow(read_original(row))
        ax_original.set_title("Original directional street view")
        ax_original.axis("off")
        ax_map = fig.add_subplot(grid[0, 1])
        ax_map.imshow(f_maps[index], cmap=cmap, vmin=-0.5, vmax=NODES - 0.5, interpolation="nearest")
        ax_map.set_title("14×14 map after D→F mapping")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_graph = fig.add_subplot(grid[0, 2])
        image_graph_panel(ax_graph, graph_vectors[index], positions, node_scale, contact_scale)
        ax_graph.set_title("Image graph: top 12 nodes and top 18 edges")
        ax_activation = fig.add_subplot(grid[1, :2])
        activation_panel(ax_activation, z_scores[index])
        ax_activation.set_xlabel("Feature-MAE dimension D000–D511")
        ax_activation.set_title("Continuous activation: mean of strongest 20 patches, standardized globally")
        ax_winner = fig.add_subplot(grid[1, 2])
        winner_panel(ax_winner, winner_counts[index])
        ax_winner.set_xlabel("Feature-MAE dimension")
        ax_winner.set_title("Spatial support: top-1 winner patches")
        city = str(row["city"]).split("/")[-1]
        fig.suptitle(
            f"{row['sample_id']} · {row['archetype_id']} rank {int(row['rank'])} · "
            f"{city} · image {int(row['image_id'])}",
            fontsize=14,
        )
        path = directory / f"{row['sample_id']}_{row['archetype_id']}_image_{int(row['image_id'])}.png"
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    representatives = pd.read_csv(args.output / "prototype_representative_images.csv")
    metadata = pd.read_parquet(args.output / "image_graph_metadata.parquet")
    selected = select_examples(representatives)
    image_ids = selected["image_id"].to_numpy(np.int64)
    if not np.array_equal(metadata.iloc[image_ids]["image_id"].to_numpy(), image_ids):
        raise ValueError("image_id is not aligned with feature rows")

    graph_features = np.load(args.source / "image_graph_features.npy", mmap_mode="r")
    top = np.load(args.top, mmap_mode="r")
    raw_activation = np.load(args.raw_activations, mmap_mode="r")
    standardized = np.load(args.standardized_activations, mmap_mode="r")
    expected_rows = len(metadata)
    if graph_features.shape[0] != expected_rows or top.shape != (expected_rows, 196):
        raise ValueError("graph/winner-map/metadata alignment failure")
    if raw_activation.shape != (expected_rows, 512) or standardized.shape != (expected_rows, 512):
        raise ValueError("activation/metadata alignment failure")

    graph_vectors = np.asarray(graph_features[image_ids], dtype=np.float32)
    winners = np.asarray(top[image_ids], dtype=np.int64)
    fine_labels = load_fine_labels()
    f_maps = fine_labels[winners].reshape(len(selected), 14, 14)
    raw_scores = np.asarray(raw_activation[image_ids], dtype=np.float32)
    z_scores = np.asarray(standardized[image_ids], dtype=np.float32)
    winner_counts = np.stack(
        [np.bincount(row, minlength=512) for row in winners]
    ).astype(np.int16)
    if not np.all(winner_counts.sum(axis=1) == 196):
        raise ValueError("winner counts do not sum to 196")
    positions = normalize_positions(
        json.loads((args.output / "global_node_positions.json").read_text())
    )

    args.data_output.mkdir(parents=True, exist_ok=True)
    metadata_columns = [
        "sample_id", "selection_reason", "archetype_id", "rank", "image_id",
        "panorama_id", "city", "image_path", "jpg_offset", "jpg_size",
        "distance_to_centroid",
    ]
    save_csv(selected[metadata_columns], args.data_output / "example_10_image_metadata.csv")
    dimension_columns = [f"D{x:03d}" for x in range(512)]
    prefix = selected[["sample_id", "archetype_id", "image_id", "city"]].reset_index(drop=True)
    save_csv(
        pd.concat([prefix, pd.DataFrame(raw_scores, columns=dimension_columns)], axis=1),
        args.data_output / "example_10_activation_raw.csv",
    )
    save_csv(
        pd.concat([prefix, pd.DataFrame(z_scores, columns=dimension_columns)], axis=1),
        args.data_output / "example_10_activation_z.csv",
    )
    save_csv(
        pd.concat([prefix, pd.DataFrame(winner_counts, columns=dimension_columns)], axis=1),
        args.data_output / "example_10_winner_patch_counts.csv",
    )

    left, right, _ = edge_definition()
    top_rows = []
    for sample_index, row in selected.iterrows():
        node_values = graph_vectors[sample_index, :NODES]
        edge_values = graph_vectors[sample_index, NODES:]
        for rank, node in enumerate(np.argsort(-node_values)[:10], 1):
            top_rows.append({
                "sample_id": row["sample_id"], "archetype_id": row["archetype_id"],
                "feature_type": "node", "rank": rank, "feature_id": f"F{node:03d}",
                "value": float(node_values[node]), "count": int(round(node_values[node] * 196)),
            })
        for rank, edge in enumerate(np.argsort(-edge_values)[:10], 1):
            top_rows.append({
                "sample_id": row["sample_id"], "archetype_id": row["archetype_id"],
                "feature_type": "edge", "rank": rank,
                "feature_id": f"F{left[edge]:03d}-F{right[edge]:03d}",
                "value": float(edge_values[edge]),
                "count": int(round(edge_values[edge] * ADJACENCIES)),
            })
        for rank, dimension in enumerate(np.argsort(-z_scores[sample_index])[:10], 1):
            top_rows.append({
                "sample_id": row["sample_id"], "archetype_id": row["archetype_id"],
                "feature_type": "activation", "rank": rank,
                "feature_id": f"D{dimension:03d}",
                "value": float(z_scores[sample_index, dimension]),
                "count": int(winner_counts[sample_index, dimension]),
            })
    save_csv(pd.DataFrame(top_rows), args.data_output / "example_10_top_features.csv")

    atlas_path = args.figure_root / "Fig_Compact_Graph_10_Image_Examples.png"
    heatmap_path = args.figure_root / "Fig_Compact_Graph_10x512_Activations.png"
    overlay_path = args.figure_root / "Fig_Compact_Graph_10_Image_Activation_Overlays.png"
    make_atlas(
        selected, graph_vectors, f_maps, z_scores, winner_counts, positions, atlas_path
    )
    make_overlay_atlas(selected, f_maps, graph_vectors, overlay_path)
    make_activation_heatmap(selected, z_scores, heatmap_path)
    make_individual_figures(
        selected, graph_vectors, f_maps, z_scores, winner_counts, positions,
        args.figure_root / "individual",
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": 10,
        "selection": "rank-1 centroid representative from A01-A08 plus rank-2 from A01-A02",
        "panorama_unique": True,
        "graph": "image-specific F graph; node area and 4-neighbour boundary contacts",
        "displayed_graph_nodes": "top 12 area nodes plus endpoints of displayed edges",
        "displayed_graph_edges": "top 18 nonzero edges",
        "graph_layout": "shared global coordinates cropped and rescaled to each visible subgraph",
        "activation_score": "mean of strongest 20 of 196 patch responses per dimension",
        "activation_display": "global standardized activation from cached all-image z-score array",
        "spatial_support": "number of 196 patches for which each D dimension is the top-1 winner",
        "spatial_support_display_threshold": f">={MIN_WINNER_PATCHES} winner patches",
        "atlas_png": str(atlas_path),
        "atlas_pdf": str(atlas_path.with_suffix(".pdf")),
        "overlay_atlas_png": str(overlay_path),
        "overlay_atlas_pdf": str(overlay_path.with_suffix(".pdf")),
        "activation_heatmap_png": str(heatmap_path),
        "activation_heatmap_pdf": str(heatmap_path.with_suffix(".pdf")),
        "individual_figure_directory": str(args.figure_root / "individual"),
        "selected_image_ids": image_ids.tolist(),
    }
    save_json(report, args.data_output / "example_10_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--raw-activations", type=Path, default=DEFAULT_RAW_ACTIVATIONS)
    parser.add_argument("--standardized-activations", type=Path, default=DEFAULT_STANDARDIZED_ACTIVATIONS)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--data-output", type=Path, default=OUTPUT_ROOT / "examples")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
