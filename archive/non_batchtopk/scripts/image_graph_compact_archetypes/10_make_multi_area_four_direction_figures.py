#!/usr/bin/env python3
"""Build graphs and atlases for ten panoramas completed to four directions."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from scripts.image_graph_archetypes.utils import (
    ADJACENCIES,
    NODES,
    assert_safe_affinity,
    edge_definition,
    graph_descriptors,
    load_fine_labels,
    read_original,
    validate_descriptor_block,
)


DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_GRAPH_ROOT = Path("paper/data/image_graph_compact_archetypes")
DEFAULT_FIGURE_ROOT = Path(
    "paper/figures/supplementary/multi_area_four_directions"
)
DEFAULT_RAW_ACTIVATIONS = Path(
    "outputs/analysis/all_city_umap_activation/image_activations.npy"
)


def make_photo_atlas(
    metadata: pd.DataFrame,
    f_maps: np.ndarray,
    assignments: pd.DataFrame,
    output: Path,
) -> None:
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    fig, axes = plt.subplots(10, 4, figsize=(20, 35), constrained_layout=True)
    for index, (ax, (_, row)) in enumerate(zip(axes.flat, metadata.iterrows())):
        ax.imshow(examples.activation_overlay(read_original(row), f_maps[index], alpha=0.43))
        assigned = assignments.iloc[index]
        ax.set_title(
            f"{int(row['heading'])}° · {assigned['archetype_id']} · "
            f"d={assigned['distance_to_centroid']:.3f}",
            fontsize=8,
        )
        ax.axis("off")
        if int(row["heading"]) == 0:
            city = str(row["city"]).split("/")[-1]
            ax.text(
                -0.03, 0.5,
                f"{row['area_id']}\n{city}\nsource {row['source_archetype_id']}",
                transform=ax.transAxes, ha="right", va="center", fontsize=9,
                fontweight="bold",
            )
    fig.suptitle(
        "Ten representative areas × four directions\n"
        "Feature-MAE F-category activation overlays from the same panorama in each row",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_graph_atlas(
    metadata: pd.DataFrame,
    graph_features: np.ndarray,
    assignments: pd.DataFrame,
    positions: dict,
    output: Path,
) -> None:
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    node_scale = float(graph_features[:, :NODES].max())
    contact_scale = float((graph_features[:, NODES:] * ADJACENCIES).max())
    fig, axes = plt.subplots(10, 4, figsize=(18, 31), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        row = metadata.iloc[index]
        examples.image_graph_panel(
            ax, graph_features[index], positions, node_scale, contact_scale
        )
        ax.set_title(
            f"{int(row['heading'])}° · {assignments.iloc[index]['archetype_id']}",
            fontsize=8,
        )
        if int(row["heading"]) == 0:
            city = str(row["city"]).split("/")[-1]
            ax.text(
                -0.02, 0.5, f"{row['area_id']}\n{city}",
                transform=ax.transAxes, ha="right", va="center", fontsize=9,
                fontweight="bold",
            )
    fig.suptitle(
        "Image graphs for ten areas × four directions\n"
        "Shared node coordinates, node-size scale, and edge-width scale",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_archetype_matrix(
    summary: pd.DataFrame, assignments: pd.DataFrame, output: Path
) -> None:
    matrix = np.empty((len(summary), 4), dtype=np.int16)
    distances = np.empty((len(summary), 4), dtype=np.float32)
    for row_index, area_id in enumerate(summary["area_id"]):
        part = assignments[assignments["area_id"].eq(area_id)].sort_values("heading")
        matrix[row_index] = [int(value[1:]) for value in part["archetype_id"]]
        distances[row_index] = part["distance_to_centroid"].to_numpy()
    colors = plt.get_cmap("tab10")(np.arange(8))
    fig, ax = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
    image = ax.imshow(matrix - 1, cmap=ListedColormap(colors), vmin=-0.5, vmax=7.5)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(
                x, y, f"A{matrix[y, x]:02d}\nd={distances[y, x]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if matrix[y, x] in {1, 2, 4, 6, 7, 8} else "black",
                fontweight="bold",
            )
    ax.set_xticks(range(4), ["0° North", "90° East", "180° South", "270° West"])
    labels = [
        f"{row.area_id} · {str(row.city).split('/')[-1]} · source {row.source_archetype_id}"
        for row in summary.itertuples()
    ]
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Directional graph-archetype assignments within each panorama")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    single = importlib.import_module(
        "scripts.image_graph_compact_archetypes.08_make_four_direction_figure"
    )
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    metadata = pd.read_csv(args.output / "multi_area_direction_metadata.csv")
    winners = np.load(args.output / "top1_dimensions.npy")
    raw_activation = np.load(args.output / "activation_raw.npy")
    if len(metadata) != 40 or winners.shape != (40, 196):
        raise ValueError("expected ten complete four-direction panoramas")
    expected_headings = np.tile([0, 90, 180, 270], 10)
    if not np.array_equal(metadata["heading"].to_numpy(), expected_headings):
        raise ValueError("directions are not ordered 0/90/180/270 within each area")
    if metadata.groupby("area_id")["panorama_id"].nunique().ne(1).any():
        raise ValueError("an area contains more than one panorama")

    fine_labels = load_fine_labels()
    f_maps = fine_labels[winners.astype(np.int64)].reshape(40, 14, 14)
    _, _, lookup = edge_definition()
    graph_features, cross_boundaries = graph_descriptors(winners, fine_labels, lookup)
    graph_qa = validate_descriptor_block(graph_features, cross_boundaries)
    compact, archetypes, distances = single.assign_archetypes(
        graph_features, args.graph_root
    )
    winner_counts = np.stack(
        [np.bincount(row.astype(np.int64), minlength=512) for row in winners]
    ).astype(np.int16)
    mean, scale = single.activation_statistics(args.raw_activations)
    z_scores = np.clip((raw_activation - mean) / scale, -6, 6).astype(np.float32)

    np.save(args.output / "f_category_maps.npy", f_maps.astype(np.uint8))
    np.save(args.output / "graph_features_2080d.npy", graph_features)
    np.save(args.output / "compact_graph_features_201d.npy", compact)
    np.save(args.output / "activation_z.npy", z_scores)
    np.save(args.output / "winner_patch_counts.npy", winner_counts)

    assignments = metadata[
        [
            "result_row", "area_id", "source_archetype_id", "city",
            "panorama_id", "heading", "lat", "lon", "was_in_analysis_cache",
            "cached_top1_match_fraction",
        ]
    ].copy()
    assignments["archetype_id"] = archetypes
    assignments["distance_to_centroid"] = distances
    assignments["active_f_nodes"] = (graph_features[:, :NODES] > 0).sum(axis=1)
    assignments["cross_category_boundaries"] = cross_boundaries
    assignments.to_csv(args.output / "multi_area_direction_assignments.csv", index=False)

    summary_rows = []
    for area_id, part in assignments.groupby("area_id", sort=False):
        first = part.iloc[0]
        row = {
            "area_id": area_id,
            "source_archetype_id": first["source_archetype_id"],
            "city": first["city"],
            "panorama_id": first["panorama_id"],
            "lat": first["lat"], "lon": first["lon"],
            "cached_directions": int(part["was_in_analysis_cache"].sum()),
            "completed_directions": int((~part["was_in_analysis_cache"]).sum()),
        }
        for direction in (0, 90, 180, 270):
            value = part[part["heading"].eq(direction)].iloc[0]
            row[f"heading_{direction}_archetype"] = value["archetype_id"]
            row[f"heading_{direction}_distance"] = value["distance_to_centroid"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "multi_area_summary.csv", index=False)

    positions = examples.normalize_positions(
        json.loads((args.graph_root / "global_node_positions.json").read_text())
    )
    photo_path = args.figure_root / "Fig_Multi_Area_Four_Direction_Atlas.png"
    graph_path = args.figure_root / "Fig_Multi_Area_Four_Direction_Graphs.png"
    matrix_path = args.figure_root / "Fig_Multi_Area_Archetype_Matrix.png"
    make_photo_atlas(metadata, f_maps, assignments, photo_path)
    make_graph_atlas(metadata, graph_features, assignments, positions, graph_path)
    make_archetype_matrix(summary, assignments, matrix_path)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "areas": int(metadata["area_id"].nunique()),
        "cities": int(metadata["city"].nunique()),
        "images": len(metadata),
        "same_panorama_within_each_area": True,
        "headings_per_area": [0, 90, 180, 270],
        "directions_previously_cached": int(metadata["was_in_analysis_cache"].sum()),
        "directions_completed_by_mae": int((~metadata["was_in_analysis_cache"]).sum()),
        "all_40_directions_recomputed_by_mae": True,
        "graph_qa": graph_qa,
        "photo_atlas_png": str(photo_path),
        "graph_atlas_png": str(graph_path),
        "archetype_matrix_png": str(matrix_path),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "multi_area_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 多地区四方向 Feature-MAE / Graph 结果",
        "",
        "本实验选择 10 个不同城市中的代表性 panorama，每处均使用同一点的 0°、90°、180°、270° 四张方向图。原缓存缺少的方向从原始 TAR 补齐，并将全部 40 张图统一通过相同的 DINOv3 与冻结 Feature-MAE 重算。",
        "",
        f"- 地区数：`{report['areas']}`",
        f"- 城市数：`{report['cities']}`",
        f"- 方向图总数：`{report['images']}`",
        f"- 原缓存已有方向：`{report['directions_previously_cached']}`",
        f"- 补齐方向：`{report['directions_completed_by_mae']}`",
        "- 代表性覆盖：A01–A08，并增加伊斯坦布尔、巴黎两个地区",
        "",
        f"照片激活总图：`{photo_path}`",
        "",
        f"Graph 总图：`{graph_path}`",
        "",
        f"方向 archetype 矩阵：`{matrix_path}`",
        "",
        "详细结果见 `multi_area_summary.csv` 和 `multi_area_direction_assignments.csv`。",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--raw-activations", type=Path, default=DEFAULT_RAW_ACTIVATIONS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
