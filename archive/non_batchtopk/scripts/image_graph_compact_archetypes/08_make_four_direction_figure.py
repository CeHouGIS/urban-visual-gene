#!/usr/bin/env python3
"""Encode and visualize four completed directions from one panorama."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from scripts.image_graph_compact_archetypes.utils import compact_encode


DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/four_direction_area"
)
DEFAULT_GRAPH_ROOT = Path("paper/data/image_graph_compact_archetypes")
DEFAULT_RAW_ACTIVATIONS = Path(
    "outputs/analysis/all_city_umap_activation/image_activations.npy"
)
DEFAULT_FIGURE_ROOT = Path(
    "paper/figures/supplementary/four_direction_area"
)
MIN_WINNER_PATCHES = 4
DIRECTION_NAMES = {0: "North", 90: "East", 180: "South", 270: "West"}


def activation_statistics(path: Path, block_size: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    scores = np.load(path, mmap_mode="r")
    total = np.zeros(512, dtype=np.float64)
    for start in range(0, len(scores), block_size):
        total += np.asarray(scores[start : start + block_size], dtype=np.float32).sum(
            axis=0, dtype=np.float64
        )
    mean = total / len(scores)
    variance = np.zeros(512, dtype=np.float64)
    for start in range(0, len(scores), block_size):
        block = np.asarray(scores[start : start + block_size], dtype=np.float32)
        variance += np.sum((block - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.sqrt(variance / max(1, len(scores) - 1)) + 1e-6
    return mean.astype(np.float32), scale.astype(np.float32)


def assign_archetypes(
    graph_features: np.ndarray, graph_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.load(graph_root / "global_graph_spectral_basis.npy")
    compact, _ = compact_encode(graph_features, basis)
    pca = joblib.load(graph_root / "pca_model.joblib")
    model = joblib.load(graph_root / "kmeans_K8.joblib")
    projected = pca.transform(compact)
    distances = model.transform(projected).astype(np.float32)
    raw = distances.argmin(axis=1).astype(np.int8)
    mapping = pd.read_csv(graph_root / "cluster_id_mapping.csv")
    raw_to_id = dict(zip(mapping["raw_cluster_id"], mapping["archetype_id"]))
    archetypes = np.asarray([raw_to_id[int(label)] for label in raw], dtype=object)
    return compact, archetypes, distances[np.arange(len(raw)), raw]


def save_dimension_table(
    metadata: pd.DataFrame,
    values: np.ndarray,
    path: Path,
) -> None:
    prefix = metadata[["direction_id", "heading", "panorama_id", "city"]].reset_index(drop=True)
    columns = [f"D{x:03d}" for x in range(512)]
    pd.concat([prefix, pd.DataFrame(values, columns=columns)], axis=1).to_csv(
        path, index=False
    )


def make_result_stitch(
    metadata: pd.DataFrame,
    graph_features: np.ndarray,
    f_maps: np.ndarray,
    winner_counts: np.ndarray,
    archetypes: np.ndarray,
    distances: np.ndarray,
    positions: dict,
    output: Path,
) -> None:
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    node_scale = float(graph_features[:, :NODES].max())
    contact_scale = float((graph_features[:, NODES:] * ADJACENCIES).max())
    cmap = plt.get_cmap("turbo", NODES)
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=[2.2, 1.15, 1.8, 1.25])
    for column, row in metadata.iterrows():
        heading = int(row["heading"])
        original = read_original(row)
        title = (
            f"{heading}° · {DIRECTION_NAMES[heading]} · {archetypes[column]}\n"
            f"centroid distance {distances[column]:.3f}"
        )
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(examples.activation_overlay(original, f_maps[column], alpha=0.43))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

        ax = fig.add_subplot(grid[1, column])
        ax.imshow(
            f_maps[column], cmap=cmap, vmin=-0.5, vmax=NODES - 0.5,
            interpolation="nearest",
        )
        top = np.argsort(-graph_features[column, :NODES])[:4]
        top_text = " · ".join(
            f"F{x:03d} {graph_features[column, x]:.0%}" for x in top
        )
        ax.set_title(f"14×14 F map\n{top_text}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(grid[2, column])
        examples.image_graph_panel(
            ax, graph_features[column], positions, node_scale, contact_scale
        )
        ax.set_title("Image graph", fontsize=8)

        ax = fig.add_subplot(grid[3, column])
        examples.winner_panel(
            ax, winner_counts[column], minimum_patches=MIN_WINNER_PATCHES
        )
        ax.set_title(
            f"512D winner support (shown at ≥{MIN_WINNER_PATCHES}/196 patches)",
            fontsize=7,
        )
        ax.set_xlabel("Feature-MAE dimension", fontsize=6)
    first = metadata.iloc[0]
    fig.suptitle(
        "One panorama completed to four directions with the frozen Feature-MAE\n"
        f"{first['city']} · panorama {first['panorama_id']} · "
        f"{float(first['lat']):.6f}, {float(first['lon']):.6f}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_photo_stitch(
    metadata: pd.DataFrame, f_maps: np.ndarray, output: Path
) -> None:
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
    for index, (ax, (_, row)) in enumerate(zip(axes, metadata.iterrows())):
        heading = int(row["heading"])
        ax.imshow(examples.activation_overlay(read_original(row), f_maps[index], alpha=0.43))
        ax.set_title(f"{heading}° · {DIRECTION_NAMES[heading]}", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "Four-direction Feature-MAE activation-overlay stitch (same panorama)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    metadata = pd.read_csv(args.output / "four_direction_metadata.csv")
    if tuple(metadata["heading"].astype(int)) != (0, 90, 180, 270):
        raise ValueError("four-direction metadata is not ordered 0/90/180/270")
    winners = np.load(args.output / "top1_dimensions.npy")
    latent = np.load(args.output / "feature_mae_latent.npy", mmap_mode="r")
    raw_activation = np.load(args.output / "activation_raw.npy")
    if winners.shape != (4, 196) or latent.shape != (4, 196, 512):
        raise ValueError("unexpected four-direction Feature-MAE array shape")

    fine_labels = load_fine_labels()
    f_maps = fine_labels[winners.astype(np.int64)].reshape(4, 14, 14)
    _, _, lookup = edge_definition()
    graph_features, cross_boundaries = graph_descriptors(winners, fine_labels, lookup)
    graph_qa = validate_descriptor_block(graph_features, cross_boundaries)
    compact, archetypes, distances = assign_archetypes(graph_features, args.graph_root)
    winner_counts = np.stack(
        [np.bincount(row.astype(np.int64), minlength=512) for row in winners]
    ).astype(np.int16)
    if not np.all(winner_counts.sum(axis=1) == 196):
        raise ValueError("winner patch counts do not sum to 196")
    mean, scale = activation_statistics(args.raw_activations)
    z_scores = np.clip((raw_activation - mean) / scale, -6, 6).astype(np.float32)

    np.save(args.output / "f_category_maps.npy", f_maps.astype(np.uint8))
    np.save(args.output / "graph_features_2080d.npy", graph_features)
    np.save(args.output / "compact_graph_features_201d.npy", compact)
    np.save(args.output / "activation_z.npy", z_scores)
    save_dimension_table(metadata, raw_activation, args.output / "activation_raw.csv")
    save_dimension_table(metadata, z_scores, args.output / "activation_z.csv")
    save_dimension_table(metadata, winner_counts, args.output / "winner_patch_counts.csv")

    assignments = metadata[
        [
            "direction_id", "heading", "panorama_id", "city", "lat", "lon",
            "was_in_analysis_cache", "cached_top1_match_fraction",
        ]
    ].copy()
    assignments["archetype_id"] = archetypes
    assignments["distance_to_centroid"] = distances
    assignments["active_f_nodes"] = (graph_features[:, :NODES] > 0).sum(axis=1)
    assignments["cross_category_boundaries"] = cross_boundaries
    assignments.to_csv(args.output / "four_direction_assignments.csv", index=False)

    top_rows = []
    left, right, _ = edge_definition()
    for index, row in metadata.iterrows():
        for rank, node in enumerate(np.argsort(-graph_features[index, :NODES])[:10], 1):
            top_rows.append(
                {
                    "direction_id": row["direction_id"], "heading": row["heading"],
                    "feature_type": "node", "rank": rank,
                    "feature_id": f"F{node:03d}",
                    "value": graph_features[index, node],
                    "count": round(graph_features[index, node] * 196),
                }
            )
        edge_values = graph_features[index, NODES:]
        for rank, edge in enumerate(np.argsort(-edge_values)[:10], 1):
            top_rows.append(
                {
                    "direction_id": row["direction_id"], "heading": row["heading"],
                    "feature_type": "edge", "rank": rank,
                    "feature_id": f"F{left[edge]:03d}-F{right[edge]:03d}",
                    "value": edge_values[edge],
                    "count": round(edge_values[edge] * ADJACENCIES),
                }
            )
    pd.DataFrame(top_rows).to_csv(args.output / "top_graph_features.csv", index=False)

    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    positions = examples.normalize_positions(
        json.loads((args.graph_root / "global_node_positions.json").read_text())
    )
    result_path = args.figure_root / "Fig_Four_Direction_Result_Stitch.png"
    photo_path = args.figure_root / "Fig_Four_Direction_Photo_Activation_Stitch.png"
    make_result_stitch(
        metadata, graph_features, f_maps, winner_counts, archetypes,
        distances, positions, result_path,
    )
    make_photo_stitch(metadata, f_maps, photo_path)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city": str(metadata.iloc[0]["city"]),
        "panorama_id": str(metadata.iloc[0]["panorama_id"]),
        "coordinates": [float(metadata.iloc[0]["lat"]), float(metadata.iloc[0]["lon"])],
        "headings": metadata["heading"].astype(int).tolist(),
        "same_panorama_for_all_directions": metadata["panorama_id"].nunique() == 1,
        "directions_previously_cached": int(metadata["was_in_analysis_cache"].sum()),
        "directions_completed_by_mae": int((~metadata["was_in_analysis_cache"]).sum()),
        "all_four_directions_recomputed_by_mae": True,
        "archetypes": archetypes.tolist(),
        "distance_to_centroid": distances.astype(float).tolist(),
        "winner_display_threshold": MIN_WINNER_PATCHES,
        "graph_qa": graph_qa,
        "result_stitch_png": str(result_path),
        "result_stitch_pdf": str(result_path.with_suffix(".pdf")),
        "photo_activation_stitch_png": str(photo_path),
        "photo_activation_stitch_pdf": str(photo_path.with_suffix(".pdf")),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "four_direction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    guide = f"""# 同一地点四方向 Feature-MAE / Graph 结果

- 城市：`{report['city']}`
- Panorama：`{report['panorama_id']}`
- 坐标：`{report['coordinates'][0]:.6f}, {report['coordinates'][1]:.6f}`
- 方向：`0° / 90° / 180° / 270°`
- 原分析缓存已有：`{report['directions_previously_cached']}` 个方向
- 从原始 TAR 补算：`{report['directions_completed_by_mae']}` 个方向
- 一致性处理：四个方向均重新通过同一 DINOv3 + 冻结 Feature-MAE
- 四方向 archetype：`{' / '.join(report['archetypes'])}`

主结果图：`{result_path}`

照片激活拼接图：`{photo_path}`

主图按列固定为北、东、南、西；按行依次是照片与 F 激活叠加、14×14 F map、image graph、过滤小支持后的 512D winner 激活。512D 图只显示至少占 4/196 patches 的维度，完整未过滤计数保留在 `winner_patch_counts.csv`。
"""
    (args.output / "README.md").write_text(guide)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--raw-activations", type=Path, default=DEFAULT_RAW_ACTIVATIONS)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
