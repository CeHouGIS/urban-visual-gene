#!/usr/bin/env python3
"""Build location-level four-view stitches, heatmaps, graphs, and web assets."""
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
from PIL import Image, ImageOps

from scripts.image_graph_archetypes.utils import (
    ADJACENCIES,
    NODES,
    assert_safe_affinity,
    edge_definition,
    read_original,
)


DEFAULT_DATA = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_GRAPH_ROOT = Path("paper/data/image_graph_compact_archetypes")
DEFAULT_FIGURE_ROOT = Path(
    "paper/figures/supplementary/multi_area_four_directions"
)
DEFAULT_WEB_ROOT = Path("dashboard/multi_area")
HEADINGS = (0, 90, 180, 270)
DIRECTION_NAMES = ("North", "East", "South", "West")


def stitch_images(images: list[Image.Image], size: tuple[int, int] = (480, 360)) -> Image.Image:
    fitted = [
        ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        for image in images
    ]
    output = Image.new("RGB", (size[0] * len(fitted), size[1]), "white")
    for index, image in enumerate(fitted):
        output.paste(image, (index * size[0], 0))
    return output


def node_color(node: int) -> str:
    colour = plt.get_cmap("turbo")(node / (NODES - 1))
    return "#%02x%02x%02x" % tuple(round(value * 255) for value in colour[:3])


def make_overview(
    summary: pd.DataFrame,
    stitches: list[Image.Image],
    stitched_maps: np.ndarray,
    area_graphs: np.ndarray,
    positions: dict,
    output: Path,
) -> None:
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    node_scale = float(area_graphs[:, :NODES].max())
    contact_scale = float((area_graphs[:, NODES:] * ADJACENCIES).max())
    cmap = plt.get_cmap("turbo", NODES)
    fig = plt.figure(figsize=(23, 29), constrained_layout=True)
    grid = fig.add_gridspec(10, 3, width_ratios=[6.7, 2.7, 3.2], hspace=0.13)
    for index, row in summary.iterrows():
        ax = fig.add_subplot(grid[index, 0])
        ax.imshow(stitches[index])
        ax.set_xticks(
            [240, 720, 1200, 1680],
            ["0° N", "90° E", "180° S", "270° W"],
            fontsize=7,
        )
        ax.set_yticks([])
        city = str(row["city"]).split("/")[-1]
        ax.set_ylabel(
            f"{row['area_id']}\n{city}", rotation=0, ha="right", va="center",
            fontsize=8, fontweight="bold", labelpad=8,
        )
        ax.set_title("Four-view image stitch", fontsize=8)

        ax = fig.add_subplot(grid[index, 1])
        ax.imshow(
            stitched_maps[index], cmap=cmap, vmin=-0.5, vmax=NODES - 0.5,
            interpolation="nearest", aspect="auto",
        )
        for boundary in (13.5, 27.5, 41.5):
            ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.85)
        ax.set_xticks([6.5, 20.5, 34.5, 48.5], ["N", "E", "S", "W"], fontsize=7)
        ax.set_yticks([])
        ax.set_title("14×56 stitched F heatmap", fontsize=8)

        ax = fig.add_subplot(grid[index, 2])
        examples.image_graph_panel(
            ax, area_graphs[index], positions, node_scale, contact_scale
        )
        ax.set_title("Four-view mean graph", fontsize=8)
    fig.suptitle(
        "Location-level four-direction composition\n"
        "Stitched street views · stitched Feature-MAE heatmap · aggregated image graph",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    metadata = pd.read_csv(args.data / "multi_area_direction_metadata.csv")
    assignments = pd.read_csv(args.data / "multi_area_direction_assignments.csv")
    summary = pd.read_csv(args.data / "multi_area_summary.csv")
    f_maps = np.load(args.data / "f_category_maps.npy")
    graph_features = np.load(args.data / "graph_features_2080d.npy")
    if f_maps.shape != (40, 14, 14) or graph_features.shape != (40, 2080):
        raise ValueError("multi-area feature arrays have unexpected shapes")

    positions = examples.normalize_positions(
        json.loads((args.graph_root / "global_node_positions.json").read_text())
    )
    left, right, _ = edge_definition()
    stitched_maps = []
    area_graphs = []
    original_stitches = []
    web_areas = []
    assets = args.web_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    for area_index, area in enumerate(summary.itertuples()):
        rows = metadata[metadata["area_id"].eq(area.area_id)].sort_values("heading")
        assigned = assignments[assignments["area_id"].eq(area.area_id)].sort_values("heading")
        indices = rows["result_row"].to_numpy(np.int64)
        if rows["heading"].astype(int).tolist() != list(HEADINGS):
            raise ValueError(f"{area.area_id}: incomplete heading order")
        originals = [read_original(row) for _, row in rows.iterrows()]
        overlays = [
            Image.fromarray(examples.activation_overlay(image, f_maps[index], alpha=0.43))
            for image, index in zip(originals, indices)
        ]
        original_stitch = stitch_images(originals)
        overlay_stitch = stitch_images(overlays)
        original_name = f"{area.area_id}_four_view_original.webp"
        overlay_name = f"{area.area_id}_four_view_activation.webp"
        original_stitch.save(assets / original_name, "WEBP", quality=88, method=6)
        overlay_stitch.save(assets / overlay_name, "WEBP", quality=88, method=6)
        original_stitches.append(original_stitch)

        stitched_map = np.concatenate([f_maps[index] for index in indices], axis=1)
        area_graph = graph_features[indices].mean(axis=0, dtype=np.float64).astype(np.float32)
        stitched_maps.append(stitched_map)
        area_graphs.append(area_graph)

        node_values = area_graph[:NODES]
        edge_values = area_graph[NODES:]
        edges = [
            {
                "source": int(left[edge]),
                "target": int(right[edge]),
                "weight": round(float(edge_values[edge]), 7),
                "mean_contacts": round(float(edge_values[edge] * ADJACENCIES), 3),
            }
            for edge in np.flatnonzero(edge_values > 0)
        ]
        nodes = [
            {
                "id": node,
                "label": f"F{node:03d}",
                "area": round(float(node_values[node]), 7),
                "position": [round(float(value), 6) for value in positions[f"F{node:03d}"]],
                "color": node_color(node),
            }
            for node in range(NODES)
        ]
        directions = [
            {
                "heading": int(row.heading),
                "name": DIRECTION_NAMES[index],
                "archetype": str(row.archetype_id),
                "distance": round(float(row.distance_to_centroid), 4),
            }
            for index, row in enumerate(assigned.itertuples())
        ]
        web_areas.append(
            {
                "id": area.area_id,
                "city": str(area.city).split("/")[-1],
                "city_key": area.city,
                "panorama_id": area.panorama_id,
                "coordinates": [round(float(area.lat), 7), round(float(area.lon), 7)],
                "source_archetype": area.source_archetype_id,
                "cached_directions": int(area.cached_directions),
                "completed_directions": int(area.completed_directions),
                "original_image": f"multi_area/assets/{original_name}",
                "activation_image": f"multi_area/assets/{overlay_name}",
                "heatmap": stitched_map.astype(int).ravel().tolist(),
                "heatmap_shape": [14, 56],
                "nodes": nodes,
                "edges": edges,
                "directions": directions,
                "top_nodes": [f"F{node:03d}" for node in np.argsort(-node_values)[:8]],
                "mean_cross_contacts": round(
                    float(edge_values.sum() * ADJACENCIES), 2
                ),
            }
        )

    stitched_maps_array = np.stack(stitched_maps).astype(np.uint8)
    area_graphs_array = np.stack(area_graphs).astype(np.float32)
    np.save(args.data / "area_stitched_f_maps.npy", stitched_maps_array)
    np.save(args.data / "area_aggregated_graph_features.npy", area_graphs_array)
    web_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "areas": 10,
            "cities": 10,
            "direction_images": 40,
            "headings": list(HEADINGS),
            "heatmap_shape": [14, 56],
            "graph_aggregation": "mean node area and mean edge weight over four directions",
        },
        "areas": web_areas,
    }
    (args.web_root / "data.json").write_text(
        json.dumps(web_payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    overview = args.figure_root / "Fig_Multi_Area_Stitched_Heatmap_Graph.png"
    make_overview(
        summary,
        original_stitches,
        stitched_maps_array,
        area_graphs_array,
        positions,
        overview,
    )
    report = {
        "created_utc": web_payload["created_utc"],
        "areas": 10,
        "stitched_images": 10,
        "stitched_heatmap_shape": [10, 14, 56],
        "aggregated_graph_shape": list(area_graphs_array.shape),
        "graph_aggregation": web_payload["experiment"]["graph_aggregation"],
        "overview_png": str(overview),
        "web_data": str(args.web_root / "data.json"),
        "web_assets": str(assets),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.data / "stitched_area_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
