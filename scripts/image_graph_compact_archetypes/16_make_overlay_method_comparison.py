#!/usr/bin/env python3
"""Plot source panoramas and F-category overlays for three inference methods."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from PIL import Image

from scripts.image_graph_archetypes.utils import assert_safe_affinity, read_original


DEFAULT_DATA = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_FIGURE = Path(
    "paper/figures/supplementary/multi_area_four_directions/"
    "Fig_Panorama_F_Category_Overlay_Comparison.png"
)
ALPHA = 0.43


def direction_stitch(
    images: list[Image.Image], f_map: np.ndarray | None, alpha: float
) -> Image.Image:
    examples = importlib.import_module(
        "scripts.image_graph_compact_archetypes.06_make_example_graphs"
    )
    stitched = importlib.import_module(
        "scripts.image_graph_compact_archetypes.11_build_stitched_area_results"
    )
    if f_map is None:
        panels = images
    else:
        if f_map.shape != (14, 56):
            raise ValueError(f"expected a 14x56 F map, got {f_map.shape}")
        panels = [
            Image.fromarray(
                examples.activation_overlay(
                    image, f_map[:, direction * 14 : (direction + 1) * 14],
                    alpha=alpha,
                )
            )
            for direction, image in enumerate(images)
        ]
    return stitched.stitch_images(panels)


def configure_axis(ax, image: Image.Image, title: str | None, ylabel: str) -> None:
    ax.imshow(image)
    for boundary in (479.5, 959.5, 1439.5):
        ax.axvline(boundary, color="white", linewidth=0.65, alpha=0.6, linestyle=":")
    ax.set_xticks(
        [240, 720, 1200, 1680], ["0° N", "90° E", "180° S", "270° W"],
        fontsize=5.5,
    )
    ax.set_yticks([])
    ax.set_ylabel(
        ylabel, rotation=0, ha="right", va="center", fontsize=7.5,
        fontweight="bold", labelpad=8,
    )
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=7)


def run(args: argparse.Namespace) -> None:
    assert_safe_affinity()
    metadata = pd.read_csv(args.data / "multi_area_direction_metadata.csv")
    summary = pd.read_csv(args.data / "multi_area_summary.csv")
    independent = np.load(args.data / "f_category_maps.npy")
    independent = independent.reshape(-1, 4, 14, 14).transpose(0, 2, 1, 3)
    independent = independent.reshape(-1, 14, 56)
    token_overlap = np.load(args.data / "panorama_f_category_maps.npy")
    pixel_overlap = np.load(args.data / "pixel_overlap_panorama_f_category_maps.npy")
    if not (
        independent.shape == token_overlap.shape == pixel_overlap.shape == (10, 14, 56)
    ):
        raise ValueError("F-category maps are not aligned across methods")

    methods = (
        ("Original four-direction panorama", None),
        ("Independent inference overlay", independent),
        ("Token-overlap overlay", token_overlap),
        ("Pixel-overlap DINOv3 + MAE overlay", pixel_overlap),
    )
    fig, axes = plt.subplots(
        10, 4, figsize=(31, 26), constrained_layout=True,
        gridspec_kw={"wspace": 0.025, "hspace": 0.12},
    )
    for area_index, row in summary.iterrows():
        rows = metadata[metadata["area_id"].eq(row["area_id"])].sort_values("heading")
        if rows["heading"].astype(int).tolist() != [0, 90, 180, 270]:
            raise ValueError(f"{row['area_id']}: incomplete direction images")
        images = [read_original(source) for _, source in rows.iterrows()]
        city = str(row["city"]).split("/")[-1]
        label = f"{row['area_id']}\n{city}"
        for method_index, (title, maps) in enumerate(methods):
            panorama = direction_stitch(
                images, None if maps is None else maps[area_index], args.alpha
            )
            configure_axis(
                axes[area_index, method_index], panorama,
                title if area_index == 0 else None,
                label if method_index == 0 else "",
            )

    cmap = plt.get_cmap("turbo", 64)
    normalizer = BoundaryNorm(np.arange(-0.5, 64.5), cmap.N)
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colorbar = fig.colorbar(
        scalar, ax=axes, location="bottom", fraction=0.012, pad=0.012,
        aspect=90, ticks=[0, 8, 16, 24, 32, 40, 48, 56, 63],
    )
    colorbar.ax.set_xticklabels(
        ["F000", "F008", "F016", "F024", "F032", "F040", "F048", "F056", "F063"]
    )
    colorbar.set_label("Fine visual-element category (shared colour scale)", fontsize=9)
    fig.suptitle(
        "Original panorama and F-category activation overlays\n"
        f"Shared 64-category colours · overlay opacity {args.alpha:.0%}",
        fontsize=16,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=190, facecolor="white")
    fig.savefig(args.figure.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(args.figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    return args


if __name__ == "__main__":
    run(parse_args())
