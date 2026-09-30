#!/usr/bin/env python3
"""Publication dendrogram for the Feature-MAE 32/64 hierarchy."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram


DEFAULT_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc/mae/"
    "hierarchy_32_64"
)
DEFAULT_OUTPUT = Path(
    "paper/figures/main/Fig_MAE_Hierarchy_Dendrogram_32_64"
)


NATURE_WIDTH_MM = 183
NATURE_HEIGHT_MM = 240

# Restrained, colour-blind-friendly categorical colours. Repetition is safe here
# because category IDs and physical group boundaries carry the identity encoding.
BASE_COLOURS = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#6F4C9B",  # purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#882E72",  # wine
    "#009E73",  # bluish green
    "#6B6B6B",  # grey
)


def color_palette(size: int) -> list[tuple[float, float, float]]:
    colours = []
    for index in range(size):
        base = mpl_colors.to_rgb(BASE_COLOURS[index % len(BASE_COLOURS)])
        cycle = index // len(BASE_COLOURS)
        colours.append(lighten(base, (0.10, 0.20, 0.30, 0.40)[cycle]))
    return colours


def to_hex(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in color)


def lighten(color: tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    return tuple(min(1.0, channel * (1 - amount) + amount) for channel in color)


def english_label(raw: str) -> str:
    """Normalise model-generated labels for publication typography."""
    return " ".join(raw.replace("_", " ").replace("-", " ").split()).capitalize()


def cut_height(linkage: np.ndarray, clusters: int) -> float:
    leaves = linkage.shape[0] + 1
    included = leaves - clusters - 1
    excluded = leaves - clusters
    return float((linkage[included, 2] + linkage[excluded, 2]) / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    linkage = np.load(root / "ward_linkage.npy").astype(np.float64)
    hierarchy = pd.read_csv(root / "feature_hierarchy.csv").sort_values("feature_id")
    labels_payload = json.loads(
        (root / "semantic_labels_qwen" / "category_labels.json").read_text()
    )
    labels = labels_payload["categories"]
    coarse_ids = hierarchy["coarse_cluster"].tolist()
    fine_ids = hierarchy["fine_cluster"].tolist()
    coarse_unique = sorted(set(coarse_ids))
    fine_unique = sorted(set(fine_ids))
    if linkage.shape != (511, 4) or len(coarse_unique) != 32 or len(fine_unique) != 64:
        raise ValueError("expected a 512-leaf Ward tree with 32 coarse and 64 fine clusters")

    # Nature requests Arial or Helvetica. Liberation Sans is the installed,
    # metrically compatible Arial substitute used consistently on this host.
    liberation_dir = Path("/usr/share/fonts/truetype/liberation2")
    for font_file in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"):
        font_path = liberation_dir / font_file
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans"],
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.transparent": False,
    })

    colors = color_palette(32)
    coarse_color = {cluster: colors[index] for index, cluster in enumerate(coarse_unique)}

    node_members: dict[int, set[str]] = {
        feature: {coarse_ids[feature]} for feature in range(512)
    }
    for row_index, row in enumerate(linkage):
        left, right = int(row[0]), int(row[1])
        node_members[512 + row_index] = node_members[left] | node_members[right]

    neutral = "#A6A6A6"

    def branch_color(node: int) -> str:
        members = node_members[int(node)]
        return to_hex(coarse_color[next(iter(members))]) if len(members) == 1 else neutral

    figure = plt.figure(
        figsize=(NATURE_WIDTH_MM / 25.4, NATURE_HEIGHT_MM / 25.4),
        dpi=180,
        facecolor="#FFFFFF",
    )
    grid = GridSpec(
        1, 4,
        width_ratios=[8.7, 0.82, 0.36, 5.5],
        wspace=0.025,
        left=0.065,
        right=0.985,
        top=0.965,
        bottom=0.055,
    )
    ax_tree = figure.add_subplot(grid[0, 0])
    ax_coarse = figure.add_subplot(grid[0, 1], sharey=ax_tree)
    ax_fine = figure.add_subplot(grid[0, 2], sharey=ax_tree)
    ax_labels = figure.add_subplot(grid[0, 3], sharey=ax_tree)
    for axis in (ax_tree, ax_coarse, ax_fine, ax_labels):
        axis.set_facecolor("#FFFFFF")

    result = dendrogram(
        linkage, ax=ax_tree, no_labels=True, link_color_func=branch_color,
        above_threshold_color=neutral, orientation="left",
    )
    for collection in ax_tree.collections:
        collection.set_linewidth(0.38)
        collection.set_alpha(1.0)
    ax_tree.tick_params(
        axis="x", colors="#333333", labelsize=5.5, width=0.4, length=2.0, pad=1.5,
    )
    ax_tree.tick_params(axis="y", left=False, labelleft=False)
    ax_tree.spines[["top", "right", "left"]].set_visible(False)
    ax_tree.spines["bottom"].set_color("#666666")
    ax_tree.spines["bottom"].set_linewidth(0.45)
    ax_tree.set_xlabel("Ward linkage distance", color="#222222", fontsize=6, labelpad=3)
    ax_tree.grid(axis="x", color="#E5E5E5", linewidth=0.3, alpha=1.0)

    height_64 = cut_height(linkage, 64)
    height_32 = cut_height(linkage, 32)
    cut_color = "#222222"
    ax_tree.axvline(
        height_64, color=cut_color, linestyle=(0, (2.0, 1.5)), linewidth=0.55,
    )
    ax_tree.axvline(
        height_32, color=cut_color, linestyle=(0, (5.0, 2.0)), linewidth=0.55,
    )
    ymax = 5120
    ax_tree.text(
        height_64, 0.997, "64-cluster cut", transform=ax_tree.get_xaxis_transform(),
        rotation=90, ha="right", va="top", color="#222222", fontsize=5.5,
    )
    ax_tree.text(
        height_32, 0.997, "32-cluster cut", transform=ax_tree.get_xaxis_transform(),
        rotation=90, ha="right", va="top", color="#222222", fontsize=5.5,
    )

    order = result["leaves"]

    def contiguous_groups(cluster_values: list[str]) -> list[tuple[str, int, int]]:
        ordered = [cluster_values[feature] for feature in order]
        groups = []
        start = 0
        for position in range(1, len(ordered) + 1):
            if position == len(ordered) or ordered[position] != ordered[start]:
                groups.append((ordered[start], start, position))
                start = position
        return groups

    coarse_groups = contiguous_groups(coarse_ids)
    fine_groups = contiguous_groups(fine_ids)
    if len(coarse_groups) != 32 or len(fine_groups) != 64:
        raise ValueError("cluster assignments are not contiguous in Ward leaf order")

    fine_sibling_index: dict[str, tuple[int, int]] = {}
    for coarse_id in coarse_unique:
        children = [
            fine_id for fine_id in fine_unique
            if hierarchy.loc[hierarchy["fine_cluster"] == fine_id, "coarse_cluster"].iloc[0]
            == coarse_id
        ]
        for index, fine_id in enumerate(children):
            fine_sibling_index[fine_id] = (index, len(children))

    for axis in (ax_coarse, ax_fine, ax_labels):
        axis.set_ylim(0, ymax)
        axis.axis("off")
    ax_coarse.set_xlim(0, 1)
    ax_fine.set_xlim(0, 1)
    ax_labels.set_xlim(0, 1)

    for coarse_id, start, end in coarse_groups:
        y0, height = start * 10, (end - start) * 10
        color = coarse_color[coarse_id]
        ax_coarse.add_patch(Rectangle(
            (0, y0), 1, height, facecolor=color, edgecolor="#FFFFFF", linewidth=0.35,
        ))
        if end - start >= 5:
            ax_coarse.text(
                0.5, y0 + height / 2, coarse_id, ha="center", va="center",
                rotation=0, fontsize=5.0, color="#111111", weight="bold",
                clip_on=True,
            )

    for fine_id, start, end in fine_groups:
        y0, height = start * 10, (end - start) * 10
        parent = hierarchy.loc[
            hierarchy["fine_cluster"] == fine_id, "coarse_cluster"
        ].iloc[0]
        sibling, siblings = fine_sibling_index[fine_id]
        amount = 0.12 + (0.28 * sibling / max(1, siblings - 1))
        color = lighten(coarse_color[parent], amount)
        ax_fine.add_patch(Rectangle(
            (0, y0), 1, height, facecolor=color, edgecolor="#FFFFFF", linewidth=0.3,
        ))

    ax_coarse.set_title("32", color="#111111", fontsize=6, pad=5)
    ax_fine.set_title("64", color="#111111", fontsize=6, pad=5)
    ax_labels.set_title(
        "Fine category  (dimensions; parent)", color="#111111", fontsize=6,
        pad=5, loc="left",
    )

    # Even label spacing prevents collisions for small clusters; fine grey leaders
    # retain the exact connection to each cluster's centre on the tree.
    label_rows = []
    for fine_id, start, end in fine_groups:
        label_rows.append((fine_id, (start + end) * 5, end - start))
    target_centres = np.linspace(42, ymax - 42, len(label_rows))
    for (fine_id, true_center, size), label_center in zip(label_rows, target_centres):
        parent = hierarchy.loc[
            hierarchy["fine_cluster"] == fine_id, "coarse_cluster"
        ].iloc[0]
        ax_labels.plot(
            [0.0, 0.075], [true_center, label_center],
            color="#B8B8B8", linewidth=0.28, clip_on=False,
        )
        ax_labels.text(
            0.09, label_center,
            f"{fine_id}  {english_label(labels[fine_id]['name_en'])}  "
            f"({size}; {parent})",
            ha="left", va="center", fontsize=5.0, color="#111111",
        )

    for suffix in (".png", ".svg", ".pdf"):
        figure.savefig(
            output.with_suffix(suffix), dpi=600 if suffix == ".png" else None,
            facecolor=figure.get_facecolor(),
        )
    plt.close(figure)
    for suffix in (".png", ".svg", ".pdf"):
        path = output.with_suffix(suffix)
        print(f"{suffix[1:]}={path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
