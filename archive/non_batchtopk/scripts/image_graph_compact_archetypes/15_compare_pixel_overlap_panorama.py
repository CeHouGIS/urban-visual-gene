#!/usr/bin/env python3
"""Compare independent, token-overlap, and pixel-overlap panorama maps."""
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
from PIL import Image

from scripts.image_graph_archetypes.utils import edge_definition


GRID = 14
WIDTH = 56
DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_FIGURE_ROOT = Path(
    "paper/figures/supplementary/multi_area_four_directions"
)
DEFAULT_HIERARCHY = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc/"
    "mae/hierarchy_edp_32_64/hierarchy_arrays.npz"
)
DEFAULT_WEB_ASSETS = Path("dashboard/multi_area/assets")


def agreement(maps: np.ndarray, columns: list[int]) -> np.ndarray:
    return np.mean(
        [maps[:, :, column] == maps[:, :, (column + 1) % WIDTH] for column in columns],
        axis=(0, 2),
    )


def draw_map(ax, values: np.ndarray, title: str) -> None:
    ax.imshow(
        values, cmap=plt.get_cmap("turbo", 64), vmin=-0.5, vmax=63.5,
        interpolation="nearest", aspect="auto",
    )
    for boundary in (13.5, 27.5, 41.5):
        ax.axvline(boundary, color="white", linewidth=0.7, alpha=0.45, linestyle=":")
    ax.set_xticks([6.5, 20.5, 34.5, 48.5], ["N", "E", "S", "W"], fontsize=6)
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def make_figure(
    summary: pd.DataFrame,
    independent: np.ndarray,
    token_overlap: np.ndarray,
    pixel_overlap: np.ndarray,
    assets: Path,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(22, 29), constrained_layout=True)
    grid = fig.add_gridspec(10, 4, width_ratios=[5.2, 2.3, 2.3, 2.3], hspace=0.15)
    for index, row in summary.iterrows():
        ax = fig.add_subplot(grid[index, 0])
        image = Image.open(assets / f"{row['area_id']}_four_view_original.webp")
        ax.imshow(image)
        ax.set_xticks(
            [240, 720, 1200, 1680], ["0° N", "90° E", "180° S", "270° W"],
            fontsize=6,
        )
        ax.set_yticks([])
        ax.set_ylabel(
            f"{row['area_id']}\n{str(row['city']).split('/')[-1]}", rotation=0,
            ha="right", va="center", fontsize=8, fontweight="bold", labelpad=8,
        )
        ax.set_title("Four source direction images", fontsize=8)
        draw_map(ax=fig.add_subplot(grid[index, 1]), values=independent[index],
                 title="Independent 4-view inference")
        draw_map(ax=fig.add_subplot(grid[index, 2]), values=token_overlap[index],
                 title="Token-overlap Feature-MAE")
        draw_map(ax=fig.add_subplot(grid[index, 3]), values=pixel_overlap[index],
                 title="Pixel-overlap DINOv3 + Feature-MAE")
    fig.suptitle(
        "Cross-direction consistency comparison\n"
        "Independent views vs token overlap vs pixel-level overlap",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    if hasattr(os, "sched_getaffinity") and set(os.sched_getaffinity(0)).intersection({8, 9}):
        raise RuntimeError("unsafe CPU affinity includes CPU 8 or 9")
    graph_module = importlib.import_module(
        "scripts.image_graph_compact_archetypes.13_build_overlap_panorama_graphs"
    )
    winners = np.load(args.output / "pixel_overlap_panorama_top1_dimensions.npy")
    with np.load(args.hierarchy) as hierarchy:
        fine_labels = hierarchy["fine_labels"].astype(np.int64)
    pixel_maps = fine_labels[winners.astype(np.int64)].astype(np.uint8)
    independent = np.load(args.output / "f_category_maps.npy")
    independent = independent.reshape(-1, 4, GRID, GRID).transpose(0, 2, 1, 3)
    independent = independent.reshape(-1, GRID, WIDTH)
    token_maps = np.load(args.output / "panorama_f_category_maps.npy")
    if not (pixel_maps.shape == independent.shape == token_maps.shape == (10, GRID, WIDTH)):
        raise ValueError("comparison F maps are not aligned")

    _, _, lookup = edge_definition()
    graph, cross_boundaries = graph_module.panorama_graph_descriptors(pixel_maps, lookup)
    node_error = np.abs(graph[:, :64].sum(axis=1) - 1.0)
    if node_error.max(initial=0) >= 1e-6 or not np.isfinite(graph).all():
        raise ValueError("pixel-overlap graph QA failed")
    np.save(args.output / "pixel_overlap_panorama_f_category_maps.npy", pixel_maps)
    np.save(args.output / "pixel_overlap_panorama_graph_features_2080d.npy", graph)
    np.save(
        args.output / "pixel_overlap_panorama_cross_category_boundaries.npy",
        cross_boundaries,
    )

    seam_columns = [13, 27, 41, 55]
    internal_columns = [x for x in range(WIDTH) if x not in seam_columns]
    summary = pd.read_csv(args.output / "multi_area_summary.csv")
    records = []
    methods = {
        "independent": independent,
        "token_overlap": token_maps,
        "pixel_overlap": pixel_maps,
    }
    for method, maps in methods.items():
        seam = agreement(maps, seam_columns)
        internal = agreement(maps, internal_columns)
        for index, row in summary.iterrows():
            records.append(
                {
                    "area_id": row["area_id"],
                    "city": row["city"],
                    "method": method,
                    "f_seam_agreement": float(seam[index]),
                    "f_internal_agreement": float(internal[index]),
                    "seam_gap": float(internal[index] - seam[index]),
                }
            )
    comparison = pd.DataFrame(records)
    comparison.to_csv(args.output / "pixel_overlap_method_comparison.csv", index=False)
    global_metrics = {
        method: {
            "f_seam_agreement": float(agreement(maps, seam_columns).mean()),
            "f_internal_agreement": float(agreement(maps, internal_columns).mean()),
        }
        for method, maps in methods.items()
    }
    for metrics in global_metrics.values():
        metrics["seam_gap"] = metrics["f_internal_agreement"] - metrics["f_seam_agreement"]

    figure = args.figure_root / "Fig_Pixel_Overlap_Panorama_Comparison.png"
    make_figure(summary, independent, token_maps, pixel_maps, args.web_assets, figure)
    inference = json.loads(
        (args.output / "pixel_overlap_panorama_inference_report.json").read_text()
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "areas": 10,
        "methods": list(methods),
        "global_metrics": global_metrics,
        "independent_512d_seam_cosine": inference["independent_seam_cosine"],
        "pixel_overlap_512d_seam_cosine": inference["pixel_overlap_seam_cosine"],
        "independent_winner_seam_agreement": inference[
            "independent_seam_winner_agreement"
        ],
        "pixel_overlap_winner_seam_agreement": inference[
            "pixel_overlap_seam_winner_agreement"
        ],
        "graph_shape": list(graph.shape),
        "maximum_node_sum_error": float(node_error.max(initial=0)),
        "minimum_cross_category_boundaries": int(cross_boundaries.min()),
        "maximum_cross_category_boundaries": int(cross_boundaries.max()),
        "figure": str(figure),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "pixel_overlap_comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Pixel-overlap DINOv3 + Feature-MAE 对照实验",
        "",
        "上一组 10 个地点的四方向图片被构造成 8 个像素级窗口。四个新增窗口分别跨越 0°/90°、90°/180°、180°/270°、270°/0° 接缝；每个窗口完整经过 DINOv3 和冻结 Feature-MAE，然后在 512D 层融合。",
        "",
        f"- 独立方向 512D 接缝余弦相似度：`{inference['independent_seam_cosine']:.4f}`",
        f"- 像素重叠 512D 接缝余弦相似度：`{inference['pixel_overlap_seam_cosine']:.4f}`",
        f"- 独立方向 winner 接缝一致率：`{inference['independent_seam_winner_agreement']:.2%}`",
        f"- 像素重叠 winner 接缝一致率：`{inference['pixel_overlap_seam_winner_agreement']:.2%}`",
        f"- 独立方向 F 接缝一致率：`{global_metrics['independent']['f_seam_agreement']:.2%}`",
        f"- token-overlap F 接缝一致率：`{global_metrics['token_overlap']['f_seam_agreement']:.2%}`",
        f"- pixel-overlap F 接缝一致率：`{global_metrics['pixel_overlap']['f_seam_agreement']:.2%}`",
        f"- 峰值 GPU 显存：`{inference['peak_gpu_gib']:.3f} GiB`",
        "",
        "注意：本原型用相邻方向图各一半组成跨缝窗口，并非从原始 equirectangular panorama 重新渲染真实 45° 透视图。pixel-overlap 的接缝一致率略高于普通内部邻接，说明方法有效，但也可能存在跨缝过度一致化；正式替换前应再用真实 45° 视图验证。",
        "",
        f"对照图：`{figure}`",
        "",
        "逐地点指标：`pixel_overlap_method_comparison.csv`",
    ]
    (args.output / "PIXEL_OVERLAP_EXPERIMENT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--web-assets", type=Path, default=DEFAULT_WEB_ASSETS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
