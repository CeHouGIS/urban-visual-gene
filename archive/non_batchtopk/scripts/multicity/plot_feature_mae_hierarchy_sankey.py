#!/usr/bin/env python3
"""Draw the Feature-MAE 32 coarse -> 64 fine hierarchy as a Sankey diagram."""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath


DEFAULT_INPUT = Path(
    "paper/data/feature_mae_hierarchy/category_transitions_32_to_64.csv"
)
DEFAULT_OUTPUT = Path(
    "paper/figures/supplementary/Fig_MAE_Hierarchy_Sankey_32_to_64"
)
DEFAULT_HTML = Path(
    "paper/figures/interactive/Fig_MAE_Hierarchy_Sankey_32_to_64.html"
)


def palette(size: int) -> dict[str, tuple[float, float, float]]:
    colors = {}
    for index in range(size):
        hue = (index * 0.618033988749895 + 0.03) % 1.0
        colors[f"C{index:03d}"] = colorsys.hsv_to_rgb(hue, 0.58, 0.95)
    return colors


def positions(
    ids: list[str], sizes: dict[str, float], top: float, bottom: float, gap: float,
    unit_scale: float,
) -> dict[str, tuple[float, float]]:
    total = sum(sizes[item] * unit_scale for item in ids) + gap * (len(ids) - 1)
    y = top + ((bottom - top) - total) / 2
    result = {}
    for item in ids:
        height = sizes[item] * unit_scale
        result[item] = (y, y + height)
        y += height + gap
    return result


def ribbon(
    ax, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float,
    color: tuple[float, float, float], label: str,
) -> None:
    bend = (x1 - x0) * 0.48
    vertices = [
        (x0, y0a), (x0 + bend, y0a), (x1 - bend, y1a), (x1, y1a),
        (x1, y1b), (x1 - bend, y1b), (x0 + bend, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(vertices, codes), facecolor=color, edgecolor=color,
        linewidth=0.35, alpha=0.34, label=label,
    )
    ax.add_patch(patch)


def draw(data: pd.DataFrame, output: Path, html_output: Path) -> None:
    data = data.sort_values(["粗类编号", "细类编号"]).reset_index(drop=True)
    coarse_ids = data["粗类编号"].drop_duplicates().tolist()
    fine_ids = data["细类编号"].tolist()
    coarse_sizes = data.groupby("粗类编号")["细类维度数"].sum().to_dict()
    fine_sizes = data.set_index("细类编号")["细类维度数"].to_dict()
    coarse_names = data.drop_duplicates("粗类编号").set_index("粗类编号")["粗类名称"].to_dict()
    fine_names = data.set_index("细类编号")["细类名称"].to_dict()

    top, bottom = 0.065, 0.965
    left_gap, right_gap = 0.0060, 0.0026
    total_units = float(data["细类维度数"].sum())
    scale = min(
        ((bottom - top) - left_gap * (len(coarse_ids) - 1)) / total_units,
        ((bottom - top) - right_gap * (len(fine_ids) - 1)) / total_units,
    )
    left_pos = positions(coarse_ids, coarse_sizes, top, bottom, left_gap, scale)
    right_pos = positions(fine_ids, fine_sizes, top, bottom, right_gap, scale)
    colors = palette(len(coarse_ids))

    font_path = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font = font_manager.FontProperties(fname=font_path).get_name()
    else:
        font = "DejaVu Sans"
    plt.rcParams["font.family"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    figure, ax = plt.subplots(figsize=(24, 34), dpi=150)
    figure.patch.set_facecolor("#080d19")
    ax.set_facecolor("#080d19")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")
    x_left, x_right, node_width = 0.285, 0.705, 0.017

    source_offsets = {item: left_pos[item][0] for item in coarse_ids}
    for row in data.itertuples(index=False):
        coarse_id = str(row.粗类编号)
        fine_id = str(row.细类编号)
        value = float(row.细类维度数)
        y0a = source_offsets[coarse_id]
        y0b = y0a + value * scale
        source_offsets[coarse_id] = y0b
        y1a, y1b = right_pos[fine_id]
        ribbon(
            ax, x_left + node_width, x_right, y0a, y0b, y1a, y1b,
            colors[coarse_id],
            f"{coarse_id} {coarse_names[coarse_id]} → {fine_id} {fine_names[fine_id]}: {int(value)}维",
        )

    for coarse_id in coarse_ids:
        y0, y1 = left_pos[coarse_id]
        color = colors[coarse_id]
        ax.add_patch(Rectangle(
            (x_left, y0), node_width, y1 - y0,
            facecolor=color, edgecolor=(1, 1, 1, 0.45), linewidth=0.7,
        ))
        ax.text(
            x_left - 0.008, (y0 + y1) / 2,
            f"{coarse_id}  {coarse_names[coarse_id]}  · {int(coarse_sizes[coarse_id])}维",
            ha="right", va="center", fontsize=7.3, color="#E7EDF7",
        )

    for row in data.itertuples(index=False):
        coarse_id = str(row.粗类编号)
        fine_id = str(row.细类编号)
        y0, y1 = right_pos[fine_id]
        color = colors[coarse_id]
        light = tuple(min(1.0, channel * 0.78 + 0.22) for channel in color)
        ax.add_patch(Rectangle(
            (x_right, y0), node_width, y1 - y0,
            facecolor=light, edgecolor=(1, 1, 1, 0.5), linewidth=0.65,
        ))
        ax.text(
            x_right + node_width + 0.008, (y0 + y1) / 2,
            f"{fine_id}  {fine_names[fine_id]}  · {int(row.细类维度数)}维",
            ha="left", va="center", fontsize=7.0, color="#E7EDF7",
        )

    figure.text(
        0.5, 0.982, "Feature-MAE 类别演化：32 个粗类 → 64 个细类",
        ha="center", va="top", color="#F3F7FC", fontsize=24, weight="bold",
    )
    figure.text(
        0.5, 0.966,
        "同一棵 Ward 层级树 · 连线宽度 = 细类包含的潜变量维度数 · 总计 512 维",
        ha="center", va="top", color="#9FB0C8", fontsize=11,
    )
    figure.text(
        0.27, 0.946, "32 个粗类", ha="center", color="#47D7FF", fontsize=12, weight="bold"
    )
    figure.text(
        0.73, 0.946, "64 个细类", ha="center", color="#55E6AE", fontsize=12, weight="bold"
    )
    figure.text(
        0.5, 0.012,
        "类别名称：Qwen3-VL-2B 对每类统计中心维度的 Top-10 图像进行后验命名；名称不参与聚类。",
        ha="center", va="bottom", color="#7F91AA", fontsize=9,
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.945, bottom=0.025)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=180, facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".pdf"), facecolor=figure.get_facecolor())
    plt.close(figure)

    svg = output.with_suffix(".svg").read_text(encoding="utf-8")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feature-MAE 32→64 Sankey</title><style>
html,body{{margin:0;background:#080d19;color:#e7edf7;font-family:system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:18px}}
.tip{{position:sticky;top:8px;z-index:2;width:max-content;max-width:90%;margin:0 auto 12px;padding:8px 13px;border:1px solid #26354e;border-radius:9px;background:#101a2dcc;color:#9fb0c8;font-size:12px;backdrop-filter:blur(8px)}}
.chart{{overflow:auto;border:1px solid #1c2940;border-radius:12px;background:#080d19}}svg{{display:block;width:100%;height:auto;min-width:1100px}}
</style></head><body><main><div class="tip">可缩放浏览器查看；连线宽度表示细类包含的维度数量。</div><div class="chart">{svg}</div></main></body></html>"""
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html-output", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    required = {"粗类编号", "粗类名称", "细类编号", "细类名称", "细类维度数"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if data["细类维度数"].sum() != 512:
        raise ValueError("link values must sum to 512 dimensions")
    draw(data, args.output, args.html_output)
    for suffix in (".png", ".svg", ".pdf"):
        path = args.output.with_suffix(suffix)
        print(f"{suffix[1:]}={path.resolve()} ({path.stat().st_size:,} bytes)")
    print(f"html={args.html_output.resolve()} ({args.html_output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
