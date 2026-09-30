#!/usr/bin/env python3
"""Name the 32/64 Feature-MAE hierarchy with a compact local Qwen VLM.

The statistical hierarchy is fixed before this script runs.  Qwen only sees ten
representative ORIGINAL/HEAT/HOTSPOT examples per category and attaches a
post-hoc human-readable name.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import io
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageStat

from scripts.multicity.config import city_slug
from scripts.multicity.interpret_sae_dimensions import (
    DEFAULT_MODEL_DIR,
    DimensionInterpreter,
    extract_json_object,
    write_json_atomic,
)


DEFAULT_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)
PROMPT_VERSION = "feature-mae-hierarchy-qwen-v1-10-images"
PATCH_GRID = 14
SEMANTIC_TYPES = {
    "object", "material", "color_light", "texture", "geometry",
    "spatial_layout", "environment", "artifact", "mixed", "unclear",
}


class OriginalResolver:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames: dict[str, pd.DataFrame] = {}

    def row(self, city: str, image_index: int) -> pd.Series:
        if city not in self.frames:
            path = self.root / "filtered_manifests" / f"{city_slug(city)}.parquet"
            self.frames[city] = pd.read_parquet(path).set_index(
                "source_image_index", drop=False
            )
        return self.frames[city].loc[image_index]

    def read(self, city: str, image_index: int) -> Image.Image:
        row = self.row(city, image_index)
        descriptor = os.open(str(row["tar_path"]), os.O_RDONLY)
        try:
            payload = os.pread(
                descriptor, int(row["jpg_size"]), int(row["jpg_offset"])
            )
        finally:
            os.close(descriptor)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")


def category_nodes(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for coarse in taxonomy["children"]:
        for fine in coarse["children"]:
            nodes.append({
                "id": fine["id"], "level": "fine", "parent": coarse["id"],
                "features": fine["features"], "n": fine["n"],
                "stability": fine["stability"],
                "medoid_feature": fine["medoid_feature"],
            })
    for coarse in taxonomy["children"]:
        nodes.append({
            "id": coarse["id"], "level": "coarse", "parent": "ROOT",
            "features": [f for child in coarse["children"] for f in child["features"]],
            "n": coarse["n"], "stability": coarse["stability"],
            "child_ids": [child["id"] for child in coarse["children"]],
            "medoid_feature": coarse["medoid_feature"],
        })
    return nodes


def select_examples(
    node: dict[str, Any], fused: np.ndarray, heatmap_root: Path, limit: int
) -> list[dict[str, Any]]:
    del fused
    feature = int(node["medoid_feature"])
    examples = json.loads(
        (heatmap_root / f"feature_{feature:03d}" / "examples.json").read_text()
    )
    selected: list[dict[str, Any]] = []
    seen_panos: set[tuple[str, str]] = set()
    for rank_index, source in enumerate(examples):
        row = dict(source)
        pano_key = (str(row["city"]), str(row["panoid"]))
        if pano_key in seen_panos:
            continue
        seen_panos.add(pano_key)
        row["feature_id"] = feature
        row["rank_index"] = rank_index
        selected.append(row)
        if len(selected) == limit:
            return selected
    raise RuntimeError(f"{node['id']}: only selected {len(selected)} unique examples")


def build_evidence_sheet(
    node: dict[str, Any], examples: list[dict[str, Any]], root: Path,
    resolver: OriginalResolver, patch_indices: np.ndarray, target: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not overwrite and target.is_file() and target.with_suffix(".stats.json").is_file():
        return json.loads(target.with_suffix(".stats.json").read_text())
    cell_w, cell_h, columns = 600, 220, 2
    rows = (len(examples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    low_information = 0
    cities: Counter[str] = Counter()
    represented_features: Counter[int] = Counter()
    for position, example in enumerate(examples):
        feature = int(example["feature_id"])
        rank_index = int(example["rank_index"])
        city = str(example["city"])
        image_index = int(example["source_image_index"])
        original = resolver.read(city, image_index)
        gray = original.resize((32, 24), Image.Resampling.BILINEAR).convert("L")
        low_information += int(float(ImageStat.Stat(gray).stddev[0]) < 2.0)
        cities[city] += 1
        represented_features[feature] += 1

        overlay_path = (
            root / "heatmaps" / f"feature_{feature:03d}" / str(example["file"])
        )
        with Image.open(overlay_path) as source:
            overlay = source.convert("RGB")
        patch_index = int(patch_indices[feature, rank_index])
        patch_row, patch_column = divmod(patch_index, PATCH_GRID)
        center_x = (patch_column + 0.5) * original.width / PATCH_GRID
        center_y = (patch_row + 0.5) * original.height / PATCH_GRID
        crop_size = max(32, int(min(original.size) * 0.36))
        left = max(0, min(original.width - crop_size, int(center_x - crop_size / 2)))
        top = max(0, min(original.height - crop_size, int(center_y - crop_size / 2)))
        hotspot = original.crop((left, top, left + crop_size, top + crop_size))

        pane_w = cell_w // 3
        original.thumbnail((pane_w, cell_h - 24), Image.Resampling.LANCZOS)
        overlay.thumbnail((pane_w, cell_h - 24), Image.Resampling.LANCZOS)
        hotspot.thumbnail((pane_w, cell_h - 24), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell_w, cell_h), "white")
        tile.paste(original, ((pane_w - original.width) // 2, 24))
        tile.paste(overlay, (pane_w + (pane_w - overlay.width) // 2, 24))
        tile.paste(hotspot, (2 * pane_w + (pane_w - hotspot.width) // 2, 24))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, cell_w, 23), fill=(0, 0, 0))
        draw.text((6, 6), f"{node['id']} #{position + 1:02d} DIM {feature:03d} ORIGINAL", fill="white")
        draw.text((pane_w + 6, 6), "HEAT", fill="white")
        draw.text((2 * pane_w + 6, 6), "HOTSPOT", fill="white")
        x = (position % columns) * cell_w
        y = (position // columns) * cell_h
        sheet.paste(tile, (x, y))

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jpg.tmp")
    sheet.save(temporary, format="JPEG", quality=90)
    temporary.replace(target)
    stats = {
        "n_examples": len(examples),
        "n_cities": len(cities),
        "city_counts": dict(cities.most_common()),
        "represented_features": {str(k): v for k, v in represented_features.items()},
        "low_information_count": low_information,
        "category_size_dimensions": int(node["n"]),
        "category_stability": float(node["stability"]),
    }
    write_json_atomic(target.with_suffix(".stats.json"), stats)
    return stats


def prompt_for(
    node: dict[str, Any], stats: dict[str, Any], child_labels: list[dict[str, Any]]
) -> str:
    level_zh = "粗类" if node["level"] == "coarse" else "细类"
    specificity = (
        "名称应概括较宽的共同视觉家族。"
        if node["level"] == "coarse"
        else "名称应比父粗类更具体，指出这个子类的区分性对象、材质、形态或空间模式。"
    )
    hierarchy_note = f"它属于统计父类 {node['parent']}。"
    if child_labels:
        children = "；".join(
            f"{row['category_id']}={row['name_zh']}（{row['summary_zh']}）"
            for row in child_labels
        )
        hierarchy_note = (
            f"该粗类包含这些已独立看图命名的细类：{children}。"
            "请结合图像证据归纳共同上位概念，不要简单拼接细类名称。"
        )
    return f"""你是一名城市街景视觉表征研究助手。图中是 Feature-MAE 层级类别 {node['id']}（{level_zh}）的 10 张代表样本。

每格有 ORIGINAL（原图）、HEAT（激活叠加）和 HOTSPOT（最高激活 patch 周围的原图裁剪）。红黄只表示强响应，绝不能把人工热力颜色当成真实物体颜色。10 张图是该类别统计中心维度（medoid）的十张最高响应图像；类别共有 {node['n']} 个维度，本图来自 {stats['n_cities']} 个城市。{hierarchy_note}

请根据多数样本中反复出现的局部视觉证据给类别命名。{specificity}
要求：
1. name_zh 为 2—12 字、可辨识的视觉概念，禁止使用“类别、特征、热区、热点、激活、维度”等空泛词，也不要只写“街景、城市街景、城市街道、道路与建筑”；
   以下名称同样禁止，因为它们不能区分类别：“建筑与道路、建筑与街道、道路与车辆”。必须指出可见的材质、形态、颜色、对象或空间关系中至少一项区分性线索；
2. 同时考虑对象、材料、纹理、几何、空间布局和环境；若内部多样，只命名最稳健的共同上位模式；
3. 若热点主要是黑边、拼接缝、模糊、遮挡或拍摄设备，明确标为 artifact；
4. 这是训练后的后验解释，不要声称模型以该语义训练；
5. semantic_type 只能取一个枚举值，不能用竖线组合多个值；
6. 三个评分必须根据图中证据估计，不能机械填写 0；本图均含可见街景，因此 confidence 和 cross_image_consistency 不得填写 0；
7. name_en 必须完全使用简洁自然的英文，不得混入中文字符；
8. 仅输出一个合法 JSON 对象，不要 Markdown 或额外文字。

JSON 严格包含：
{{
  "category_id": "{node['id']}",
  "level": "{node['level']}",
  "name_zh": "2到12字中文名",
  "name_en": "short English label",
  "semantic_type": "object|material|color_light|texture|geometry|spatial_layout|environment|artifact|mixed|unclear",
  "summary_zh": "不超过45字的共同视觉模式总结",
  "visual_cues_zh": ["证据1，不超过25字", "证据2，不超过25字", "证据3，不超过25字"],
  "urban_meaning_zh": "不超过35字的可能城市景观含义，无把握写不明确",
  "cross_image_consistency": 0.0,
  "artifact_probability": 0.0,
  "confidence": 0.0
}}

最后三个数值必须在 0 到 1 之间。"""


def probability(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def normalize(value: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    semantic_type = str(value.get("semantic_type", "unclear")).strip().lower()
    if semantic_type not in SEMANTIC_TYPES:
        semantic_type = "unclear"
    cues = value.get("visual_cues_zh", [])
    if not isinstance(cues, list):
        cues = [str(cues)] if cues else []
    cues = [str(x).strip() for x in cues if str(x).strip()][:3]
    return {
        "category_id": node["id"],
        "level": node["level"],
        "parent": node["parent"],
        "name_zh": str(value.get("name_zh", "语义不明确")).strip() or "语义不明确",
        "name_en": str(value.get("name_en", "unclear pattern")).strip() or "unclear pattern",
        "semantic_type": semantic_type,
        "summary_zh": str(value.get("summary_zh", "")).strip(),
        "visual_cues_zh": cues,
        "urban_meaning_zh": str(value.get("urban_meaning_zh", "不明确")).strip(),
        "cross_image_consistency": probability(value.get("cross_image_consistency"), 0.0),
        "artifact_probability": probability(value.get("artifact_probability"), 0.5),
        "confidence": probability(value.get("confidence"), 0.0),
        "n_dimensions": int(node["n"]),
        "statistical_stability": float(node["stability"]),
    }


def interpret(
    interpreter: DimensionInterpreter, node: dict[str, Any], image: Path,
    stats: dict[str, Any], child_labels: list[dict[str, Any]], model_dir: Path,
) -> dict[str, Any]:
    prompt = prompt_for(node, stats, child_labels)
    started = time.perf_counter()
    raw = interpreter.generate(image, prompt)
    parse_error = ""
    try:
        parsed = extract_json_object(raw)
    except ValueError as first_error:
        raw = interpreter.generate(
            image, prompt + "\n上一次回答无法解析。请缩短并严格只输出合法 JSON。"
        )
        try:
            parsed = extract_json_object(raw)
        except ValueError as second_error:
            parse_error = f"{first_error}; retry: {second_error}"
            parsed = {"name_zh": "自动解释失败", "confidence": 0.0}
    result = normalize(parsed, node)
    banned = ("类别", "特征", "热区", "热点", "激活", "维度")
    generic = {"街景", "城市街景", "城市街道", "道路与建筑"}
    quality_issues = []
    if any(word in result["name_zh"] for word in banned) or result["name_zh"] in generic:
        quality_issues.append("名称过于空泛或包含禁用词")
    if result["confidence"] <= 0.05:
        quality_issues.append("置信度被机械填写为零")
    if len(set(result["visual_cues_zh"])) < 2:
        quality_issues.append("视觉证据重复")
    if quality_issues:
        revision = (
            prompt + "\n初稿未通过质量检查：" + "；".join(quality_issues) +
            "。请重新仔细比较十张图，给出更具体且保守的名称、互不重复的证据和真实评分，只输出 JSON。"
        )
        revised_raw = interpreter.generate(image, revision)
        try:
            revised = normalize(extract_json_object(revised_raw), node)
            raw = revised_raw
            result = revised
        except ValueError:
            pass
    result["evidence_stats"] = stats
    result["provenance"] = {
        "interpretation_model": model_dir.parents[1].name.replace("--", "/"),
        "prompt_version": PROMPT_VERSION,
        "evidence_sheet": str(image),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_response": raw,
        "parse_error": parse_error,
    }
    return result


def consolidate(output: Path, records: list[dict[str, Any]]) -> None:
    records.sort(key=lambda x: (x["level"] != "coarse", x["category_id"]))
    payload = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "prompt_version": PROMPT_VERSION,
        "examples_per_category": 10,
        "categories": {row["category_id"]: row for row in records},
    }
    write_json_atomic(output / "category_labels.json", payload)
    fields = [
        "category_id", "level", "parent", "name_zh", "name_en", "semantic_type",
        "summary_zh", "visual_cues_zh", "urban_meaning_zh",
        "cross_image_consistency", "artifact_probability", "confidence",
        "n_dimensions", "statistical_stability",
    ]
    with (output / "category_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            flat = {key: row.get(key, "") for key in fields}
            flat["visual_cues_zh"] = "；".join(row.get("visual_cues_zh", []))
            writer.writerow(flat)
    report = {
        "categories": len(records),
        "coarse_categories": sum(row["level"] == "coarse" for row in records),
        "fine_categories": sum(row["level"] == "fine" for row in records),
        "parse_failures": sum(bool(row["provenance"]["parse_error"]) for row in records),
        "mean_confidence": float(np.mean([row["confidence"] for row in records])),
        "mean_cross_image_consistency": float(np.mean([
            row["cross_image_consistency"] for row in records
        ])),
        "semantic_type_counts": dict(Counter(row["semantic_type"] for row in records)),
        "duplicate_names": sorted(
            name for name, count in Counter(row["name_zh"] for row in records).items()
            if count > 1
        ),
        "model": payload["model"],
        "prompt_version": PROMPT_VERSION,
        "examples_per_category": 10,
    }
    write_json_atomic(output / "interpretation_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--hierarchy-dir", type=Path,
        help="Frozen hierarchy to interpret (default: ROOT/mae/hierarchy_32_64)",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--image-max-side", type=int, default=1200)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--min-free-gib", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.examples != 10:
        parser.error("this experiment requires exactly 10 representative images per category")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local Qwen interpretation")
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gib < args.min_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB GPU memory free")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)

    root = args.root.resolve()
    hierarchy = (
        args.hierarchy_dir.resolve()
        if args.hierarchy_dir is not None
        else root / "mae" / "hierarchy_32_64"
    )
    output = hierarchy / "semantic_labels_qwen"
    per_category = output / "per_category"
    per_category.mkdir(parents=True, exist_ok=True)
    taxonomy = json.loads((hierarchy / "taxonomy.json").read_text())
    nodes = category_nodes(taxonomy)
    with np.load(hierarchy / "hierarchy_arrays.npz") as arrays:
        fused = arrays["fused_similarity"].astype(np.float32)
    with np.load(root / "mae" / "top_image_activations.npz") as ranking:
        patch_indices = ranking["patch_indices"].astype(int)

    resolver = OriginalResolver(root)
    selected: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for node in nodes:
        category_id = node["id"]
        selected[category_id] = select_examples(
            node, fused, root / "heatmaps", args.examples
        )
        evidence = output / "evidence" / node["level"] / f"{category_id}.jpg"
        stats[category_id] = build_evidence_sheet(
            node, selected[category_id], root, resolver, patch_indices, evidence,
            overwrite=args.overwrite,
        )

    pending = [
        node for node in nodes
        if args.overwrite or not (per_category / f"{node['id']}.json").is_file()
    ]
    print(
        f"hierarchy interpretation: categories={len(nodes)} pending={len(pending)} "
        f"free_gpu={free_gib:.2f}GiB", flush=True,
    )
    interpreter = None
    if pending:
        interpreter = DimensionInterpreter(
            args.model_dir, args.max_new_tokens, args.image_max_side
        )
    labels: dict[str, dict[str, Any]] = {}
    for node in nodes:
        path = per_category / f"{node['id']}.json"
        if path.is_file() and not args.overwrite:
            result = json.loads(path.read_text())
        else:
            child_labels = [
                labels[child_id] for child_id in node.get("child_ids", [])
                if child_id in labels
            ]
            evidence = output / "evidence" / node["level"] / f"{node['id']}.jpg"
            result = interpret(
                interpreter, node, evidence, stats[node["id"]], child_labels,
                args.model_dir,
            )
            write_json_atomic(path, result)
            print(
                f"[{len(labels)+1:02d}/{len(nodes):02d}] {node['id']}: "
                f"{result['name_zh']} confidence={result['confidence']:.2f}",
                flush=True,
            )
        labels[node["id"]] = result

    consolidate(output, list(labels.values()))
    print(f"saved 32 coarse and 64 fine labels to {output}", flush=True)


if __name__ == "__main__":
    main()
