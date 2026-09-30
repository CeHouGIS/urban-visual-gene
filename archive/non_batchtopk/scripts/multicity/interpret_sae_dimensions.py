"""Post-hoc semantic interpretation of SAE heatmaps with a compact VLM.

The learned dimensions remain data-driven.  This script only attaches tentative
human-readable labels after training by showing a VLM the top-activating image
contact sheet for each dimension.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat

from scripts.multicity.patch_config import OUTPUT_ROOT, PATCH_GRID, WIDTH
from scripts.multicity.patch_config import image_manifest_path


DEFAULT_MODEL_DIR = Path(
    "/workplace/models/modelscope/models/"
    "Qwen--Qwen3-VL-2B-Instruct/snapshots/master"
)
PROMPT_VERSION = "sae-dimension-interpretation-v4-top10"
SEMANTIC_TYPES = {
    "object",
    "material",
    "color_light",
    "texture",
    "geometry",
    "spatial_layout",
    "environment",
    "artifact",
    "mixed",
    "unclear",
}


def build_prompt(feature_id: int, stats: dict[str, Any] | None = None) -> str:
    stats = stats or {}
    n_examples = stats.get("n_examples", 10)
    evidence_note = (
        f"这 {n_examples} 个样本来自 {stats.get('n_cities', '未知')} 个城市；最高激活分数范围为 "
        f"{stats.get('min_selected_score', '未知')} 到 {stats.get('top_score', '未知')}；"
        f"自动质量检查发现 {stats.get('original_quality', {}).get('low_information_count', '未知')} "
        f"张原图接近纯色/低信息。"
    )
    return f"""你是一名城市街景视觉表征研究助手。图中是稀疏自编码器 SAE 维度 {feature_id:03d} 的 {n_examples} 张最高激活样本。

版面为 2 列×5 行，每个样本格依次包含三个面板：ORIGINAL 是无热力图原图，HEAT 是激活叠加图，HOTSPOT 是从原图裁出的最高激活 patch 周围局部。HEAT 中红/黄区域表示强激活，青/蓝/暗区域表示弱激活。你的任务是先在 HEAT 定位热点，再重点比较 HOTSPOT，并回看 ORIGINAL 的上下文，解释热点跨样本反复对应的共同局部视觉模式，而不是笼统描述整张街景。热力图颜色是人工叠加，绝不能把叠加的红色或黄色本身解释为秋色、暖色或原图语义，除非 ORIGINAL/HOTSPOT 中确实存在这种颜色。

{evidence_note} 城市过度集中、分数完全并列、原图全黑/近黑或相同占位图都是潜在数据伪影信号。

请：
1. 比较至少多数样本的热点区域，识别对象、材料、纹理、颜色/光照、几何、空间布局或环境语境；
2. 若热点不一致、主要落在黑边、暗角、空白占位图、模糊、遮挡、拼接缝、文字或拍摄设备上，应判为 artifact 或 unclear；
3. 解释应保守。维度可能是混合概念，不要硬套单一城市功能；
4. urban_meaning_zh 只写可能的城市景观含义，不作因果推断；
5. name_zh/name_en 必须命名 ORIGINAL/HOTSPOT 中的视觉概念，禁止使用“热区、热点、高亮、激活、heat、hotspot、activation”等循环描述；
6. 文字务必精炼：summary 不超过 40 字，visual_cues 恰好 2 条且每条不超过 25 字，其余文字字段各不超过 30 字；
7. 仅输出一个合法 JSON 对象，不要 Markdown，不要额外文字。

JSON 必须严格包含以下字段：
{{
  "feature_id": {feature_id},
  "name_zh": "2到10字的概念名",
  "name_en": "short English label",
  "semantic_type": "object|material|color_light|texture|geometry|spatial_layout|environment|artifact|mixed|unclear",
  "summary_zh": "一句话概括热点共同模式",
  "visual_cues_zh": ["证据1", "证据2"],
  "hot_regions_zh": "热点通常位于画面中的什么对象或位置",
  "spatial_pattern_zh": "热点的形状、尺度或空间分布",
  "urban_meaning_zh": "该模式可能反映的城市景观含义；无把握则写不明确",
  "cross_image_consistency": 0.0,
  "artifact_probability": 0.0,
  "artifact_reason_zh": "伪影判断依据，无明显伪影则写无",
  "confidence": 0.0
}}

后三个数值必须在 0 到 1 之间。"""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first decodable JSON object, tolerating Markdown fences."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a valid JSON object")


def _probability(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_interpretation(value: dict[str, Any], feature_id: int) -> dict[str, Any]:
    """Enforce a stable output schema without inventing semantic content."""
    semantic_type = str(value.get("semantic_type", "unclear")).strip().lower()
    if semantic_type not in SEMANTIC_TYPES:
        semantic_type = "unclear"
    cues = value.get("visual_cues_zh", [])
    if not isinstance(cues, list):
        cues = [str(cues)] if cues else []
    cues = list(dict.fromkeys(str(item).strip() for item in cues if str(item).strip()))[:5]
    result = {
        "feature_id": feature_id,
        "name_zh": str(value.get("name_zh", "未明确特征")).strip() or "未明确特征",
        "name_en": str(value.get("name_en", "unclear feature")).strip() or "unclear feature",
        "semantic_type": semantic_type,
        "summary_zh": str(value.get("summary_zh", "")).strip(),
        "visual_cues_zh": cues,
        "hot_regions_zh": str(value.get("hot_regions_zh", "")).strip(),
        "spatial_pattern_zh": str(value.get("spatial_pattern_zh", "")).strip(),
        "urban_meaning_zh": str(value.get("urban_meaning_zh", "不明确")).strip(),
        "cross_image_consistency": _probability(
            value.get("cross_image_consistency"), 0.0
        ),
        "artifact_probability": _probability(value.get("artifact_probability"), 0.5),
        "artifact_reason_zh": str(value.get("artifact_reason_zh", "")).strip(),
        "confidence": _probability(value.get("confidence"), 0.0),
    }
    return result


def load_example_stats(examples_path: Path, limit: int) -> dict[str, Any]:
    examples = json.loads(examples_path.read_text(encoding="utf-8"))[:limit]
    cities = Counter(str(row["city"]) for row in examples)
    scores = [float(row["score"]) for row in examples]
    return {
        "n_examples": len(examples),
        "n_cities": len(cities),
        "city_counts": dict(cities.most_common()),
        "top_score": max(scores),
        "min_selected_score": min(scores),
    }


class OriginalImageResolver:
    """Random-access original images through the existing tar manifests."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self._manifests: dict[str, Any] = {}

    def _manifest(self, city: str) -> Any:
        if city not in self._manifests:
            import pandas as pd

            self._manifests[city] = (
                pd.read_parquet(image_manifest_path(city, self.output_root))
                .sort_values("image_index")
                .reset_index(drop=True)
            )
        return self._manifests[city]

    def read(self, city: str, image_index: int) -> Image.Image:
        row = self._manifest(city).iloc[image_index]
        descriptor = os.open(str(row["tar_path"]), os.O_RDONLY)
        try:
            payload = os.pread(
                descriptor, int(row["jpg_size"]), int(row["jpg_offset"])
            )
        finally:
            os.close(descriptor)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")


def build_evidence_sheet(
    feature_dir: Path,
    target_path: Path,
    resolver: OriginalImageResolver,
    patch_indices: list[int],
    limit: int,
) -> tuple[Path, dict[str, Any]]:
    """Build ORIGINAL/HEAT/HOTSPOT evidence for localized interpretation."""
    examples = json.loads(
        (feature_dir / "examples.json").read_text(encoding="utf-8")
    )[:limit]
    stats_path = target_path.with_suffix(".stats.json")
    if target_path.is_file() and stats_path.is_file():
        return target_path, json.loads(stats_path.read_text(encoding="utf-8"))
    cell_w, cell_h = 600, 220
    columns = 2
    rows = (len(examples) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    low_information_count = 0
    blank_extreme_count = 0
    thumbnail_hashes = set()
    for index, example in enumerate(examples):
        original = resolver.read(str(example["city"]), int(example["image_index"]))
        gray = original.resize((32, 24), Image.Resampling.BILINEAR).convert("L")
        gray_stats = ImageStat.Stat(gray)
        mean = float(gray_stats.mean[0])
        std = float(gray_stats.stddev[0])
        low_information_count += int(std < 2.0)
        blank_extreme_count += int(std < 2.0 and (mean < 8.0 or mean > 247.0))
        thumbnail = original.resize((16, 12), Image.Resampling.BILINEAR).convert("RGB")
        thumbnail_hashes.add(hashlib.sha1(thumbnail.tobytes()).hexdigest())
        with Image.open(feature_dir / str(example["file"])) as source:
            overlay = source.convert("RGB")
        patch_index = int(patch_indices[index])
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
        tile.paste(
            overlay,
            (pane_w + (pane_w - overlay.width) // 2, 24),
        )
        tile.paste(hotspot, (2 * pane_w + (pane_w - hotspot.width) // 2, 24))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, cell_w, 23), fill=(0, 0, 0))
        draw.text((6, 6), f"rank {int(example['rank']):02d}  ORIGINAL", fill="white")
        draw.text((pane_w + 6, 6), "HEAT", fill="white")
        draw.text((2 * pane_w + 6, 6), "HOTSPOT", fill="white")
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        sheet.paste(tile, (x, y))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(".jpg.tmp")
    sheet.save(temporary, format="JPEG", quality=90)
    temporary.replace(target_path)
    quality = {
        "low_information_count": low_information_count,
        "blank_extreme_count": blank_extreme_count,
        "unique_thumbnail_count": len(thumbnail_hashes),
    }
    write_json_atomic(stats_path, quality)
    return target_path, quality


def apply_quality_guard(
    result: dict[str, Any], stats: dict[str, Any]
) -> dict[str, Any]:
    """Prevent a small VLM from inventing semantics for blank/placeholder images."""
    quality = stats.get("original_quality", {})
    n_examples = max(1, int(stats.get("n_examples", 0)))
    low_count = int(quality.get("low_information_count", 0))
    blank_count = int(quality.get("blank_extreme_count", 0))
    unique_count = int(quality.get("unique_thumbnail_count", n_examples))
    placeholder_dominated = blank_count / n_examples >= 0.75 or (
        low_count / n_examples >= 0.75 and unique_count <= 2
    )
    result["quality_guard_applied"] = placeholder_dominated
    if placeholder_dominated:
        result.update({
            "name_zh": "空白占位图伪影",
            "name_en": "blank placeholder artifact",
            "semantic_type": "artifact",
            "summary_zh": "最高激活样本主要是空白或近纯色占位图，该维度不具有可靠的城市景观语义。",
            "visual_cues_zh": [
                f"{blank_count}/{n_examples} 张原图为极端明暗的近纯色图像",
                f"16×12 缩略图仅有 {unique_count} 种不同内容",
                "激活集中在图像边缘或统一画框结构上",
            ],
            "hot_regions_zh": "空白图像的边缘、画框或固定位置",
            "spatial_pattern_zh": "跨样本位置固定的边框状激活",
            "urban_meaning_zh": "无可靠城市景观含义，应在后续分析中剔除或单独标记。",
            "cross_image_consistency": 1.0,
            "artifact_probability": 1.0,
            "artifact_reason_zh": "原图自动质量检查表明样本由重复空白/低信息图像主导。",
            "confidence": 0.99,
        })
    return result


class DimensionInterpreter:
    def __init__(self, model_dir: Path, max_new_tokens: int, image_max_side: int) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.image_max_side = image_max_side
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, local_files_only=True
        ).eval().to("cuda")
        self.model.requires_grad_(False)

    def generate(self, image_path: Path, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(
            (self.image_max_side, self.image_max_side), Image.Resampling.LANCZOS
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def interpret_one(
    interpreter: DimensionInterpreter,
    heatmap_root: Path,
    feature_id: int,
    model_dir: Path,
    evidence_dir: Path,
    resolver: OriginalImageResolver,
    patch_indices: list[int],
    examples_per_dimension: int,
) -> dict[str, Any]:
    feature_dir = heatmap_root / f"feature_{feature_id:03d}"
    contact_sheet_path = feature_dir / "contact_sheet.jpg"
    examples_path = feature_dir / "examples.json"
    if not contact_sheet_path.is_file() or not examples_path.is_file():
        raise FileNotFoundError(f"missing heatmap inputs for feature {feature_id:03d}")

    image_path, original_quality = build_evidence_sheet(
        feature_dir,
        evidence_dir / f"feature_{feature_id:03d}.jpg",
        resolver,
        patch_indices,
        examples_per_dimension,
    )
    stats = load_example_stats(examples_path, examples_per_dimension)
    stats["original_quality"] = original_quality
    started = time.perf_counter()
    quality_only = apply_quality_guard(
        normalize_interpretation({}, feature_id), stats
    )
    parse_error = ""
    if quality_only["quality_guard_applied"]:
        result = quality_only
        raw = "VLM generation skipped: deterministic blank-image quality guard."
    else:
        raw = interpreter.generate(image_path, build_prompt(feature_id, stats))
        try:
            parsed = extract_json_object(raw)
        except ValueError as first_error:
            retry_prompt = build_prompt(feature_id, stats) + (
                "\n上一次响应不是合法 JSON。请缩短回答并严格只输出 JSON 对象。"
            )
            raw = interpreter.generate(image_path, retry_prompt)
            try:
                parsed = extract_json_object(raw)
            except ValueError as second_error:
                parse_error = f"{first_error}; retry: {second_error}"
                parsed = {
                    "name_zh": "自动解释失败",
                    "name_en": "automatic interpretation failed",
                    "semantic_type": "unclear",
                    "summary_zh": "模型未返回可解析的结构化解释，需人工复核。",
                    "artifact_probability": 0.5,
                    "confidence": 0.0,
                }
        result = apply_quality_guard(
            normalize_interpretation(parsed, feature_id), stats
        )
    result["evidence_stats"] = stats
    result["provenance"] = {
        "interpretation_model": model_dir.parents[1].name.replace("--", "/"),
        "prompt_version": PROMPT_VERSION,
        "evidence_sheet": str(image_path),
        "contact_sheet": str(contact_sheet_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_response": raw,
        "parse_error": parse_error,
    }
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def consolidate(feature_output_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(feature_output_dir.glob("feature_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(key=lambda row: int(row["feature_id"]))
    write_json_atomic(output_dir / "dimension_interpretations.json", records)

    csv_path = output_dir / "dimension_interpretations.csv"
    fields = [
        "feature_id", "name_zh", "name_en", "semantic_type", "summary_zh",
        "visual_cues_zh", "hot_regions_zh", "spatial_pattern_zh", "urban_meaning_zh",
        "cross_image_consistency", "artifact_probability", "artifact_reason_zh", "confidence",
        "n_cities", "top_score", "quality_guard_applied",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            flat = {key: row.get(key, "") for key in fields}
            flat["visual_cues_zh"] = "；".join(row.get("visual_cues_zh", []))
            flat["n_cities"] = row.get("evidence_stats", {}).get("n_cities", "")
            flat["top_score"] = row.get("evidence_stats", {}).get("top_score", "")
            writer.writerow(flat)

    lines = [
        "# SAE 维度语义目录",
        "",
        "> 这些名称是轻量视觉语言模型根据最高激活热力图给出的后验解释，不是 SAE 训练标签。",
        "",
        "| ID | 中文名称 | English | 类型 | 一致性 | 伪影概率 | 置信度 | 摘要 |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for row in records:
        feature_id = int(row["feature_id"])
        sheet = f"evidence/top10/feature_{feature_id:03d}.jpg"
        summary = str(row.get("summary_zh", "")).replace("|", "／").replace("\n", " ")
        lines.append(
            f"| [{feature_id:03d}]({sheet}) | {row['name_zh']} | {row['name_en']} | "
            f"{row['semantic_type']} | {row['cross_image_consistency']:.2f} | "
            f"{row['artifact_probability']:.2f} | {row['confidence']:.2f} | {summary} |"
        )
    (output_dir / "dimension_catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    semantic_counts = Counter(row.get("semantic_type", "unclear") for row in records)
    parse_failures = sum(bool(row.get("provenance", {}).get("parse_error")) for row in records)
    report = {
        "n_dimensions": len(records),
        "feature_id_min": min((int(row["feature_id"]) for row in records), default=None),
        "feature_id_max": max((int(row["feature_id"]) for row in records), default=None),
        "examples_per_dimension": sorted({
            int(row.get("evidence_stats", {}).get("n_examples", 0)) for row in records
        }),
        "semantic_type_counts": dict(sorted(semantic_counts.items())),
        "quality_guard_count": sum(bool(row.get("quality_guard_applied")) for row in records),
        "artifact_probability_ge_0_5_count": sum(
            float(row.get("artifact_probability", 0.0)) >= 0.5 for row in records
        ),
        "parse_failure_count": parse_failures,
        "mean_confidence": (
            sum(float(row.get("confidence", 0.0)) for row in records) / len(records)
            if records else 0.0
        ),
        "prompt_versions": sorted({
            str(row.get("provenance", {}).get("prompt_version", "")) for row in records
        }),
        "interpretation_models": sorted({
            str(row.get("provenance", {}).get("interpretation_model", "")) for row in records
        }),
    }
    write_json_atomic(output_dir / "interpretation_report.json", report)
    return records


def parse_feature_ids(args: argparse.Namespace) -> list[int]:
    if args.features:
        ids = sorted(set(args.features))
    else:
        ids = list(range(args.start_feature, args.end_feature))
    invalid = [feature_id for feature_id in ids if not 0 <= feature_id < WIDTH]
    if invalid:
        raise ValueError(f"feature IDs outside [0, {WIDTH - 1}]: {invalid}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--features", type=int, nargs="*")
    ap.add_argument("--start-feature", type=int, default=0)
    ap.add_argument("--end-feature", type=int, default=WIDTH)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--image-max-side", type=int, default=1200)
    ap.add_argument("--examples-per-dimension", type=int, default=10)
    ap.add_argument("--min-free-gib", type=float, default=6.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not 1 <= args.examples_per_dimension <= 16:
        ap.error("--examples-per-dimension must be between 1 and 16")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL dimension interpretation")
    free_gib = torch.cuda.mem_get_info()[0] / (1024**3)
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB GPU memory free; need {args.min_free_gib:.2f} GiB"
        )
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)

    feature_ids = parse_feature_ids(args)
    output_dir = args.output_root / "dimension_interpretations"
    feature_output_dir = output_dir / "per_feature"
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        feature_id for feature_id in feature_ids
        if args.overwrite or not (feature_output_dir / f"feature_{feature_id:03d}.json").is_file()
    ]
    print(
        f"dimension interpretation: selected={len(feature_ids)} pending={len(pending)} "
        f"free_gpu={free_gib:.2f}GiB",
        flush=True,
    )
    interpreter = None
    resolver = OriginalImageResolver(args.output_root)
    import numpy as np

    with np.load(args.output_root / "patch_sae" / "top_image_activations.npz") as ranking:
        all_patch_indices = ranking["patch_indices"].astype(int)
    if pending:
        interpreter = DimensionInterpreter(
            args.model_dir, args.max_new_tokens, args.image_max_side
        )
    for position, feature_id in enumerate(pending, start=1):
        result = interpret_one(
            interpreter,
            args.output_root / "heatmaps",
            feature_id,
            args.model_dir,
            output_dir / "evidence" / "top10",
            resolver,
            all_patch_indices[feature_id].tolist(),
            args.examples_per_dimension,
        )
        write_json_atomic(
            feature_output_dir / f"feature_{feature_id:03d}.json", result
        )
        print(
            f"[{position:03d}/{len(pending):03d}] feature {feature_id:03d}: "
            f"{result['name_zh']} ({result['semantic_type']}, "
            f"confidence={result['confidence']:.2f})",
            flush=True,
        )
    records = consolidate(feature_output_dir, output_dir)
    print(f"consolidated {len(records)} interpretations in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
