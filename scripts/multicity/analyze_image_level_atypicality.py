#!/usr/bin/env python3
"""Rank same-image atypical visual co-activations and render examples.

An element is present only when it wins at least ``minimum_patches`` of the
196 DINOv3 patches in a directional image. Pair-level unexpectedness is the
positive Pearson residual multiplied by the frozen E+D+P visual distance.
Within a co-activating image, the pair score is additionally weighted by the
smaller of the two patch shares. Image-level scores are the maximum and the
mean of the three strongest active pair scores.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.multicity.analyze_atypical_cooccurrence import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MAIN_HIERARCHY,
    DEFAULT_SOURCE_HIERARCHY,
    element_distances,
    global_statistics,
    load_semantic_names,
    save_csv,
    save_json,
)


PATCHES = 196
ELEMENTS = 64
DEFAULT_RESULTS = Path("results/image_level_atypicality")
DEFAULT_FIGURES = Path("figures/image_level_atypicality")
DEFAULT_METADATA = Path("results/image_dimension_pixel_proportions.parquet")


def fine_counts(block: np.ndarray, fine_labels: np.ndarray) -> np.ndarray:
    winners = fine_labels[np.asarray(block, dtype=np.int64)]
    rows = np.repeat(np.arange(len(winners), dtype=np.int64), winners.shape[1])
    keys = rows * ELEMENTS + winners.ravel()
    counts = np.bincount(
        keys, minlength=len(winners) * ELEMENTS
    ).reshape(len(winners), ELEMENTS)
    if not np.all(counts.sum(axis=1) == PATCHES):
        raise ValueError("fine-element patch counts do not sum to 196")
    return counts.astype(np.uint16)


def strict_cooccurrence(
    top: np.ndarray,
    fine_labels: np.ndarray,
    minimum_patches: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    cooccurrence = np.zeros((ELEMENTS, ELEMENTS), dtype=np.int64)
    active_per_image = np.empty(len(top), dtype=np.uint8)
    for start in range(0, len(top), batch_size):
        block = np.asarray(top[start : start + batch_size], dtype=np.int64)
        counts = fine_counts(block, fine_labels)
        active = counts >= minimum_patches
        active_per_image[start : start + len(block)] = active.sum(axis=1)
        values = active.astype(np.int64, copy=False)
        cooccurrence += values.T @ values
        if start == 0 or start + len(block) == len(top) or start % 50000 == 0:
            print(f"strict co-occurrence {start + len(block):,}/{len(top):,}", flush=True)
    return cooccurrence, active_per_image


def score_images(
    top: np.ndarray,
    fine_labels: np.ndarray,
    pair_frame: pd.DataFrame,
    minimum_patches: int,
    batch_size: int,
    representative_pairs: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    upper = np.triu_indices(ELEMENTS, 1)
    significant = pair_frame["significant_positive"].to_numpy(bool)
    requested = np.where(
        significant, pair_frame["unexpected_score"].to_numpy(np.float32), 0.0
    )
    effect = np.where(
        significant, pair_frame["visual_bridge_index"].to_numpy(np.float32), 0.0
    )
    leading = (
        pair_frame[pair_frame["significant_positive"]]
        .sort_values("unexpected_score", ascending=False)
        .head(representative_pairs)
        .index.to_numpy(np.int64)
    )
    if not len(leading):
        raise ValueError("no significant positive pairs available for image scoring")

    n = len(top)
    output = {
        "image_unexpected_max": np.zeros(n, dtype=np.float32),
        "image_unexpected_top3_mean": np.zeros(n, dtype=np.float32),
        "image_visual_bridge_max": np.zeros(n, dtype=np.float32),
        "image_visual_bridge_top3_mean": np.zeros(n, dtype=np.float32),
        "top_pair_index": np.full(n, -1, dtype=np.int16),
        "top_pair_joint_share": np.zeros(n, dtype=np.float32),
    }
    representative_min_share = np.zeros((n, len(leading)), dtype=np.float32)
    representative_sum_share = np.zeros((n, len(leading)), dtype=np.float32)
    threshold = minimum_patches / PATCHES

    for start in range(0, n, batch_size):
        block = np.asarray(top[start : start + batch_size], dtype=np.int64)
        counts = fine_counts(block, fine_labels)
        shares = counts.astype(np.float32) / PATCHES
        joint = np.minimum(shares[:, upper[0]], shares[:, upper[1]])
        joint[joint < threshold] = 0.0

        requested_scores = joint * requested[None, :]
        effect_scores = joint * effect[None, :]
        requested_top = np.partition(requested_scores, -3, axis=1)[:, -3:]
        effect_top = np.partition(effect_scores, -3, axis=1)[:, -3:]
        best = np.argmax(requested_scores, axis=1)
        rows = np.arange(len(block))
        destination = slice(start, start + len(block))
        output["image_unexpected_max"][destination] = requested_scores[rows, best]
        output["image_unexpected_top3_mean"][destination] = requested_top.mean(axis=1)
        output["image_visual_bridge_max"][destination] = effect_scores.max(axis=1)
        output["image_visual_bridge_top3_mean"][destination] = effect_top.mean(axis=1)
        has_score = requested_scores[rows, best] > 0
        output["top_pair_index"][destination] = np.where(has_score, best, -1)
        output["top_pair_joint_share"][destination] = np.where(
            has_score, joint[rows, best], 0.0
        )
        representative_min_share[destination] = joint[:, leading]
        representative_sum_share[destination] = (
            shares[:, upper[0][leading]] + shares[:, upper[1][leading]]
        )
        if start == 0 or start + len(block) == n or start % 50000 == 0:
            print(f"image scores {start + len(block):,}/{n:,}", flush=True)

    return output, leading, representative_min_share, representative_sum_share


def metadata_frame(path: Path, images: int) -> pd.DataFrame:
    columns = [
        "global_image_index", "city_key", "image_index", "source_image_index",
        "pano_index", "panoid", "direction_index", "heading", "lat", "lon",
        "year", "month", "tar_path", "jpg_offset", "jpg_size",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != images:
        raise ValueError(f"metadata/image mismatch: {len(frame)} != {images}")
    expected = np.arange(images, dtype=np.int64)
    if not np.array_equal(frame["global_image_index"].to_numpy(), expected):
        raise ValueError("metadata is not ordered by global_image_index")
    return frame


def attach_pair_labels(
    metadata: pd.DataFrame,
    scores: dict[str, np.ndarray],
    pair_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = metadata.drop(columns=["tar_path", "jpg_offset", "jpg_size"]).copy()
    for name, values in scores.items():
        frame[name] = values
    top = scores["top_pair_index"].astype(np.int64)
    valid = top >= 0
    for source, destination in (
        ("element_a", "top_element_a"),
        ("element_b", "top_element_b"),
        ("name_a", "top_name_a"),
        ("name_b", "top_name_b"),
    ):
        values = np.full(len(frame), "", dtype=object)
        values[valid] = pair_frame[source].to_numpy()[top[valid]]
        frame[destination] = values
    return frame


def representative_samples(
    metadata: pd.DataFrame,
    top: np.ndarray,
    fine_labels: np.ndarray,
    pair_frame: pd.DataFrame,
    leading: np.ndarray,
    min_share: np.ndarray,
    sum_share: np.ndarray,
    samples_per_pair: int,
) -> pd.DataFrame:
    upper = np.triu_indices(ELEMENTS, 1)
    selected_rows = []
    for pair_rank, pair_index in enumerate(leading, 1):
        pair = pair_frame.iloc[pair_index]
        sample_score = min_share[:, pair_rank - 1] * float(pair["unexpected_score"])
        order = np.lexsort((-sum_share[:, pair_rank - 1], -sample_score))
        seen_panoramas: set[str] = set()
        chosen = []
        for image_index in order:
            if sample_score[image_index] <= 0:
                break
            panoid = str(metadata.iloc[image_index]["panoid"])
            if panoid in seen_panoramas:
                continue
            seen_panoramas.add(panoid)
            chosen.append(int(image_index))
            if len(chosen) == samples_per_pair:
                break
        counts = fine_counts(np.asarray(top[chosen]), fine_labels)
        element_a = int(upper[0][pair_index])
        element_b = int(upper[1][pair_index])
        for sample_rank, (image_index, count_row) in enumerate(zip(chosen, counts), 1):
            row = metadata.iloc[image_index]
            share_a = float(count_row[element_a] / PATCHES)
            share_b = float(count_row[element_b] / PATCHES)
            selected_rows.append(
                {
                    "pair_rank": pair_rank,
                    "sample_rank": sample_rank,
                    "element_a": pair["element_a"],
                    "name_a": pair["name_a"],
                    "element_b": pair["element_b"],
                    "name_b": pair["name_b"],
                    "pair_unexpected_score": pair["unexpected_score"],
                    "visual_distance": pair["visual_distance"],
                    "observed": int(pair["observed"]),
                    "expected": pair["expected"],
                    "global_image_index": image_index,
                    "city_key": row["city_key"],
                    "panoid": row["panoid"],
                    "heading": row["heading"],
                    "patches_a": int(count_row[element_a]),
                    "patches_b": int(count_row[element_b]),
                    "share_a": share_a,
                    "share_b": share_b,
                    "joint_min_share": min(share_a, share_b),
                    "sample_pair_score": float(
                        pair["unexpected_score"] * min(share_a, share_b)
                    ),
                }
            )
    return pd.DataFrame(selected_rows)


def read_original(row: pd.Series) -> Image.Image:
    fd = os.open(str(row["tar_path"]), os.O_RDONLY)
    try:
        payload = os.pread(fd, int(row["jpg_size"]), int(row["jpg_offset"]))
    finally:
        os.close(fd)
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def overlay_pair(
    original: Image.Image,
    winners: np.ndarray,
    element_a: int,
    element_b: int,
) -> Image.Image:
    grid = winners.reshape(14, 14)
    mask_a = Image.fromarray((grid == element_a).astype(np.uint8) * 255, "L").resize(
        original.size, Image.Resampling.NEAREST
    )
    mask_b = Image.fromarray((grid == element_b).astype(np.uint8) * 255, "L").resize(
        original.size, Image.Resampling.NEAREST
    )
    base = original.convert("RGBA")
    red = Image.new("RGBA", original.size, (245, 70, 70, 135))
    cyan = Image.new("RGBA", original.size, (20, 210, 245, 135))
    base.alpha_composite(Image.composite(red, Image.new("RGBA", original.size), mask_a))
    base.alpha_composite(Image.composite(cyan, Image.new("RGBA", original.size), mask_b))
    return base.convert("RGB")


def render_contact_sheet(
    samples: pd.DataFrame,
    metadata: pd.DataFrame,
    top: np.ndarray,
    fine_labels: np.ndarray,
    output: Path,
    displayed_pairs: int,
    displayed_samples: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    label_width, tile_width, tile_height = 330, 270, 215
    header_height = 48
    rows = min(displayed_pairs, samples["pair_rank"].nunique())
    sheet = Image.new(
        "RGB",
        (label_width + displayed_samples * tile_width, header_height + rows * tile_height),
        "#f7f7f5",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "Red = element A    Cyan = element B    Same directional image", fill="#202020", font=font(20, True))
    for row_index, pair_rank in enumerate(range(1, rows + 1)):
        part = samples[samples["pair_rank"] == pair_rank].head(displayed_samples)
        if part.empty:
            continue
        first = part.iloc[0]
        y0 = header_height + row_index * tile_height
        fill = "#ecece8" if row_index % 2 == 0 else "#f6f6f3"
        draw.rectangle((0, y0, sheet.width, y0 + tile_height), fill=fill)
        label_lines = [
            f"{first.element_a}  {first.name_a}",
            f"{first.element_b}  {first.name_b}",
            f"U={first.pair_unexpected_score:.2f}  D={first.visual_distance:.2f}",
            f"O={int(first.observed):,}  E={first.expected:,.1f}",
        ]
        for line_index, text in enumerate(label_lines):
            draw.text(
                (12, y0 + 18 + line_index * 36), text,
                fill="#202020", font=font(17, bold=line_index < 2),
            )
        element_a = int(str(first.element_a)[1:])
        element_b = int(str(first.element_b)[1:])
        for column, (_, sample) in enumerate(part.iterrows()):
            image_index = int(sample.global_image_index)
            original = read_original(metadata.iloc[image_index])
            winners = fine_labels[np.asarray(top[image_index], dtype=np.int64)]
            overlay = overlay_pair(original, winners, element_a, element_b)
            tile = ImageOps.fit(
                overlay, (tile_width - 8, tile_height - 34),
                method=Image.Resampling.LANCZOS,
            )
            x0 = label_width + column * tile_width + 4
            sheet.paste(tile, (x0, y0 + 4))
            caption = (
                f"{str(sample.city_key).split('/')[-1]}  "
                f"{int(sample.patches_a)}/{int(sample.patches_b)} patches"
            )
            draw.text((x0 + 3, y0 + tile_height - 26), caption, fill="#202020", font=font(13))
    sheet.save(output / "Fig_Image_Level_Atypical_Coactivation.png")
    sheet.save(output / "Fig_Image_Level_Atypical_Coactivation.pdf", "PDF", resolution=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-hierarchy", type=Path, default=DEFAULT_SOURCE_HIERARCHY)
    parser.add_argument("--main-hierarchy", type=Path, default=DEFAULT_MAIN_HIERARCHY)
    parser.add_argument("--base-results", type=Path, default=Path("results"))
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--minimum-patches", type=int, default=4)
    parser.add_argument("--minimum-support", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--representative-pairs", type=int, default=20)
    parser.add_argument("--samples-per-pair", type=int, default=5)
    args = parser.parse_args()

    source_report = json.loads((args.source_hierarchy / "scan_report.json").read_text())
    images = int(source_report["images"])
    minimum_support = args.minimum_support
    if minimum_support is None:
        minimum_support = max(100, int(math.ceil(0.0005 * images)))
    top = np.load(args.source_hierarchy / "top1_dimensions.npy", mmap_mode="r")
    arrays = np.load(args.main_hierarchy / "hierarchy_arrays.npz")
    fine_labels = arrays["fine_labels"].astype(np.int64)

    print("[1/6] strict same-image occurrence (>=4 patches per element)", flush=True)
    cooccurrence, active_per_image = strict_cooccurrence(
        top, fine_labels, args.minimum_patches, args.batch_size
    )
    print("[2/6] pair-level unexpectedness", flush=True)
    distances, _ = element_distances(
        args.main_hierarchy, args.base_results / "fine_clusters.csv"
    )
    names = load_semantic_names(args.base_results / "semantic_labels_fine.json")
    pair_frame = global_statistics(
        cooccurrence, images, distances, names, minimum_support
    )
    save_csv(pair_frame, args.results / "global_pair_statistics.csv")
    pair_top = (
        pair_frame[pair_frame["significant_positive"]]
        .sort_values("unexpected_score", ascending=False)
        .head(100)
    )
    save_csv(pair_top, args.results / "global_top_pairs.csv")

    print("[3/6] per-image pair and image atypicality", flush=True)
    scores, leading, representative_min, representative_sum = score_images(
        top, fine_labels, pair_frame, args.minimum_patches, args.batch_size,
        args.representative_pairs,
    )
    print("[4/6] image metadata and representative samples", flush=True)
    metadata = metadata_frame(args.metadata, images)
    image_frame = attach_pair_labels(metadata, scores, pair_frame)
    args.results.mkdir(parents=True, exist_ok=True)
    image_frame.to_parquet(
        args.results / "image_atypicality.parquet", index=False,
        compression="zstd",
    )
    top_images = (
        image_frame.sort_values("image_unexpected_max", ascending=False)
        .drop_duplicates("panoid")
        .head(500)
    )
    save_csv(top_images, args.results / "top_atypical_images.csv")
    samples = representative_samples(
        metadata, top, fine_labels, pair_frame, leading,
        representative_min, representative_sum, args.samples_per_pair,
    )
    save_csv(samples, args.results / "representative_pair_samples.csv")

    print("[5/6] activation-overlay contact sheet", flush=True)
    render_contact_sheet(
        samples, metadata, top, fine_labels, args.figures,
        displayed_pairs=min(8, args.representative_pairs),
        displayed_samples=min(4, args.samples_per_pair),
    )
    print("[6/6] report", flush=True)
    supported_positive = (
        (pair_frame["observed"] >= minimum_support) & (pair_frame["log_lift"] > 0)
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": images,
        "elements": ELEMENTS,
        "pairs": int(ELEMENTS * (ELEMENTS - 1) / 2),
        "presence_rule": f"fine element wins at least {args.minimum_patches} of {PATCHES} patches in the same directional image",
        "minimum_patch_share": args.minimum_patches / PATCHES,
        "mean_active_elements_per_image": float(active_per_image.mean()),
        "median_active_elements_per_image": float(np.median(active_per_image)),
        "minimum_pair_support": minimum_support,
        "supported_positive_lift_pairs": int(supported_positive.sum()),
        "significant_positive_pairs_bh_005": int(pair_frame["significant_positive"].sum()),
        "sample_pair_score": "pair unexpected_score * min(element_a patch share, element_b patch share)",
        "image_score_max": "maximum significant sample-pair score in the image",
        "image_score_top3_mean": "mean of the three strongest significant sample-pair scores in the image",
        "images_with_positive_score": int((scores["image_unexpected_max"] > 0).sum()),
        "representative_pairs": int(len(leading)),
        "samples_per_pair": args.samples_per_pair,
        "semantic_labels_used_for_scoring": False,
    }
    save_json(report, args.results / "report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
