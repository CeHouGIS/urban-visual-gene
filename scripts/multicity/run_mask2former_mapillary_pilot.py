#!/usr/bin/env python3
"""Run a memory-safe Mask2Former Swin-L Mapillary Vistas pilot.

The script samples panoramas evenly across the existing multi-city manifests,
runs the four cardinal views independently, stores full-resolution uint8 class
masks, and aggregates each prediction into 14x14x65 semantic fractions for
future alignment with Feature-MAE patch activations.
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (must precede numpy/pandas/torch)

import argparse
import colorsys
import io
import json
import os
import random
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
DEFAULT_MANIFEST_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "rectangular_panorama_frozen_mae_n30/manifests"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "mask2former_swin_l_mapillary_pilot"
)
DEFAULT_PAPER_DATA = Path(
    "/workplace/urban_visual_gene/paper/data/semantic_alignment/"
    "mask2former_swin_l_pilot"
)
DEFAULT_FIGURE = Path(
    "/workplace/urban_visual_gene/paper/figures/supplementary/"
    "Fig_Mask2Former_Mapillary_Pilot.png"
)
HEADINGS = (0, 90, 180, 270)
PATCH_GRID = 14
NUM_CLASSES = 65


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_safe_affinity() -> None:
    if hasattr(os, "sched_getaffinity"):
        forbidden = set(os.sched_getaffinity(0)).intersection({8, 9})
        if forbidden:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(forbidden)}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


class TarReader:
    """Small LRU cache of TAR file descriptors using manifest byte offsets."""

    def __init__(self, max_open_files: int = 8) -> None:
        self.max_open_files = max_open_files
        self.descriptors: OrderedDict[str, int] = OrderedDict()

    def descriptor(self, path: str) -> int:
        if path in self.descriptors:
            descriptor = self.descriptors.pop(path)
            self.descriptors[path] = descriptor
            return descriptor
        descriptor = os.open(path, os.O_RDONLY)
        self.descriptors[path] = descriptor
        if len(self.descriptors) > self.max_open_files:
            _, old_descriptor = self.descriptors.popitem(last=False)
            os.close(old_descriptor)
        return descriptor

    def image(self, path: str, offset: int, size: int) -> Image.Image:
        payload = os.pread(self.descriptor(path), size, offset)
        if len(payload) != size:
            raise OSError(f"short TAR read: {path} at {offset}: {len(payload)} != {size}")
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()


def sample_balanced_panoramas(
    manifest_root: Path, panorama_count: int, seed: int
) -> pd.DataFrame:
    paths = sorted(manifest_root.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no manifests found under {manifest_root}")
    if panorama_count < len(paths):
        # Useful for the required one-panorama smoke test.  The production
        # pilot uses >= one panorama per city and therefore keeps every path.
        paths = paths[:panorama_count]

    base, remainder = divmod(panorama_count, len(paths))
    selected = []
    required = {
        "panoid", "heading", "jpg_offset", "jpg_size", "tar_path", "city_key",
        "lat", "lon", "year", "month", "place_id", "split",
    }
    for city_index, path in enumerate(paths):
        frame = pd.read_parquet(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        valid = frame.groupby("panoid")["heading"].agg(
            lambda values: tuple(sorted(set(map(int, values)))) == HEADINGS
        )
        valid_panoids = valid[valid].index.to_numpy()
        count = base + int(city_index < remainder)
        if len(valid_panoids) < count:
            raise ValueError(f"{path.name}: only {len(valid_panoids)} complete panoramas")
        generator = np.random.default_rng(seed + city_index)
        chosen = generator.choice(valid_panoids, size=count, replace=False)
        city_rows = frame[frame["panoid"].isin(chosen)].copy()
        city_rows = city_rows[city_rows["heading"].isin(HEADINGS)]
        city_rows["city_sample_index"] = city_index
        selected.append(city_rows)

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["city_key", "panoid", "heading"]).reset_index(drop=True)
    result["sample_index"] = np.arange(len(result), dtype=np.int32)
    result["panorama_sample_index"] = result.groupby(
        ["city_key", "panoid"], sort=True
    ).ngroup().astype(np.int32)
    counts = result.groupby(["city_key", "panoid"]).size()
    if len(result) != panorama_count * 4 or not (counts == 4).all():
        raise ValueError("balanced sample does not contain exactly four views per panorama")
    return result


def patch_fractions(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"expected 2D semantic mask, got {mask.shape}")
    height, width = mask.shape
    y_edges = np.linspace(0, height, PATCH_GRID + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, PATCH_GRID + 1, dtype=np.int64)
    result = np.zeros((PATCH_GRID, PATCH_GRID, num_classes), dtype=np.float32)
    for row in range(PATCH_GRID):
        for column in range(PATCH_GRID):
            cell = mask[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]]
            counts = np.bincount(cell.reshape(-1), minlength=num_classes)[:num_classes]
            result[row, column] = counts / max(1, cell.size)
    if not np.allclose(result.sum(axis=-1), 1.0, atol=1e-6):
        raise ValueError("patch semantic fractions do not sum to one")
    return result.astype(np.float16)


def label_palette(num_classes: int) -> np.ndarray:
    """Stable, high-contrast palette; class identity remains in the uint8 mask."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for index in range(num_classes):
        hue = (index * 0.618033988749895) % 1.0
        saturation = 0.58 + 0.32 * ((index % 3) / 2)
        value = 0.82 + 0.16 * (index % 2)
        palette[index] = np.array(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return palette


def mask_filename(row: pd.Series) -> str:
    city = str(row.city_key).replace("/", "__")
    panoid = str(row.panoid).replace("/", "_")
    return f"{int(row.sample_index):04d}__{city}__{panoid}__h{int(row.heading):03d}.png"


def make_qa_figure(
    qa_records: list,
    palette: np.ndarray,
    id2label: dict,
    output_path: Path,
) -> None:
    if not qa_records:
        return
    thumb = 320
    header = 64
    row_label = 32
    columns = ("Original", "65-class prediction", "Prediction overlay")
    canvas = Image.new("RGB", (thumb * 3, header + (thumb + row_label) * len(qa_records)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = "Mask2Former Swin-L · Mapillary Vistas (65 classes)"
    draw.text((12, 8), title, fill="black", font=font)
    for column, label in enumerate(columns):
        draw.text((column * thumb + 10, 38), label, fill="black", font=font)

    for row_index, record in enumerate(qa_records):
        y = header + row_index * (thumb + row_label)
        original = record["image"].resize((thumb, thumb), Image.Resampling.LANCZOS)
        mask = record["mask"]
        color = Image.fromarray(palette[mask], "RGB").resize(
            (thumb, thumb), Image.Resampling.NEAREST
        )
        overlay = Image.blend(original, color, 0.48)
        canvas.paste(original, (0, y))
        canvas.paste(color, (thumb, y))
        canvas.paste(overlay, (thumb * 2, y))
        top_ids = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES).argsort()[-3:][::-1]
        top_text = ", ".join(id2label[int(index)] for index in top_ids)
        caption = f"{record['city']} · h={record['heading']}° · {top_text}"
        draw.text((8, y + thumb + 8), caption, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--paper-data", type=Path, default=DEFAULT_PAPER_DATA)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--panoramas", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qa-images", type=int, default=10)
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_safe_affinity()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Swin-L pilot")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_root = args.output_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    args.paper_data.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = utc_now()

    sample = sample_balanced_panoramas(args.manifest_root, args.panoramas, args.seed)
    sample.to_csv(args.paper_data / "sample_manifest.csv", index=False)
    atomic_json(
        args.output_root / "progress.json",
        {"status": "loading_model", "started_at": started_at, "total_images": len(sample)},
    )

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_id)
    model.eval().requires_grad_(False).to("cuda")
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    if len(id2label) != NUM_CLASSES or sorted(id2label) != list(range(NUM_CLASSES)):
        raise ValueError(f"expected class IDs 0..64, got {sorted(id2label)}")
    palette = label_palette(NUM_CLASSES)

    fractions = np.empty((len(sample), PATCH_GRID, PATCH_GRID, NUM_CLASSES), dtype=np.float16)
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.uint64)
    class_images = np.zeros(NUM_CLASSES, dtype=np.uint32)
    metadata = []
    qa_records = []
    qa_indices = set(np.linspace(0, len(sample) - 1, min(args.qa_images, len(sample)), dtype=int))
    reader = TarReader()
    failures = []
    peak_memory = 0

    try:
        for position, (_, row) in enumerate(sample.iterrows()):
            mask_path = mask_root / mask_filename(row)
            image = reader.image(str(row.tar_path), int(row.jpg_offset), int(row.jpg_size))
            width, height = image.size
            inference_seconds = 0.0
            resumed = mask_path.exists()
            if resumed:
                with Image.open(mask_path) as saved_mask:
                    mask = np.asarray(saved_mask, dtype=np.uint8)
                if mask.shape != (height, width) or int(mask.max()) >= NUM_CLASSES:
                    raise ValueError(f"invalid resumed mask {mask_path}: {mask.shape}, max={mask.max()}")
            else:
                inference_started = time.monotonic()
                try:
                    inputs = processor(images=image, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to("cuda", non_blocking=True)
                    pixel_mask = inputs.get("pixel_mask")
                    if pixel_mask is not None:
                        pixel_mask = pixel_mask.to("cuda", non_blocking=True)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                    prediction = processor.post_process_semantic_segmentation(
                        outputs, target_sizes=[(height, width)]
                    )[0]
                    mask = prediction.to("cpu", dtype=torch.uint8).numpy()
                    del inputs, pixel_values, pixel_mask, outputs, prediction
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    failures.append({"sample_index": int(row.sample_index), "error": str(error)})
                    atomic_json(
                        args.output_root / "progress.json",
                        {
                            "status": "failed_cuda_oom",
                            "processed_images": position,
                            "total_images": len(sample),
                            "failures": failures,
                        },
                    )
                    raise
                inference_seconds = time.monotonic() - inference_started
                Image.fromarray(mask, mode="L").save(mask_path, optimize=True)

            if mask.shape != (height, width):
                raise ValueError(f"prediction shape {mask.shape} != source {(height, width)}")
            fractions[position] = patch_fractions(mask)
            counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
            class_pixels += counts.astype(np.uint64)
            class_images += (counts > 0).astype(np.uint32)
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
            metadata.append(
                {
                    "sample_index": int(row.sample_index),
                    "panorama_sample_index": int(row.panorama_sample_index),
                    "city_key": str(row.city_key),
                    "panoid": str(row.panoid),
                    "heading": int(row.heading),
                    "width": width,
                    "height": height,
                    "mask_path": str(mask_path),
                    "inference_seconds": inference_seconds,
                    "resumed": resumed,
                    "predicted_class_count": int((counts > 0).sum()),
                }
            )
            if position in qa_indices:
                qa_records.append(
                    {"image": image.copy(), "mask": mask.copy(), "city": row.city_key, "heading": row.heading}
                )
            if (position + 1) % 10 == 0 or position + 1 == len(sample):
                elapsed = time.monotonic() - started
                atomic_json(
                    args.output_root / "progress.json",
                    {
                        "status": "running",
                        "processed_images": position + 1,
                        "total_images": len(sample),
                        "elapsed_seconds": elapsed,
                        "images_per_second": (position + 1) / elapsed,
                        "peak_cuda_memory_gib": peak_memory / 2**30,
                        "updated_at": utc_now(),
                    },
                )
    finally:
        reader.close()

    np.save(args.output_root / "semantic_patch_fractions_65.f16.npy", fractions)
    panorama_fractions = fractions.reshape(args.panoramas, 4, PATCH_GRID, PATCH_GRID, NUM_CLASSES)
    panorama_fractions = panorama_fractions.transpose(0, 2, 1, 3, 4).reshape(
        args.panoramas, PATCH_GRID, 4 * PATCH_GRID, NUM_CLASSES
    )
    np.save(args.output_root / "panorama_semantic_fractions_14x56x65.f16.npy", panorama_fractions)
    metadata_frame = pd.DataFrame(metadata)
    metadata_frame.to_csv(args.paper_data / "prediction_metadata.csv", index=False)

    total_pixels = int(class_pixels.sum())
    prevalence = pd.DataFrame(
        {
            "class_id": np.arange(NUM_CLASSES),
            "class_name": [id2label[index] for index in range(NUM_CLASSES)],
            "pixel_count": class_pixels,
            "pixel_share": class_pixels / total_pixels,
            "images_present": class_images,
            "image_presence_share": class_images / len(sample),
        }
    ).sort_values("pixel_share", ascending=False)
    prevalence.to_csv(args.paper_data / "class_prevalence_65.csv", index=False)
    pd.DataFrame(
        {"class_id": np.arange(NUM_CLASSES), "class_name": [id2label[i] for i in range(NUM_CLASSES)]}
    ).to_csv(args.paper_data / "mapillary_vistas_class_index.csv", index=False)
    make_qa_figure(qa_records, palette, id2label, args.figure)

    elapsed = time.monotonic() - started
    report = {
        "status": "complete",
        "model_id": args.model_id,
        "task": "semantic_segmentation",
        "classes": NUM_CLASSES,
        "panoramas": args.panoramas,
        "direction_images": len(sample),
        "cities": int(sample.city_key.nunique()),
        "seed": args.seed,
        "processor_size": dict(processor.size),
        "source_resolution_counts": metadata_frame.groupby(["width", "height"]).size().rename("count").reset_index().to_dict("records"),
        "patch_fraction_shape": list(fractions.shape),
        "panorama_fraction_shape": list(panorama_fractions.shape),
        "peak_cuda_memory_gib": peak_memory / 2**30,
        "elapsed_seconds": elapsed,
        "images_per_second": len(sample) / elapsed,
        "started_at": started_at,
        "completed_at": utc_now(),
        "output_root": str(args.output_root),
        "paper_data": str(args.paper_data),
        "figure": str(args.figure),
        "failures": failures,
    }
    atomic_json(args.paper_data / "inference_report.json", report)
    atomic_json(args.output_root / "run_complete.json", report)
    atomic_json(args.output_root / "progress.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
