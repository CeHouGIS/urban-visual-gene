#!/usr/bin/env python3
"""Build web assets for spatially comparing 512D activations and semantics."""
from __future__ import annotations

import scripts._env  # noqa: F401  (must precede numpy / pandas)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.multicity.analyze_feature_mae_semantic_alignment import (
    TarReader,
    city_slug,
    palette,
    read_panorama,
    read_semantic_mask,
)


ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT_DATA = (
    ROOT
    / "paper/data/semantic_alignment/feature_mae_mapillary_alignment_n60000"
)
SEMANTIC_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "mask2former_swin_l_mapillary_n2000_per_city"
)
SEMANTIC_PAPER = Path(
    "/workplace/urban_visual_gene/paper/data/semantic_alignment/"
    "mask2former_swin_l_mapillary_n2000_per_city"
)
FEATURE_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "rectangular_panorama_frozen_mae_n30/predictions"
)
DEFAULT_OUTPUT = ROOT / "dashboard/semantic_alignment/spatial"
ROWS = 14
COLS = 56
DIMENSIONS = 512


def build_assets(output: Path, width: int = 1280) -> dict:
    selected = pd.read_csv(ALIGNMENT_DATA / "selected_20_points.csv")
    metrics = pd.read_csv(ALIGNMENT_DATA / "dimension_semantic_profiles.csv").sort_values(
        "dimension_id"
    )
    manifest = pd.read_csv(SEMANTIC_PAPER / "sample_manifest.csv")
    classes = pd.read_csv(SEMANTIC_PAPER / "mapillary_vistas_class_index.csv").sort_values(
        "class_id"
    )
    thresholds = metrics.high_activation_threshold.to_numpy(np.float32)
    if thresholds.shape != (DIMENSIONS,):
        raise ValueError("expected one activation threshold for every D dimension")

    output.mkdir(parents=True, exist_ok=True)
    reader = TarReader()
    records = []
    latent_cache: dict[str, np.ndarray] = {}
    try:
        for sample in selected.itertuples(index=False):
            sample_rows = manifest[
                manifest.panorama_sample_index.eq(int(sample.panorama_sample_index))
            ]
            if len(sample_rows) != 4:
                raise ValueError(f"{sample.sample_id} does not have four directions")
            original = read_panorama(sample_rows, reader)
            semantic_mask = read_semantic_mask(sample_rows, SEMANTIC_ROOT / "masks")
            height = round(original.height * width / original.width)
            size = (width, height)
            original = original.resize(size, Image.Resampling.LANCZOS)
            semantic_mask = semantic_mask.resize(size, Image.Resampling.NEAREST)

            original_name = f"{sample.sample_id}_streetview.webp"
            mask_name = f"{sample.sample_id}_semantic_ids.png"
            activation_name = f"{sample.sample_id}_activation_512x14x56.u8"
            original.save(output / original_name, "WEBP", quality=84, method=6)
            semantic_mask.save(output / mask_name, "PNG", optimize=True)

            slug = city_slug(str(sample.city_key))
            if slug not in latent_cache:
                latent_cache[slug] = np.load(
                    FEATURE_ROOT
                    / slug
                    / "full"
                    / "panorama_feature_mae_latent.f16.npy",
                    mmap_mode="r",
                )
            feature = np.asarray(
                latent_cache[slug][int(sample.pano_index)], dtype=np.float32
            )
            if feature.shape != (ROWS, COLS, DIMENSIONS):
                raise ValueError(
                    f"unexpected feature shape for {sample.sample_id}: {feature.shape}"
                )
            dimension_major = feature.transpose(2, 0, 1)
            medians = np.median(dimension_major, axis=(1, 2))
            scales = np.maximum(thresholds - medians, 1e-6)
            intensity = np.clip(
                (dimension_major - medians[:, None, None])
                / scales[:, None, None],
                0,
                1,
            )
            quantized = np.rint(intensity * 255).astype(np.uint8)
            (output / activation_name).write_bytes(quantized.tobytes(order="C"))

            records.append(
                {
                    "id": str(sample.sample_id),
                    "city": str(sample.city_key).replace("/", " · "),
                    "panoid": str(sample.panoid),
                    "latitude": round(float(sample.lat), 7),
                    "longitude": round(float(sample.lon), 7),
                    "year": int(sample.year),
                    "month": int(sample.month),
                    "original": f"semantic_alignment/spatial/{original_name}",
                    "semantic_mask": f"semantic_alignment/spatial/{mask_name}",
                    "activation": f"semantic_alignment/spatial/{activation_name}",
                    "width": width,
                    "height": height,
                }
            )
            print(f"built {sample.sample_id}: {sample.city_key}", flush=True)
    finally:
        reader.close()

    colours = palette(len(classes)).tolist()
    payload = {
        "version": 1,
        "grid": [ROWS, COLS],
        "dimensions": DIMENSIONS,
        "headings": [0, 90, 180, 270],
        "activation_encoding": (
            "uint8 dimension-major; 0=image median or below; "
            "255=global high-activation threshold or above"
        ),
        "classes": [
            {
                "id": int(row.class_id),
                "name": str(row.class_name),
                "colour": colours[int(row.class_id)],
            }
            for row in classes.itertuples(index=False)
        ],
        "samples": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()
    payload = build_assets(args.output, args.width)
    total_bytes = sum(path.stat().st_size for path in args.output.iterdir())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(payload["samples"]),
                "classes": len(payload["classes"]),
                "bytes": total_bytes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
