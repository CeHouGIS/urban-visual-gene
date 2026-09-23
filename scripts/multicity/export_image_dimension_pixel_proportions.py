#!/usr/bin/env python3
"""Export one row per image with 512 patch-area proportions.

Each of the 196 equal-area DINOv3 patches is assigned to the Feature-MAE
dimension with the largest latent response. D000--D511 store the fraction of
patches assigned to each dimension, so every row sums to one. Because the
224x224 input has a regular 14x14 patch grid, this is also a discrete estimate
of image-pixel-area share at patch resolution.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.multicity.config import CITIES, city_slug


ROOT = Path("outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc")
DEFAULT_TOP = ROOT / "mae/hierarchy_32_64/top1_dimensions.npy"
DEFAULT_MANIFESTS = ROOT / "filtered_manifests"
DEFAULT_OUTPUT = Path("results/image_dimension_pixel_proportions.parquet")
DEFAULT_CACHE = Path("outputs/analysis/image_dimension_pixel_proportion_shards")
PATCHES = 196
DIMENSIONS = 512
DIMENSION_COLUMNS = [f"D{x:03d}" for x in range(DIMENSIONS)]
METADATA_COLUMNS = [
    "global_image_index",
    "city_key",
    "image_index",
    "source_image_index",
    "pano_index",
    "panoid",
    "direction_index",
    "heading",
    "lat",
    "lon",
    "year",
    "month",
    "tar_path",
    "jpg_offset",
    "jpg_size",
]


def normalized_metadata(frame: pd.DataFrame, global_start: int) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(frame)))
    result["global_image_index"] = np.arange(global_start, global_start + len(frame), dtype=np.int64)
    result["city_key"] = frame["city_key"].astype(str).to_numpy()
    for column in (
        "image_index", "source_image_index", "pano_index", "direction_index",
        "jpg_offset", "jpg_size",
    ):
        result[column] = frame[column].fillna(-1).astype(np.int64).to_numpy()
    result["panoid"] = frame["panoid"].fillna("").astype(str).to_numpy()
    result["heading"] = frame["heading"].fillna(-1).astype(np.float32).to_numpy()
    result["lat"] = frame["lat"].astype(np.float64).to_numpy()
    result["lon"] = frame["lon"].astype(np.float64).to_numpy()
    result["year"] = frame["year"].fillna(-1).astype(np.int16).to_numpy()
    result["month"] = frame["month"].fillna(-1).astype(np.int8).to_numpy()
    result["tar_path"] = frame["tar_path"].fillna("").astype(str).to_numpy()
    return result[METADATA_COLUMNS]


def proportions(block: np.ndarray) -> np.ndarray:
    winners = np.asarray(block, dtype=np.int64)
    rows = np.repeat(np.arange(len(winners), dtype=np.int64), PATCHES)
    counts = np.bincount(
        rows * DIMENSIONS + winners.ravel(), minlength=len(winners) * DIMENSIONS
    ).reshape(len(winners), DIMENSIONS)
    if not np.all(counts.sum(axis=1) == PATCHES):
        raise ValueError("patch counts do not sum to 196")
    return (counts / PATCHES).astype(np.float32)


def write_city_shard(
    path: Path,
    frame: pd.DataFrame,
    top: np.ndarray,
    global_start: int,
    chunk_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    metadata = normalized_metadata(frame, global_start)
    writer = None
    try:
        for start in range(0, len(frame), chunk_size):
            end = min(start + chunk_size, len(frame))
            values = proportions(top[global_start + start : global_start + end])
            dimension_frame = pd.DataFrame(values, columns=DIMENSION_COLUMNS, copy=False)
            block = pd.concat(
                [metadata.iloc[start:end].reset_index(drop=True), dimension_frame], axis=1
            )
            table = pa.Table.from_pandas(block, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    compression_level=7,
                    use_dictionary=True,
                )
            writer.write_table(table, row_group_size=chunk_size)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(path)


def valid_shard(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows == expected_rows and parquet.metadata.num_columns == (
        len(METADATA_COLUMNS) + DIMENSIONS
    )


def merge_shards(shards: list[Path], output: Path, chunk_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = None
    try:
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=chunk_size):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    metadata = dict(table.schema.metadata or {})
                    metadata.update(
                        {
                            b"proportion_definition": (
                                b"fraction of 196 patches whose strongest Feature-MAE response is the dimension"
                            ),
                            b"value_scale": b"0 to 1; D000--D511 sum to 1 for every image",
                            b"spatial_resolution": b"14x14 patch grid from 224x224 model input",
                        }
                    )
                    schema = table.schema.with_metadata(metadata)
                    writer = pq.ParquetWriter(
                        temporary,
                        schema,
                        compression="zstd",
                        compression_level=7,
                        use_dictionary=True,
                    )
                    table = table.cast(schema)
                writer.write_table(table, row_group_size=chunk_size)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    top = np.load(args.top, mmap_mode="r")
    if top.shape != (378818, PATCHES):
        raise ValueError(f"unexpected top1 array shape: {top.shape}")
    shards = []
    cursor = 0
    manifest_rows = 0
    for city_index, city in enumerate(CITIES):
        manifest_path = args.manifests / f"{city_slug(city)}.parquet"
        frame = pd.read_parquet(manifest_path, columns=[
            "city_key", "image_index", "source_image_index", "pano_index", "panoid",
            "direction_index", "heading", "lat", "lon", "year", "month", "tar_path",
            "jpg_offset", "jpg_size",
        ])
        shard = args.cache / f"{city_index:02d}_{city_slug(city)}.parquet"
        if valid_shard(shard, len(frame)):
            print(f"shard [{city_index + 1:02d}/30] cached {city}: {len(frame):,}", flush=True)
        else:
            write_city_shard(shard, frame, top, cursor, args.chunk_size)
            print(f"shard [{city_index + 1:02d}/30] wrote  {city}: {len(frame):,}", flush=True)
        shards.append(shard)
        cursor += len(frame)
        manifest_rows += len(frame)
    if manifest_rows != len(top):
        raise ValueError(f"manifest rows {manifest_rows} != activation rows {len(top)}")

    merge_shards(shards, args.output, args.chunk_size)
    parquet = pq.ParquetFile(args.output)
    if parquet.metadata.num_rows != len(top):
        raise ValueError("final parquet row count mismatch")
    sampled_errors = []
    row_groups = sorted(set([0, parquet.num_row_groups // 2, parquet.num_row_groups - 1]))
    for row_group in row_groups:
        sample = parquet.read_row_group(row_group, columns=DIMENSION_COLUMNS).to_pandas()
        sampled_errors.append(np.max(np.abs(sample.sum(axis=1).to_numpy() - 1.0)))
    maximum_error = float(max(sampled_errors))
    if maximum_error > 1e-5:
        raise ValueError(f"dimension proportions fail row-sum check: {maximum_error}")

    report = {
        "rows": parquet.metadata.num_rows,
        "metadata_columns": len(METADATA_COLUMNS),
        "dimension_columns": DIMENSIONS,
        "total_columns": parquet.metadata.num_columns,
        "value_scale": "0 to 1",
        "row_sum": 1.0,
        "patches_per_image": PATCHES,
        "maximum_sampled_row_sum_error": maximum_error,
        "output": str(args.output.resolve()),
        "input": str(args.top.resolve()),
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
