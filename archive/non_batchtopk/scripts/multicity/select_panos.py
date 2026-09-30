"""Select exactly 10,000 complete four-heading panoramas per city.

The source packages contain one parquet sidecar per TAR shard.  We sample
shards in a deterministic random order (which spreads the sample over the
archive), retain only complete 0/90/180/270 panoramas, then sample panorama
IDs with a city-specific deterministic seed.  No image bytes are read here.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.multicity.config import (
    CITIES,
    DATA_ROOT,
    HEADINGS,
    OUTPUT_ROOT,
    PANOS_PER_CITY,
    SEED,
    city_slug,
    manifest_path,
)

READ_COLUMNS = (
    "panoid", "heading", "shard", "jpg_offset", "jpg_size",
    "lat", "lon", "year", "month", "place_id", "split",
)


def _city_seed(city_key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{city_key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _usable_shards(city_dir: Path) -> list[str]:
    names = set(os.listdir(city_dir))
    parquet_stems = {
        name[:-len(".parquet")]
        for name in names
        if name.startswith("shard-") and name.endswith(".parquet")
    }
    tar_stems = {
        name[:-len(".tar")]
        for name in names
        if name.startswith("shard-") and name.endswith(".tar")
    }
    return sorted(parquet_stems & tar_stems)


def _complete_pano_ids(rows: pd.DataFrame) -> np.ndarray:
    rows = rows[rows["heading"].isin(HEADINGS)]
    counts = rows.groupby("panoid", sort=False)["heading"].nunique()
    return counts.index[counts.eq(len(HEADINGS))].to_numpy(dtype=object)


def select_city(
    city_key: str,
    n_panos: int = PANOS_PER_CITY,
    seed: int = SEED,
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    oversample: float = 1.25,
) -> dict:
    """Build one city's image-level manifest and return summary statistics."""
    city_dir = data_root / city_key
    shards = _usable_shards(city_dir)
    if not shards:
        raise FileNotFoundError(f"No complete parquet/TAR shard pairs in {city_dir}")

    rng = np.random.default_rng(_city_seed(city_key, seed))
    shard_order = np.asarray(shards, dtype=object)
    rng.shuffle(shard_order)
    chunks: list[pd.DataFrame] = []
    complete_ids = np.empty(0, dtype=object)
    target_candidates = int(np.ceil(n_panos * oversample))

    for shard_i, stem in enumerate(shard_order, start=1):
        table = pq.read_table(city_dir / f"{stem}.parquet", columns=list(READ_COLUMNS))
        chunk = table.to_pandas()
        # Source packages may include held-out rows.  This is an unsupervised
        # dictionary fit, but using the source training split keeps held-out
        # imagery untouched for future evaluation.
        if "split" in chunk and chunk["split"].notna().any():
            chunk = chunk[chunk["split"].eq("train")]
        chunk = chunk[chunk["heading"].isin(HEADINGS)].copy()
        chunk["tar_path"] = str(city_dir / f"{stem}.tar")
        chunks.append(chunk)

        # Avoid a full groupby after every small Hong Kong shard.
        if shard_i % 8 == 0 or shard_i == len(shard_order):
            combined = pd.concat(chunks, ignore_index=True)
            combined = combined.drop_duplicates(["panoid", "heading"], keep="first")
            complete_ids = _complete_pano_ids(combined)
            if len(complete_ids) >= target_candidates:
                break

    if len(complete_ids) < n_panos:
        raise RuntimeError(
            f"{city_key}: only {len(complete_ids):,} complete training panos "
            f"found in {len(shards):,} usable shards; need {n_panos:,}"
        )

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.drop_duplicates(["panoid", "heading"], keep="first")
    complete_ids = np.sort(_complete_pano_ids(combined))
    chosen = rng.choice(complete_ids, size=n_panos, replace=False)
    pano_order = {panoid: i for i, panoid in enumerate(chosen.tolist())}

    selected = combined[combined["panoid"].isin(pano_order)].copy()
    selected["pano_index"] = selected["panoid"].map(pano_order).astype("int32")
    heading_order = {heading: i for i, heading in enumerate(HEADINGS)}
    selected["direction_index"] = selected["heading"].map(heading_order).astype("int8")
    selected["city_key"] = city_key
    selected = selected.sort_values(["pano_index", "direction_index"])

    expected_rows = n_panos * len(HEADINGS)
    if len(selected) != expected_rows:
        raise AssertionError(
            f"{city_key}: selected {len(selected):,} rows, expected {expected_rows:,}"
        )
    observed = selected.groupby("pano_index")["direction_index"].nunique()
    if not observed.eq(len(HEADINGS)).all():
        raise AssertionError(f"{city_key}: incomplete panorama survived selection")

    keep = [
        "city_key", "pano_index", "panoid", "direction_index", "heading",
        "tar_path", "jpg_offset", "jpg_size", "lat", "lon", "year",
        "month", "place_id", "split",
    ]
    output = manifest_path(city_key, output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected[keep].to_parquet(output, index=False, compression="zstd")
    return {
        "city_key": city_key,
        "city_slug": city_slug(city_key),
        "panos": n_panos,
        "images": expected_rows,
        "candidate_panos": int(len(complete_ids)),
        "shards_read": len(chunks),
        "usable_shards": len(shards),
        "manifest": str(output),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--panos-per-city", type=int, default=PANOS_PER_CITY)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reports = []
    for i, city in enumerate(args.cities, start=1):
        out = manifest_path(city, args.output_root)
        if out.exists() and not args.force:
            frame = pd.read_parquet(out, columns=["pano_index"])
            n_panos = int(frame["pano_index"].nunique())
            if n_panos == args.panos_per_city and len(frame) == n_panos * 4:
                report = {
                    "city_key": city, "panos": n_panos, "images": len(frame),
                    "manifest": str(out), "status": "existing",
                }
                reports.append(report)
                print(f"[{i:02d}/{len(args.cities)}] {city}: existing manifest", flush=True)
                continue
        report = select_city(
            city, n_panos=args.panos_per_city, seed=args.seed,
            output_root=args.output_root,
        )
        report["status"] = "created"
        reports.append(report)
        print(
            f"[{i:02d}/{len(args.cities)}] {city}: {report['panos']:,} panos, "
            f"{report['images']:,} images from {report['shards_read']} shards",
            flush=True,
        )

    summary = {
        "seed": args.seed,
        "panos_per_city": args.panos_per_city,
        "headings": list(HEADINGS),
        "n_cities": len(reports),
        "total_panos": sum(r["panos"] for r in reports),
        "total_images": sum(r["images"] for r in reports),
        "cities": reports,
    }
    summary_path = args.output_root / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"manifest selection complete -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
