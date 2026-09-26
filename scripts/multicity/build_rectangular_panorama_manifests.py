#!/usr/bin/env python3
"""Recover complete four-heading panoramas for the clean training image set.

The quality-controlled training manifests select images independently, so
most panorama IDs occur with fewer than four selected directions.  The source
TAR sidecars still contain all four captures.  This script deduplicates the
training set by panorama ID, reads only the sidecars referenced by those
training rows, and writes one complete 0/90/180/270 manifest per city.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.multicity.config import CITIES, HEADINGS, city_slug


DEFAULT_DATA_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "feature_mae_n30x12800_qc"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "rectangular_panorama_frozen_mae_n30"
)
SOURCE_COLUMNS = [
    "panoid", "heading", "jpg_offset", "jpg_size", "lat", "lon", "year",
    "month", "place_id", "split",
]


def assert_safe_affinity() -> None:
    if hasattr(os, "sched_getaffinity"):
        forbidden = set(os.sched_getaffinity(0)).intersection({8, 9})
        if forbidden:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(forbidden)}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def build_city(city_key: str, data_root: Path, output_root: Path, force: bool) -> dict:
    slug = city_slug(city_key)
    source_path = data_root / "filtered_manifests" / f"{slug}.parquet"
    output_path = output_root / "manifests" / f"{slug}.parquet"
    report_path = output_root / "manifest_reports" / f"{slug}.json"
    if output_path.exists() and report_path.exists() and not force:
        report = json.loads(report_path.read_text())
        report["status"] = "existing"
        return report

    selected = pd.read_parquet(source_path)
    required = {"panoid", "tar_path", "heading"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"{city_key}: filtered manifest lacks {sorted(missing)}")
    selected["panoid"] = selected["panoid"].astype(str)
    selected_counts = selected.groupby("panoid").size().rename("clean_direction_count")
    # A panorama is wholly contained in one source shard in the package
    # contract.  Keep the first path deterministically and check this below.
    pano_sources = (
        selected.sort_values(["panoid", "heading"])
        .drop_duplicates("panoid")[["panoid", "tar_path"]]
        .reset_index(drop=True)
    )

    recovered = []
    sidecars_read: set[Path] = set()
    for tar_string, targets in pano_sources.groupby("tar_path", sort=True):
        tar_path = Path(tar_string)
        sidecar = tar_path.with_suffix(".parquet")
        if not sidecar.exists():
            raise FileNotFoundError(f"missing source sidecar: {sidecar}")
        frame = pd.read_parquet(sidecar, columns=SOURCE_COLUMNS)
        wanted = set(targets["panoid"].astype(str))
        frame["panoid"] = frame["panoid"].astype(str)
        frame = frame[
            frame["panoid"].isin(wanted) & frame["heading"].isin(HEADINGS)
        ].copy()
        frame["tar_path"] = str(tar_path)
        recovered.append(frame)
        sidecars_read.add(sidecar)

    directions = pd.concat(recovered, ignore_index=True)
    directions["heading"] = directions["heading"].astype(int)
    directions = directions.drop_duplicates(["panoid", "heading"], keep="first")
    direction_counts = directions.groupby("panoid")["heading"].nunique()
    complete_ids = set(direction_counts[direction_counts.eq(len(HEADINGS))].index)
    missing_ids = sorted(set(pano_sources["panoid"]) - complete_ids)
    # A panorama can straddle two sequential TAR shards at an archive boundary.
    # Read adjacent sidecars for the rare incomplete cases, then fall back to a
    # city-wide sidecar search only if adjacency did not recover all headings.
    if missing_ids:
        source_by_pano = pano_sources.set_index("panoid")["tar_path"].to_dict()
        candidates: set[Path] = set()
        for panoid in missing_ids:
            tar_path = Path(source_by_pano[panoid])
            match = re.fullmatch(r"shard-(\d+)", tar_path.stem)
            if match:
                shard = int(match.group(1))
                for neighbour in (shard - 1, shard + 1):
                    if neighbour >= 0:
                        candidates.add(tar_path.parent / f"shard-{neighbour:06d}.parquet")
        fallback = []
        remaining = set(missing_ids)
        for sidecar in sorted(candidates):
            if not sidecar.exists():
                continue
            frame = pd.read_parquet(sidecar, columns=SOURCE_COLUMNS)
            frame["panoid"] = frame["panoid"].astype(str)
            frame = frame[
                frame["panoid"].isin(remaining) & frame["heading"].isin(HEADINGS)
            ].copy()
            if len(frame):
                frame["tar_path"] = str(sidecar.with_suffix(".tar"))
                fallback.append(frame)
            sidecars_read.add(sidecar)
        if fallback:
            directions = pd.concat([directions, *fallback], ignore_index=True)
            directions = directions.drop_duplicates(["panoid", "heading"], keep="first")
        direction_counts = directions.groupby("panoid")["heading"].nunique()
        complete_ids = set(direction_counts[direction_counts.eq(len(HEADINGS))].index)
        missing_ids = sorted(set(pano_sources["panoid"]) - complete_ids)
    if missing_ids:
        city_directory = Path(pano_sources.iloc[0]["tar_path"]).parent
        fallback = []
        remaining = set(missing_ids)
        for sidecar in sorted(city_directory.glob("shard-*.parquet")):
            frame = pd.read_parquet(sidecar, columns=SOURCE_COLUMNS)
            frame["panoid"] = frame["panoid"].astype(str)
            frame = frame[
                frame["panoid"].isin(remaining) & frame["heading"].isin(HEADINGS)
            ].copy()
            if len(frame):
                frame["tar_path"] = str(sidecar.with_suffix(".tar"))
                fallback.append(frame)
            sidecars_read.add(sidecar)
        if fallback:
            directions = pd.concat([directions, *fallback], ignore_index=True)
            directions = directions.drop_duplicates(["panoid", "heading"], keep="first")
        direction_counts = directions.groupby("panoid")["heading"].nunique()
        complete_ids = set(direction_counts[direction_counts.eq(len(HEADINGS))].index)
        missing_ids = sorted(set(pano_sources["panoid"]) - complete_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"{city_key}: {len(missing_ids)} training panoramas cannot be recovered "
            f"with four headings; first IDs: {preview}"
        )
    directions = directions[directions["panoid"].isin(complete_ids)].copy()
    directions["city_key"] = city_key
    directions["clean_direction_count"] = directions["panoid"].map(selected_counts).astype("uint8")
    directions["direction_index"] = directions["heading"].map(
        {heading: index for index, heading in enumerate(HEADINGS)}
    ).astype("uint8")
    pano_order = {panoid: index for index, panoid in enumerate(sorted(complete_ids))}
    directions["pano_index"] = directions["panoid"].map(pano_order).astype("int32")
    directions = directions.sort_values(["pano_index", "direction_index"]).reset_index(drop=True)
    expected_headings = list(HEADINGS) * len(complete_ids)
    if directions["heading"].tolist() != expected_headings:
        raise AssertionError(f"{city_key}: recovered heading order is invalid")
    if len(directions) != len(complete_ids) * len(HEADINGS):
        raise AssertionError(f"{city_key}: recovered manifest is not four rows per panorama")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    directions.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city_key": city_key,
        "selected_clean_images": int(len(selected)),
        "unique_training_panoramas": int(selected["panoid"].nunique()),
        "complete_panoramas": int(len(complete_ids)),
        "recovered_direction_images": int(len(directions)),
        "source_sidecars_read": int(len(sidecars_read)),
        "headings": list(HEADINGS),
        "manifest": str(output_path),
        "status": "created",
    }
    save_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--city", action="append", choices=CITIES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_safe_affinity()
    cities = args.city or list(CITIES)
    reports = []
    for index, city in enumerate(cities, 1):
        report = build_city(city, args.data_root, args.output_root, args.force)
        reports.append(report)
        print(
            f"[{index:02d}/{len(cities)}] {city}: "
            f"{report['complete_panoramas']:,} complete panoramas ({report['status']})",
            flush=True,
        )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cities": len(reports),
        "unique_training_panoramas": sum(x["unique_training_panoramas"] for x in reports),
        "direction_images": sum(x["recovered_direction_images"] for x in reports),
        "reports": reports,
    }
    save_json(args.output_root / "manifest_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
