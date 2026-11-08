"""Count actual on-disk GoogleSV directional images for the 300-city cohort."""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CITY_MAP = (ROOT / "experiments/global307_baseline_w1024_k8/configs"
            / "final_global_300_google_city_map.json")
IMAGE_ROOT = Path("/host/root/mnt/nas/huangyj/GoogleSV/images")
INVENTORY = Path(__file__).resolve().parent / "results/google_city_counts.csv"
OUTPUT = Path(__file__).resolve().parent / "results/google_actual_images.csv"
HEADINGS = {"0": 1, "90": 2, "180": 4, "270": 8}
FIELDS = ["city", "metadata_candidate_panoids", "actual_jpg_files",
          "directional_jpg_files", "disk_panoids_any_image",
          "disk_panoids_four_images", "disk_panoids_partial",
          "other_jpg_files"]


def scan_city(city: str, root: Path, metadata_count: int):
    totals = [0] * 5
    for current, _, filenames in os.walk(root):
        # os.walk lists every filename once; count each shard independently to
        # keep memory bounded even for multimillion-panorama cities.
        masks = {}
        for name in filenames:
            if not name.lower().endswith(".jpg"):
                continue
            totals[0] += 1
            pid, separator, heading = name[:-4].rpartition("_")
            bit = HEADINGS.get(heading)
            if not separator or not pid or bit is None:
                totals[2] += 1
                continue
            totals[1] += 1
            masks[pid] = masks.get(pid, 0) | bit
        totals[3] += sum(mask == 15 for mask in masks.values())
        totals[4] += sum(mask != 15 for mask in masks.values())
    return {"city": city, "metadata_candidate_panoids": metadata_count,
            "actual_jpg_files": totals[0], "directional_jpg_files": totals[1],
            "disk_panoids_any_image": totals[3] + totals[4],
            "disk_panoids_four_images": totals[3],
            "disk_panoids_partial": totals[4], "other_jpg_files": totals[2]}


def run(output: Path, workers: int):
    if workers < 1:
        raise ValueError("workers must be positive")
    city_map = json.loads(CITY_MAP.read_text())
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        metadata = {row["city"]: int(row["sample_count"]) for row in csv.DictReader(stream)}
    if set(city_map.values()) != set(metadata):
        raise ValueError("City inventory and city map differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".progress.csv")
    prior = {}
    if checkpoint.exists():
        with checkpoint.open(newline="", encoding="utf-8") as stream:
            prior = {row["city"]: row for row in csv.DictReader(stream)}
        for city, row in prior.items():
            if city not in metadata or int(row["metadata_candidate_panoids"]) != metadata[city]:
                raise ValueError(f"Stale checkpoint row: {city}")
    remaining = sorted(set(metadata) - set(prior))
    with checkpoint.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not prior:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(scan_city, city, IMAGE_ROOT / city, metadata[city]): city
                       for city in remaining}
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                stream.flush()
                print(f"{len(prior) + 1}/{len(metadata)} {row['city']}: "
                      f"{row['actual_jpg_files']:,} images, "
                      f"{row['disk_panoids_four_images']:,} complete panoramas", flush=True)
                prior[row["city"]] = row
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for city in sorted(metadata):
            writer.writerow(prior[city])
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.workers)
