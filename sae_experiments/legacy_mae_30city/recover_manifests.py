"""Recover original Feature-MAE QC row identities from its archived image table.

This restores only the fields needed to select cached DINOv3 token rows. It
does not fabricate missing token arrays or original JPEG/TAR paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .train import CITIES, DEFAULT_DATA, EXPECTED_IMAGES, ROOT, write_json

SOURCE = ROOT / "archive/non_batchtopk/paper/data/image_layout_archetypes/image_layout_archetypes.parquet"


def recover(source: Path, output: Path) -> dict:
    frame = pd.read_parquet(source, columns=["city_key", "source_image_index", "panoid",
                                                  "heading", "image_index", "global_image_index"])
    if len(frame) != EXPECTED_IMAGES or set(frame.city_key) != set(CITIES):
        raise ValueError("archived image rows do not match the old 30-city QC cohort")
    if frame.global_image_index.nunique() != EXPECTED_IMAGES:
        raise ValueError("duplicate global image index")
    prepared = []
    for city in CITIES:
        rows = frame[frame.city_key.eq(city)].sort_values("image_index").copy()
        if rows.panoid.isna().any() or rows.source_image_index.isna().any():
            raise ValueError(city + ": missing panoid or source image index")
        if rows.image_index.to_numpy().tolist() != list(range(len(rows))):
            raise ValueError(city + ": image order is not contiguous")
        if rows.source_image_index.duplicated().any():
            raise ValueError(city + ": duplicate source image index")
        prepared.append((city, rows))
    output.mkdir(parents=True, exist_ok=True)
    counts = []
    for city, rows in prepared:
        target = output / (city.replace("/", "__") + ".parquet")
        if target.exists():
            actual = pd.read_parquet(target, columns=["source_image_index", "panoid"])
            if not actual.reset_index(drop=True).equals(rows[["source_image_index", "panoid"]].reset_index(drop=True)):
                raise ValueError(str(target) + ": existing file disagrees with archive")
        else:
            temporary = target.with_suffix(".parquet.tmp")
            rows[["source_image_index", "panoid", "heading", "image_index", "global_image_index"]].to_parquet(
                temporary, index=False, compression="zstd")
            temporary.replace(target)
        counts.append({"city": city, "images": len(rows), "panoramas": rows.panoid.nunique()})
    report = {"status": "restored_legacy_qc_row_mapping", "source": str(source.resolve()),
              "source_rows": len(frame), "output": str(output.resolve()), "cities": counts,
              "limit": "no JPEG/TAR paths and no DINOv3 token arrays in this archive"}
    write_json(output.parent / "recovery_report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_DATA / "filtered_manifests")
    args = p.parse_args()
    result = recover(args.source, args.output)
    print(json.dumps({"status": result["status"], "rows": result["source_rows"],
                      "cities": len(result["cities"]), "output": result["output"]}, indent=2))


if __name__ == "__main__":
    main()
