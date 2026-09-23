"""Choose 12,800 images while maximizing unique-panorama coverage per city."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multicity.config import CITIES, city_slug
from scripts.multicity.patch_config import (
    IMAGES_PER_CITY,
    OUTPUT_ROOT,
    SEED,
    SOURCE_ROOT,
    image_manifest_path,
)


def _seed(city: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"image:{seed}:{city}".encode()).digest()[:8], "little")


def select_city(city: str, source_root: Path, output_root: Path, seed: int) -> dict:
    source = source_root / "manifests" / f"{city_slug(city)}.parquet"
    frame = pd.read_parquet(source).sort_values(["pano_index", "direction_index"])
    groups = [np.asarray(v) for v in frame.groupby("pano_index", sort=True).indices.values()]
    if sum(len(v) for v in groups) < IMAGES_PER_CITY:
        raise ValueError(f"{city}: only {len(frame):,} images available")
    rng = np.random.default_rng(_seed(city, seed))
    # Round-robin across panoramas: take one random heading from every pano
    # before taking a second heading from any pano.  This yields the broadest
    # spatial coverage possible while still reaching 12,800 images from the
    # 10,000 complete panoramas in the source manifest.
    per_pano = [rng.permutation(indices) for indices in groups]
    chosen_rows: list[int] = []
    for direction_round in range(max(map(len, per_pano))):
        for pano_i in rng.permutation(len(per_pano)):
            if direction_round < len(per_pano[pano_i]):
                chosen_rows.append(int(per_pano[pano_i][direction_round]))
                if len(chosen_rows) == IMAGES_PER_CITY:
                    break
        if len(chosen_rows) == IMAGES_PER_CITY:
            break
    selected = frame.iloc[chosen_rows].copy().reset_index(drop=True)
    selected["image_index"] = np.arange(len(selected), dtype=np.int32)
    output = image_manifest_path(city, output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output, index=False, compression="zstd")
    return {
        "city_key": city,
        "images": len(selected),
        "unique_panos": int(selected["panoid"].nunique()),
        "heading_counts": {
            str(int(k)): int(v) for k, v in selected["heading"].value_counts().sort_index().items()
        },
        "manifest": str(output),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    reports = []
    for i, city in enumerate(args.cities, 1):
        report = select_city(city, args.source_root, args.output_root, args.seed)
        reports.append(report)
        print(f"[{i:02d}/{len(args.cities)}] {city}: {report['images']:,} images", flush=True)
    summary = {
        "n_cities": len(reports),
        "images_per_city": IMAGES_PER_CITY,
        "total_images": sum(r["images"] for r in reports),
        "sampling": "round-robin random headings, maximizing unique panoramas first",
        "seed": args.seed,
        "cities": reports,
    }
    path = args.output_root / "image_manifest_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"selected {summary['total_images']:,} images -> {path}", flush=True)


if __name__ == "__main__":
    main()
