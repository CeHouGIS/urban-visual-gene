#!/usr/bin/env python3
"""Create deterministic, panorama-disjoint manifests for scale-up experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.multicity.config import CITIES, city_slug


DEFAULT_SOURCE = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc/filtered_manifests"
)
DEFAULT_OUTPUT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_scaleup/splits"
)
SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000, 10000)


def _city_seed(city: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{city}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _take_panorama_groups(
    frame: pd.DataFrame, ordered_panoids: np.ndarray, target_images: int
) -> tuple[pd.DataFrame, np.ndarray]:
    selected = []
    count = 0
    sizes = frame.groupby("panoid", sort=False).size()
    for panoid in ordered_panoids:
        selected.append(panoid)
        count += int(sizes.loc[panoid])
        if count >= target_images:
            break
    chosen = np.asarray(selected, dtype=object)
    subset = frame[frame["panoid"].isin(chosen)].copy()
    return subset, chosen


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _hash_indices(frame: pd.DataFrame) -> str:
    values = np.sort(frame["source_image_index"].to_numpy(np.int64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--validation-images", type=int, default=500)
    parser.add_argument("--evaluation-images", type=int, default=500)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=list(SAMPLE_SIZES))
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    city_hashes = {}
    for city in CITIES:
        slug = city_slug(city)
        source = pd.read_parquet(args.source_root / f"{slug}.parquet")
        required = {"panoid", "source_image_index", "heading"}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"{city}: missing columns {sorted(missing)}")
        if source["panoid"].isna().any():
            raise ValueError(f"{city}: null panoid values are not permitted")

        rng = np.random.default_rng(_city_seed(city, args.seed))
        panoids = source["panoid"].drop_duplicates().to_numpy(object)
        ordered = panoids[rng.permutation(len(panoids))]
        evaluation, evaluation_panos = _take_panorama_groups(
            source, ordered, args.evaluation_images
        )
        remaining_order = ordered[len(evaluation_panos):]
        validation, validation_panos = _take_panorama_groups(
            source, remaining_order, args.validation_images
        )
        training_order = remaining_order[len(validation_panos):]
        training = source[source["panoid"].isin(training_order)].copy()
        pano_rank = {panoid: i for i, panoid in enumerate(training_order)}
        training["_pano_rank"] = training["panoid"].map(pano_rank)
        training = training.sort_values(
            ["_pano_rank", "direction_index", "source_image_index"], kind="stable"
        ).drop(columns="_pano_rank").reset_index(drop=True)

        split_panos = {
            "evaluation": set(evaluation["panoid"]),
            "validation": set(validation["panoid"]),
            "training": set(training["panoid"]),
        }
        if split_panos["evaluation"] & split_panos["validation"]:
            raise AssertionError(f"{city}: evaluation/validation panorama leakage")
        if split_panos["evaluation"] & split_panos["training"]:
            raise AssertionError(f"{city}: evaluation/training panorama leakage")
        if split_panos["validation"] & split_panos["training"]:
            raise AssertionError(f"{city}: validation/training panorama leakage")
        if len(training) < max(args.sample_sizes):
            raise ValueError(
                f"{city}: only {len(training)} training images after holdout; "
                f"need {max(args.sample_sizes)}"
            )

        evaluation = evaluation.sort_values("source_image_index").reset_index(drop=True)
        validation = validation.sort_values("source_image_index").reset_index(drop=True)
        _atomic_parquet(evaluation, args.output_root / "evaluation" / f"{slug}.parquet")
        _atomic_parquet(validation, args.output_root / "validation" / f"{slug}.parquet")
        city_hashes[city] = {
            "evaluation": _hash_indices(evaluation),
            "validation": _hash_indices(validation),
            "training_pool": _hash_indices(training),
        }

        for sample_size in sorted(args.sample_sizes):
            subset = training.iloc[:sample_size].copy().sort_values(
                "source_image_index"
            ).reset_index(drop=True)
            _atomic_parquet(
                subset,
                args.output_root / f"train_n{sample_size:05d}" / f"{slug}.parquet",
            )
            heading_counts = subset["heading"].value_counts().to_dict()
            audit_rows.append(
                {
                    "city": city,
                    "sample_size": sample_size,
                    "train_images": len(subset),
                    "train_panoramas": subset["panoid"].nunique(),
                    "validation_images": len(validation),
                    "validation_panoramas": validation["panoid"].nunique(),
                    "evaluation_images": len(evaluation),
                    "evaluation_panoramas": evaluation["panoid"].nunique(),
                    "heading_0": int(heading_counts.get(0, 0)),
                    "heading_90": int(heading_counts.get(90, 0)),
                    "heading_180": int(heading_counts.get(180, 0)),
                    "heading_270": int(heading_counts.get(270, 0)),
                    "train_index_sha256": _hash_indices(subset),
                    "panoid_overlap": 0,
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_root / "split_audit.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "cities": list(CITIES),
        "sample_sizes": sorted(args.sample_sizes),
        "validation_target_images_per_city": args.validation_images,
        "evaluation_target_images_per_city": args.evaluation_images,
        "grouping": "panoid-disjoint",
        "nested_training_subsets": True,
        "city_hashes": city_hashes,
    }
    temporary = args.output_root / "split_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.output_root / "split_manifest.json")
    print(
        f"prepared {len(CITIES)} cities x {len(args.sample_sizes)} sample sizes "
        f"at {args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
