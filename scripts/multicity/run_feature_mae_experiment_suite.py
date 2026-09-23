#!/usr/bin/env python3
"""Run the resumable Feature-MAE sensitivity and seed suite sequentially.

Every run uses the same city-balanced 3,200-image-per-city epoch budget.  The
published full-data seed-42 checkpoint remains untouched and is evaluated
separately as the final model.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_controlled_suite"
)

CONFIGURATIONS = (
    # Mask-ratio sensitivity at W=512.
    {"experiment_id": "mask050_w512_d4_seed42", "mask_ratio": 0.50, "width": 512, "depth": 4, "seed": 42},
    {"experiment_id": "mask075_w512_d4_seed42", "mask_ratio": 0.75, "width": 512, "depth": 4, "seed": 42},
    {"experiment_id": "mask090_w512_d4_seed42", "mask_ratio": 0.90, "width": 512, "depth": 4, "seed": 42},
    # Latent-width sensitivity at mask=75%; centre configuration above is shared.
    {"experiment_id": "mask075_w256_d4_seed42", "mask_ratio": 0.75, "width": 256, "depth": 4, "seed": 42},
    {"experiment_id": "mask075_w1024_d4_seed42", "mask_ratio": 0.75, "width": 1024, "depth": 4, "seed": 42},
    # Independent initialisations at the selected centre configuration.
    {"experiment_id": "mask075_w512_d4_seed43", "mask_ratio": 0.75, "width": 512, "depth": 4, "seed": 43},
    {"experiment_id": "mask075_w512_d4_seed44", "mask_ratio": 0.75, "width": 512, "depth": 4, "seed": 44},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--images-per-city", type=int, default=16)
    parser.add_argument("--images-per-city-per-epoch", type=int, default=3200)
    parser.add_argument("--validation-images", type=int, default=128)
    parser.add_argument("--only", nargs="*", help="optional experiment IDs")
    args = parser.parse_args()
    if args.images_per_city_per_epoch % args.images_per_city:
        parser.error("images-per-city-per-epoch must divide by images-per-city")
    batches = args.images_per_city_per_epoch // args.images_per_city
    selected = [
        config for config in CONFIGURATIONS
        if not args.only or config["experiment_id"] in set(args.only)
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": args.epochs,
        "images_per_city_per_epoch": args.images_per_city_per_epoch,
        "validation_images_per_city": args.validation_images,
        "configurations": selected,
        "resumable": True,
    }
    (args.output_root / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for index, config in enumerate(selected, 1):
        output = args.output_root / config["experiment_id"]
        report = output / "training_report.json"
        if report.is_file():
            print(f"[{index}/{len(selected)}] skip complete {config['experiment_id']}", flush=True)
            continue
        # Wider models receive a smaller microbatch; all remain far below the
        # 12-GiB device limit and preserve the same optimizer batch definition.
        microbatch = 80 if config["width"] == 1024 else 160
        command = [
            sys.executable, "-m", "scripts.multicity.train_feature_mae",
            "--output-dir", str(output),
            "--width", str(config["width"]),
            "--mask-ratio", str(config["mask_ratio"]),
            "--encoder-layers", str(config["depth"]),
            "--epochs", str(args.epochs),
            "--images-per-city", str(args.images_per_city),
            "--microbatch", str(microbatch),
            "--gradient-accumulation", "1",
            "--max-batches-per-epoch", str(batches),
            "--validation-images", str(args.validation_images),
            "--seed", str(config["seed"]),
        ]
        print(f"[{index}/{len(selected)}] run {config['experiment_id']}", flush=True)
        subprocess.run(command, check=True)
    print("Feature-MAE controlled suite complete", flush=True)


if __name__ == "__main__":
    main()
