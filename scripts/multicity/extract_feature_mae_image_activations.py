#!/usr/bin/env python3
"""Extract one robust 512-D Feature-MAE activation profile per image.

For each latent dimension, the image score is the mean of its 20 strongest
spatial patches. This retains local visual-element responses while being less
sensitive than a single maximum patch.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import mmap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.multicity.config import CITIES, city_slug
from scripts.multicity.patch_config import OUTPUT_ROOT as TOKEN_ROOT, token_path
from scripts.multicity.train_feature_mae import FeatureMAE


DEFAULT_DATA_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/analysis/all_city_umap_activation/image_activations.npy"),
    )
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--top-patches", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint_path = args.data_root / "mae" / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FeatureMAE(
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_width=int(config["decoder_width"]),
        decoder_layers=int(config["decoder_layers"]),
        mask_ratio=float(config["mask_ratio"]),
    )
    model.load_state_dict(checkpoint["model"])
    model = model.eval().requires_grad_(False).to("cuda")

    manifests = []
    counts = []
    for city in CITIES:
        frame = pd.read_parquet(
            args.data_root / "filtered_manifests" / f"{city_slug(city)}.parquet",
            columns=["source_image_index"],
        )
        manifests.append(frame)
        counts.append(len(frame))
    total = sum(counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.json")
    completed_cities = 0
    if args.output.is_file() and progress_path.is_file():
        activations = np.load(args.output, mmap_mode="r+")
        progress = json.loads(progress_path.read_text())
        completed_cities = int(progress["completed_cities"])
        if activations.shape != (total, model.width):
            raise ValueError(f"unexpected activation cache shape {activations.shape}")
    elif args.output.is_file():
        raise RuntimeError(f"activation cache exists without progress file: {args.output}")
    else:
        activations = np.lib.format.open_memmap(
            args.output, mode="w+", dtype=np.float16, shape=(total, model.width)
        )

    cursor = sum(counts[:completed_cities])
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for city_i in range(completed_cities, len(CITIES)):
            city = CITIES[city_i]
            ids = manifests[city_i]["source_image_index"].to_numpy(np.int64)
            tokens = np.load(token_path(city, args.token_root), mmap_mode="r")
            if hasattr(tokens._mmap, "madvise"):
                tokens._mmap.madvise(mmap.MADV_SEQUENTIAL)
            for start in range(0, len(ids), args.batch):
                image_ids = ids[start : start + args.batch]
                block = np.asarray(tokens[image_ids], dtype=np.float32)
                batch = torch.from_numpy(block).to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent = model.encode_full(batch)
                scores = latent.float().topk(args.top_patches, dim=1).values.mean(dim=1)
                activations[cursor + start : cursor + start + len(image_ids)] = (
                    scores.cpu().numpy().astype(np.float16)
                )
            cursor += len(ids)
            activations.flush()
            progress_path.write_text(
                json.dumps(
                    {
                        "completed_cities": city_i + 1,
                        "images_written": cursor,
                        "total_images": total,
                    },
                    indent=2,
                ) + "\n"
            )
            print(f"activation [{city_i+1:02d}/{len(CITIES)}] {city}: {len(ids):,}", flush=True)

    report = {
        "images": total,
        "dimensions": int(model.width),
        "score": f"mean of strongest {args.top_patches} of 196 spatial patches",
        "checkpoint": str(checkpoint_path.resolve()),
        "output": str(args.output.resolve()),
        "dtype": "float16",
        "elapsed_seconds": time.time() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
