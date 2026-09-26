#!/usr/bin/env python3
"""Run DINOv3 + frozen Feature-MAE on eight pixel-level panorama windows."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import importlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from scripts.multicity.config import MODEL_DIR
from scripts.multicity.dinov3_vit_backport import load_dinov3_vit
from scripts.multicity.extract_dinov3_features import MEAN, STD


GRID = 14
WINDOW_HEADINGS = tuple(range(0, 360, 45))
DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_CHECKPOINT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "feature_mae_n30x12800_qc/mae/model_best.pt"
)


def assert_safe_affinity() -> None:
    if hasattr(os, "sched_getaffinity"):
        forbidden = set(os.sched_getaffinity(0)).intersection({8, 9})
        if forbidden:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(forbidden)}")


def read_original(row: pd.Series) -> Image.Image:
    descriptor = os.open(str(row["image_path"]), os.O_RDONLY)
    try:
        payload = os.pread(
            descriptor, int(row["jpg_size"]), int(row["jpg_offset"])
        )
    finally:
        os.close(descriptor)
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def seam_window(first: Image.Image, second: Image.Image) -> Image.Image:
    """Join the right half of one 90° view to the left half of the next."""
    first = first.convert("RGB")
    second = second.convert("RGB")
    if first.size != second.size:
        raise ValueError(f"direction image sizes differ: {first.size} and {second.size}")
    width, height = first.size
    if width % 2:
        raise ValueError(f"direction image width must be even, got {width}")
    half = width // 2
    output = Image.new("RGB", (width, height))
    output.paste(first.crop((half, 0, width, height)), (0, 0))
    output.paste(second.crop((0, 0, half, height)), (half, 0))
    return output


def make_pixel_windows(images: list[Image.Image]) -> list[Image.Image]:
    if len(images) != 4:
        raise ValueError(f"expected four direction images, got {len(images)}")
    windows = []
    for direction in range(4):
        windows.append(images[direction])
        windows.append(seam_window(images[direction], images[(direction + 1) % 4]))
    return windows


@torch.inference_mode()
def infer_windows(
    images: list[Image.Image],
    dino_model: torch.nn.Module,
    mae_model: torch.nn.Module,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = v2.Compose(
        [
            v2.Resize(
                (224, 224), interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )
    token_blocks = []
    latent_blocks = []
    for start_index in range(0, len(images), batch_size):
        pixels = torch.stack(
            [transform(image) for image in images[start_index : start_index + batch_size]]
        ).to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sequence = dino_model(pixel_values=pixels).last_hidden_state
        start = 1 + int(dino_model.config.num_register_tokens)
        tokens = sequence[:, start:].float()
        if tokens.shape[1:] != (GRID * GRID, 768):
            raise ValueError(f"unexpected DINO token shape {tuple(tokens.shape)}")
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        tokens = tokens.half().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = mae_model.encode_full(tokens)
        token_blocks.append(tokens.cpu().numpy().astype(np.float16))
        latent_blocks.append(latent.float().cpu().numpy().astype(np.float16))
        del pixels, sequence, tokens, latent
    return np.concatenate(token_blocks), np.concatenate(latent_blocks)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pixel-overlap inference")
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("batch-size must be between 1 and 8")
    overlap = importlib.import_module(
        "scripts.image_graph_compact_archetypes.12_infer_overlap_panorama"
    )
    metadata = pd.read_csv(args.output / "multi_area_direction_metadata.csv")
    area_ids = metadata["area_id"].drop_duplicates().tolist()
    if args.max_areas:
        area_ids = area_ids[: args.max_areas]

    dino_model = load_dinov3_vit(args.model_dir)
    dino_model = dino_model.eval().requires_grad_(False).to("cuda")
    mae_model, config = overlap.load_model(args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    token_blocks = []
    latent_blocks = []
    for area_id in area_ids:
        rows = metadata[metadata["area_id"].eq(area_id)].sort_values("heading")
        if rows["heading"].astype(int).tolist() != [0, 90, 180, 270]:
            raise ValueError(f"{area_id}: incomplete direction images")
        images = [read_original(row) for _, row in rows.iterrows()]
        windows = make_pixel_windows(images)
        tokens, latent = infer_windows(
            windows, dino_model, mae_model, args.batch_size
        )
        token_blocks.append(tokens)
        latent_blocks.append(latent)
        print(f"completed {area_id}: 8 pixel windows", flush=True)

    window_tokens = np.stack(token_blocks)
    window_latent = np.stack(latent_blocks).reshape(
        len(area_ids), 8, GRID, GRID, 512
    )
    fused, weight_sum = overlap.fuse_windows(window_latent)
    winners = fused.argmax(axis=-1).astype(np.uint16)
    independent = np.concatenate(
        [window_latent[:, index] for index in (0, 2, 4, 6)], axis=2
    )
    independent_winners = independent.argmax(axis=-1).astype(np.uint16)

    cached_winners = np.load(args.output / "top1_dimensions.npy")
    cached_winners = cached_winners[: len(area_ids) * 4].reshape(
        len(area_ids), 4, GRID, GRID
    ).transpose(0, 2, 1, 3).reshape(len(area_ids), GRID, 4 * GRID)
    original_match = float(np.mean(independent_winners == cached_winners))
    # The existing four-direction run reproduced cached winners at 95.4--99%
    # because near-tied dimensions can flip under bfloat16 batch arithmetic.
    if original_match < 0.95:
        raise ValueError(
            f"original pixel windows do not reproduce cached winners: {original_match}"
        )
    if not np.isfinite(fused).all():
        raise ValueError("pixel-overlap fused activation contains NaN or Inf")

    np.save(args.output / "pixel_overlap_window_dino_tokens.npy", window_tokens)
    np.save(args.output / "pixel_overlap_window_feature_mae_latent.npy", window_latent)
    np.save(args.output / "pixel_overlap_panorama_feature_mae_latent.npy", fused.astype(np.float16))
    np.save(args.output / "pixel_overlap_panorama_top1_dimensions.npy", winners)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "areas": len(area_ids),
        "window_headings": list(WINDOW_HEADINGS),
        "pixel_windows_per_area": 8,
        "pipeline": "pixel window -> DINOv3 -> frozen Feature-MAE -> 512D overlap-add -> argmax",
        "pixel_window_construction": (
            "cardinal views plus right-half/left-half seam-centred windows"
        ),
        "window_token_shape": list(window_tokens.shape),
        "window_latent_shape": list(window_latent.shape),
        "fused_activation_shape": list(fused.shape),
        "fused_winner_shape": list(winners.shape),
        "weight_sum_minimum": float(weight_sum.min()),
        "weight_sum_maximum": float(weight_sum.max()),
        "original_window_cached_winner_match": original_match,
        "independent_seam_cosine": overlap.mean_boundary_cosine(independent),
        "pixel_overlap_seam_cosine": overlap.mean_boundary_cosine(fused),
        "independent_seam_winner_agreement": overlap.mean_boundary_winner_agreement(
            independent_winners
        ),
        "pixel_overlap_seam_winner_agreement": overlap.mean_boundary_winner_agreement(
            winners
        ),
        "winner_change_fraction": float(np.mean(winners != independent_winners)),
        "batch_size": args.batch_size,
        "dino_model": str(args.model_dir),
        "feature_mae_checkpoint": str(args.checkpoint),
        "feature_mae_checkpoint_sha256": overlap.sha256(args.checkpoint),
        "feature_mae_config": config,
        "peak_gpu_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "pixel_overlap_panorama_inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-areas", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
