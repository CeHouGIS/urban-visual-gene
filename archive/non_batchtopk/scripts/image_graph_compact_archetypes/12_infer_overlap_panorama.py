#!/usr/bin/env python3
"""Fuse Feature-MAE activations from circular, half-view-overlap windows.

The four cached DINO token grids remain unchanged. Four additional seam-centred
14x14 token windows are formed by joining the right seven columns of one view
to the left seven columns of the following view. The frozen Feature-MAE is run
on all eight windows, and their raw 512D activations are overlap-added on a
14x56 circular panorama before winner selection.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.multicity.train_feature_mae import FeatureMAE


GRID = 14
PANORAMA_WIDTH = 56
WINDOW_HEADINGS = tuple(range(0, 360, 45))
DEFAULT_OUTPUT = Path(
    "paper/data/image_graph_compact_archetypes/multi_area_four_directions"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc/mae/model_best.pt"
)


def assert_safe_affinity() -> None:
    if hasattr(os, "sched_getaffinity"):
        forbidden = set(os.sched_getaffinity(0)).intersection({8, 9})
        if forbidden:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(forbidden)}")


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: Path) -> tuple[FeatureMAE, dict]:
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
    return model.eval().requires_grad_(False).to("cuda"), config


def make_overlap_windows(tokens: np.ndarray) -> np.ndarray:
    """Convert A x 4 x 196 x 768 tokens into A x 8 x 196 x 768 windows."""
    values = np.asarray(tokens)
    if values.ndim != 3 or values.shape[1:] != (GRID * GRID, 768):
        raise ValueError(f"unexpected DINO token shape {values.shape}")
    if len(values) % 4:
        raise ValueError("DINO token rows must contain four directions per area")
    areas = values.reshape(-1, 4, GRID, GRID, 768)
    windows = np.empty(
        (len(areas), 8, GRID, GRID, 768), dtype=values.dtype
    )
    for direction in range(4):
        windows[:, 2 * direction] = areas[:, direction]
        windows[:, 2 * direction + 1] = np.concatenate(
            (
                areas[:, direction, :, GRID // 2 :, :],
                areas[:, (direction + 1) % 4, :, : GRID // 2, :],
            ),
            axis=2,
        )
    return windows.reshape(len(areas), 8, GRID * GRID, 768)


@torch.inference_mode()
def infer_windows(
    windows: np.ndarray, model: FeatureMAE, batch_size: int
) -> np.ndarray:
    flat = windows.reshape(-1, GRID * GRID, 768)
    blocks = []
    for start in range(0, len(flat), batch_size):
        batch = torch.from_numpy(flat[start : start + batch_size].astype(np.float32))
        batch = batch.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode_full(batch)
        blocks.append(latent.float().cpu().numpy().astype(np.float16))
        del batch, latent
    return np.concatenate(blocks).reshape(
        windows.shape[0], 8, GRID, GRID, 512
    )


def fuse_windows(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if latent.ndim != 5 or latent.shape[1:] != (8, GRID, GRID, 512):
        raise ValueError(f"unexpected overlap latent shape {latent.shape}")
    # Two windows cover every panorama column. A half-period sine-squared taper
    # forms an exact complementary partition of unity for the seven-column hop.
    x = np.arange(GRID, dtype=np.float32)
    weights = np.sin(np.pi * (x + 0.5) / GRID) ** 2
    accumulator = np.zeros(
        (len(latent), GRID, PANORAMA_WIDTH, 512), dtype=np.float32
    )
    denominator = np.zeros((PANORAMA_WIDTH,), dtype=np.float32)
    values = latent.astype(np.float32)
    for window in range(8):
        columns = (window * (GRID // 2) + np.arange(GRID)) % PANORAMA_WIDTH
        accumulator[:, :, columns, :] += values[:, window] * weights[None, None, :, None]
        denominator[columns] += weights
    if not np.allclose(denominator, 1.0, atol=2e-6):
        raise ValueError(
            f"overlap weights do not sum to one: {denominator.min()}--{denominator.max()}"
        )
    return accumulator / denominator[None, None, :, None], denominator


def mean_boundary_cosine(values: np.ndarray) -> float:
    seams = ((13, 14), (27, 28), (41, 42), (55, 0))
    scores = []
    source = values.astype(np.float32)
    for left, right in seams:
        a, b = source[:, :, left], source[:, :, right]
        numerator = np.sum(a * b, axis=-1)
        denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        scores.append(numerator / np.maximum(denominator, 1e-12))
    return float(np.mean(scores))


def mean_boundary_winner_agreement(winners: np.ndarray) -> float:
    seams = ((13, 14), (27, 28), (41, 42), (55, 0))
    return float(
        np.mean([winners[:, :, left] == winners[:, :, right] for left, right in seams])
    )


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for overlap Feature-MAE inference")
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("batch-size must be between 1 and 8")
    tokens = np.load(args.tokens)
    windows = make_overlap_windows(tokens)
    if args.max_areas:
        windows = windows[: args.max_areas]
    norms = np.linalg.norm(windows.astype(np.float32), axis=-1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError(f"DINO token norms are invalid: max error {abs(norms - 1).max()}")

    model, config = load_model(args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    overlap_latent = infer_windows(windows, model, args.batch_size)
    fused, weight_sum = fuse_windows(overlap_latent)
    winners = fused.argmax(axis=-1).astype(np.uint16)

    original = np.concatenate(
        [overlap_latent[:, index] for index in (0, 2, 4, 6)], axis=2
    )
    original_winners = original.argmax(axis=-1).astype(np.uint16)
    if not np.isfinite(fused).all() or winners.max(initial=0) >= 512:
        raise ValueError("invalid fused Feature-MAE output")

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "overlap_window_feature_mae_latent.npy", overlap_latent)
    np.save(args.output / "panorama_feature_mae_latent.npy", fused.astype(np.float16))
    np.save(args.output / "panorama_top1_dimensions.npy", winners)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "areas": int(len(windows)),
        "input_direction_windows": 4,
        "total_overlap_windows_per_area": 8,
        "window_headings": list(WINDOW_HEADINGS),
        "window_shape": [GRID, GRID, 768],
        "fused_activation_shape": list(fused.shape),
        "fused_winner_shape": list(winners.shape),
        "horizontal_hop_patches": GRID // 2,
        "fusion": "raw 512D sine-squared weighted overlap-add before argmax",
        "weight_sum_minimum": float(weight_sum.min()),
        "weight_sum_maximum": float(weight_sum.max()),
        "original_seam_cosine": mean_boundary_cosine(original),
        "fused_seam_cosine": mean_boundary_cosine(fused),
        "original_seam_winner_agreement": mean_boundary_winner_agreement(
            original_winners
        ),
        "fused_seam_winner_agreement": mean_boundary_winner_agreement(winners),
        "winner_change_fraction": float(np.mean(winners != original_winners)),
        "batch_size": int(args.batch_size),
        "feature_mae_checkpoint": str(args.checkpoint),
        "feature_mae_checkpoint_sha256": sha256(args.checkpoint),
        "feature_mae_config": config,
        "peak_gpu_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "overlap_panorama_inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=Path, default=DEFAULT_OUTPUT / "dino_patch_tokens.npy"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-areas", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
