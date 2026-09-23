"""Extract and store normalized 14x14 DINOv3 patch tokens for each image."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from scripts.multicity.config import CITIES, MODEL_DIR, city_slug
from scripts.multicity.extract_dinov3_features import TarOffsetImageDataset
from scripts.multicity.patch_config import (
    INPUT_DIM,
    OUTPUT_ROOT,
    PATCHES_PER_IMAGE,
    image_manifest_path,
    token_path,
)


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


@torch.inference_mode()
def infer_patch_tokens(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sequence = model(pixel_values=pixels).last_hidden_state
        start = 1 + int(model.config.num_register_tokens)
        patches = sequence[:, start:]
        if patches.shape[1:] != (PATCHES_PER_IMAGE, INPUT_DIM):
            raise ValueError(f"unexpected patch shape {tuple(patches.shape)}")
        patches = torch.nn.functional.normalize(patches.float(), dim=-1)
        return patches.cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(pixels) == 1:
            raise
        middle = len(pixels) // 2
        return torch.cat(
            (infer_patch_tokens(model, pixels[:middle]), infer_patch_tokens(model, pixels[middle:])),
            dim=0,
        )


def extract_city(
    city: str,
    model: torch.nn.Module,
    output_root: Path,
    batch_size: int,
    workers: int,
) -> dict:
    manifest = pd.read_parquet(image_manifest_path(city, output_root)).sort_values("image_index")
    n_images = len(manifest)
    final = token_path(city, output_root)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and np.load(final, mmap_mode="r").shape == (
        n_images, PATCHES_PER_IMAGE, INPUT_DIM
    ):
        return {"city_key": city, "images": n_images, "status": "existing", "path": str(final)}

    partial = final.with_name(final.name + ".partial.npy")
    progress_path = final.with_name(final.name + ".progress.json")
    if partial.exists() and progress_path.exists():
        tokens = np.lib.format.open_memmap(partial, mode="r+")
        done = int(json.loads(progress_path.read_text())["completed_images"])
        if tokens.shape != (n_images, PATCHES_PER_IMAGE, INPUT_DIM):
            raise ValueError(f"bad resume shape for {city}: {tokens.shape}")
    else:
        tokens = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float16,
            shape=(n_images, PATCHES_PER_IMAGE, INPUT_DIM),
        )
        done = 0
        _atomic_json(progress_path, {"city_key": city, "completed_images": 0})

    loader = DataLoader(
        TarOffsetImageDataset(manifest, start=done),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    started = time.time()
    initial_done = done
    torch.cuda.reset_peak_memory_stats()
    saved_at = done
    for batch_i, (pixels, row_indices) in enumerate(loader, 1):
        pixels = pixels.to("cuda", non_blocking=True)
        batch_tokens = infer_patch_tokens(model, pixels).numpy().astype(np.float16)
        rows = row_indices.numpy()
        tokens[rows] = batch_tokens
        done = int(rows[-1]) + 1
        if done - saved_at >= batch_size * 25 or done == n_images:
            tokens.flush()
            _atomic_json(progress_path, {"city_key": city, "completed_images": done})
            saved_at = done
        if batch_i % 50 == 0 or done == n_images:
            rate = (done - initial_done) / max(time.time() - started, 1e-6)
            print(
                f"  {city}: {done:,}/{n_images:,} images, {rate:.1f} img/s, "
                f"peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                flush=True,
            )
    tokens.flush()
    os.replace(partial, final)
    progress_path.unlink(missing_ok=True)
    check = np.load(final, mmap_mode="r")
    norms = np.linalg.norm(np.asarray(check[:16], dtype=np.float32), axis=-1)
    report = {
        "city_key": city,
        "images": n_images,
        "tokens": n_images * PATCHES_PER_IMAGE,
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "batch_size": batch_size,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.time() - started,
        "path": str(final),
        "status": "created",
    }
    _atomic_json(final.with_suffix(".report.json"), report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    model.eval().requires_grad_(False).to("cuda")
    reports = []
    for i, city in enumerate(args.cities, 1):
        print(f"[{i:02d}/{len(args.cities)}] extracting patch tokens: {city}", flush=True)
        reports.append(extract_city(city, model, args.output_root, args.batch_size, args.workers))
    _atomic_json(args.output_root / "patch_extraction_summary.json", {"cities": reports})


if __name__ == "__main__":
    main()
