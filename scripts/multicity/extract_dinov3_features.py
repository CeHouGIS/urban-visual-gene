"""Extract resumable four-view DINOv3 ViT-B/16 panorama features."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import AutoModel

from scripts.multicity.config import (
    CITIES,
    FEATURES_PER_HEADING,
    FEATURE_DIM,
    HEADINGS,
    MODEL_DIR,
    OUTPUT_ROOT,
    city_slug,
    feature_path,
    manifest_path,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TarOffsetImageDataset(Dataset):
    """Read JPEG payloads directly from uncompressed TAR byte offsets."""

    def __init__(self, frame: pd.DataFrame, start: int = 0, max_open_files: int = 16):
        self.paths = frame["tar_path"].astype(str).tolist()[start:]
        self.offsets = frame["jpg_offset"].astype(np.int64).tolist()[start:]
        self.sizes = frame["jpg_size"].astype(np.int64).tolist()[start:]
        self.row_indices = np.arange(start, len(frame), dtype=np.int64)
        self.max_open_files = max_open_files
        self._fds: OrderedDict[str, int] = OrderedDict()
        self.transform = v2.Compose([
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC,
                      antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self) -> int:
        return len(self.row_indices)

    def _fd(self, path: str) -> int:
        if path in self._fds:
            fd = self._fds.pop(path)
            self._fds[path] = fd
            return fd
        fd = os.open(path, os.O_RDONLY)
        self._fds[path] = fd
        if len(self._fds) > self.max_open_files:
            _, old_fd = self._fds.popitem(last=False)
            os.close(old_fd)
        return fd

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        payload = os.pread(self._fd(self.paths[item]), self.sizes[item], self.offsets[item])
        if len(payload) != self.sizes[item]:
            raise OSError(
                f"short read from {self.paths[item]} at {self.offsets[item]}: "
                f"{len(payload)} != {self.sizes[item]}"
            )
        with Image.open(io.BytesIO(payload)) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, int(self.row_indices[item])

    def __del__(self):
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass


@torch.inference_mode()
def infer_safely(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Infer a batch, recursively splitting it if CUDA reports an OOM."""
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(pixel_values=pixels).last_hidden_state[:, 0].float().cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(pixels) == 1:
            raise
        midpoint = len(pixels) // 2
        left = infer_safely(model, pixels[:midpoint])
        right = infer_safely(model, pixels[midpoint:])
        return torch.cat((left, right), dim=0)


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def extract_city(
    city_key: str,
    model: torch.nn.Module,
    batch_size: int,
    workers: int,
    output_root: Path,
    limit_panos: int | None = None,
) -> dict:
    manifest = pd.read_parquet(manifest_path(city_key, output_root))
    manifest = manifest.sort_values(["pano_index", "direction_index"]).reset_index(drop=True)
    if limit_panos is not None:
        manifest = manifest[manifest["pano_index"] < limit_panos].reset_index(drop=True)
    n_panos = int(manifest["pano_index"].nunique())
    if len(manifest) != n_panos * len(HEADINGS):
        raise ValueError(f"{city_key}: manifest does not contain four rows per panorama")

    city_out = output_root / "features" / city_slug(city_key)
    city_out.mkdir(parents=True, exist_ok=True)
    final = feature_path(city_key, output_root)
    if limit_panos is not None:
        final = city_out / f"smoke_{limit_panos}_pano_features.f16.npy"
    report_path = final.with_suffix(".report.json")
    if final.exists():
        existing = np.load(final, mmap_mode="r")
        if existing.shape == (n_panos, FEATURE_DIM):
            return {
                "city_key": city_key, "panos": n_panos, "images": len(manifest),
                "feature_path": str(final), "status": "existing",
            }

    partial = final.with_name(final.name + ".directions.partial.npy")
    progress_path = final.with_name(final.name + ".progress.json")
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        done = int(progress.get("completed_rows", 0))
        directions = np.lib.format.open_memmap(partial, mode="r+")
        if directions.shape != (n_panos * 4, FEATURES_PER_HEADING):
            raise ValueError(f"resume array has unexpected shape: {directions.shape}")
    else:
        done = 0
        directions = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float16,
            shape=(n_panos * 4, FEATURES_PER_HEADING),
        )
        _save_json(progress_path, {"completed_rows": 0, "city_key": city_key})

    dataset = TarOffsetImageDataset(manifest, start=done)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    last_saved = done
    for batch_i, (pixels, row_indices) in enumerate(loader, start=1):
        pixels = pixels.to("cuda", non_blocking=True)
        embeddings = infer_safely(model, pixels).numpy().astype(np.float16)
        rows = row_indices.numpy()
        directions[rows] = embeddings
        done = int(rows[-1]) + 1
        if done - last_saved >= batch_size * 50 or done == len(manifest):
            directions.flush()
            _save_json(progress_path, {"completed_rows": done, "city_key": city_key})
            last_saved = done
        if batch_i % 100 == 0 or done == len(manifest):
            elapsed = max(time.time() - started, 1e-6)
            rate = (done - dataset.row_indices[0]) / elapsed if len(dataset) else 0.0
            print(
                f"  {city_key}: {done:,}/{len(manifest):,} images "
                f"({rate:.1f} img/s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB)",
                flush=True,
            )

    output_tmp = final.with_name(final.name + ".tmp.npy")
    features = np.lib.format.open_memmap(
        output_tmp, mode="w+", dtype=np.float16, shape=(n_panos, FEATURE_DIM),
    )
    direction_view = directions.reshape(n_panos, 4, FEATURES_PER_HEADING)
    for start in range(0, n_panos, 1024):
        end = min(start + 1024, n_panos)
        block = np.asarray(direction_view[start:end], dtype=np.float32).reshape(end - start, -1)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if not np.isfinite(block).all() or np.any(norms <= 0):
            raise ValueError(f"{city_key}: non-finite or zero DINOv3 feature")
        features[start:end] = (block / norms).astype(np.float16)
    features.flush()
    os.replace(output_tmp, final)

    check = np.load(final, mmap_mode="r")
    check_norms = np.linalg.norm(np.asarray(check[: min(2048, n_panos)], dtype=np.float32), axis=1)
    report = {
        "city_key": city_key,
        "model": str(MODEL_DIR),
        "panos": n_panos,
        "images": len(manifest),
        "shape": list(check.shape),
        "dtype": str(check.dtype),
        "sample_norm_min": float(check_norms.min()),
        "sample_norm_max": float(check_norms.max()),
        "batch_size": batch_size,
        "workers": workers,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.time() - started,
        "feature_path": str(final),
        "status": "created",
    }
    _save_json(report_path, report)
    del directions, check, features
    partial.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    # 128 was measured at 0.95 GiB peak allocated memory on the 12 GiB RTX
    # 3060.  Larger batches provide little extra throughput and are avoided.
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-panos", type=int)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full DINOv3 extraction")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True)
    if model.config.hidden_size != FEATURES_PER_HEADING:
        raise ValueError(
            f"expected hidden size {FEATURES_PER_HEADING}, got {model.config.hidden_size}"
        )
    model.eval().requires_grad_(False).to("cuda")
    reports = []
    for i, city in enumerate(args.cities, start=1):
        print(f"[{i:02d}/{len(args.cities)}] extracting {city}", flush=True)
        report = extract_city(
            city, model, args.batch_size, args.workers, args.output_root,
            limit_panos=args.limit_panos,
        )
        reports.append(report)
        print(
            f"  done: {report['panos']:,} panos, status={report['status']}",
            flush=True,
        )
    suffix = "smoke_extraction_summary.json" if args.limit_panos else "extraction_summary.json"
    _save_json(args.output_root / suffix, {"cities": reports})


if __name__ == "__main__":
    main()
