#!/usr/bin/env python3
"""Resumable DINOv3 + frozen Feature-MAE inference for 4:1 panoramas.

Four 640x640 cardinal views are joined as 2560x640, resized without changing
aspect ratio to 896x224, and passed through DINOv3 once.  The resulting 14x56
patch-token map is sampled by eight circular 14x14 windows (seven-column hop).
The frozen square-window Feature-MAE processes those windows and the 512D
activations are fused by sine-squared overlap-add before winner selection.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import hashlib
import io
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision.transforms import v2

from scripts.multicity.config import CITIES, HEADINGS, MODEL_DIR, city_slug
from scripts.multicity.dinov3_vit_backport import load_dinov3_vit
from scripts.multicity.extract_dinov3_features import MEAN, STD
from scripts.multicity.train_feature_mae import FeatureMAE


ImageFile.LOAD_TRUNCATED_IMAGES = True
GRID = 14
PANORAMA_COLUMNS = 56
WINDOWS = 8
HOP = 7
DIMENSIONS = 512
TOP_PATCHES = 20
DEFAULT_DATA_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "feature_mae_n30x12800_qc"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "rectangular_panorama_frozen_mae_n30"
)


def assert_safe_affinity() -> None:
    if hasattr(os, "sched_getaffinity"):
        forbidden = set(os.sched_getaffinity(0)).intersection({8, 9})
        if forbidden:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(forbidden)}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class TarReader:
    def __init__(self, max_open_files: int = 16) -> None:
        self.max_open_files = max_open_files
        self.descriptors: OrderedDict[str, int] = OrderedDict()

    def descriptor(self, path: str) -> int:
        if path in self.descriptors:
            descriptor = self.descriptors.pop(path)
            self.descriptors[path] = descriptor
            return descriptor
        descriptor = os.open(path, os.O_RDONLY)
        self.descriptors[path] = descriptor
        if len(self.descriptors) > self.max_open_files:
            _, old_descriptor = self.descriptors.popitem(last=False)
            os.close(old_descriptor)
        return descriptor

    def image(self, path: str, offset: int, size: int) -> Image.Image:
        payload = os.pread(self.descriptor(path), size, offset)
        if len(payload) != size:
            raise OSError(f"short TAR read: {path} at {offset}: {len(payload)} != {size}")
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()


def load_mae(checkpoint_path: Path) -> tuple[FeatureMAE, dict]:
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


def compose_panorama(rows: pd.DataFrame, reader: TarReader) -> Image.Image:
    if rows["heading"].astype(int).tolist() != list(HEADINGS):
        raise ValueError("panorama rows are not ordered 0/90/180/270")
    images = [
        reader.image(str(row.tar_path), int(row.jpg_offset), int(row.jpg_size))
        for row in rows.itertuples(index=False)
    ]
    if any(image.size != (640, 640) for image in images):
        sizes = [image.size for image in images]
        raise ValueError(f"expected four 640x640 source images, got {sizes}")
    panorama = Image.new("RGB", (2560, 640))
    for index, image in enumerate(images):
        panorama.paste(image, (index * 640, 0))
    return panorama


def circular_windows(tokens: torch.Tensor) -> torch.Tensor:
    if tuple(tokens.shape) != (1, GRID, PANORAMA_COLUMNS, 768):
        raise ValueError(f"unexpected rectangular DINO tokens {tuple(tokens.shape)}")
    windows = []
    columns = torch.arange(GRID, device=tokens.device)
    for window in range(WINDOWS):
        selected = (window * HOP + columns) % PANORAMA_COLUMNS
        windows.append(tokens[0, :, selected, :].reshape(GRID * GRID, 768))
    return torch.stack(windows)


def fuse_windows(latent: torch.Tensor) -> torch.Tensor:
    if tuple(latent.shape) != (WINDOWS, GRID, GRID, DIMENSIONS):
        raise ValueError(f"unexpected Feature-MAE window shape {tuple(latent.shape)}")
    x = torch.arange(GRID, dtype=torch.float32, device=latent.device)
    weights = torch.sin(torch.pi * (x + 0.5) / GRID).square()
    accumulator = torch.zeros(
        (GRID, PANORAMA_COLUMNS, DIMENSIONS), dtype=torch.float32, device=latent.device
    )
    denominator = torch.zeros(PANORAMA_COLUMNS, dtype=torch.float32, device=latent.device)
    for window in range(WINDOWS):
        columns = (window * HOP + torch.arange(GRID, device=latent.device)) % PANORAMA_COLUMNS
        accumulator[:, columns, :] += latent[window].float() * weights[None, :, None]
        denominator[columns] += weights
    if not torch.allclose(denominator, torch.ones_like(denominator), atol=2e-6):
        raise ValueError("overlap-add weights do not form a partition of unity")
    return accumulator / denominator[None, :, None]


@torch.inference_mode()
def infer_one(
    image: Image.Image,
    transform: v2.Compose,
    dino: torch.nn.Module,
    mae: FeatureMAE,
    mae_batch: int,
) -> torch.Tensor:
    pixels = transform(image).unsqueeze(0).to("cuda", non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sequence = dino(pixel_values=pixels).last_hidden_state
    prefix = 1 + int(dino.config.num_register_tokens)
    tokens = sequence[:, prefix:].float()
    if tuple(tokens.shape) != (1, GRID * PANORAMA_COLUMNS, 768):
        raise ValueError(f"DINO output has unexpected shape {tuple(tokens.shape)}")
    # Preserve the original training pipeline's normalization and fp16 cache
    # round-trip before Feature-MAE.
    tokens = torch.nn.functional.normalize(tokens, dim=-1).half().float()
    windows = circular_windows(tokens.reshape(1, GRID, PANORAMA_COLUMNS, 768))
    latent_parts = []
    for start in range(0, WINDOWS, mae_batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent_parts.append(mae.encode_full(windows[start : start + mae_batch]).float())
    latent = torch.cat(latent_parts).reshape(WINDOWS, GRID, GRID, DIMENSIONS)
    return fuse_windows(latent)


def open_output(path: Path, shape: tuple[int, ...], dtype: np.dtype, resume: bool):
    if resume:
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"resume output mismatch for {path}: {array.shape}/{array.dtype}, "
                f"expected {shape}/{np.dtype(dtype)}"
            )
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def run_city(args: argparse.Namespace, city_key: str, dino, mae, mae_config, fine_labels) -> dict:
    slug = city_slug(city_key)
    manifest_path = args.output_root / "manifests" / f"{slug}.parquet"
    manifest = pd.read_parquet(manifest_path).sort_values(
        ["pano_index", "direction_index"]
    ).reset_index(drop=True)
    if len(manifest) % 4 or manifest.groupby("pano_index").size().ne(4).any():
        raise ValueError(f"{city_key}: panorama manifest is not four rows per panorama")
    total_panoramas = int(manifest["pano_index"].nunique())
    n_panoramas = min(total_panoramas, args.limit) if args.limit else total_panoramas
    manifest = manifest[manifest["pano_index"].lt(n_panoramas)].copy()

    suffix = f"smoke_{n_panoramas}" if args.limit else "full"
    city_output = args.output_root / "predictions" / slug / suffix
    city_output.mkdir(parents=True, exist_ok=True)
    progress_path = city_output / "progress.json"
    report_path = city_output / "inference_report.json"
    if report_path.exists() and not args.force:
        report = json.loads(report_path.read_text())
        report["status"] = "existing"
        return report
    resume = progress_path.exists() and not args.force
    if resume:
        progress = json.loads(progress_path.read_text())
        completed = int(progress["completed_panoramas"])
    else:
        completed = 0

    outputs = {
        "latent": open_output(
            city_output / "panorama_feature_mae_latent.f16.npy",
            (n_panoramas, GRID, PANORAMA_COLUMNS, DIMENSIONS), np.float16, resume,
        ),
        "winners": open_output(
            city_output / "top1_dimensions.u16.npy",
            (n_panoramas, GRID, PANORAMA_COLUMNS), np.uint16, resume,
        ),
        "fine": open_output(
            city_output / "f_category_maps.u8.npy",
            (n_panoramas, GRID, PANORAMA_COLUMNS), np.uint8, resume,
        ),
        "activation": open_output(
            city_output / "activation_top20.f16.npy",
            (n_panoramas, DIMENSIONS), np.float16, resume,
        ),
        "status": open_output(
            city_output / "status.u8.npy", (n_panoramas,), np.uint8, resume,
        ),
    }
    if completed < 0 or completed > n_panoramas:
        raise ValueError(f"invalid resume cursor {completed} for {n_panoramas} panoramas")
    errors_path = city_output / "errors.jsonl"
    transform = v2.Compose([
        v2.Resize((224, 896), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])
    reader = TarReader()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    interval_started = started
    interval_start = completed
    errors = int(np.count_nonzero(np.asarray(outputs["status"][:completed]) == 2))
    try:
        for pano_index in range(completed, n_panoramas):
            rows = manifest[manifest["pano_index"].eq(pano_index)]
            try:
                image = compose_panorama(rows, reader)
                fused = infer_one(image, transform, dino, mae, args.mae_batch)
                winner = fused.argmax(dim=-1).to(torch.int64).cpu().numpy().astype(np.uint16)
                values = fused.reshape(-1, DIMENSIONS)
                activation = values.topk(TOP_PATCHES, dim=0).values.mean(dim=0)
                latent = fused.half().cpu().numpy()
                if not np.isfinite(latent).all() or winner.max(initial=0) >= DIMENSIONS:
                    raise ValueError("non-finite activation or invalid winner ID")
                outputs["latent"][pano_index] = latent
                outputs["winners"][pano_index] = winner
                outputs["fine"][pano_index] = fine_labels[winner]
                outputs["activation"][pano_index] = activation.cpu().numpy().astype(np.float16)
                outputs["status"][pano_index] = 1
            except (OSError, ValueError, RuntimeError) as error:
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    raise
                outputs["status"][pano_index] = 2
                errors += 1
                with errors_path.open("a") as handle:
                    handle.write(json.dumps({
                        "pano_index": pano_index,
                        "panoid": str(rows.iloc[0]["panoid"]) if len(rows) else None,
                        "error": repr(error),
                    }) + "\n")
            completed = pano_index + 1
            if completed % args.checkpoint_every == 0 or completed == n_panoramas:
                for output in outputs.values():
                    output.flush()
                elapsed = max(time.time() - interval_started, 1e-6)
                rate = (completed - interval_start) / elapsed
                progress = {
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "city_key": city_key,
                    "completed_panoramas": completed,
                    "total_panoramas": n_panoramas,
                    "errors": errors,
                    "recent_panoramas_per_second": rate,
                    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                }
                save_json(progress_path, progress)
                print(
                    f"  {city_key}: {completed:,}/{n_panoramas:,} panoramas, "
                    f"{rate:.2f} pano/s, {errors} errors, "
                    f"peak {progress['peak_gpu_gib']:.2f} GiB",
                    flush=True,
                )
                interval_started = time.time()
                interval_start = completed
            if errors > max(10, int(n_panoramas * args.max_error_fraction)):
                raise RuntimeError(f"{city_key}: error limit exceeded ({errors}/{completed})")
    finally:
        reader.close()

    valid = np.asarray(outputs["status"]) == 1
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city_key": city_key,
        "status": "complete",
        "panoramas": n_panoramas,
        "successful_panoramas": int(valid.sum()),
        "errors": int(errors),
        "input_construction": "four 640x640 views -> 2560x640 -> 896x224",
        "dino_token_grid": [GRID, PANORAMA_COLUMNS, 768],
        "feature_mae_windows": [WINDOWS, GRID, GRID, 768],
        "window_start_columns": [index * HOP for index in range(WINDOWS)],
        "fusion": "512D sine-squared circular overlap-add before argmax",
        "latent_shape": list(outputs["latent"].shape),
        "latent_dtype": str(outputs["latent"].dtype),
        "activation_summary": f"mean of strongest {TOP_PATCHES} of 784 panorama patches",
        "elapsed_seconds_this_run": time.time() - started,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "mae_config": mae_config,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    save_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hierarchy", type=Path)
    parser.add_argument("--city", action="append", choices=CITIES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mae-batch", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--max-error-fraction", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_safe_affinity()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mae_batch < 1 or args.mae_batch > 4:
        raise ValueError("--mae-batch must be between 1 and 4")
    args.checkpoint = args.checkpoint or args.data_root / "mae" / "model_best.pt"
    args.hierarchy = args.hierarchy or (
        args.data_root / "mae" / "hierarchy_edp_32_64" / "hierarchy_arrays.npz"
    )
    with np.load(args.hierarchy) as hierarchy:
        fine_labels = hierarchy["fine_labels"].astype(np.uint8)
    if fine_labels.shape != (DIMENSIONS,) or fine_labels.min() != 0 or fine_labels.max() != 63:
        raise ValueError("hierarchy must map 512 dimensions onto F000--F063")

    dino = load_dinov3_vit(args.model_dir).eval().requires_grad_(False).to("cuda")
    mae, mae_config = load_mae(args.checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = True
    reports = []
    cities = args.city or list(CITIES)
    for index, city in enumerate(cities, 1):
        print(f"[{index:02d}/{len(cities)}] starting {city}", flush=True)
        reports.append(run_city(args, city, dino, mae, mae_config, fine_labels))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cities": len(reports),
        "panoramas": sum(x["panoramas"] for x in reports),
        "successful_panoramas": sum(x["successful_panoramas"] for x in reports),
        "reports": reports,
    }
    name = f"smoke_{args.limit}_summary.json" if args.limit else "inference_summary.json"
    save_json(args.output_root / name, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))


if __name__ == "__main__":
    main()
