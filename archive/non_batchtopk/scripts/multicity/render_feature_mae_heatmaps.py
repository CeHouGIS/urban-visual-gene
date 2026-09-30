#!/usr/bin/env python3
"""Rank clean street-view images and render Feature-MAE latent heatmaps."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import mmap
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.multicity.config import CITIES, city_slug
from scripts.multicity.patch_config import INPUT_DIM, OUTPUT_ROOT as TOKEN_ROOT, PATCH_GRID, token_path
from scripts.multicity.train_feature_mae import FeatureMAE


DEFAULT_ROOT = TOKEN_ROOT.parent / "feature_mae_n30x12800_qc"


def load_model(root: Path) -> FeatureMAE:
    checkpoint = torch.load(
        root / "mae" / "model_best.pt", map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = FeatureMAE(
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_width=int(config["decoder_width"]),
        decoder_layers=int(config["decoder_layers"]),
        mask_ratio=float(config["mask_ratio"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval().requires_grad_(False).to("cuda")


def manifest_path(root: Path, city: str) -> Path:
    return root / "filtered_manifests" / f"{city_slug(city)}.parquet"


def _save_scan(
    path: Path,
    scores: np.ndarray,
    city_indices: np.ndarray,
    image_indices: np.ndarray,
    patch_indices: np.ndarray,
    next_city: int,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        scores=scores,
        city_indices=city_indices,
        image_indices=image_indices,
        patch_indices=patch_indices,
        next_city=np.asarray(next_city, dtype=np.int16),
    )
    os.replace(temporary, path)


@torch.inference_mode()
def find_top_images(
    model: FeatureMAE,
    cities: list[str],
    root: Path,
    token_root: Path,
    candidates: int,
    image_batch: int,
    checkpoint_path: Path,
    max_images_per_city: int = 0,
) -> dict[str, np.ndarray]:
    width = model.width
    best_scores = np.full((width, candidates), -np.inf, dtype=np.float32)
    best_city = np.full((width, candidates), -1, dtype=np.int16)
    best_image = np.full((width, candidates), -1, dtype=np.int32)
    best_patch = np.full((width, candidates), -1, dtype=np.int16)
    next_city = 0
    if checkpoint_path.exists():
        with np.load(checkpoint_path) as saved:
            best_scores = saved["scores"]
            best_city = saved["city_indices"]
            best_image = saved["image_indices"]
            best_patch = saved["patch_indices"]
            next_city = int(saved["next_city"])
        print(f"resuming rank scan at city {next_city + 1}", flush=True)

    for city_i, city in enumerate(cities[next_city:], next_city):
        frame = pd.read_parquet(
            manifest_path(root, city), columns=["source_image_index"]
        )
        image_ids = frame["source_image_index"].to_numpy(np.int64)
        if max_images_per_city:
            image_ids = image_ids[:max_images_per_city]
        tokens = np.load(token_path(city, token_root), mmap_mode="r")
        started = time.time()
        for start in range(0, len(image_ids), image_batch):
            ids = image_ids[start : start + image_batch]
            block = np.asarray(tokens[ids], dtype=np.float32)
            batch = torch.from_numpy(block).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = model.encode_full(batch)
            scores, patches = latent.max(dim=1)
            scores_np = scores.float().cpu().numpy().T
            patches_np = patches.cpu().numpy().T.astype(np.int16)
            new_images = np.broadcast_to(ids.astype(np.int32), (width, len(ids)))
            new_cities = np.full((width, len(ids)), city_i, dtype=np.int16)
            all_scores = np.concatenate((best_scores, scores_np), axis=1)
            all_city = np.concatenate((best_city, new_cities), axis=1)
            all_image = np.concatenate((best_image, new_images), axis=1)
            all_patch = np.concatenate((best_patch, patches_np), axis=1)
            keep = np.argpartition(-all_scores, candidates - 1, axis=1)[:, :candidates]
            best_scores = np.take_along_axis(all_scores, keep, axis=1)
            best_city = np.take_along_axis(all_city, keep, axis=1)
            best_image = np.take_along_axis(all_image, keep, axis=1)
            best_patch = np.take_along_axis(all_patch, keep, axis=1)
        mapping = getattr(tokens, "_mmap", None)
        if mapping is not None and hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            mapping.madvise(mmap.MADV_DONTNEED)
        _save_scan(
            checkpoint_path, best_scores, best_city, best_image, best_patch, city_i + 1
        )
        print(
            f"rank scan [{city_i+1:02d}/{len(cities)}] {city}: "
            f"{len(image_ids):,} images, {time.time()-started:.1f}s",
            flush=True,
        )

    order = np.argsort(-best_scores, axis=1)
    return {
        "scores": np.take_along_axis(best_scores, order, axis=1),
        "city_indices": np.take_along_axis(best_city, order, axis=1),
        "image_indices": np.take_along_axis(best_image, order, axis=1),
        "patch_indices": np.take_along_axis(best_patch, order, axis=1),
    }


def select_unique_panos(
    ranking: dict[str, np.ndarray], manifests: list[pd.DataFrame], top_n: int
) -> dict[str, np.ndarray]:
    selected = {key: [] for key in ranking}
    lookups = [frame.set_index("source_image_index", drop=False) for frame in manifests]
    for feature in range(ranking["scores"].shape[0]):
        rows = []
        seen = set()
        for rank in range(ranking["scores"].shape[1]):
            city_i = int(ranking["city_indices"][feature, rank])
            image_i = int(ranking["image_indices"][feature, rank])
            if city_i < 0 or image_i < 0:
                continue
            panoid = str(lookups[city_i].loc[image_i, "panoid"])
            key = (city_i, panoid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(rank)
            if len(rows) == top_n:
                break
        if len(rows) != top_n:
            raise RuntimeError(f"dimension {feature}: only {len(rows)} unique panoramas")
        for key, value in ranking.items():
            selected[key].append(value[feature, rows])
    return {key: np.stack(value) for key, value in selected.items()}


def read_original(row: pd.Series) -> Image.Image:
    fd = os.open(str(row["tar_path"]), os.O_RDONLY)
    try:
        payload = os.pread(fd, int(row["jpg_size"]), int(row["jpg_offset"]))
    finally:
        os.close(fd)
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def colorize(values: np.ndarray) -> np.ndarray:
    positions = np.asarray([0.0, 0.20, 0.45, 0.70, 1.0], dtype=np.float32)
    colors = np.asarray(
        [[13, 19, 43], [20, 83, 205], [0, 218, 210], [255, 221, 38], [232, 34, 20]],
        dtype=np.float32,
    )
    return np.stack(
        [np.interp(values, positions, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype(np.uint8)


def overlay_heatmap(original: Image.Image, activation: np.ndarray, scale: float) -> Image.Image:
    # Feature-MAE latents are signed. Positive evidence is the relevant direction
    # for the top-activation gallery; negative/near-zero patches remain unchanged.
    normalized = np.clip(np.maximum(activation, 0.0) / max(scale, 1e-8), 0.0, 1.0)
    heat = Image.fromarray(colorize(normalized), "RGB").resize(
        original.size, Image.Resampling.BICUBIC
    )
    alpha_values = np.clip((normalized - 0.04) / 0.96, 0.0, 1.0) ** 0.72
    alpha = Image.fromarray((alpha_values * 190).astype(np.uint8), "L").resize(
        original.size, Image.Resampling.BICUBIC
    )
    return Image.composite(heat, original, alpha)


@torch.inference_mode()
def render(
    model: FeatureMAE,
    ranking: dict[str, np.ndarray],
    cities: list[str],
    root: Path,
    token_root: Path,
) -> None:
    manifests = [pd.read_parquet(manifest_path(root, city)) for city in cities]
    lookups = [frame.set_index("source_image_index", drop=False) for frame in manifests]
    token_arrays = [np.load(token_path(city, token_root), mmap_mode="r") for city in cities]
    heat_root = root / "heatmaps"
    heat_root.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for feature in range(model.width):
        feature_dir = heat_root / f"feature_{feature:03d}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        city_ids = ranking["city_indices"][feature].astype(int)
        image_ids = ranking["image_indices"][feature].astype(int)
        blocks = [
            np.asarray(token_arrays[city_i][image_i], dtype=np.float32)
            for city_i, image_i in zip(city_ids, image_ids)
        ]
        batch = torch.from_numpy(np.stack(blocks)).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode_full(batch)
        activation_maps = latent[:, :, feature].float().cpu().numpy().reshape(
            -1, PATCH_GRID, PATCH_GRID
        )
        scale = float(ranking["scores"][feature, 0])
        rendered = []
        examples = []
        for rank, (city_i, image_i, activation) in enumerate(
            zip(city_ids, image_ids, activation_maps), 1
        ):
            row = lookups[city_i].loc[image_i]
            original = read_original(row)
            overlay = overlay_heatmap(original, activation, scale)
            draw = ImageDraw.Draw(overlay)
            label = (
                f"MAE dim {feature:03d} | rank {rank:02d} | "
                f"{cities[city_i].split('/')[-1]} | {ranking['scores'][feature, rank-1]:.3f}"
            )
            draw.rectangle((0, 0, min(overlay.width, 760), 28), fill=(0, 0, 0))
            draw.text((8, 7), label, fill=(255, 255, 255), font=ImageFont.load_default())
            filename = f"rank_{rank:02d}_{city_slug(cities[city_i])}_{image_i:05d}.jpg"
            overlay.save(feature_dir / filename, quality=90)
            rendered.append(overlay)
            examples.append(
                {
                    "rank": rank,
                    "city": cities[city_i],
                    "source_image_index": int(image_i),
                    "panoid": str(row["panoid"]),
                    "heading": int(row["heading"]),
                    "score": float(ranking["scores"][feature, rank - 1]),
                    "file": filename,
                }
            )

        thumb_w, thumb_h, columns = 320, 240, 5
        rows = int(np.ceil(len(rendered) / columns))
        sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "#05080d")
        for position, image in enumerate(rendered):
            tile = image.copy()
            tile.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x0 = (position % columns) * thumb_w + (thumb_w - tile.width) // 2
            y0 = (position // columns) * thumb_h + (thumb_h - tile.height) // 2
            sheet.paste(tile, (x0, y0))
        sheet.save(feature_dir / "contact_sheet.jpg", quality=90)
        (feature_dir / "examples.json").write_text(
            json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        city_counts: dict[str, int] = {}
        for city_i in city_ids:
            city_counts[cities[city_i]] = city_counts.get(cities[city_i], 0) + 1
        index_rows.append(
            {
                "feature": feature,
                "top_score": scale,
                "min_selected_score": float(ranking["scores"][feature, -1]),
                "cities": len(city_counts),
                "city_counts": json.dumps(city_counts, ensure_ascii=False),
                "contact_sheet": str(feature_dir / "contact_sheet.jpg"),
            }
        )
        if (feature + 1) % 32 == 0:
            print(f"rendered {feature+1}/{model.width} MAE dimensions", flush=True)
    pd.DataFrame(index_rows).to_csv(heat_root / "feature_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", default=list(CITIES))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument("--top-images", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--scan-image-batch", type=int, default=192)
    parser.add_argument("--max-images-per-city", type=int, default=0)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.candidates < args.top_images:
        raise ValueError("--candidates must be at least --top-images")
    model = load_model(args.data_root)
    ranking_path = args.data_root / "mae" / "top_image_activations.npz"
    scan_path = args.data_root / "mae" / "top_image_scan_checkpoint.npz"
    if ranking_path.exists() and not args.rerank:
        with np.load(ranking_path) as saved:
            ranking = {key: saved[key] for key in saved.files}
    else:
        if args.rerank:
            scan_path.unlink(missing_ok=True)
        candidates = find_top_images(
            model, args.cities, args.data_root, args.token_root,
            args.candidates, args.scan_image_batch, scan_path,
            args.max_images_per_city,
        )
        manifests = [pd.read_parquet(manifest_path(args.data_root, city)) for city in args.cities]
        ranking = select_unique_panos(candidates, manifests, args.top_images)
        np.savez_compressed(ranking_path, **ranking)
    render(model, ranking, args.cities, args.data_root, args.token_root)


if __name__ == "__main__":
    main()
