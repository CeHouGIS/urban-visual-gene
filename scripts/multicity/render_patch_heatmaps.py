"""Find top images per SAE dimension and overlay patch activations on originals."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from scripts.multicity.config import CITIES, city_slug
from scripts.multicity.patch_config import (
    INPUT_DIM,
    OUTPUT_ROOT,
    PATCH_GRID,
    TOPK,
    WIDTH,
    image_manifest_path,
    token_path,
)
from scripts.multicity.train_topk_sae import TopKSAE


def load_model(root: Path) -> TopKSAE:
    checkpoint = torch.load(
        root / "patch_sae" / "model_best.pt", map_location="cuda", weights_only=False
    )
    config = checkpoint.get("config", {})
    model = TopKSAE(
        INPUT_DIM, int(config.get("width", WIDTH)), int(config.get("topk", TOPK))
    ).to("cuda")
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model


@torch.inference_mode()
def find_top_images(
    model: TopKSAE,
    cities: list[str],
    root: Path,
    top_n: int,
    image_batch: int,
) -> dict[str, np.ndarray]:
    width = model.width
    best_scores = np.full((width, top_n), -np.inf, dtype=np.float32)
    best_city = np.full((width, top_n), -1, dtype=np.int16)
    best_image = np.full((width, top_n), -1, dtype=np.int32)
    best_patch = np.full((width, top_n), -1, dtype=np.int16)

    for city_i, city in enumerate(cities):
        tokens = np.load(token_path(city, root), mmap_mode="r")
        for start in range(0, len(tokens), image_batch):
            block = np.asarray(tokens[start:start + image_batch], dtype=np.float32)
            batch_n = len(block)
            x = torch.from_numpy(block.reshape(-1, INPUT_DIM)).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                codes = model.encode(x)
            codes = codes.float().reshape(batch_n, PATCH_GRID * PATCH_GRID, width)
            scores, patches = codes.max(dim=1)
            scores_np = scores.cpu().numpy().T
            patches_np = patches.cpu().numpy().T.astype(np.int16)

            new_images = np.broadcast_to(
                np.arange(start, start + batch_n, dtype=np.int32), (width, batch_n)
            )
            new_cities = np.full((width, batch_n), city_i, dtype=np.int16)
            all_scores = np.concatenate((best_scores, scores_np), axis=1)
            all_city = np.concatenate((best_city, new_cities), axis=1)
            all_image = np.concatenate((best_image, new_images), axis=1)
            all_patch = np.concatenate((best_patch, patches_np), axis=1)
            keep = np.argpartition(-all_scores, top_n - 1, axis=1)[:, :top_n]
            best_scores = np.take_along_axis(all_scores, keep, axis=1)
            best_city = np.take_along_axis(all_city, keep, axis=1)
            best_image = np.take_along_axis(all_image, keep, axis=1)
            best_patch = np.take_along_axis(all_patch, keep, axis=1)
        print(f"rank scan [{city_i+1:02d}/{len(cities)}] {city}", flush=True)

    order = np.argsort(-best_scores, axis=1)
    return {
        "scores": np.take_along_axis(best_scores, order, axis=1),
        "city_indices": np.take_along_axis(best_city, order, axis=1),
        "image_indices": np.take_along_axis(best_image, order, axis=1),
        "patch_indices": np.take_along_axis(best_patch, order, axis=1),
    }


def read_original(row: pd.Series) -> Image.Image:
    fd = os.open(str(row["tar_path"]), os.O_RDONLY)
    try:
        payload = os.pread(fd, int(row["jpg_size"]), int(row["jpg_offset"]))
    finally:
        os.close(fd)
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def colorize(values: np.ndarray) -> np.ndarray:
    """Small blue/cyan/yellow/red colormap without a matplotlib dependency."""
    positions = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    colors = np.asarray(
        [[18, 18, 70], [20, 90, 210], [0, 220, 210], [255, 225, 30], [220, 20, 10]],
        dtype=np.float32,
    )
    rgb = np.stack(
        [np.interp(values, positions, colors[:, channel]) for channel in range(3)], axis=-1
    )
    return rgb.astype(np.uint8)


def overlay_heatmap(original: Image.Image, activation: np.ndarray, scale: float) -> Image.Image:
    normalized = np.clip(activation / max(scale, 1e-8), 0.0, 1.0)
    small_rgb = colorize(normalized)
    heat = Image.fromarray(small_rgb, mode="RGB").resize(original.size, Image.Resampling.BICUBIC)
    alpha_small = Image.fromarray((normalized * 180).astype(np.uint8), mode="L")
    alpha = alpha_small.resize(original.size, Image.Resampling.BICUBIC)
    return Image.composite(heat, original, alpha)


@torch.inference_mode()
def render(
    model: TopKSAE,
    ranking: dict[str, np.ndarray],
    cities: list[str],
    root: Path,
) -> None:
    manifests = [
        pd.read_parquet(image_manifest_path(city, root)).sort_values("image_index").reset_index(drop=True)
        for city in cities
    ]
    token_arrays = [np.load(token_path(city, root), mmap_mode="r") for city in cities]
    heat_root = root / "heatmaps"
    heat_root.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for feature in range(model.width):
        feature_dir = heat_root / f"feature_{feature:03d}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        rendered = []
        scale = float(ranking["scores"][feature, 0])
        examples = []
        for rank in range(ranking["scores"].shape[1]):
            city_i = int(ranking["city_indices"][feature, rank])
            image_i = int(ranking["image_indices"][feature, rank])
            row = manifests[city_i].iloc[image_i]
            original = read_original(row)
            tokens = np.asarray(token_arrays[city_i][image_i], dtype=np.float32)
            x = torch.from_numpy(tokens).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                activation = model.encode(x)[:, feature]
            heat = activation.float().cpu().numpy().reshape(PATCH_GRID, PATCH_GRID)
            overlay = overlay_heatmap(original, heat, scale)
            draw = ImageDraw.Draw(overlay)
            label = (
                f"feature {feature:03d} | rank {rank+1:02d} | "
                f"{cities[city_i]} | score {ranking['scores'][feature, rank]:.4f}"
            )
            draw.rectangle((0, 0, min(overlay.width, 760), 28), fill=(0, 0, 0))
            draw.text((8, 7), label, fill=(255, 255, 255))
            filename = f"rank_{rank+1:02d}_{city_slug(cities[city_i])}_{image_i:05d}.jpg"
            overlay.save(feature_dir / filename, quality=92)
            rendered.append(overlay.copy())
            examples.append({
                "rank": rank + 1, "city": cities[city_i], "image_index": image_i,
                "panoid": str(row["panoid"]), "heading": int(row["heading"]),
                "score": float(ranking["scores"][feature, rank]), "file": filename,
            })

        thumb_w, thumb_h = 320, 240
        columns = 4
        rows = int(np.ceil(len(rendered) / columns))
        sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
        for i, image in enumerate(rendered):
            tile = image.copy(); tile.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x0 = (i % columns) * thumb_w + (thumb_w - tile.width) // 2
            y0 = (i // columns) * thumb_h + (thumb_h - tile.height) // 2
            sheet.paste(tile, (x0, y0))
        sheet.save(feature_dir / "contact_sheet.jpg", quality=92)
        (feature_dir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
        index_rows.append({
            "feature": feature, "top_score": scale,
            "contact_sheet": str(feature_dir / "contact_sheet.jpg"),
        })
        if (feature + 1) % 32 == 0:
            print(f"rendered {feature+1}/{model.width} SAE dimensions", flush=True)
    pd.DataFrame(index_rows).to_csv(heat_root / "feature_index.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--top-images", type=int, default=16)
    ap.add_argument("--scan-image-batch", type=int, default=32)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    model = load_model(args.output_root)
    ranking_path = args.output_root / "patch_sae" / "top_image_activations.npz"
    if ranking_path.exists() and not args.rerank:
        with np.load(ranking_path) as data:
            ranking = {key: data[key] for key in data.files}
    else:
        ranking = find_top_images(
            model, args.cities, args.output_root, args.top_images, args.scan_image_batch
        )
        np.savez_compressed(ranking_path, **ranking)
    render(model, ranking, args.cities, args.output_root)


if __name__ == "__main__":
    main()
