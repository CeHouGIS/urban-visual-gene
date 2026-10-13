"""Render dimension-level BatchTopK activation examples for Chapter 5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from sae_experiments.models.base_sae import BatchTopKSAE
from .nas_batch import OUTPUT


def ranked_images(output: Path, features: list[int], pool: int = 64) -> dict[int, list[tuple[int, float]]]:
    scores = np.load(output / "image_activations.f16.npy", mmap_mode="r")
    if scores.shape[1] != 4096 or any(not 0 <= feature < 4096 for feature in features):
        raise ValueError("Expected W4096 activation cache and valid feature IDs")
    best = np.empty((0, len(features)), dtype=np.float32)
    indices = np.empty((0, len(features)), dtype=np.int64)
    for start in range(0, len(scores), 2048):
        block = np.asarray(scores[start:start + 2048, features], dtype=np.float32)
        candidates = np.concatenate((best, block), axis=0)
        candidate_ids = np.concatenate((indices, np.broadcast_to(
            np.arange(start, start + len(block))[:, None], block.shape)), axis=0)
        top = np.argpartition(candidates, -min(pool, len(candidates)), axis=0)[
            -min(pool, len(candidates)):]
        best = np.take_along_axis(candidates, top, axis=0)
        indices = np.take_along_axis(candidate_ids, top, axis=0)
    result = {}
    for column, feature in enumerate(features):
        order = np.argsort(-best[:, column], kind="stable")
        result[feature] = [(int(indices[pos, column]), float(best[pos, column]))
                           for pos in order if best[pos, column] > 0]
    return result


def render(output: Path, figures: Path, features: list[int], device: str) -> None:
    if len(features) % 3 or len(set(features)) != len(features):
        raise ValueError("Use a distinct feature ID for each of the three rows per figure")
    inference = json.loads((output / "inference_report.json").read_text())
    audit = json.loads((output / "data_audit.json").read_text())
    if inference["data_identity"] != audit["dataset_identity"] or inference["images"] != 363700:
        raise ValueError("Inference cache does not match the screened NAS cohort")
    checkpoint_path = output / "model_best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if inference["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Activation cache and SAE checkpoint do not match")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("D"), checkpoint.get("K"), checkpoint.get("k")) != (768, 4096, 32):
        raise ValueError("Expected the NAS W4096/K32 checkpoint")
    model = BatchTopKSAE(768, 4096, 32).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    image_index = pd.read_parquet(output / "inference_image_index.parquet")
    cluster = np.load(output / "feature_to_cluster64.npy")
    ranking = ranked_images(output, features)
    manifests: dict[str, pd.DataFrame] = {}
    tokens: dict[str, np.ndarray] = {}
    records = []
    figures.mkdir(parents=True, exist_ok=True)
    for page in range(len(features) // 3):
        fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
        for row, feature in enumerate(features[page * 3:page * 3 + 3]):
            selected = []
            used_panos = set()
            for rank, (image_id, score) in enumerate(ranking[feature], 1):
                city = str(image_index.iloc[image_id].city)
                source_id = int(image_index.iloc[image_id].source_image_index)
                if city not in manifests:
                    city_key = city.replace("/", "__")
                    manifests[city] = pd.read_parquet(output / "manifests" / f"{city_key}.parquet")
                    tokens[city] = np.load(output / "tokens" / "patch_tokens" /
                                           city_key / "tokens.f16.npy", mmap_mode="r")
                manifest = manifests[city]
                source = manifest.iloc[source_id]
                panoid = str(source.panoid)
                if (city, panoid) in used_panos:
                    continue
                used_panos.add((city, panoid))
                pano = manifest[manifest.panoid.astype(str) == panoid].sort_values("heading")
                if len(pano) != 4 or pano.heading.tolist() != [0, 90, 180, 270]:
                    raise ValueError(f"Incomplete four-view panorama: {city}/{panoid}")
                patches = np.asarray(tokens[city][pano.source_image_index.to_numpy()], dtype=np.float32)
                with torch.inference_mode():
                    x = torch.from_numpy(patches.reshape(-1, 768)).to(device)
                    active = model.encode(x)[:, feature].float().cpu().numpy().reshape(4, 14, 14)
                heading_index = pano.heading.tolist().index(int(source.heading))
                heat = active[heading_index]
                pooled = np.sort(heat.ravel())[-20:].mean()
                if not np.isclose(pooled, score, rtol=.02, atol=.003):
                    raise ValueError(f"Recomputed activation differs from cache: f{feature}, image {image_id}")
                with Image.open(source.image_path) as image:
                    original = np.asarray(image.convert("RGB").resize((320, 320), Image.Resampling.BICUBIC))
                heat_up = np.asarray(Image.fromarray(heat).resize((320, 320), Image.Resampling.BILINEAR))
                selected.append((original, heat_up, city, panoid, int(source.heading),
                                 image_id, rank, score, str(source.image_path)))
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                raise ValueError(f"Feature {feature} has fewer than two distinct panoramas")
            scale = max(float(np.quantile(np.concatenate([item[1].ravel() for item in selected]), .98)),
                        1e-6)
            for pair, (original, heat, city, panoid, heading, image_id, rank, score, path) in enumerate(selected):
                strength = np.clip(heat / scale, 0, 1)
                rgba = plt.get_cmap("inferno")(strength)[..., :3]
                alpha = (.74 * strength)[..., None]
                overlay = (original * (1 - alpha) + 255 * rgba * alpha).clip(0, 255).astype(np.uint8)
                left, right = axes[row, pair * 2:pair * 2 + 2]
                left.imshow(original)
                right.imshow(overlay)
                left.set_title(f"f{feature:04d}  {city.split('/')[-1]}  {heading}°  (rank {rank})", fontsize=9)
                right.set_title("SAE patch activation", fontsize=9)
                for ax in (left, right):
                    ax.axis("off")
                records.append({"figure": page + 1, "feature": feature, "cluster64": int(cluster[feature]),
                                "rank": rank, "global_image_index": image_id,
                                "pooled_activation": score, "city": city, "panoid": panoid,
                                "heading": heading, "image_path": path})
        target = figures / f"Fig_PatchSAE_Examples_{page + 1}.png"
        fig.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    pd.DataFrame(records).to_csv(figures / "patch_examples_provenance.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--features", required=True, help="Comma-separated feature IDs, in groups of three")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    features = [int(item) for item in args.features.split(",")]
    render(args.output, args.figures or args.output / "figures" / "chapter5" / "patch_examples",
           features, args.device)


if __name__ == "__main__":
    main()
