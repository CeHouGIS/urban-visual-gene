"""UMAP and four-heading image evidence for the replacement BatchTopK cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image

from sae_experiments.models.base_sae import BatchTopKSAE
from .figures import save
from .nas_batch import CITIES, OUTPUT
from .train import write_json


def embeddings(output: Path, figures: Path, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    image_index = pd.read_parquet(output / "inference_image_index.parquet")
    counts = np.load(output / "image_cluster64_patch_winners.u16.npy", mmap_mode="r")
    labels64 = np.load(output / "feature_to_cluster64.npy")
    labels32 = np.load(output / "feature_to_cluster32.npy")
    selected = []
    for city in CITIES:
        ids = np.flatnonzero(image_index.city.to_numpy() == city)
        selected.extend(rng.choice(ids, min(200, len(ids)), replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    x = np.asarray(counts[selected], dtype=np.float32)
    x /= x.sum(axis=1, keepdims=True).clip(min=1)
    coord = umap.UMAP(n_neighbors=30, min_dist=.15, metric="cosine",
                      random_state=seed).fit_transform(x)
    city_values = image_index.city.iloc[selected].to_numpy()
    frame = pd.DataFrame({"global_image_index": selected, "city": city_values,
                          "umap_x": coord[:, 0], "umap_y": coord[:, 1],
                          "dominant_cluster": np.argmax(x, axis=1)})
    frame.to_parquet(output / "results" / "image_sample_umap.parquet", index=False)
    generated = []
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = plt.get_cmap("tab20", len(CITIES))
    for i, city in enumerate(CITIES):
        mask = city_values == city
        ax.scatter(coord[mask, 0], coord[mask, 1], s=3, alpha=.45,
                   color=palette(i), label=city.split("/")[-1], rasterized=True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), ncol=2, fontsize=7, markerscale=3)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
           title="Street-view image composition, 200 images per city")
    save(fig, figures, "Fig_All_City_Samples_UMAP")
    generated.append("Fig_All_City_Samples_UMAP")

    parent = np.array([np.bincount(labels32[labels64 == j], minlength=32).argmax()
                       for j in range(64)])
    dominant = parent[np.argmax(x, axis=1)]
    fig, ax = plt.subplots(figsize=(9, 8))
    dots = ax.scatter(coord[:, 0], coord[:, 1], s=3, c=dominant,
                      cmap="turbo", alpha=.7, rasterized=True)
    fig.colorbar(dots, ax=ax, label="Dominant coarse SAE cluster", fraction=.035)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
           title="Same image UMAP colored by dominant SAE cluster")
    save(fig, figures, "Fig_BatchTopK_Image_UMAP_Dominant_Cluster")
    generated.append("Fig_BatchTopK_Image_UMAP_Dominant_Cluster")

    checkpoint = torch.load(output / "model_best.pt", map_location="cpu", weights_only=False)
    decoder = checkpoint["state"]["decoder.weight"].float().numpy().T
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True).clip(min=1e-8)
    features = umap.UMAP(n_neighbors=30, min_dist=.1, metric="cosine",
                         random_state=seed).fit_transform(decoder)
    pd.DataFrame({"feature": np.arange(4096), "umap_x": features[:, 0],
                  "umap_y": features[:, 1], "cluster64": labels64}).to_csv(
                      output / "results" / "dictionary_umap.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    dots = ax.scatter(features[:, 0], features[:, 1], c=labels64, cmap="turbo",
                      s=4, alpha=.7, rasterized=True)
    fig.colorbar(dots, ax=ax, label="64-cluster assignment", fraction=.04)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
           title="All 4,096 SAE decoder directions (cosine UMAP)")
    save(fig, figures, "Fig_BatchTopK_UMAP_4096_Dimensions")
    generated.append("Fig_BatchTopK_UMAP_4096_Dimensions")
    return generated


def examples(output: Path, figures: Path, device: str) -> list[str]:
    composition = pd.read_csv(output / "results" / "city_visual_composition_fine.csv", index_col=0)
    selected_clusters = np.argsort(-composition.to_numpy().std(axis=0))[:16]
    counts = np.load(output / "image_cluster64_patch_winners.u16.npy", mmap_mode="r")
    image_index = pd.read_parquet(output / "inference_image_index.parquet")
    labels64 = np.load(output / "feature_to_cluster64.npy")
    checkpoint = torch.load(output / "model_best.pt", map_location="cpu", weights_only=False)
    model = BatchTopKSAE(768, 4096, 32).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    manifest_cache: dict[str, pd.DataFrame] = {}
    token_cache = {}
    rows = []
    originals = []
    overlays = []
    for cluster in selected_clusters:
        global_idx = int(np.argmax(np.asarray(counts[:, cluster])))
        score = int(counts[global_idx, cluster])
        city = str(image_index.iloc[global_idx].city)
        source_idx = int(image_index.iloc[global_idx].source_image_index)
        if city not in manifest_cache:
            manifest_cache[city] = pd.read_parquet(
                output / "manifests" / (city.replace("/", "__") + ".parquet"))
            token_cache[city] = np.load(
                output / "tokens" / "patch_tokens" / city.replace("/", "__") / "tokens.f16.npy",
                mmap_mode="r")
        manifest = manifest_cache[city]
        panoid = str(manifest.iloc[source_idx].panoid)
        pano_rows = manifest[manifest.panoid.astype(str) == panoid].sort_values("heading")
        if len(pano_rows) != 4 or pano_rows.heading.tolist() != [0, 90, 180, 270]:
            raise ValueError(f"Incomplete panorama for example: {city}/{panoid}")
        paths = pano_rows.image_path.tolist()
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB").resize((224, 224))))
        original = np.concatenate(images, axis=1)
        blocks = np.asarray(token_cache[city][pano_rows.source_image_index.to_numpy()], dtype=np.float32)
        with torch.inference_mode():
            x = torch.from_numpy(blocks.reshape(-1, 768)).to(device)
            active = model.encode(x).float()
            active_cluster = active[:, labels64 == cluster].sum(dim=1).cpu().numpy()
        heat = np.concatenate([active_cluster.reshape(4, 14, 14)[i] for i in range(4)], axis=1)
        heat = np.asarray(Image.fromarray(heat.astype(np.float32)).resize((896, 224),
                         Image.Resampling.BILINEAR))
        cap = float(np.quantile(heat, .98))
        heat = (heat / max(cap, 1e-6)).clip(0, 1)
        color = plt.get_cmap("inferno")(heat)[..., :3]
        alpha = (.72 * heat)[..., None]
        overlay = (original * (1 - alpha) + color * 255 * alpha).clip(0, 255).astype(np.uint8)
        rows.append({"cluster64": int(cluster), "city": city, "panoid": panoid,
                     "global_image_index": global_idx, "winning_patches": score,
                     "image_paths": "|".join(paths)})
        originals.append(original)
        overlays.append(overlay)
    pd.DataFrame(rows).to_csv(output / "results" / "cluster_example_panoramas.csv", index=False)
    generated = []
    for images, name, title in (
        (originals, "Fig_BatchTopK_Panorama_Examples", "SAE cluster examples: four headings per panorama"),
        (overlays, "Fig_BatchTopK_Panorama_Activation_Overlays",
         "SAE cluster activation over original four-heading panoramas"),
    ):
        fig, axes = plt.subplots(4, 4, figsize=(18, 6.7), constrained_layout=True)
        for ax, picture, row in zip(axes.flat, images, rows):
            ax.imshow(picture)
            ax.set_title(f"F{row['cluster64']:02d}  {row['city'].split('/')[-1]}",
                         fontsize=9)
            ax.axis("off")
        fig.suptitle(title, fontsize=13)
        save(fig, figures, name)
        generated.append(name)
    return generated


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    figures = args.output / "figures"
    generated = embeddings(args.output, figures, args.seed)
    generated.extend(examples(args.output, figures, args.device))
    inventory_path = args.output / "figure_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    mapping = {
        "Fig_All_City_UMAP_512_Dimensions": "Fig_BatchTopK_UMAP_4096_Dimensions",
        "Fig_All_City_UMAP_Visual_Families": "Fig_BatchTopK_Image_UMAP_Dominant_Cluster",
    }
    for row in inventory["figures"]:
        target = mapping.get(row["legacy_figure"], row["legacy_figure"])
        if target in generated:
            note = ("Dominant SAE cluster colors replace the old labeled visual families."
                    if row["legacy_figure"] == "Fig_All_City_UMAP_Visual_Families" else
                    "Embeds 4,096 SAE decoder directions rather than 512 MAE dimensions."
                    if row["legacy_figure"] == "Fig_All_City_UMAP_512_Dimensions" else None)
            row.update(status="recomputed_from_SAE", new_figure=target, reason=note)
    inventory["generated"] = sorted(set(inventory.get("generated", [])) | set(generated))
    inventory["recomputed_count"] = sum(row["status"] == "recomputed_from_SAE"
                                        for row in inventory["figures"])
    write_json(inventory_path, inventory)
    print(json.dumps({"generated": generated, "recomputed": inventory["recomputed_count"]}, indent=2))


if __name__ == "__main__":
    main()
