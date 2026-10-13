"""City-specific SAE cluster pairs and same-image activation examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from PIL import Image

from sae_experiments.models.base_sae import BatchTopKSAE
from .figures import save
from .nas_batch import OUTPUT
from .train import write_json


def cooccurrence(counts: np.ndarray, city_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count pairs with at least four winning patches in one directional image."""
    cities, city_ids = np.unique(city_values, return_inverse=True)
    matrices = np.zeros((len(cities), 64, 64), dtype=np.int64)
    city_sizes = np.bincount(city_ids)
    for city_index in range(len(cities)):
        ids = np.flatnonzero(city_ids == city_index)
        for start in range(0, len(ids), 2048):
            present = sp.csr_matrix(np.asarray(counts[ids[start:start + 2048]]) >= 4,
                                    dtype=np.int32)
            matrices[city_index] += (present.T @ present).toarray()
    return cities, city_sizes, matrices


def city_pairs(cities: np.ndarray, sizes: np.ndarray, matrices: np.ndarray) -> pd.DataFrame:
    total = matrices.sum(axis=0)
    rows = []
    for city, n, matrix in zip(cities, sizes, matrices):
        rest = total - matrix
        rest_n = int(sizes.sum() - n)
        for a, b in zip(*np.triu_indices(64, k=1)):
            local, elsewhere = int(matrix[a, b]), int(rest[a, b])
            if local < 30 or elsewhere < 100:
                continue
            local_rate = local / n
            rest_rate = elsewhere / rest_n
            rows.append({"city": city, "cluster_a": a, "cluster_b": b,
                         "city_images": int(n), "city_joint_images": local,
                         "rest_joint_images": elsewhere,
                         "city_rate": local_rate, "rest_rate": rest_rate,
                         "log2_city_vs_rest_rate": float(np.log2(local_rate / rest_rate))})
    return pd.DataFrame(rows).sort_values("log2_city_vs_rest_rate", ascending=False)


def global_pairs(matrices: np.ndarray, images: int) -> pd.DataFrame:
    total = matrices.sum(axis=0)
    support = np.diag(total)
    rows = []
    for a, b in zip(*np.triu_indices(64, k=1)):
        joint = int(total[a, b])
        if joint < 500:
            continue
        expected = float(support[a]) * float(support[b]) / images
        rows.append({"cluster_a": a, "cluster_b": b, "joint_images": joint,
                     "expected_independent": expected,
                     "ppmi": float(max(0, np.log2(joint / expected)))})
    return pd.DataFrame(rows).sort_values(["ppmi", "joint_images"], ascending=False)


def example_rows(pairs: pd.DataFrame, counts: np.ndarray, index: pd.DataFrame,
                 limit: int = 6) -> pd.DataFrame:
    rows = []
    used_panos = set()
    for pair in pairs.itertuples(index=False):
        if pair.ppmi <= 0 or len(rows) >= limit:
            break
        a, b = int(pair.cluster_a), int(pair.cluster_b)
        strengths = np.minimum(np.asarray(counts[:, a]), np.asarray(counts[:, b]))
        for image_id in np.argsort(strengths)[-100:][::-1]:
            if strengths[image_id] < 4:
                break
            block_start = int(image_id) // 4 * 4
            key = (str(index.iloc[image_id].city), block_start)
            if key in used_panos:
                continue
            used_panos.add(key)
            rows.append({"cluster_a": a, "cluster_b": b, "ppmi": pair.ppmi,
                         "joint_images": int(pair.joint_images), "global_image_index": int(image_id),
                         "block_start": block_start, "city": key[0],
                         "winning_patches_a": int(counts[image_id, a]),
                         "winning_patches_b": int(counts[image_id, b])})
            break
    if not rows:
        raise ValueError("No supported co-activating image examples")
    return pd.DataFrame(rows)


def draw_examples(output: Path, figures: Path, rows: pd.DataFrame,
                  index: pd.DataFrame, device: str) -> None:
    checkpoint = torch.load(output / "model_best.pt", map_location="cpu", weights_only=False)
    labels = np.load(output / "feature_to_cluster64.npy")
    model = BatchTopKSAE(768, 4096, 32).to(device).eval()
    model.load_state_dict(checkpoint["state"])
    manifests = {}
    tokens = {}
    fig, axes = plt.subplots(len(rows), 2, figsize=(16, 2.7 * len(rows)), squeeze=False,
                             constrained_layout=True)
    for row_idx, row in enumerate(rows.itertuples(index=False)):
        group = index.iloc[row.block_start:row.block_start + 4]
        if len(group) != 4 or group.city.nunique() != 1 or group.city.iloc[0] != row.city:
            raise ValueError("BatchTopK inference group does not contain one four-view panorama")
        city = row.city
        if city not in manifests:
            slug = city.replace("/", "__")
            manifests[city] = pd.read_parquet(output / "manifests" / (slug + ".parquet"))
            tokens[city] = np.load(output / "tokens" / "patch_tokens" / slug / "tokens.f16.npy",
                                   mmap_mode="r")
        source_ids = group.source_image_index.to_numpy(np.int64)
        manifest_rows = manifests[city].iloc[source_ids]
        if (manifest_rows.panoid.nunique() != 1
                or manifest_rows.heading.tolist() != [0, 90, 180, 270]):
            raise ValueError("Manifest group is not a complete panorama")
        pictures = []
        for path in manifest_rows.image_path:
            with Image.open(path) as image:
                pictures.append(np.asarray(image.convert("RGB").resize((224, 224))))
        original = np.concatenate(pictures, axis=1)
        block = np.asarray(tokens[city][source_ids], dtype=np.float32)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                active = model.encode(torch.from_numpy(block.reshape(-1, 768)).to(device))
        active = active.float()
        winners = active.argmax(-1).cpu().numpy()
        winner_clusters = labels[winners]
        is_active = active.amax(-1).cpu().numpy() > 0
        overlay = original.astype(np.float32).copy()
        for cluster, color in ((row.cluster_a, np.array([236, 64, 60])),
                               (row.cluster_b, np.array([35, 171, 204]))):
            mask = ((winner_clusters == cluster) & is_active).reshape(4, 14, 14)
            mask = np.concatenate(mask, axis=1).astype(np.uint8)
            mask = np.asarray(Image.fromarray(mask).resize((896, 224), Image.Resampling.NEAREST),
                              dtype=bool)
            overlay[mask] = .5 * overlay[mask] + .5 * color
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 1].imshow(overlay.astype(np.uint8))
        axes[row_idx, 0].set_title(f"{city} | F{row.cluster_a:02d} + F{row.cluster_b:02d}", fontsize=9)
        axes[row_idx, 1].set_title(f"Red: F{row.cluster_a:02d}  Cyan: F{row.cluster_b:02d} | "
                                   f"joint images: {row.joint_images}", fontsize=9)
        for ax in axes[row_idx]:
            ax.axis("off")
    save(fig, figures, "Fig_Image_Level_Atypical_Coactivation")


def run(output: Path, figures: Path, device: str = "cpu") -> dict:
    counts = np.load(output / "image_cluster64_patch_winners.u16.npy", mmap_mode="r")
    index = pd.read_parquet(output / "inference_image_index.parquet")
    if len(counts) != len(index) or counts.shape[1] != 64:
        raise ValueError("Image index and patch winner counts do not match")
    figures.mkdir(parents=True, exist_ok=True)
    data = output / "results"
    data.mkdir(exist_ok=True)
    cities, sizes, matrices = cooccurrence(counts, index.city.to_numpy())
    ranked = city_pairs(cities, sizes, matrices)
    ranked.to_csv(data / "city_specific_sae_pairs.csv", index=False)
    best = ranked.groupby("city", sort=False).head(1).sort_values(
        "log2_city_vs_rest_rate", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f"{r.city.split('/')[-1]}: F{r.cluster_a:02d}+F{r.cluster_b:02d}"
              for r in best.itertuples(index=False)]
    ax.barh(labels, best.log2_city_vs_rest_rate, color="#247d76")
    ax.set(xlabel="log2(city joint rate / other-city joint rate)",
           title="City-enriched SAE cluster pairs (at least four winning patches each)")
    save(fig, figures, "Fig_City_Specific_Atypical_Combinations")
    global_ranked = global_pairs(matrices, len(index))
    global_ranked.to_csv(data / "global_strict_sae_pairs.csv", index=False)
    examples = example_rows(global_ranked, counts, index)
    examples.to_csv(data / "pair_example_panoramas.csv", index=False)
    draw_examples(output, figures, examples, index, device)
    payload = json.loads((output / "figure_inventory.json").read_text())
    generated = ["Fig_City_Specific_Atypical_Combinations", "Fig_Image_Level_Atypical_Coactivation"]
    for row in payload["figures"]:
        if row["legacy_figure"] in generated:
            row.update(new_figure=row["legacy_figure"], status="recomputed_from_SAE",
                       reason="SAE analogue: >=4 winning patches per image; rate-ratio or PPMI replaces MAE E+D+P distance weighting")
    payload["generated"] = sorted(set(payload.get("generated", [])) | set(generated))
    payload["recomputed_count"] = sum(row["status"] == "recomputed_from_SAE" for row in payload["figures"])
    write_json(output / "figure_inventory.json", payload)
    return {"generated": generated, "recomputed": payload["recomputed_count"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--figures", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print(json.dumps(run(args.output, args.figures or args.output / "figures", args.device), indent=2))


if __name__ == "__main__":
    main()
