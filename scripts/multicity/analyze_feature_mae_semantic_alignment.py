#!/usr/bin/env python3
"""Profile every Feature-MAE dimension using its own highest activations.

For each of the 512 dense Feature-MAE dimensions, the analysis first finds the
50 panorama locations with the strongest image-level activation summary.  It
then reads that dimension's full 14x56 maps in those panoramas and retains its
1,000 strongest patches.  Their fractional Mapillary-65 labels define the
dimension's semantic profile.  No per-patch winner/argmax assignment is used.
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (must precede numpy / sklearn)

import argparse
import colorsys
import io
import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SEMANTIC_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "mask2former_swin_l_mapillary_n2000_per_city"
)
SEMANTIC_PAPER = Path(
    "/workplace/urban_visual_gene/paper/data/semantic_alignment/"
    "mask2former_swin_l_mapillary_n2000_per_city"
)
FEATURE_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "rectangular_panorama_frozen_mae_n30/predictions"
)
OUTPUT_DATA = Path(
    "/workplace/urban_visual_gene/paper/data/semantic_alignment/"
    "feature_mae_mapillary_alignment_n60000"
)
OUTPUT_FIGURES = Path(
    "/workplace/urban_visual_gene/paper/figures/supplementary/"
    "feature_mae_semantic_alignment_20"
)
HIERARCHY_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "feature_mae_n30x12800_qc/mae/hierarchy_edp_32_64"
)
ROWS = 14
COLS = 56
DIMENSIONS = 512
CLASSES = 65
HEADINGS = (0, 90, 180, 270)


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


def city_slug(city_key: str) -> str:
    return city_key.replace("/", "__")


class TarReader:
    def __init__(self, max_open_files: int = 8) -> None:
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


def palette(count: int) -> np.ndarray:
    values = np.zeros((count, 3), dtype=np.uint8)
    for index in range(count):
        hue = (index * 0.618033988749895) % 1.0
        saturation = 0.58 + 0.32 * ((index % 3) / 2)
        value = 0.82 + 0.16 * (index % 2)
        values[index] = np.asarray(colorsys.hsv_to_rgb(hue, saturation, value)) * 255
    return values


def load_f64_mapping(hierarchy_root: Path) -> pd.DataFrame:
    with np.load(hierarchy_root / "hierarchy_arrays.npz") as arrays:
        fine_ids = arrays["fine_labels"].astype(np.int64)
    if fine_ids.shape != (DIMENSIONS,) or sorted(np.unique(fine_ids).tolist()) != list(range(64)):
        raise ValueError("frozen hierarchy must map 512 dimensions onto F000--F063")
    labels = pd.read_csv(hierarchy_root / "semantic_labels_qwen" / "category_labels.csv")
    labels = labels[labels["level"].eq("fine")].copy()
    labels["fine_id"] = labels["category_id"].str.replace("^F", "", regex=True).astype(int)
    labels = labels.set_index("fine_id").reindex(range(64))
    if labels["name_en"].isna().any():
        raise ValueError("semantic labels are missing one or more F000--F063 categories")
    return pd.DataFrame(
        {
            "dimension_id": np.arange(DIMENSIONS),
            "dimension": [f"D{x:03d}" for x in range(DIMENSIONS)],
            "fine_id": fine_ids,
            "fine_category": [f"F{x:03d}" for x in fine_ids],
            "fine_label_en": labels.loc[fine_ids, "name_en"].to_numpy(),
            "fine_label_zh": labels.loc[fine_ids, "name_zh"].to_numpy(),
            "fine_semantic_type": labels.loc[fine_ids, "semantic_type"].to_numpy(),
            "fine_label_confidence": labels.loc[fine_ids, "confidence"].to_numpy(),
        }
    )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    normalized = probabilities / np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-12)
    return -np.where(
        normalized > 0, normalized * np.log(normalized + 1e-20), 0
    ).sum(axis=-1)


def js_divergence(profiles: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    baseline = baseline / baseline.sum()
    mixture = 0.5 * (profiles + baseline[None, :])
    first = np.where(
        profiles > 0, profiles * np.log2((profiles + 1e-20) / (mixture + 1e-20)), 0
    ).sum(axis=1)
    second = np.where(
        baseline[None, :] > 0,
        baseline[None, :] * np.log2((baseline[None, :] + 1e-20) / (mixture + 1e-20)),
        0,
    ).sum(axis=1)
    return 0.5 * (first + second)


def load_aligned_index(
    panos: pd.DataFrame,
    semantic: np.ndarray,
    feature_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load valid panorama rows, 512D image summaries, and semantic shares."""
    records = []
    activation_blocks = []
    semantic_blocks = []
    for number, (city, frame) in enumerate(panos.groupby("city_key", sort=True), 1):
        root = feature_root / city_slug(city) / "full"
        status = np.load(root / "status.u8.npy", mmap_mode="r")
        activation = np.load(root / "activation_top20.f16.npy", mmap_mode="r")
        frame = frame.sort_values("panorama_sample_index").copy()
        pano_indices = frame["pano_index"].to_numpy(np.int64)
        valid_mask = np.asarray(status[pano_indices]) == 1
        valid = frame.loc[valid_mask].reset_index(drop=True)
        source = valid["pano_index"].to_numpy(np.int64)
        semantic_indices = valid["panorama_sample_index"].to_numpy(np.int64)
        activation_blocks.append(np.asarray(activation[source], dtype=np.float32))
        shares = np.empty((len(valid), CLASSES), dtype=np.float32)
        for start in range(0, len(valid), 128):
            indices = semantic_indices[start:start + 128]
            shares[start:start + len(indices)] = np.asarray(
                semantic[indices], dtype=np.float32
            ).mean(axis=(1, 2), dtype=np.float32)
        semantic_blocks.append(shares)
        records.append(valid)
        print(
            f"[{number:02d}/30] index {city}: {len(valid):,} valid, "
            f"{int((~valid_mask).sum())} excluded",
            flush=True,
        )
    valid = pd.concat(records, ignore_index=True)
    activations = np.concatenate(activation_blocks, axis=0)
    semantics = np.concatenate(semantic_blocks, axis=0)
    valid["descriptor_row"] = np.arange(len(valid), dtype=np.int32)
    if activations.shape != (len(valid), DIMENSIONS) or semantics.shape != (len(valid), CLASSES):
        raise ValueError("aligned descriptor shape mismatch")
    return valid, activations, semantics


def dimension_high_activation_profiles(
    valid: pd.DataFrame,
    activations: np.ndarray,
    semantic: np.ndarray,
    feature_root: Path,
    top_images: int,
    top_patches: int,
    class_names: np.ndarray,
    baseline: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute each dimension's semantics from its own strongest patches."""
    if top_patches > top_images * ROWS * COLS:
        raise ValueError("top_patches exceeds available patches in top images")
    split = len(activations) - top_images
    top_image_rows = np.argpartition(activations, split, axis=0)[split:]
    profiles = np.zeros((DIMENSIONS, CLASSES), dtype=np.float64)
    thresholds = np.zeros(DIMENSIONS, dtype=np.float32)
    means = np.zeros(DIMENSIONS, dtype=np.float32)
    row_counts = np.zeros((DIMENSIONS, ROWS), dtype=np.uint32)
    column_counts = np.zeros((DIMENSIONS, COLS), dtype=np.uint32)
    city_counts = np.zeros(DIMENSIONS, dtype=np.uint16)
    image_counts = np.zeros(DIMENSIONS, dtype=np.uint16)
    exemplars = []

    for dimension in range(DIMENSIONS):
        candidate_rows = top_image_rows[:, dimension]
        candidate_meta = valid.iloc[candidate_rows].copy()
        values_parts = []
        semantic_parts = []
        descriptor_parts = []
        row_parts = []
        column_parts = []
        for city, group in candidate_meta.groupby("city_key", sort=False):
            array = np.load(
                feature_root / city_slug(city) / "full" / "panorama_feature_mae_latent.f16.npy",
                mmap_mode="r",
            )
            pano_indices = group["pano_index"].to_numpy(np.int64)
            semantic_indices = group["panorama_sample_index"].to_numpy(np.int64)
            dimension_values = np.asarray(
                array[pano_indices, :, :, dimension], dtype=np.float32
            )
            semantic_values = np.asarray(semantic[semantic_indices], dtype=np.float32)
            values_parts.append(dimension_values.reshape(-1))
            semantic_parts.append(semantic_values.reshape(-1, CLASSES))
            descriptor_parts.append(
                np.repeat(group["descriptor_row"].to_numpy(np.int32), ROWS * COLS)
            )
            row_parts.append(np.tile(np.repeat(np.arange(ROWS), COLS), len(group)))
            column_parts.append(np.tile(np.arange(COLS), len(group) * ROWS))
        values = np.concatenate(values_parts)
        semantic_values = np.concatenate(semantic_parts)
        descriptor_rows = np.concatenate(descriptor_parts)
        patch_rows = np.concatenate(row_parts)
        patch_columns = np.concatenate(column_parts)
        top = np.argpartition(values, len(values) - top_patches)[-top_patches:]
        top = top[np.argsort(-values[top])]
        selected_semantics = semantic_values[top]
        profiles[dimension] = selected_semantics.mean(axis=0, dtype=np.float64)
        thresholds[dimension] = float(values[top[-1]])
        means[dimension] = float(values[top].mean())
        row_counts[dimension] = np.bincount(patch_rows[top], minlength=ROWS)
        column_counts[dimension] = np.bincount(patch_columns[top], minlength=COLS)
        chosen_descriptor_rows = descriptor_rows[top]
        image_counts[dimension] = len(np.unique(chosen_descriptor_rows))
        city_counts[dimension] = valid.iloc[np.unique(chosen_descriptor_rows)]["city_key"].nunique()
        for rank, patch_index in enumerate(top[:12], 1):
            record = valid.iloc[int(descriptor_rows[patch_index])]
            patch_semantic = semantic_values[patch_index]
            semantic_id = int(patch_semantic.argmax())
            exemplars.append(
                {
                    "dimension_id": dimension,
                    "dimension": f"D{dimension:03d}",
                    "rank": rank,
                    "activation": float(values[patch_index]),
                    "descriptor_row": int(descriptor_rows[patch_index]),
                    "panorama_sample_index": int(record.panorama_sample_index),
                    "city_key": record.city_key,
                    "panoid": record.panoid,
                    "patch_row": int(patch_rows[patch_index]),
                    "patch_column": int(patch_columns[patch_index]),
                    "dominant_semantic_id": semantic_id,
                    "dominant_semantic": class_names[semantic_id],
                    "dominant_semantic_fraction": float(patch_semantic[semantic_id]),
                }
            )
        if (dimension + 1) % 32 == 0 or dimension + 1 == DIMENSIONS:
            print(f"profiled dimensions {dimension + 1}/{DIMENSIONS}", flush=True)

    ordered = np.argsort(-profiles, axis=1)
    top1, top2, top3 = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    top1_share = profiles[np.arange(DIMENSIONS), top1]
    top2_share = profiles[np.arange(DIMENSIONS), top2]
    top3_individual = profiles[np.arange(DIMENSIONS), top3]
    top3_share = np.take_along_axis(profiles, ordered[:, :3], axis=1).sum(axis=1)
    specificity = 1 - entropy(profiles) / math.log(CLASSES)
    divergence = js_divergence(profiles, baseline)
    lift = top1_share / np.maximum(baseline[top1], 1e-12)
    row_prob = row_counts / top_patches
    column_prob = column_counts / top_patches
    row_entropy = entropy(row_prob) / math.log(ROWS)
    column_entropy = entropy(column_prob) / math.log(COLS)
    seam_columns = np.asarray([0, 13, 14, 27, 28, 41, 42, 55])
    seam_share = column_prob[:, seam_columns].sum(axis=1)
    seam_lift = seam_share / (len(seam_columns) / COLS)

    clear = (top1_share >= 0.55) & (lift >= 1.5)
    coherent = ~clear & (top3_share >= 0.75) & (divergence >= 0.04)
    artifact = (
        ~clear
        & ~coherent
        & (top1_share < 0.35)
        & ((seam_lift >= 1.75) | (np.minimum(row_entropy, column_entropy) < 0.55))
    )
    assessment = np.full(DIMENSIONS, "low_mapillary_specificity", dtype=object)
    assessment[coherent] = "coherent_semantic_mixture"
    assessment[clear] = "clear_single_semantic"
    assessment[artifact] = "spatial_artifact_candidate"
    metrics = pd.DataFrame(
        {
            "dimension_id": np.arange(DIMENSIONS),
            "dimension": [f"D{x:03d}" for x in range(DIMENSIONS)],
            "profile_top_images": top_images,
            "profile_top_patches": top_patches,
            "high_activation_threshold": thresholds,
            "mean_selected_activation": means,
            "selected_image_count": image_counts,
            "selected_city_count": city_counts,
            "top1_class_id": top1,
            "top1_class": class_names[top1],
            "top1_share": top1_share,
            "top1_lift_vs_global": lift,
            "top2_class_id": top2,
            "top2_class": class_names[top2],
            "top2_share": top2_share,
            "top3_class_id": top3,
            "top3_class": class_names[top3],
            "top3_individual_share": top3_individual,
            "top3_cumulative_share": top3_share,
            "semantic_specificity": specificity,
            "js_divergence_vs_global": divergence,
            "row_entropy_normalized": row_entropy,
            "column_entropy_normalized": column_entropy,
            "row_peak_lift": row_prob.max(axis=1) * ROWS,
            "column_peak_lift": column_prob.max(axis=1) * COLS,
            "seam_share": seam_share,
            "seam_lift": seam_lift,
            "assessment": assessment,
        }
    )
    distribution = pd.DataFrame(
        {
            "dimension_id": np.repeat(np.arange(DIMENSIONS), CLASSES),
            "dimension": np.repeat([f"D{x:03d}" for x in range(DIMENSIONS)], CLASSES),
            "class_id": np.tile(np.arange(CLASSES), DIMENSIONS),
            "class_name": np.tile(class_names, DIMENSIONS),
            "semantic_fraction": profiles.reshape(-1),
            "global_semantic_fraction": np.tile(baseline, DIMENSIONS),
        }
    )
    return metrics, distribution, pd.DataFrame(exemplars)


def select_diverse_points(
    activations: np.ndarray,
    semantics: np.ndarray,
    valid: pd.DataFrame,
    count: int,
    seed: int,
) -> pd.DataFrame:
    activation_z = StandardScaler().fit_transform(activations)
    semantic_z = StandardScaler().fit_transform(semantics)
    activation_compact = PCA(
        n_components=24, whiten=True, random_state=seed, svd_solver="randomized"
    ).fit_transform(activation_z)
    semantic_compact = PCA(
        n_components=12, whiten=True, random_state=seed, svd_solver="randomized"
    ).fit_transform(semantic_z)
    descriptor = np.concatenate(
        [activation_compact / math.sqrt(24), semantic_compact / math.sqrt(12)], axis=1
    ).astype(np.float32)
    model = MiniBatchKMeans(
        n_clusters=count, batch_size=4096, n_init=20, max_iter=300, random_state=seed
    ).fit(descriptor)
    distances = model.transform(descriptor)
    assignments = model.labels_
    sizes = np.bincount(assignments, minlength=count)
    selected = []
    used_cities = set()
    for cluster in np.argsort(-sizes):
        candidates = np.flatnonzero(assignments == cluster)
        candidates = candidates[np.argsort(distances[candidates, cluster])]
        chosen = next(
            (int(index) for index in candidates if valid.iloc[index].city_key not in used_cities),
            int(candidates[0]),
        )
        row = valid.iloc[chosen].copy()
        row["selection_cluster"] = int(cluster)
        row["cluster_size"] = int(sizes[cluster])
        row["distance_to_cluster_centroid"] = float(distances[chosen, cluster])
        selected.append(row)
        used_cities.add(row.city_key)
    result = pd.DataFrame(selected).reset_index(drop=True)
    result.insert(0, "sample_id", [f"P{x:02d}" for x in range(1, count + 1)])
    return result


def read_panorama(rows: pd.DataFrame, reader: TarReader) -> Image.Image:
    rows = rows.sort_values("heading")
    if tuple(rows["heading"].astype(int)) != HEADINGS:
        raise ValueError("panorama does not have 0/90/180/270 headings")
    images = [
        reader.image(str(row.tar_path), int(row.jpg_offset), int(row.jpg_size))
        for row in rows.itertuples(index=False)
    ]
    if len({image.size for image in images}) != 1:
        raise ValueError("direction images have inconsistent sizes")
    width, height = images[0].size
    panorama = Image.new("RGB", (width * 4, height))
    for index, image in enumerate(images):
        panorama.paste(image, (index * width, 0))
    return panorama


def read_semantic_mask(rows: pd.DataFrame, mask_root: Path) -> Image.Image:
    images = []
    for row in rows.sort_values("heading").itertuples(index=False):
        panoid = str(row.panoid).replace("/", "_")
        filename = f"{int(row.sample_index):07d}__{panoid}__h{int(row.heading):03d}.png"
        path = mask_root / city_slug(str(row.city_key)) / filename
        with Image.open(path) as image:
            images.append(image.convert("L"))
    width, height = images[0].size
    panorama = Image.new("L", (width * 4, height))
    for index, image in enumerate(images):
        panorama.paste(image, (index * width, 0))
    return panorama


def semantic_overlay(image: Image.Image, mask: Image.Image, colours: np.ndarray) -> Image.Image:
    colour = Image.fromarray(colours[np.asarray(mask, dtype=np.uint8)], "RGB")
    return Image.blend(image.convert("RGB"), colour, 0.46)


def activation_overlay(image: Image.Image, values: np.ndarray, threshold: float) -> Image.Image:
    median = float(np.median(values))
    scale = max(threshold - median, 1e-6)
    intensity = np.clip((values - median) / scale, 0, 1)
    heat_rgb = plt.get_cmap("inferno")(intensity)[..., :3]
    heat = Image.fromarray(np.uint8(heat_rgb * 255), "RGB").resize(
        image.size, Image.Resampling.NEAREST
    )
    alpha = Image.fromarray(np.uint8((0.12 + 0.68 * intensity) * 255), "L").resize(
        image.size, Image.Resampling.NEAREST
    )
    return Image.composite(heat, image.convert("RGB"), alpha)


def render_cases(
    selected: pd.DataFrame,
    manifest: pd.DataFrame,
    activations: np.ndarray,
    semantic_root: Path,
    feature_root: Path,
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    semantic_colours = palette(CLASSES)
    profiles = distribution.pivot(
        index="dimension_id", columns="class_id", values="semantic_fraction"
    ).to_numpy()
    class_names = (
        distribution.drop_duplicates("class_id").sort_values("class_id")["class_name"].to_numpy()
    )
    reader = TarReader()
    detail_rows = []
    atlas_records = []
    try:
        for _, sample in selected.iterrows():
            rows = manifest[manifest["panorama_sample_index"].eq(sample.panorama_sample_index)]
            original = read_panorama(rows, reader)
            mask = read_semantic_mask(rows, semantic_root / "masks")
            semantic = semantic_overlay(original, mask, semantic_colours)
            city_root = feature_root / city_slug(str(sample.city_key)) / "full"
            latent = np.load(
                city_root / "panorama_feature_mae_latent.f16.npy", mmap_mode="r"
            )
            feature = np.asarray(latent[int(sample.pano_index)], dtype=np.float32)
            descriptor_row = int(sample.descriptor_row)
            top_dimensions = np.argsort(-activations[descriptor_row])[:6]
            heatmaps = [
                activation_overlay(
                    original,
                    feature[:, :, dimension],
                    float(metrics.iloc[int(dimension)].high_activation_threshold),
                )
                for dimension in top_dimensions
            ]

            figure = plt.figure(figsize=(18, 12), constrained_layout=True)
            grid = figure.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])
            for row_index, (panel, title) in enumerate(
                [(original, "Original panorama"), (semantic, "Mask2Former Mapillary-65 overlay")]
            ):
                axis = figure.add_subplot(grid[row_index, :])
                axis.imshow(panel)
                axis.set_title(title, fontsize=10, loc="left")
                axis.axis("off")
            for index, (dimension, heatmap) in enumerate(zip(top_dimensions, heatmaps)):
                axis = figure.add_subplot(grid[2 + index // 3, index % 3])
                axis.imshow(heatmap)
                profile = profiles[dimension]
                order = np.argsort(-profile)[:3]
                semantic_text = " · ".join(
                    f"{class_names[x]} {profile[x]:.0%}" for x in order
                )
                metric = metrics.iloc[int(dimension)]
                axis.set_title(
                    f"D{dimension:03d} · {metric.fine_category} {metric.fine_label_en}"
                    f"\n{semantic_text}",
                    fontsize=8,
                )
                axis.axis("off")
                detail_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "rank": index + 1,
                        "dimension_id": int(dimension),
                        "fine_category": metric.fine_category,
                        "fine_label_en": metric.fine_label_en,
                        "fine_label_zh": metric.fine_label_zh,
                        "image_activation_top20": float(activations[descriptor_row, dimension]),
                        "global_top_semantic": metric.top1_class,
                        "global_top_semantic_share": float(metric.top1_share),
                        "global_top3_semantic_share": float(metric.top3_cumulative_share),
                        "assessment": metric.assessment,
                    }
                )
            figure.suptitle(
                f"{sample.sample_id} · {str(sample.city_key).replace('/', ' · ')} · {sample.panoid}",
                fontsize=13, fontweight="bold",
            )
            figure.savefig(
                output_root / f"{sample.sample_id}_semantic_feature_alignment.jpg",
                dpi=140, bbox_inches="tight", format="jpg",
                pil_kwargs={"quality": 90, "optimize": True},
            )
            plt.close(figure)
            atlas_records.append(
                {
                    "sample_id": sample.sample_id,
                    "city": sample.city_key,
                    "original": original.resize((640, 160), Image.Resampling.LANCZOS),
                    "semantic": semantic.resize((640, 160), Image.Resampling.LANCZOS),
                    "activation": heatmaps[0].resize((640, 160), Image.Resampling.LANCZOS),
                    "dimension": int(top_dimensions[0]),
                }
            )
    finally:
        reader.close()

    atlas = plt.figure(figsize=(18, 42), constrained_layout=True)
    grid = atlas.add_gridspec(len(atlas_records), 4, width_ratios=[1, 1, 1, 0.3])
    for row_index, record in enumerate(atlas_records):
        for column, key in enumerate(("original", "semantic", "activation")):
            axis = atlas.add_subplot(grid[row_index, column])
            axis.imshow(record[key])
            if row_index == 0:
                axis.set_title(
                    ["Original", "Semantic segmentation", "Strongest image dimension"][column],
                    fontsize=9,
                )
            axis.axis("off")
        axis = atlas.add_subplot(grid[row_index, 3])
        axis.text(
            0, 0.5,
            f"{record['sample_id']}\n{record['city']}\nD{record['dimension']:03d}",
            va="center", fontsize=7,
        )
        axis.axis("off")
    atlas.suptitle(
        "20 diverse panorama points: semantics and per-dimension Feature-MAE activations",
        fontsize=15, fontweight="bold",
    )
    atlas.savefig(
        output_root / "Fig_FeatureMAE_Semantic_Alignment_20_Atlas.jpg",
        dpi=120, format="jpg", pil_kwargs={"quality": 90, "optimize": True},
    )
    plt.close(atlas)
    return pd.DataFrame(detail_rows)


def plot_summary(metrics: pd.DataFrame, output: Path) -> None:
    colours = {
        "clear_single_semantic": "#2a9d8f",
        "coherent_semantic_mixture": "#457b9d",
        "low_mapillary_specificity": "#adb5bd",
        "spatial_artifact_candidate": "#e76f51",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for label, group in metrics.groupby("assessment"):
        axes[0].scatter(
            group.top1_share, group.top3_cumulative_share,
            s=14, alpha=0.78, color=colours[label], label=f"{label} (n={len(group)})",
        )
        axes[1].scatter(
            group.seam_lift, group.js_divergence_vs_global,
            s=14, alpha=0.78, color=colours[label], label=label,
        )
    axes[0].set(
        xlabel="Top semantic class share among high-activation patches",
        ylabel="Top-three semantic share",
        title="Semantic concentration of all 512 dimensions",
    )
    axes[1].axvline(1, color="#555", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Cardinal-boundary seam lift among high activations",
        ylabel="Jensen–Shannon divergence from dataset semantics",
        title="Semantic distinctiveness versus seam preference",
    )
    axes[0].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_all_dimension_heatmap(
    metrics: pd.DataFrame, distribution: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = distribution.pivot(
        index="dimension_id", columns="class_id", values="semantic_fraction"
    ).to_numpy()
    class_names = (
        distribution.drop_duplicates("class_id").sort_values("class_id").class_name.to_numpy()
    )
    # Hellinger geometry is appropriate for probability compositions and does
    # not let a few large semantic fractions dominate Euclidean distances.
    row_distance = pdist(np.sqrt(np.clip(profiles, 0, 1)), metric="euclidean") / math.sqrt(2)
    row_linkage = linkage(row_distance, method="average")
    row_linkage = optimal_leaf_ordering(row_linkage, row_distance)
    cluster_labels = fcluster(row_linkage, t=16, criterion="maxclust").astype(np.int64)

    # Put related Mapillary concepts next to one another instead of relying on
    # raw class IDs.  Group names and boundaries are shown above the heatmap.
    semantic_groups = [
        ("Nature", [27, 30, 29, 25, 26, 28, 31, 0, 1]),
        ("Ground / road", [13, 15, 7, 8, 9, 10, 11, 12, 14, 23, 24]),
        ("Built structure", [17, 2, 3, 4, 5, 6, 16, 18]),
        ("People / riders", [19, 20, 21, 22]),
        ("Street objects", list(range(32, 52))),
        ("Vehicles", list(range(52, 65))),
    ]
    grouped_class_ids = np.asarray(
        [class_id for _, class_ids in semantic_groups for class_id in class_ids], dtype=np.int64
    )
    if sorted(grouped_class_ids.tolist()) != list(range(CLASSES)):
        raise ValueError("semantic display groups must cover class IDs 0..64 exactly once")
    class_to_group = {
        class_id: group_index
        for group_index, (_, class_ids) in enumerate(semantic_groups)
        for class_id in class_ids
    }

    # Cluster semantic classes by the dimensions in which they co-occur. Each
    # class column is normalized over dimensions before applying Hellinger
    # distance, so rare classes can still form meaningful branches.
    column_profiles = profiles.T
    column_profiles = column_profiles / np.maximum(
        column_profiles.sum(axis=1, keepdims=True), 1e-12
    )
    column_distance = pdist(
        np.sqrt(np.clip(column_profiles, 0, 1)), metric="euclidean"
    ) / math.sqrt(2)
    column_linkage = linkage(column_distance, method="average")
    column_linkage = optimal_leaf_ordering(column_linkage, column_distance)
    column_clusters = fcluster(column_linkage, t=10, criterion="maxclust").astype(np.int64)
    assessment_names = [
        "clear_single_semantic",
        "coherent_semantic_mixture",
        "low_mapillary_specificity",
        "spatial_artifact_candidate",
    ]
    assessment_colours = np.asarray(
        [[42, 157, 143], [69, 123, 157], [173, 181, 189], [231, 111, 81]],
        dtype=np.uint8,
    )
    semantic_group_colours = np.asarray(
        [[65, 182, 196], [244, 162, 97], [138, 117, 99], [231, 111, 81], [144, 190, 109], [87, 117, 144]],
        dtype=np.uint8,
    )

    fine_ids_by_dimension = metrics.set_index("dimension_id").loc[
        np.arange(DIMENSIONS), "fine_id"
    ].to_numpy(np.int64)
    fine_colours = palette(64)

    figure = plt.figure(figsize=(22, 22), constrained_layout=True)
    grid = figure.add_gridspec(
        3, 4,
        width_ratios=[2.8, 0.18, 0.18, 14],
        height_ratios=[2.7, 0.22, 15],
        wspace=0.02,
        hspace=0.02,
    )
    column_axis = figure.add_subplot(grid[0, 3])
    column_tree = dendrogram(
        column_linkage,
        ax=column_axis,
        orientation="top",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#52606d",
        link_color_func=lambda _: "#52606d",
    )
    column_order = np.asarray(column_tree["leaves"], dtype=np.int64)
    column_axis.set_xticks([])
    column_axis.set_yticks([])
    column_axis.spines[:].set_visible(False)
    column_axis.set_title("Semantic-class hierarchy", fontsize=9)

    row_axis = figure.add_subplot(grid[2, 0])
    row_tree = dendrogram(
        row_linkage,
        ax=row_axis,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#52606d",
        link_color_func=lambda _: "#52606d",
    )
    ordered = np.asarray(row_tree["leaves"], dtype=np.int64)
    row_axis.invert_yaxis()
    row_axis.set_xticks([])
    row_axis.set_yticks([])
    row_axis.spines[:].set_visible(False)
    row_axis.set_ylabel("Dimension hierarchy", fontsize=9)

    assessments = metrics.set_index("dimension_id").loc[ordered, "assessment"].to_numpy()
    assessment_ids = np.asarray(
        [assessment_names.index(value) for value in assessments], dtype=np.int64
    )
    strip = assessment_colours[assessment_ids][:, None, :]
    strip_axis = figure.add_subplot(grid[2, 1])
    strip_axis.imshow(strip, aspect="auto")
    strip_axis.set_xticks([])
    strip_axis.set_yticks([])
    strip_axis.set_title("type", fontsize=8)
    fine_strip_axis = figure.add_subplot(grid[2, 2])
    fine_strip_axis.imshow(
        fine_colours[fine_ids_by_dimension[ordered]][:, None, :], aspect="auto"
    )
    fine_strip_axis.set_xticks([])
    fine_strip_axis.set_yticks([])
    fine_strip_axis.set_title("F64", fontsize=8)

    semantic_strip_axis = figure.add_subplot(grid[1, 3])
    semantic_group_ids = np.asarray(
        [class_to_group[int(class_id)] for class_id in column_order], dtype=np.int64
    )
    semantic_strip_axis.imshow(
        semantic_group_colours[semantic_group_ids][None, :, :], aspect="auto"
    )
    semantic_strip_axis.set_xticks([])
    semantic_strip_axis.set_yticks([])
    semantic_strip_axis.set_ylabel("role", fontsize=7, rotation=0, labelpad=16)

    axis = figure.add_subplot(grid[2, 3])
    image = axis.imshow(
        profiles[np.ix_(ordered, column_order)], aspect="auto", interpolation="nearest",
        cmap="magma", vmin=0, vmax=1,
    )
    axis.set_xticks(
        np.arange(CLASSES), class_names[column_order], rotation=90, fontsize=6
    )
    tick_positions = np.arange(0, DIMENSIONS, 8)
    axis.set_yticks(
        tick_positions,
        [
            f"D{ordered[x]:03d} · F{fine_ids_by_dimension[ordered[x]]:03d}"
            for x in tick_positions
        ],
        fontsize=4.7,
    )
    axis.set_xlabel("Mapillary Vistas semantic classes, hierarchically clustered")
    axis.set_ylabel("Feature-MAE dimensions, clustered by complete 65D semantic profile")
    axis.set_title(
        "512D × 65-class hierarchical clustered heatmap",
        fontsize=13,
    )
    colourbar = figure.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    colourbar.set_label("Semantic fraction")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=colour / 255.0, label=name)
        for colour, name in zip(assessment_colours, assessment_names)
    ]
    role_handles = [
        plt.Line2D(
            [0], [0], marker="s", linestyle="", color=semantic_group_colours[index] / 255.0,
            label=name,
        )
        for index, (name, _) in enumerate(semantic_groups)
    ]
    first_legend = axis.legend(
        handles=handles, loc="upper right", fontsize=7, frameon=True, title="Dimension type"
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=role_handles, loc="lower right", fontsize=7, frameon=True,
        title="Semantic role strip",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    dimension_order = pd.DataFrame(
        {
            "heatmap_row": np.arange(DIMENSIONS),
            "dimension_id": ordered,
            "dimension": [f"D{x:03d}" for x in ordered],
            "semantic_cluster_16": cluster_labels[ordered],
            "fine_id": fine_ids_by_dimension[ordered],
            "fine_category": [f"F{x:03d}" for x in fine_ids_by_dimension[ordered]],
            "fine_label_en": metrics.set_index("dimension_id").loc[
                ordered, "fine_label_en"
            ].to_numpy(),
            "fine_label_zh": metrics.set_index("dimension_id").loc[
                ordered, "fine_label_zh"
            ].to_numpy(),
            "top1_class": metrics.set_index("dimension_id").loc[ordered, "top1_class"].to_numpy(),
            "assessment": assessments,
        }
    )
    semantic_order = pd.DataFrame(
        {
            "heatmap_column": np.arange(CLASSES),
            "class_id": column_order,
            "class_name": class_names[column_order],
            "semantic_cluster_10": column_clusters[column_order],
            "scene_role": [semantic_groups[class_to_group[int(x)]][0] for x in column_order],
        }
    )
    return dimension_order, semantic_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-root", type=Path, default=SEMANTIC_ROOT)
    parser.add_argument("--semantic-paper", type=Path, default=SEMANTIC_PAPER)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--hierarchy-root", type=Path, default=HIERARCHY_ROOT)
    parser.add_argument("--output-data", type=Path, default=OUTPUT_DATA)
    parser.add_argument("--output-figures", type=Path, default=OUTPUT_FIGURES)
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--top-images-per-dimension", type=int, default=50)
    parser.add_argument("--top-patches-per-dimension", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_safe_affinity()
    args.output_data.mkdir(parents=True, exist_ok=True)
    args.output_figures.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.semantic_paper / "sample_manifest.csv")
    sizes = manifest.groupby("panorama_sample_index").size()
    if not (sizes == 4).all():
        raise ValueError("sample manifest is not four directions per panorama")
    panos = (
        manifest.sort_values(["panorama_sample_index", "heading"])
        .groupby("panorama_sample_index", as_index=False)
        .first()
    )
    semantic = np.load(
        args.semantic_root / "panorama_semantic_fractions_14x56x65.f16.npy",
        mmap_mode="r",
    )
    if semantic.shape != (len(panos), ROWS, COLS, CLASSES):
        raise ValueError(f"unexpected semantic shape {semantic.shape}")
    classes = pd.read_csv(args.semantic_paper / "mapillary_vistas_class_index.csv")
    class_names = classes.sort_values("class_id").class_name.to_numpy()
    prevalence = pd.read_csv(args.semantic_paper / "class_prevalence_65.csv")
    baseline = (
        prevalence.set_index("class_id").reindex(range(CLASSES)).pixel_share.to_numpy(np.float64)
    )

    valid, activations, image_semantics = load_aligned_index(
        panos, semantic, args.feature_root
    )
    metrics, distribution, exemplars = dimension_high_activation_profiles(
        valid,
        activations,
        semantic,
        args.feature_root,
        args.top_images_per_dimension,
        args.top_patches_per_dimension,
        class_names,
        baseline,
    )
    f64_mapping = load_f64_mapping(args.hierarchy_root)
    metrics = metrics.merge(
        f64_mapping.drop(columns="dimension"), on="dimension_id", how="left", validate="one_to_one"
    ).sort_values("dimension_id").reset_index(drop=True)
    if metrics["fine_id"].isna().any():
        raise ValueError("one or more dimensions are missing the frozen F64 mapping")
    metrics.to_csv(args.output_data / "dimension_semantic_profiles.csv", index=False)
    f64_mapping.to_csv(args.output_data / "dimension_f64_mapping.csv", index=False)
    distribution.to_csv(args.output_data / "dimension_semantic_distribution_65.csv", index=False)
    exemplars.to_csv(args.output_data / "dimension_top_patch_exemplars.csv", index=False)
    selected = select_diverse_points(
        activations, image_semantics, valid, args.points, args.seed
    )
    selected.to_csv(args.output_data / "selected_20_points.csv", index=False)
    details = render_cases(
        selected,
        manifest,
        activations,
        args.semantic_root,
        args.feature_root,
        metrics,
        distribution,
        args.output_figures,
    )
    details.to_csv(args.output_data / "selected_point_top_dimensions.csv", index=False)
    plot_summary(metrics, args.output_figures / "Fig_Dimension_Semantic_Clarity.png")
    dimension_order, semantic_order = plot_all_dimension_heatmap(
        metrics,
        distribution,
        args.output_figures / "Fig_512D_Semantic_Profile_Heatmap.png",
    )
    dimension_order.to_csv(args.output_data / "heatmap_dimension_order.csv", index=False)
    semantic_order.to_csv(args.output_data / "heatmap_semantic_order.csv", index=False)
    counts = metrics.assessment.value_counts().to_dict()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "per-dimension high-activation patches; no winner/argmax assignment",
        "semantic_model": "facebook/mask2former-swin-large-mapillary-vistas-semantic",
        "feature_representation": "frozen dense Feature-MAE 512D bottleneck",
        "f64_mapping": str(args.hierarchy_root / "hierarchy_arrays.npz"),
        "f64_semantic_labels": str(
            args.hierarchy_root / "semantic_labels_qwen" / "category_labels.csv"
        ),
        "sampled_panoramas": int(len(panos)),
        "valid_aligned_panoramas": int(len(valid)),
        "excluded_feature_mae_errors": int(len(panos) - len(valid)),
        "dimensions": DIMENSIONS,
        "semantic_classes": CLASSES,
        "top_images_per_dimension": args.top_images_per_dimension,
        "top_patches_per_dimension": args.top_patches_per_dimension,
        "visual_case_studies": int(len(selected)),
        "case_study_unique_cities": int(selected.city_key.nunique()),
        "assessment_counts": {str(key): int(value) for key, value in counts.items()},
        "assessment_thresholds": {
            "clear_single_semantic": "top1_share >= 0.55 and top1_lift_vs_global >= 1.5",
            "coherent_semantic_mixture": (
                "not clear; top3_cumulative_share >= 0.75 and "
                "js_divergence_vs_global >= 0.04"
            ),
            "spatial_artifact_candidate": (
                "not clear/coherent; top1_share < 0.35 and either seam_lift >= 1.75 "
                "or min(row_entropy, column_entropy) < 0.55"
            ),
        },
        "interpretation_warning": (
            "Low Mapillary-class specificity is not proof of artifact; dimensions may encode "
            "texture, material, lighting, geometry, or concepts absent from the taxonomy."
        ),
    }
    save_json(args.output_data / "alignment_summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
