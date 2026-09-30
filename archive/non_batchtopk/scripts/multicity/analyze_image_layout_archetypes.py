#!/usr/bin/env python3
"""Discover representative spatial-composition archetypes of street-view images.

The unit of analysis is one directional image.  Its 14x14 Feature-MAE winner
map is converted to 64 frozen E+D+P fine elements and pooled into a 3x3
canvas.  Every spatial cell contributes equally, so the representation is a
64-element distribution in each of nine locations rather than a city-level
aggregate.  Hellinger-transformed features are reduced with PCA and clustered
on a city-balanced, panorama-deduplicated sample.  The selected model is then
used to assign every image.

A composition-only 64-element representation is evaluated as a baseline.  It
can distinguish what is present, while the main 64x3x3 representation also
distinguishes where it appears on the image canvas.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import io
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from scripts.multicity.analyze_atypical_cooccurrence import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MAIN_HIERARCHY,
    DEFAULT_SOURCE_HIERARCHY,
    load_semantic_names,
    save_csv,
    save_json,
)


PATCH_GRID = 14
PATCHES = PATCH_GRID * PATCH_GRID
ELEMENTS = 64
DIMENSIONS = 512
REGION_EDGES = (0, 5, 9, 14)
REGIONS = 9
DEFAULT_METADATA = Path("results/image_dimension_pixel_proportions.parquet")
DEFAULT_RESULTS = Path("results/image_layout_archetypes")
DEFAULT_FIGURES = Path("figures/image_layout_archetypes")
DEFAULT_PAPER = Path("paper")


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def category_counts(labels: np.ndarray, categories: int) -> np.ndarray:
    rows = np.repeat(np.arange(len(labels), dtype=np.int64), labels.shape[1])
    keys = rows * categories + labels.ravel()
    return np.bincount(keys, minlength=len(labels) * categories).reshape(
        len(labels), categories
    )


def fine_counts(labels: np.ndarray) -> np.ndarray:
    return category_counts(labels, ELEMENTS)


def pooled_composition_features(
    top: np.ndarray, category_labels: np.ndarray, categories: int
) -> np.ndarray:
    labels = category_labels[np.asarray(top, dtype=np.int64)]
    counts = category_counts(labels, categories)
    return (counts / PATCHES).astype(np.float32)


def composition_features(
    top: np.ndarray, fine_labels: np.ndarray
) -> np.ndarray:
    output = pooled_composition_features(top, fine_labels, ELEMENTS)
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("element counts do not sum to 196")
    return output


def pooled_spatial_features(
    top: np.ndarray, category_labels: np.ndarray, categories: int
) -> np.ndarray:
    labels = category_labels[np.asarray(top, dtype=np.int64)].reshape(
        len(top), PATCH_GRID, PATCH_GRID
    )
    output = np.empty((len(top), REGIONS, categories), dtype=np.float32)
    region = 0
    for row in range(3):
        r0, r1 = REGION_EDGES[row : row + 2]
        for column in range(3):
            c0, c1 = REGION_EDGES[column : column + 2]
            cell = labels[:, r0:r1, c0:c1].reshape(len(top), -1)
            counts = category_counts(cell, categories)
            output[:, region, :] = counts / (cell.shape[1] * REGIONS)
            region += 1
    flat = output.reshape(len(top), REGIONS * categories)
    if not np.allclose(flat.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("spatial-composition features do not sum to one")
    return flat


def spatial_features(top: np.ndarray, fine_labels: np.ndarray) -> np.ndarray:
    """Return equal-cell 3x3 spatial element distributions, summing to one."""
    return pooled_spatial_features(top, fine_labels, ELEMENTS)


def balanced_sample_indices(
    metadata: pd.DataFrame, per_city: int, seed: int
) -> np.ndarray:
    """Sample one direction per panorama and an equal count from every city."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for city, group in metadata.groupby("city_key", sort=True):
        order = rng.permutation(group.index.to_numpy(np.int64))
        seen: set[str] = set()
        city_rows: list[int] = []
        panoids = metadata["panoid"].to_numpy()
        for index in order:
            panoid = str(panoids[index])
            if panoid in seen:
                continue
            seen.add(panoid)
            city_rows.append(int(index))
            if len(city_rows) == per_city:
                break
        if len(city_rows) < per_city:
            raise ValueError(
                f"{city} has only {len(city_rows)} unique panoramas; requested {per_city}"
            )
        chosen.extend(city_rows)
    return np.asarray(sorted(chosen), dtype=np.int64)


def fit_pca(
    features: np.ndarray, variance_target: float, seed: int, cap: int = 192
) -> tuple[PCA, np.ndarray, int]:
    transformed = np.sqrt(np.maximum(features, 0.0)).astype(np.float32)
    components = min(cap, transformed.shape[1] - 1, len(transformed) - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(transformed).astype(np.float32)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained = int(np.searchsorted(cumulative, variance_target) + 1)
    if retained > components:
        retained = components
    return pca, scores[:, :retained], retained


def pca_transform(features: np.ndarray, pca: PCA, retained: int) -> np.ndarray:
    values = np.sqrt(np.maximum(features, 0.0)).astype(np.float32)
    return ((values - pca.mean_) @ pca.components_[:retained].T).astype(np.float32)


def make_kmeans(k: int, seed: int, n_init: int = 8) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=2048,
        n_init=n_init,
        max_iter=300,
        max_no_improvement=30,
        reassignment_ratio=0.01,
    )


def k_sweep(
    scores: np.ndarray,
    k_values: list[int],
    seed: int,
    evaluation_size: int,
    representation: str,
) -> tuple[pd.DataFrame, dict[int, MiniBatchKMeans]]:
    rng = np.random.default_rng(seed + 991)
    evaluation = rng.choice(
        len(scores), size=min(evaluation_size, len(scores)), replace=False
    )
    evaluation_scores = scores[evaluation]
    rows = []
    models: dict[int, MiniBatchKMeans] = {}
    for k in k_values:
        model = make_kmeans(k, seed)
        labels = model.fit_predict(scores)
        models[k] = model
        eval_labels = labels[evaluation]
        counts = np.bincount(labels, minlength=k)
        alternate_aris = []
        for alternate_seed in (seed + 101, seed + 202):
            alternate = make_kmeans(k, alternate_seed, n_init=4).fit(scores)
            alternate_aris.append(
                adjusted_rand_score(eval_labels, alternate.predict(evaluation_scores))
            )
        rows.append(
            {
                "representation": representation,
                "k": k,
                "silhouette": silhouette_score(
                    evaluation_scores, eval_labels, metric="euclidean"
                ),
                "calinski_harabasz": calinski_harabasz_score(
                    evaluation_scores, eval_labels
                ),
                "davies_bouldin": davies_bouldin_score(
                    evaluation_scores, eval_labels
                ),
                "initialization_stability_ari": float(np.mean(alternate_aris)),
                "minimum_cluster_fraction": float(counts.min() / len(labels)),
                "maximum_cluster_fraction": float(counts.max() / len(labels)),
                "normalized_size_entropy": float(
                    -np.sum((counts / counts.sum()) * np.log(counts / counts.sum()))
                    / math.log(k)
                ),
                "inertia_per_image": float(model.inertia_ / len(scores)),
            }
        )
        print(f"{representation} K={k}: silhouette={rows[-1]['silhouette']:.4f}, "
              f"ARI={rows[-1]['initialization_stability_ari']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    frame["selection_score"] = (
        frame["silhouette"].rank(pct=True)
        + (-frame["davies_bouldin"]).rank(pct=True)
        + frame["initialization_stability_ari"].rank(pct=True)
        + frame["minimum_cluster_fraction"].rank(pct=True)
        + frame["normalized_size_entropy"].rank(pct=True)
    ) / 5.0
    return frame, models


def choose_k(metrics: pd.DataFrame) -> int:
    eligible = metrics[metrics["minimum_cluster_fraction"] >= 0.01]
    if eligible.empty:
        eligible = metrics
    ordered = eligible.sort_values(
        ["selection_score", "silhouette", "k"],
        ascending=[False, False, True],
    )
    return int(ordered.iloc[0]["k"])


def assign_all(
    top: np.ndarray,
    fine_labels: np.ndarray,
    pca: PCA,
    retained: int,
    model: MiniBatchKMeans,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(top)
    labels = np.empty(n, dtype=np.int16)
    nearest = np.empty(n, dtype=np.float32)
    second = np.empty(n, dtype=np.float32)
    confidence = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        x = spatial_features(top[start:stop], fine_labels)
        z = pca_transform(x, pca, retained)
        distances = model.transform(z).astype(np.float32)
        predicted = distances.argmin(axis=1)
        ordered = np.partition(distances, 1, axis=1)[:, :2]
        d1 = ordered[:, 0]
        d2 = ordered[:, 1]
        labels[start:stop] = predicted
        nearest[start:stop] = d1
        second[start:stop] = d2
        confidence[start:stop] = np.clip(1.0 - d1 / np.maximum(d2, 1e-8), 0, 1)
        if start == 0 or stop == n or start % 50000 == 0:
            print(f"assignment {stop:,}/{n:,}", flush=True)
    return labels, nearest, second, confidence


def reorder_by_prevalence(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    raw_to_rank = np.empty(k, dtype=np.int16)
    raw_to_rank[order] = np.arange(k, dtype=np.int16)
    return raw_to_rank[labels], order


def aggregate_profiles(
    top: np.ndarray,
    fine_labels: np.ndarray,
    labels: np.ndarray,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial_sum = np.zeros((k, REGIONS, ELEMENTS), dtype=np.float64)
    element_counts = np.zeros((k, ELEMENTS), dtype=np.int64)
    dimension_counts = np.zeros((k, DIMENSIONS), dtype=np.int64)
    for start in range(0, len(top), batch_size):
        stop = min(start + batch_size, len(top))
        block = np.asarray(top[start:stop], dtype=np.int64)
        block_labels = labels[start:stop]
        spatial = spatial_features(block, fine_labels).reshape(
            len(block), REGIONS, ELEMENTS
        )
        elements = fine_counts(fine_labels[block])
        for cluster in np.unique(block_labels):
            take = block_labels == cluster
            spatial_sum[cluster] += spatial[take].sum(axis=0)
            element_counts[cluster] += elements[take].sum(axis=0)
            dimension_counts[cluster] += np.bincount(
                block[take].ravel(), minlength=DIMENSIONS
            )
        if start == 0 or stop == len(top) or start % 50000 == 0:
            print(f"profiles {stop:,}/{len(top):,}", flush=True)
    return spatial_sum, element_counts, dimension_counts


def representative_rows(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    distance: np.ndarray,
    confidence: np.ndarray,
    per_archetype: int,
) -> pd.DataFrame:
    rows = []
    for cluster in range(labels.max() + 1):
        candidates = np.flatnonzero(labels == cluster)
        candidates = candidates[np.argsort(distance[candidates])]
        chosen: list[int] = []
        seen_panos: set[str] = set()
        city_counts: dict[str, int] = {}
        for index in candidates:
            panoid = str(metadata.iloc[index]["panoid"])
            city = str(metadata.iloc[index]["city_key"])
            if panoid in seen_panos or city_counts.get(city, 0) >= 2:
                continue
            chosen.append(int(index))
            seen_panos.add(panoid)
            city_counts[city] = city_counts.get(city, 0) + 1
            if len(chosen) == per_archetype:
                break
        if len(chosen) < per_archetype:
            for index in candidates:
                panoid = str(metadata.iloc[index]["panoid"])
                if panoid in seen_panos:
                    continue
                chosen.append(int(index))
                seen_panos.add(panoid)
                if len(chosen) == per_archetype:
                    break
        for rank, index in enumerate(chosen, 1):
            source = metadata.iloc[index]
            rows.append(
                {
                    "archetype_id": f"A{cluster + 1:02d}",
                    "representative_rank": rank,
                    "global_image_index": index,
                    "city_key": source["city_key"],
                    "panoid": source["panoid"],
                    "heading": source["heading"],
                    "distance_to_centroid": float(distance[index]),
                    "assignment_confidence": float(confidence[index]),
                }
            )
    return pd.DataFrame(rows)


def read_original(row: pd.Series) -> Image.Image:
    fd = os.open(str(row["tar_path"]), os.O_RDONLY)
    try:
        payload = os.pread(fd, int(row["jpg_size"]), int(row["jpg_offset"]))
    finally:
        os.close(fd)
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def element_colors() -> np.ndarray:
    cmap = plt.get_cmap("turbo")
    return (cmap(np.linspace(0.02, 0.98, ELEMENTS))[:, :3] * 255).astype(np.uint8)


def layout_panel(
    spatial_mean: np.ndarray, colors: np.ndarray, size: int = 204
) -> Image.Image:
    panel = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(panel)
    small = font(16, bold=True)
    for region in range(REGIONS):
        row, column = divmod(region, 3)
        x0, y0 = column * size // 3, row * size // 3
        x1, y1 = (column + 1) * size // 3, (row + 1) * size // 3
        distribution = spatial_mean[region] * REGIONS
        element = int(distribution.argmax())
        share = float(distribution[element])
        base = colors[element].astype(np.float32)
        color = tuple(np.clip(0.35 * base + 0.65 * 255, 0, 255).astype(np.uint8))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color, outline=(70, 70, 70), width=2)
        label = f"F{element:03d}\n{share:.0%}"
        box = draw.multiline_textbbox((0, 0), label, font=small, align="center")
        width, height = box[2] - box[0], box[3] - box[1]
        draw.multiline_text(
            ((x0 + x1 - width) / 2, (y0 + y1 - height) / 2),
            label,
            fill=(20, 20, 20),
            font=small,
            align="center",
            spacing=1,
        )
    return panel


def render_atlas(
    profiles: pd.DataFrame,
    spatial_means: np.ndarray,
    element_shares: np.ndarray,
    representatives: pd.DataFrame,
    metadata: pd.DataFrame,
    names: dict[str, str],
    output: Path,
    shown_representatives: int = 4,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    k = len(profiles)
    row_height = 205
    header_width = 360
    layout_width = 220
    image_width = 260
    canvas = Image.new(
        "RGB",
        (header_width + layout_width + shown_representatives * image_width, 54 + k * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(25, bold=True)
    header_font = font(18, bold=True)
    body_font = font(14)
    draw.text((12, 10), "Directional-image spatial composition archetypes", fill="black", font=title_font)
    colors = element_colors()
    lookup = metadata.set_index("global_image_index", drop=False)
    for cluster in range(k):
        y = 54 + cluster * row_height
        row = profiles.iloc[cluster]
        top = np.argsort(-element_shares[cluster])[:4]
        draw.text((12, y + 8), f"A{cluster + 1:02d}", fill="black", font=title_font)
        draw.text(
            (80, y + 12),
            f"n={int(row['n_images']):,}  ({row['prevalence']:.1%})",
            fill=(40, 40, 40),
            font=header_font,
        )
        lines = []
        for element in top:
            key = f"F{element:03d}"
            name = names.get(key, key).replace("_", " ")
            if len(name) > 25:
                name = name[:24] + "…"
            lines.append(f"{key} {element_shares[cluster, element]:.1%}  {name}")
        draw.multiline_text((14, y + 51), "\n".join(lines), fill=(35, 35, 35), font=body_font, spacing=8)
        panel = layout_panel(spatial_means[cluster], colors, row_height - 12)
        canvas.paste(panel, (header_width + 6, y + 5))
        reps = representatives[
            representatives["archetype_id"] == f"A{cluster + 1:02d}"
        ].head(shown_representatives)
        for position, (_, representative) in enumerate(reps.iterrows()):
            source = lookup.loc[int(representative["global_image_index"])]
            image = ImageOps.fit(
                read_original(source),
                (image_width - 8, row_height - 34),
                method=Image.Resampling.LANCZOS,
            )
            x = header_width + layout_width + position * image_width
            canvas.paste(image, (x + 4, y + 4))
            city = str(source["city_key"]).split("/")[-1]
            draw.rectangle((x + 4, y + row_height - 28, x + image_width - 4, y + row_height - 5), fill="white")
            draw.text((x + 9, y + row_height - 26), city, fill=(20, 20, 20), font=body_font)
        draw.line((0, y + row_height - 1, canvas.width, y + row_height - 1), fill=(190, 190, 190), width=1)
    png = output / "Fig_Image_Layout_Archetype_Atlas.png"
    pdf = output / "Fig_Image_Layout_Archetype_Atlas.pdf"
    canvas.save(png, optimize=True)
    canvas.save(pdf, "PDF", resolution=150.0)


def plot_selection(metrics: pd.DataFrame, selected_k: int, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    specifications = [
        ("silhouette", "Silhouette (higher is better)"),
        ("initialization_stability_ari", "Initialization stability, ARI"),
        ("minimum_cluster_fraction", "Smallest archetype share"),
        ("selection_score", "Composite selection score"),
    ]
    for ax, (column, title) in zip(axes.ravel(), specifications):
        for representation, group in metrics.groupby("representation"):
            labels = {
                "spatial": "spatial 64×3×3 (main)",
                "composition": "composition-only 64",
                "coarse_spatial": "spatial 32×3×3",
                "dimension_composition": "composition-only 512",
            }
            label = labels.get(representation, representation)
            ax.plot(group["k"], group[column], marker="o", linewidth=1.7, label=label)
        ax.axvline(selected_k, color="#c8342d", linestyle="--", linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Archetype model selection (selected spatial K={selected_k})", fontsize=14)
    fig.savefig(output / "Fig_Image_Layout_K_Selection.png", dpi=300)
    fig.savefig(output / "Fig_Image_Layout_K_Selection.pdf")
    plt.close(fig)


def snapshot_paper(results: Path, figures: Path, paper: Path) -> None:
    data = paper / "data" / "image_layout_archetypes"
    data.mkdir(parents=True, exist_ok=True)
    for name in (
        "archetype_profiles.csv",
        "archetype_interpretations.csv",
        "archetype_element_profiles.csv",
        "archetype_spatial_profiles.csv",
        "archetype_dimension_profiles.csv",
        "city_archetype_prevalence.csv",
        "representative_images.csv",
        "image_layout_archetypes.parquet",
        "k_selection_metrics.csv",
        "model_parameters.npz",
        "report.json",
        "README.md",
    ):
        shutil.copy2(results / name, data / name)
    figure_target = paper / "figures" / "main"
    figure_target.mkdir(parents=True, exist_ok=True)
    for path in figures.glob("Fig_Image_Layout_*"):
        shutil.copy2(path, figure_target / path.name)
    table_target = paper / "tables" / "main"
    table_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        results / "archetype_profiles.csv",
        table_target / "Table_Image_Layout_Archetypes.csv",
    )
    reproducibility = paper / "reproducibility"
    reproducibility.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        results / "report.json",
        reproducibility / "image_layout_archetypes_report.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-hierarchy", type=Path, default=DEFAULT_SOURCE_HIERARCHY)
    parser.add_argument("--main-hierarchy", type=Path, default=DEFAULT_MAIN_HIERARCHY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--base-results", type=Path, default=Path("results"))
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--sample-per-city", type=int, default=2000)
    parser.add_argument("--k-values", type=int, nargs="+", default=[6, 8, 10, 12, 15, 20])
    parser.add_argument("--variance-target", type=float, default=0.90)
    parser.add_argument("--evaluation-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--representatives", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    metadata_columns = [
        "global_image_index", "city_key", "image_index", "source_image_index",
        "pano_index", "panoid", "direction_index", "heading", "lat", "lon",
        "year", "month", "tar_path", "jpg_offset", "jpg_size",
    ]
    metadata = pd.read_parquet(args.metadata, columns=metadata_columns)
    top = np.load(args.source_hierarchy / "top1_dimensions.npy", mmap_mode="r")
    arrays = np.load(args.main_hierarchy / "hierarchy_arrays.npz")
    fine_labels = arrays["fine_labels"].astype(np.int64)
    coarse_labels = arrays["coarse_labels"].astype(np.int64)
    if len(metadata) != len(top):
        raise ValueError(f"metadata/top mismatch: {len(metadata)} != {len(top)}")
    if not np.array_equal(metadata["global_image_index"], np.arange(len(metadata))):
        raise ValueError("metadata must be ordered by global_image_index")
    if fine_labels.shape != (DIMENSIONS,) or fine_labels.min() != 0 or fine_labels.max() != 63:
        raise ValueError("expected 512 dimensions mapped to exactly 64 fine elements")

    print("[1/7] city-balanced, panorama-deduplicated sample", flush=True)
    sample_indices = balanced_sample_indices(metadata, args.sample_per_city, args.seed)
    sample_top = np.asarray(top[sample_indices], dtype=np.int64)
    print(f"sample: {len(sample_indices):,} images from {metadata['city_key'].nunique()} cities", flush=True)

    print("[2/7] spatial-composition representation and PCA", flush=True)
    spatial_sample = spatial_features(sample_top, fine_labels)
    spatial_pca, spatial_scores, spatial_retained = fit_pca(
        spatial_sample, args.variance_target, args.seed
    )
    print(
        f"spatial PCA retained {spatial_retained}/{spatial_pca.n_components_} components "
        f"({spatial_pca.explained_variance_ratio_[:spatial_retained].sum():.3%} variance)",
        flush=True,
    )

    print("[3/7] K sweep and composition-only baseline", flush=True)
    spatial_metrics, spatial_models = k_sweep(
        spatial_scores, args.k_values, args.seed, args.evaluation_size, "spatial"
    )
    composition_sample = composition_features(sample_top, fine_labels)
    composition_pca, composition_scores, composition_retained = fit_pca(
        composition_sample, args.variance_target, args.seed, cap=63
    )
    composition_metrics, _ = k_sweep(
        composition_scores, args.k_values, args.seed, args.evaluation_size, "composition"
    )
    coarse_spatial_sample = pooled_spatial_features(sample_top, coarse_labels, 32)
    coarse_pca, coarse_scores, coarse_retained = fit_pca(
        coarse_spatial_sample, args.variance_target, args.seed
    )
    coarse_metrics, _ = k_sweep(
        coarse_scores, args.k_values, args.seed, args.evaluation_size, "coarse_spatial"
    )
    dimension_sample = pooled_composition_features(
        sample_top, np.arange(DIMENSIONS, dtype=np.int64), DIMENSIONS
    )
    dimension_pca, dimension_scores, dimension_retained = fit_pca(
        dimension_sample, args.variance_target, args.seed
    )
    dimension_metrics, _ = k_sweep(
        dimension_scores, args.k_values, args.seed, args.evaluation_size,
        "dimension_composition",
    )
    metrics = pd.concat(
        (spatial_metrics, composition_metrics, coarse_metrics, dimension_metrics),
        ignore_index=True,
    )
    selected_k = choose_k(spatial_metrics)
    save_csv(metrics, args.results / "k_selection_metrics.csv")
    plot_selection(metrics, selected_k, args.figures)
    print(f"selected spatial K={selected_k}", flush=True)

    print("[4/7] full-dataset assignment", flush=True)
    raw_labels, nearest, second, confidence = assign_all(
        top, fine_labels, spatial_pca, spatial_retained,
        spatial_models[selected_k], args.batch_size,
    )
    labels, prevalence_order = reorder_by_prevalence(raw_labels, selected_k)
    assignments = metadata.drop(columns=["tar_path", "jpg_offset", "jpg_size"]).copy()
    assignments["archetype_index"] = labels
    assignments["archetype_id"] = [f"A{x + 1:02d}" for x in labels]
    assignments["distance_to_centroid"] = nearest
    assignments["second_centroid_distance"] = second
    assignments["assignment_confidence"] = confidence
    save_parquet(assignments, args.results / "image_layout_archetypes.parquet")

    print("[5/7] archetype profiles and representatives", flush=True)
    spatial_sum, element_counts, dimension_counts = aggregate_profiles(
        top, fine_labels, labels, selected_k, args.batch_size
    )
    counts = np.bincount(labels, minlength=selected_k)
    spatial_means = spatial_sum / counts[:, None, None]
    element_shares = element_counts / element_counts.sum(axis=1, keepdims=True)
    dimension_shares = dimension_counts / dimension_counts.sum(axis=1, keepdims=True)
    names = load_semantic_names(args.base_results / "semantic_labels_fine.json")
    profile_rows = []
    element_rows = []
    spatial_rows = []
    dimension_rows = []
    for cluster in range(selected_k):
        top_elements = np.argsort(-element_shares[cluster])[:8]
        top_dimensions = np.argsort(-dimension_shares[cluster])[:12]
        entropy = -np.sum(
            element_shares[cluster] * np.log(np.maximum(element_shares[cluster], 1e-12))
        ) / math.log(ELEMENTS)
        profile_rows.append(
            {
                "archetype_id": f"A{cluster + 1:02d}",
                "n_images": int(counts[cluster]),
                "prevalence": float(counts[cluster] / len(labels)),
                "mean_distance_to_centroid": float(nearest[labels == cluster].mean()),
                "median_assignment_confidence": float(np.median(confidence[labels == cluster])),
                "normalized_element_entropy": float(entropy),
                "top_elements": "; ".join(f"F{x:03d}" for x in top_elements),
                "top_element_names": "; ".join(names.get(f"F{x:03d}", "") for x in top_elements),
                "top_dimensions": "; ".join(f"D{x:03d}" for x in top_dimensions),
            }
        )
        element_rank = np.empty(ELEMENTS, dtype=np.int16)
        element_rank[np.argsort(-element_shares[cluster])] = np.arange(1, ELEMENTS + 1)
        for element in range(ELEMENTS):
            element_rows.append(
                {
                    "archetype_id": f"A{cluster + 1:02d}",
                    "element_id": f"F{element:03d}",
                    "element_name": names.get(f"F{element:03d}", ""),
                    "canvas_share": float(element_shares[cluster, element]),
                    "within_archetype_rank": int(element_rank[element]),
                }
            )
        for region in range(REGIONS):
            row, column = divmod(region, 3)
            for element in range(ELEMENTS):
                spatial_rows.append(
                    {
                        "archetype_id": f"A{cluster + 1:02d}",
                        "region_id": f"R{row + 1}{column + 1}",
                        "region_row": row,
                        "region_column": column,
                        "element_id": f"F{element:03d}",
                        "element_name": names.get(f"F{element:03d}", ""),
                        "within_region_share": float(spatial_means[cluster, region, element] * REGIONS),
                        "equal_weight_canvas_share": float(spatial_means[cluster, region, element]),
                    }
                )
        dimension_rows.append(
            {"archetype_id": f"A{cluster + 1:02d}", **{
                f"D{dimension:03d}": float(dimension_shares[cluster, dimension])
                for dimension in range(DIMENSIONS)
            }}
        )
    profiles = pd.DataFrame(profile_rows)
    save_csv(profiles, args.results / "archetype_profiles.csv")
    save_csv(pd.DataFrame(element_rows), args.results / "archetype_element_profiles.csv")
    save_csv(pd.DataFrame(spatial_rows), args.results / "archetype_spatial_profiles.csv")
    save_csv(pd.DataFrame(dimension_rows), args.results / "archetype_dimension_profiles.csv")
    representatives = representative_rows(
        metadata, labels, nearest, confidence, args.representatives
    )
    save_csv(representatives, args.results / "representative_images.csv")
    city_counts = pd.crosstab(metadata["city_key"], labels).reindex(
        columns=np.arange(selected_k), fill_value=0
    )
    city_rows = []
    for city, row in city_counts.iterrows():
        for cluster in range(selected_k):
            city_rows.append(
                {
                    "city_key": city,
                    "archetype_id": f"A{cluster + 1:02d}",
                    "n_images": int(row[cluster]),
                    "city_prevalence": float(row[cluster] / row.sum()),
                    "global_prevalence": float(counts[cluster] / len(labels)),
                    "prevalence_ratio": float(
                        (row[cluster] / row.sum()) / max(counts[cluster] / len(labels), 1e-12)
                    ),
                }
            )
    save_csv(pd.DataFrame(city_rows), args.results / "city_archetype_prevalence.csv")

    print("[6/7] atlas, model parameters, and validation", flush=True)
    render_atlas(
        profiles, spatial_means, element_shares, representatives, metadata,
        names, args.figures,
    )
    model = spatial_models[selected_k]
    np.savez_compressed(
        args.results / "model_parameters.npz",
        sample_indices=sample_indices,
        pca_mean=spatial_pca.mean_.astype(np.float32),
        pca_components=spatial_pca.components_[:spatial_retained].astype(np.float32),
        pca_explained_variance_ratio=spatial_pca.explained_variance_ratio_[:spatial_retained].astype(np.float32),
        cluster_centers=model.cluster_centers_.astype(np.float32),
        raw_cluster_prevalence_order=prevalence_order,
        region_edges=np.asarray(REGION_EDGES, dtype=np.int16),
    )
    if not np.array_equal(np.bincount(labels, minlength=selected_k), counts):
        raise ValueError("archetype assignment counts changed during aggregation")
    if not np.allclose(element_shares.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("element profiles do not sum to one")
    if not np.allclose(dimension_shares.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("dimension profiles do not sum to one")
    if not np.allclose(spatial_means.sum(axis=(1, 2)), 1.0, atol=1e-6):
        raise ValueError("spatial profiles do not sum to one")
    sample_city_counts = metadata.iloc[sample_indices]["city_key"].value_counts()
    selected_metrics = spatial_metrics.loc[
        spatial_metrics["k"] == selected_k
    ].iloc[0]
    best_silhouette = {}
    for representation, group in metrics.groupby("representation"):
        row = group.loc[group["silhouette"].idxmax()]
        best_silhouette[representation] = {
            "k": int(row["k"]),
            "silhouette": float(row["silhouette"]),
            "initialization_stability_ari": float(row["initialization_stability_ari"]),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "unit_of_analysis": "directional street-view image",
        "images": int(len(metadata)),
        "cities": int(metadata["city_key"].nunique()),
        "sample_images": int(len(sample_indices)),
        "sample_per_city": int(args.sample_per_city),
        "sample_panorama_deduplicated": True,
        "sample_city_balance_min_max": [int(sample_city_counts.min()), int(sample_city_counts.max())],
        "main_representation": "64 frozen fine elements x 3 x 3 equal-weight canvas regions",
        "composition_baseline": "64 frozen fine-element whole-image shares",
        "patch_grid": [PATCH_GRID, PATCH_GRID],
        "region_edges": list(REGION_EDGES),
        "transform": "square-root (Hellinger) followed by PCA",
        "pca_variance_target": args.variance_target,
        "spatial_pca_components": spatial_retained,
        "spatial_pca_variance_retained": float(spatial_pca.explained_variance_ratio_[:spatial_retained].sum()),
        "composition_pca_components": composition_retained,
        "composition_pca_variance_retained": float(composition_pca.explained_variance_ratio_[:composition_retained].sum()),
        "coarse_spatial_sensitivity": "32 frozen coarse elements x 3 x 3 equal-weight canvas regions",
        "coarse_spatial_pca_components": coarse_retained,
        "coarse_spatial_pca_variance_retained": float(coarse_pca.explained_variance_ratio_[:coarse_retained].sum()),
        "dimension_composition_sensitivity": "512 original dimension whole-image shares",
        "dimension_composition_pca_components": dimension_retained,
        "dimension_composition_pca_variance_retained": float(dimension_pca.explained_variance_ratio_[:dimension_retained].sum()),
        "algorithm": "MiniBatchKMeans",
        "k_candidates": args.k_values,
        "selected_k": selected_k,
        "selection_rule": "highest mean percentile rank of silhouette, inverse Davies-Bouldin, initialization ARI, minimum cluster fraction, and normalized size entropy; clusters below 1% excluded when possible",
        "selected_model_metrics": {
            key: float(selected_metrics[key])
            for key in (
                "silhouette", "calinski_harabasz", "davies_bouldin",
                "initialization_stability_ari", "minimum_cluster_fraction",
                "maximum_cluster_fraction", "normalized_size_entropy",
                "selection_score",
            )
        },
        "best_silhouette_by_representation": best_silhouette,
        "random_seed": args.seed,
        "archetype_order": "descending full-dataset prevalence",
        "smallest_archetype_images": int(counts.min()),
        "smallest_archetype_fraction": float(counts.min() / len(labels)),
        "median_assignment_confidence": float(np.median(confidence)),
        "assignment_confidence_quantiles": {
            str(q): float(np.quantile(confidence, q))
            for q in (0.25, 0.5, 0.75, 0.9, 0.95)
        },
        "semantic_labels_used_for_clustering": False,
        "city_used_for_clustering": False,
        "city_used_for_balanced_sampling": True,
        "interpretation_caveat": "These are directional-image visual scene archetypes, not plan-view urban morphology classes.",
    }
    save_json(report, args.results / "report.json")
    readme = """# Directional-image visual scene archetypes

聚类单元是单张方向街景，而不是城市或 1 km 网格。主表征将 14×14 patch winner map 映射到冻结的 64 个 E+D+P 细粒度元素，并汇聚成 3×3 画布。每个空间格先独立归一化、再赋予 1/9 权重，因此不会因 14 无法被 3 整除而让边缘格权重偏低。

主实验对 `64×3×3=576` 维空间组成做 Hellinger 变换和 PCA，在每城 2,000 张、panorama 去重的均衡样本上比较 K={6,8,10,12,15,20} 的 MiniBatchKMeans，然后给全部图片赋类。64 维全图元素占比是 composition-only baseline；`32×3×3` 粗元素空间布局和 512 原始维度全图占比是敏感性分析。

- `image_layout_archetypes.parquet`：全部图片的原型、质心距离和分配置信度。
- `archetype_profiles.csv`：原型规模、熵、主导元素和主导原始维度。
- `archetype_element_profiles.csv`：每个原型的 64 元素全图占比。
- `archetype_spatial_profiles.csv`：每个原型在 3×3 各格中的 64 元素条件占比。
- `archetype_dimension_profiles.csv`：每个原型的原始 512 维画布占比。
- `representative_images.csv`：最接近质心且 panorama 去重的代表图。
- `city_archetype_prevalence.csv`：事后城市分布；城市不参与聚类。
- `k_selection_metrics.csv`：空间表征与纯组成基线的 K 选择指标。
- `model_parameters.npz`：样本索引、PCA 参数、聚类中心与标签重排。

语义名称只用于事后说明，不参与聚类。结果应称为“方向街景视觉场景原型”或“街景画布原型”，不宜直接解释为平面城市形态。
"""
    (args.results / "README.md").write_text(readme)

    print("[7/7] paper snapshot", flush=True)
    snapshot_paper(args.results, args.figures, args.paper)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
