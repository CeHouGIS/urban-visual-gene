#!/usr/bin/env python3
"""Balanced-fit, all-image UMAP of the 512 Feature-MAE dimensions.

UMAP is fitted without city labels.  The fit subset contains exactly 2,000
images per city and targets at least 100 Top-3 representatives for each latent
dimension.  All retained training images are then transformed into the frozen
two-dimensional embedding.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from adjustText import adjust_text
from sklearn.decomposition import PCA

from scripts.multicity.config import CITIES, SEED, city_slug


DEFAULT_DATA_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)
DEFAULT_HIERARCHY = DEFAULT_DATA_ROOT / "mae" / "hierarchy_edp_32_64"
DEFAULT_ACTIVATIONS = Path(
    "outputs/analysis/all_city_umap_activation/image_activations.npy"
)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def dimension_counts(top: np.ndarray, path: Path, width: int = 512) -> np.memmap:
    expected = (len(top), width)
    if path.is_file():
        saved = np.load(path, mmap_mode="r+")
        if saved.shape == expected and saved.dtype == np.uint8:
            return saved
        raise ValueError(f"unexpected cached count matrix {saved.shape} {saved.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=expected)
    for start in range(0, len(top), 5000):
        block = np.asarray(top[start : start + 5000], dtype=np.int64)
        rows = np.repeat(np.arange(len(block), dtype=np.int64), block.shape[1])
        keys = rows * width + block.ravel()
        values = np.bincount(keys, minlength=len(block) * width).reshape(len(block), width)
        counts[start : start + len(block)] = values.astype(np.uint8)
        if start % 50000 == 0:
            counts.flush()
            print(f"dimension counts {min(start+len(block),len(top)):,}/{len(top):,}", flush=True)
    counts.flush()
    return counts


def top_dimensions(counts: np.ndarray, top_path: Path, dominant_path: Path):
    if top_path.is_file() and dominant_path.is_file():
        return np.load(top_path, mmap_mode="r"), np.load(dominant_path, mmap_mode="r")
    top3 = np.lib.format.open_memmap(top_path, mode="w+", dtype=np.uint16, shape=(len(counts), 3))
    dominant = np.lib.format.open_memmap(dominant_path, mode="w+", dtype=np.uint16, shape=(len(counts),))
    for start in range(0, len(counts), 10000):
        block = np.asarray(counts[start : start + 10000])
        selected = np.argpartition(block, -3, axis=1)[:, -3:]
        values = np.take_along_axis(block, selected, axis=1)
        order = np.argsort(-values, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        top3[start : start + len(block)] = selected
        dominant[start : start + len(block)] = selected[:, 0]
    top3.flush(); dominant.flush()
    return top3, dominant


def constrained_sample(
    top3: np.ndarray,
    city_ids: np.ndarray,
    per_city: int,
    per_dimension: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(top3), dtype=bool)
    city_used = np.zeros(len(CITIES), dtype=np.int64)
    available = np.bincount(np.asarray(top3).ravel(), minlength=512)
    targets = np.minimum(available, per_dimension)
    dimension_used = np.zeros(512, dtype=np.int64)

    # Rarest dimensions are protected first. Multi-label images can satisfy up
    # to three quotas, reducing distortion of the natural image distribution.
    for dimension in np.argsort(available):
        needed = int(targets[dimension] - dimension_used[dimension])
        if needed <= 0:
            continue
        candidates = np.flatnonzero(
            (~selected) & np.any(np.asarray(top3) == dimension, axis=1)
        )
        rng.shuffle(candidates)
        candidates = candidates[
            np.argsort(city_used[np.asarray(city_ids[candidates], dtype=np.int64)], kind="stable")
        ]
        chosen = []
        for index in candidates:
            city = int(city_ids[index])
            if city_used[city] >= per_city:
                continue
            chosen.append(int(index))
            if len(chosen) == needed:
                break
        if chosen:
            chosen_array = np.asarray(chosen, dtype=np.int64)
            selected[chosen_array] = True
            np.add.at(city_used, np.asarray(city_ids[chosen_array], dtype=np.int64), 1)
            dimension_used += np.bincount(
                np.asarray(top3[chosen_array]).ravel(), minlength=512
            )

    for city in range(len(CITIES)):
        needed = int(per_city - city_used[city])
        candidates = np.flatnonzero((np.asarray(city_ids) == city) & (~selected))
        if len(candidates) < needed:
            raise RuntimeError(f"city {city} has only {len(candidates)} remaining candidates")
        chosen = rng.choice(candidates, size=needed, replace=False)
        selected[chosen] = True
        city_used[city] += needed
    fit_indices = np.flatnonzero(selected)
    final_dimension_counts = np.bincount(
        np.asarray(top3[fit_indices]).ravel(), minlength=512
    )
    if not np.all(final_dimension_counts >= targets):
        missing = np.flatnonzero(final_dimension_counts < targets)
        raise RuntimeError(f"dimension quotas not satisfied: {missing.tolist()}")
    if not np.all(city_used == per_city):
        raise RuntimeError(f"city quotas not satisfied: {city_used.tolist()}")
    return fit_indices


def feature_block(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(counts[indices], dtype=np.float32) / 196.0
    return np.sqrt(probabilities, out=probabilities)


def standardized_activations(
    scores: np.ndarray, path: Path, block_size: int = 10000
) -> np.memmap:
    if path.is_file():
        values = np.load(path, mmap_mode="r+")
        if values.shape == scores.shape and values.dtype == np.float16:
            return values
        raise ValueError(f"unexpected standardized activation cache {values.shape}")
    mean = np.asarray(scores, dtype=np.float32).mean(axis=0, dtype=np.float64)
    variance = np.zeros(scores.shape[1], dtype=np.float64)
    for start in range(0, len(scores), block_size):
        block = np.asarray(scores[start : start + block_size], dtype=np.float32)
        variance += np.sum((block - mean) ** 2, axis=0)
    scale = np.sqrt(variance / max(1, len(scores) - 1)) + 1e-6
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=scores.shape
    )
    for start in range(0, len(scores), block_size):
        block = (np.asarray(scores[start : start + block_size], dtype=np.float32) - mean) / scale
        values[start : start + len(block)] = np.clip(block, -6, 6).astype(np.float16)
    values.flush()
    return values


def activation_top_dimensions(
    values: np.ndarray, top_path: Path, dominant_path: Path
):
    if top_path.is_file() and dominant_path.is_file():
        return np.load(top_path, mmap_mode="r"), np.load(dominant_path, mmap_mode="r")
    top3 = np.lib.format.open_memmap(top_path, mode="w+", dtype=np.uint16, shape=(len(values), 3))
    dominant = np.lib.format.open_memmap(dominant_path, mode="w+", dtype=np.uint16, shape=(len(values),))
    for start in range(0, len(values), 10000):
        block = np.asarray(values[start : start + 10000], dtype=np.float32)
        selected = np.argpartition(block, -3, axis=1)[:, -3:]
        selected_values = np.take_along_axis(block, selected, axis=1)
        selected = np.take_along_axis(
            selected, np.argsort(-selected_values, axis=1), axis=1
        )
        top3[start : start + len(block)] = selected
        dominant[start : start + len(block)] = selected[:, 0]
    top3.flush(); dominant.flush()
    return top3, dominant


def constrained_activation_sample(
    values: np.ndarray,
    city_ids: np.ndarray,
    per_city: int,
    per_dimension: int,
    candidate_pool: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    city_ids_array = np.asarray(city_ids, dtype=np.int64)
    selected = np.zeros(len(values), dtype=bool)
    city_used = np.zeros(len(CITIES), dtype=np.int64)
    candidates = np.empty((values.shape[1], candidate_pool), dtype=np.int32)
    concentration = np.empty(values.shape[1], dtype=np.int16)
    for dimension in range(values.shape[1]):
        column = values[:, dimension]
        indices = np.argpartition(column, -candidate_pool)[-candidate_pool:]
        indices = indices[np.argsort(-column[indices])]
        candidates[dimension] = indices.astype(np.int32)
        concentration[dimension] = len(np.unique(city_ids_array[indices]))

    # Dimensions whose strongest evidence occurs in fewer cities are allocated
    # first, reducing the chance that a relevant city quota fills prematurely.
    for dimension in np.argsort(concentration):
        pool = candidates[dimension]
        current = int(selected[pool].sum())
        needed = per_dimension - current
        if needed <= 0:
            continue
        for index in pool:
            index = int(index)
            if selected[index]:
                continue
            city = int(city_ids_array[index])
            if city_used[city] >= per_city:
                continue
            selected[index] = True
            city_used[city] += 1
            needed -= 1
            if needed == 0:
                break
        if needed:
            raise RuntimeError(
                f"could not allocate {per_dimension} representatives for D{dimension:03d}"
            )

    for city in range(len(CITIES)):
        needed = int(per_city - city_used[city])
        pool = np.flatnonzero((city_ids_array == city) & (~selected))
        chosen = rng.choice(pool, size=needed, replace=False)
        selected[chosen] = True
        city_used[city] += needed
    fit_indices = np.flatnonzero(selected)
    representative_counts = np.asarray(
        [selected[pool].sum() for pool in candidates], dtype=np.int64
    )
    if not np.all(representative_counts >= per_dimension):
        raise RuntimeError("activation representative quotas were not satisfied")
    if not np.all(city_used == per_city):
        raise RuntimeError("city quotas were not satisfied")
    return fit_indices, candidates


def centroids(
    coordinates: np.ndarray, counts: np.ndarray, block_size: int = 10000
) -> np.ndarray:
    numerator = np.zeros((512, 2), dtype=np.float64)
    denominator = np.zeros(512, dtype=np.float64)
    for start in range(0, len(counts), block_size):
        block = np.asarray(counts[start : start + block_size], dtype=np.float64)
        numerator += block.T @ coordinates[start : start + len(block)]
        denominator += block.sum(axis=0)
    return numerator / np.maximum(denominator[:, None], 1e-12)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB colours to CIE Lab for perceptual palette selection."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3 * delta ** 2) + 4.0 / 29.0,
    )
    return np.column_stack(
        [
            116 * transformed[:, 1] - 16,
            500 * (transformed[:, 0] - transformed[:, 1]),
            200 * (transformed[:, 1] - transformed[:, 2]),
        ]
    )


def categorical_palette(size: int) -> list[str]:
    """Return a deterministic Glasbey-style set of separated category colours."""
    anchors = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
        "#56B4E9", "#6A3D9A", "#8C564B",
    ]
    candidates = [matplotlib.colors.to_rgb(colour) for colour in anchors]
    for cmap_name in ("tab20", "tab20b", "tab20c", "Dark2", "Set1", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        candidates.extend(cmap(index)[:3] for index in range(cmap.N))
    # Add a dense HSV candidate set, then greedily maximize minimum CIE-Lab
    # separation. Moderate lightness limits keep points visible on white.
    for hue in np.linspace(0, 1, 72, endpoint=False):
        for saturation in (0.55, 0.72, 0.88):
            for value in (0.58, 0.72, 0.86):
                candidates.append(matplotlib.colors.hsv_to_rgb((hue, saturation, value)))
    rgb = np.unique(np.round(np.asarray(candidates), 6), axis=0)
    lab = _rgb_to_lab(rgb)
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    keep = (lab[:, 0] >= 30) & (lab[:, 0] <= 78) & (chroma >= 22)
    rgb, lab = rgb[keep], lab[keep]

    anchor_rgb = np.asarray([matplotlib.colors.to_rgb(colour) for colour in anchors])
    selected = [int(np.argmin(np.linalg.norm(rgb - anchor_rgb[0], axis=1)))]
    minimum_distance = np.linalg.norm(lab - lab[selected[0]], axis=1)
    while len(selected) < size:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(lab - lab[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return [matplotlib.colors.to_hex(rgb[index]) for index in selected]


def palette32() -> list[str]:
    return categorical_palette(32)


def plots(
    args: argparse.Namespace,
    coordinates: np.ndarray,
    fit_indices: np.ndarray,
    city_ids: np.ndarray,
    dominant: np.ndarray,
    dimension_centroids: np.ndarray,
    hierarchy: pd.DataFrame,
) -> None:
    args.figures.mkdir(parents=True, exist_ok=True)
    coarse_by_dimension = hierarchy["coarse_cluster"].str[1:].astype(int).to_numpy()
    fine_by_dimension = hierarchy["fine_cluster"].to_numpy()
    coarse = coarse_by_dimension[np.asarray(dominant, dtype=np.int64)]
    colours = palette32()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    city_colours = categorical_palette(len(CITIES))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    texts = []
    city_names = [value.split("/")[-1] for value in CITIES]
    for city, name in enumerate(city_names):
        mask = np.asarray(city_ids) == city
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=0.18, alpha=0.12,
            color=city_colours[city], linewidths=0, rasterized=True,
        )
        center = np.median(coordinates[mask], axis=0)
        ax.plot(center[0], center[1], "o", ms=2.4, color=city_colours[city], mec="black", mew=0.35)
        texts.append(ax.text(center[0], center[1], name, fontsize=5.2, color="#111111"))
    adjust_text(
        texts, ax=ax, expand=(1.08, 1.16), force_text=(0.35, 0.55),
        force_points=(0.15, 0.25), arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.3},
    )
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.figures / "all_city_samples_umap.png", dpi=500)
    fig.savefig(args.figures / "all_city_samples_umap.pdf", dpi=500)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for group in range(32):
        mask = coarse == group
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=0.18, alpha=0.12,
            color=colours[group], linewidths=0, rasterized=True,
        )
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.figures / "all_city_samples_umap_by_visual_family.png", dpi=500)
    fig.savefig(args.figures / "all_city_samples_umap_by_visual_family.pdf", dpi=500)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 18), constrained_layout=True)
    ax.scatter(
        coordinates[:, 0], coordinates[:, 1], s=0.04, alpha=0.05,
        color="#777777", linewidths=0, rasterized=True,
    )
    for dimension, point in enumerate(dimension_centroids):
        group = coarse_by_dimension[dimension]
        ax.scatter(*point, s=9, color=colours[group], edgecolor="white", linewidth=0.25)
        ax.text(point[0] + 0.015, point[1] + 0.015, f"D{dimension:03d}", fontsize=3.2)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.figures / "all_city_samples_umap_512_labels.png", dpi=500)
    fig.savefig(args.figures / "all_city_samples_umap_512_labels.pdf", dpi=500)
    plt.close(fig)

    labels_path = args.hierarchy / "semantic_labels_qwen" / "category_labels.json"
    semantic = json.loads(labels_path.read_text())["categories"] if labels_path.is_file() else {}
    sample_city = np.asarray(city_ids[fit_indices], dtype=np.int64)
    sample_dimension = np.asarray(dominant[fit_indices], dtype=np.int64)
    hover = [
        f"City: {CITIES[city].split('/')[-1]}<br>Dominant: D{dimension:03d}<br>"
        f"Fine: {fine_by_dimension[dimension]}"
        for city, dimension in zip(sample_city, sample_dimension)
    ]
    figure = go.Figure()
    for city, name in enumerate([value.split("/")[-1] for value in CITIES]):
        mask = sample_city == city
        figure.add_trace(
            go.Scattergl(
                x=coordinates[fit_indices[mask], 0], y=coordinates[fit_indices[mask], 1],
                mode="markers", name=name,
                marker={"size": 2.5, "opacity": 0.35, "color": city_colours[city]},
                text=np.asarray(hover, dtype=object)[mask], hoverinfo="text",
            )
        )
    centroid_hover = []
    for dimension in range(512):
        fine_id = fine_by_dimension[dimension]
        name = semantic.get(fine_id, {}).get("name_en", "")
        centroid_hover.append(
            f"D{dimension:03d}<br>Fine: {fine_id} {name}<br>"
            f"Coarse: C{coarse_by_dimension[dimension]:03d}"
        )
    figure.add_trace(
        go.Scatter(
            x=dimension_centroids[:, 0], y=dimension_centroids[:, 1],
            mode="markers+text", text=[f"D{x:03d}" for x in range(512)],
            textposition="top center", textfont={"size": 7}, name="512 dimensions",
            marker={"size": 5, "color": [colours[x] for x in coarse_by_dimension],
                    "line": {"color": "white", "width": 0.5}},
            customdata=centroid_hover, hovertemplate="%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        template="simple_white", width=1400, height=1000,
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        legend={"itemsizing": "constant"},
    )
    figure.write_html(
        args.figures / "all_city_samples_umap_interactive.html",
        include_plotlyjs=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--activation-scores", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path("outputs/analysis/all_city_umap_activation"),
    )
    parser.add_argument("--per-city", type=int, default=2000)
    parser.add_argument("--per-dimension", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--transform-batch", type=int, default=25000)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    source = args.data_root / "mae" / "hierarchy_32_64"
    city_ids = np.load(source / "city_indices.npy", mmap_mode="r")
    raw_scores = np.load(args.activation_scores, mmap_mode="r")
    if raw_scores.shape != (len(city_ids), 512):
        raise ValueError(f"unexpected activation score shape {raw_scores.shape}")
    standardized = standardized_activations(
        raw_scores, args.work_dir / "standardized_activations.npy"
    )
    # Keeping this float32 matrix resident avoids hundreds of strided disk
    # passes while selecting representatives for all 512 dimensions.
    features = np.asarray(standardized, dtype=np.float32)
    top3, dominant = activation_top_dimensions(
        features,
        args.work_dir / "top3_dimensions.npy",
        args.work_dir / "dominant_dimension.npy",
    )
    fit_path = args.work_dir / "fit_indices.npy"
    candidate_path = args.work_dir / "dimension_candidate_indices.npy"
    if fit_path.is_file() and candidate_path.is_file():
        fit_indices = np.load(fit_path)
        candidates = np.load(candidate_path)
    else:
        fit_indices, candidates = constrained_activation_sample(
            features, city_ids, args.per_city, args.per_dimension,
            args.candidate_pool, args.seed,
        )
        np.save(fit_path, fit_indices)
        np.save(candidate_path, candidates)
    print(f"UMAP fit subset: {len(fit_indices):,} images", flush=True)

    available_top1 = np.bincount(np.asarray(dominant), minlength=512)
    available_top3 = np.bincount(np.asarray(top3).ravel(), minlength=512)
    fit_top1 = np.bincount(np.asarray(dominant[fit_indices]), minlength=512)
    fit_top3 = np.bincount(np.asarray(top3[fit_indices]).ravel(), minlength=512)
    is_fit = np.zeros(len(features), dtype=bool); is_fit[fit_indices] = True
    audit_rows = []
    for dimension in range(512):
        representatives = candidates[dimension][is_fit[candidates[dimension]]]
        represented = len(np.unique(np.asarray(city_ids[representatives])))
        audit_rows.append(
            {
                "dimension_id": f"D{dimension:03d}",
                "available_top1_count": int(available_top1[dimension]),
                "available_top3_count": int(available_top3[dimension]),
                "umap_fit_top1_count": int(fit_top1[dimension]),
                "umap_fit_top3_count": int(fit_top3[dimension]),
                "high_response_candidate_pool": int(len(candidates[dimension])),
                "selected_high_response_representatives": int(len(representatives)),
                "representative_city_count": represented,
                "target_count": args.per_dimension,
                "quota_satisfied": bool(len(representatives) >= args.per_dimension),
            }
        )
    save_csv(pd.DataFrame(audit_rows), args.results / "umap_sampling_audit.csv")

    reducer_path = args.work_dir / "umap_reducer.pkl"
    pca_path = args.work_dir / "pca_50.pkl"
    coordinates_path = args.work_dir / "all_coordinates.npy"
    if coordinates_path.is_file():
        coordinates = np.load(coordinates_path, mmap_mode="r+")
    else:
        # Import UMAP only when the embedding must actually be fitted. This
        # keeps cached-coordinate plotting independent of numba/UMAP runtime
        # initialization and its CPU-specific JIT state.
        import umap

        fit_features = features[fit_indices]
        pca = PCA(n_components=50, svd_solver="randomized", random_state=args.seed)
        fit_reduced = pca.fit_transform(fit_features)
        reducer = umap.UMAP(
            n_neighbors=30, min_dist=0.1, n_components=2, metric="cosine",
            random_state=args.seed, low_memory=True, verbose=True,
        ).fit(fit_reduced)
        with pca_path.open("wb") as handle:
            pickle.dump(pca, handle)
        with reducer_path.open("wb") as handle:
            pickle.dump(reducer, handle)
        coordinates = np.lib.format.open_memmap(
            coordinates_path, mode="w+", dtype=np.float32, shape=(len(features), 2)
        )
        coordinates[fit_indices] = reducer.embedding_.astype(np.float32)
        remaining = np.flatnonzero(~is_fit)
        for start in range(0, len(remaining), args.transform_batch):
            indices = remaining[start : start + args.transform_batch]
            reduced = pca.transform(features[indices])
            coordinates[indices] = reducer.transform(reduced).astype(np.float32)
            coordinates.flush()
            print(f"UMAP transform {min(start+len(indices),len(remaining)):,}/{len(remaining):,}", flush=True)

    dimension_centroids = np.zeros((512, 2), dtype=np.float64)
    for dimension in range(512):
        representatives = candidates[dimension][is_fit[candidates[dimension]]]
        representatives = representatives[: args.per_dimension]
        dimension_centroids[dimension] = np.median(
            np.asarray(coordinates)[representatives], axis=0
        )
    np.save(args.work_dir / "dimension_centroids.npy", dimension_centroids.astype(np.float32))
    hierarchy = pd.read_csv(args.hierarchy / "feature_hierarchy.csv").sort_values("feature_id")

    city_names, source_indices, panoids = [], [], []
    for city in CITIES:
        frame = pd.read_parquet(
            args.data_root / "filtered_manifests" / f"{city_slug(city)}.parquet",
            columns=["source_image_index", "panoid"],
        )
        city_names.extend([city.split("/")[-1]] * len(frame))
        source_indices.extend(frame["source_image_index"].astype(int).tolist())
        panoids.extend(frame["panoid"].astype(str).tolist())
    dominant_int = np.asarray(dominant, dtype=np.int64)
    fine_by_dimension = hierarchy["fine_cluster"].to_numpy()
    coarse_by_dimension = hierarchy["coarse_cluster"].to_numpy()
    output = pd.DataFrame(
        {
            "global_image_index": np.arange(len(features)),
            "city": city_names,
            "source_image_index": source_indices,
            "panoid": panoids,
            "umap_1": np.asarray(coordinates)[:, 0],
            "umap_2": np.asarray(coordinates)[:, 1],
            "dominant_dimension": [f"D{x:03d}" for x in dominant_int],
            "top2_dimension": [f"D{x:03d}" for x in np.asarray(top3)[:, 1]],
            "top3_dimension": [f"D{x:03d}" for x in np.asarray(top3)[:, 2]],
            "fine_cluster": fine_by_dimension[dominant_int],
            "coarse_cluster": coarse_by_dimension[dominant_int],
            "used_to_fit_umap": is_fit,
        }
    )
    save_csv(output, args.results / "image_umap_coordinates.csv")
    temporary = args.results / "image_umap_coordinates.parquet.tmp"
    output.to_parquet(temporary, index=False)
    temporary.replace(args.results / "image_umap_coordinates.parquet")
    plots(
        args, np.asarray(coordinates), fit_indices, city_ids, dominant,
        dimension_centroids, hierarchy,
    )
    report = {
        "images": len(features), "cities": len(CITIES), "fit_images": len(fit_indices),
        "fit_images_per_city": args.per_city,
        "high_response_target_per_dimension": args.per_dimension,
        "high_response_candidate_pool_per_dimension": args.candidate_pool,
        "all_dimension_quotas_satisfied": bool(all(row["quota_satisfied"] for row in audit_rows)),
        "input": "standardized per-image 512-D Feature-MAE activation profiles",
        "activation_score": "mean of the 20 strongest of 196 spatial patches per dimension",
        "pca_components": 50,
        "umap": {"n_neighbors": 30, "min_dist": 0.1, "metric": "cosine", "seed": args.seed},
        "city_labels_used_for_fit": False,
    }
    (args.results / "image_umap_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
