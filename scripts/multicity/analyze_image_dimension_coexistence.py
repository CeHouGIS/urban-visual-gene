#!/usr/bin/env python3
"""Analyse coexistence of 512 Feature-MAE dimensions within individual images.

Two complementary definitions are retained:

1. Spatial coverage: a dimension is present when it is the strongest latent
   response for at least four of the 196 image patches.
2. Activation strength: a dimension is strongly active when its robust
   image-level score is at least two median absolute deviations above its
   dimension-specific reference median.

The script only consumes existing cached arrays and does not run the model.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist

from scripts.multicity.config import CITIES


ROOT = Path("outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc")
DEFAULT_TOP = ROOT / "mae/hierarchy_32_64/top1_dimensions.npy"
DEFAULT_CITY = ROOT / "mae/hierarchy_32_64/city_indices.npy"
DEFAULT_HIERARCHY = ROOT / "mae/hierarchy_edp_32_64/hierarchy_arrays.npz"
DEFAULT_ACTIVATIONS = Path("outputs/analysis/all_city_umap_activation/image_activations.npy")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def categorical_palette(size: int) -> np.ndarray:
    sources = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colours = []
    for source in sources:
        colours.extend(source(np.linspace(0, 1, 20))[:, :3])
    return np.asarray(colours[:size])


def robust_reference(
    activations: np.ndarray,
    city_ids: np.ndarray,
    sample_per_city: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = []
    for city in range(len(CITIES)):
        indices = np.flatnonzero(city_ids == city)
        selected.append(rng.choice(indices, size=min(sample_per_city, len(indices)), replace=False))
    selected = np.concatenate(selected)
    sample = np.asarray(activations[selected], dtype=np.float32)
    median = np.median(sample, axis=0)
    mad = 1.4826 * np.median(np.abs(sample - median), axis=0)
    return median, np.maximum(mad, 1e-4), selected


def scan(
    top: np.ndarray,
    activations: np.ndarray,
    city_ids: np.ndarray,
    median: np.ndarray,
    mad: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    images, dimensions = len(top), activations.shape[1]
    city_counts = np.bincount(city_ids, minlength=len(CITIES)).astype(np.float64)
    city_weight = images / (len(CITIES) * city_counts)

    thresholds = (1, 2, 4, 8)
    z_thresholds = (1.5, 2.0, 2.5)
    coverage_counts = {value: np.empty(images, dtype=np.uint16) for value in thresholds}
    activation_counts = {value: np.empty(images, dtype=np.uint16) for value in z_thresholds}
    dominant_dimension = np.empty(images, dtype=np.uint16)
    dominant_patch_share = np.empty(images, dtype=np.float32)
    patch_entropy = np.empty(images, dtype=np.float32)

    occurrence_raw = np.zeros(dimensions, dtype=np.int64)
    occurrence_weighted = np.zeros(dimensions, dtype=np.float64)
    patch_total = np.zeros(dimensions, dtype=np.int64)
    patch_present_total = np.zeros(dimensions, dtype=np.int64)
    activation_occurrence_raw = np.zeros(dimensions, dtype=np.int64)
    activation_occurrence_weighted = np.zeros(dimensions, dtype=np.float64)
    activation_share_total = np.zeros(dimensions, dtype=np.float64)
    activation_share_present_total = np.zeros(dimensions, dtype=np.float64)
    cooccurrence_raw = np.zeros((dimensions, dimensions), dtype=np.int64)
    cooccurrence_weighted = np.zeros((dimensions, dimensions), dtype=np.float64)

    for start in range(0, images, chunk_size):
        end = min(start + chunk_size, images)
        winners = np.asarray(top[start:end], dtype=np.int64)
        rows = np.repeat(np.arange(len(winners), dtype=np.int64), winners.shape[1])
        counts = np.bincount(
            rows * dimensions + winners.ravel(), minlength=len(winners) * dimensions
        ).reshape(len(winners), dimensions)
        patch_total += counts.sum(axis=0)
        for threshold in thresholds:
            coverage_counts[threshold][start:end] = (counts >= threshold).sum(axis=1)

        present = counts >= 4
        weights = city_weight[np.asarray(city_ids[start:end], dtype=np.int64)]
        occurrence_raw += present.sum(axis=0)
        occurrence_weighted += np.sum(present * weights[:, None], axis=0)
        patch_present_total += np.sum(counts * present, axis=0)
        binary = csr_matrix(present.astype(np.int32, copy=False))
        cooccurrence_raw += (binary.T @ binary).toarray().astype(np.int64)
        weighted_binary = binary.multiply(weights[:, None])
        cooccurrence_weighted += (binary.T @ weighted_binary).toarray()

        dominant_dimension[start:end] = counts.argmax(axis=1).astype(np.uint16)
        dominant_patch_share[start:end] = counts.max(axis=1) / winners.shape[1]
        proportions = counts / winners.shape[1]
        patch_entropy[start:end] = (
            -np.sum(np.where(proportions > 0, proportions * np.log(proportions + 1e-12), 0), axis=1)
            / np.log(dimensions)
        )

        image_scores = np.asarray(activations[start:end], dtype=np.float32)
        z = (image_scores - median) / mad
        for threshold in z_thresholds:
            activation_counts[threshold][start:end] = (z >= threshold).sum(axis=1)
        strong = z >= 2.0
        activation_occurrence_raw += strong.sum(axis=0)
        activation_occurrence_weighted += np.sum(strong * weights[:, None], axis=0)
        excess = np.maximum(z - 2.0, 0.0)
        excess /= np.maximum(excess.sum(axis=1, keepdims=True), 1e-12)
        activation_share_total += excess.sum(axis=0)
        activation_share_present_total += np.sum(excess * strong, axis=0)

        if start == 0 or end == images or (start // chunk_size) % 10 == 0:
            print(f"scan {end:,}/{images:,}", flush=True)

    return {
        "coverage_counts": coverage_counts,
        "activation_counts": activation_counts,
        "dominant_dimension": dominant_dimension,
        "dominant_patch_share": dominant_patch_share,
        "patch_entropy": patch_entropy,
        "occurrence_raw": occurrence_raw,
        "occurrence_weighted": occurrence_weighted,
        "patch_total": patch_total,
        "patch_present_total": patch_present_total,
        "activation_occurrence_raw": activation_occurrence_raw,
        "activation_occurrence_weighted": activation_occurrence_weighted,
        "activation_share_total": activation_share_total,
        "activation_share_present_total": activation_share_present_total,
        "cooccurrence_raw": cooccurrence_raw,
        "cooccurrence_weighted": cooccurrence_weighted,
    }


def association_matrices(
    raw: np.ndarray,
    weighted: np.ndarray,
    total_weight: float,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.diag(raw).astype(np.float64)
    union = support[:, None] + support[None, :] - raw
    jaccard = raw / np.maximum(union, 1.0)
    probability = weighted / total_weight
    marginal = np.diag(weighted) / total_weight
    expected = marginal[:, None] * marginal[None, :]
    ppmi = np.maximum(np.log2((probability + 1e-12) / (expected + 1e-12)), 0.0)
    ppmi[raw < minimum_support] = 0.0
    np.fill_diagonal(jaccard, 0.0)
    np.fill_diagonal(ppmi, 0.0)
    return jaccard, ppmi, marginal


def ordered_subset(ppmi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = ppmi[np.ix_(indices, indices)]
    distances = pdist(values, metric="cosine")
    distances[~np.isfinite(distances)] = 1.0
    return indices[leaves_list(linkage(distances, method="average"))]


def dimension_statistics(
    stats: dict[str, np.ndarray],
    fine: np.ndarray,
    coarse: np.ndarray,
    images: int,
) -> pd.DataFrame:
    occurrence = stats["occurrence_raw"]
    activation_occurrence = stats["activation_occurrence_raw"]
    conditional_patch = stats["patch_present_total"] / np.maximum(occurrence * 196, 1)
    conditional_activation = (
        stats["activation_share_present_total"] / np.maximum(activation_occurrence, 1)
    )
    return pd.DataFrame(
        {
            "dimension_id": [f"D{x:03d}" for x in range(512)],
            "dimension_index": np.arange(512),
            "fine_cluster": [f"F{x:03d}" for x in fine],
            "coarse_cluster": [f"C{x:03d}" for x in coarse],
            "image_occurrence_count": occurrence,
            "image_prevalence": occurrence / images,
            "city_balanced_prevalence": stats["occurrence_weighted"] / images,
            "mean_patch_share_all_images": stats["patch_total"] / (images * 196),
            "mean_patch_share_when_present": conditional_patch,
            "strong_activation_count": activation_occurrence,
            "strong_activation_prevalence": activation_occurrence / images,
            "city_balanced_strong_activation_prevalence": (
                stats["activation_occurrence_weighted"] / images
            ),
            "mean_activation_share_all_images": stats["activation_share_total"] / images,
            "mean_activation_share_when_strong": conditional_activation,
        }
    )


def threshold_summary(stats: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for threshold, values in stats["coverage_counts"].items():
        rows.append(
            {
                "definition": "patch_winner_count",
                "threshold": threshold,
                "mean_dimensions": values.mean(),
                "q05": np.quantile(values, 0.05),
                "q25": np.quantile(values, 0.25),
                "median": np.median(values),
                "q75": np.quantile(values, 0.75),
                "q95": np.quantile(values, 0.95),
            }
        )
    for threshold, values in stats["activation_counts"].items():
        rows.append(
            {
                "definition": "robust_activation_z",
                "threshold": threshold,
                "mean_dimensions": values.mean(),
                "q05": np.quantile(values, 0.05),
                "q25": np.quantile(values, 0.25),
                "median": np.median(values),
                "q75": np.quantile(values, 0.75),
                "q95": np.quantile(values, 0.95),
            }
        )
    return pd.DataFrame(rows)


def top_pairs(
    raw: np.ndarray,
    weighted: np.ndarray,
    ppmi: np.ndarray,
    total_weight: float,
) -> pd.DataFrame:
    left, right = np.triu_indices(raw.shape[0], 1)
    frame = pd.DataFrame(
        {
            "dimension_a": [f"D{x:03d}" for x in left],
            "dimension_b": [f"D{x:03d}" for x in right],
            "cooccurrence_count": raw[left, right],
            "city_balanced_joint_probability": weighted[left, right] / total_weight,
            "ppmi": ppmi[left, right],
        }
    )
    return frame.sort_values(["ppmi", "cooccurrence_count"], ascending=False)


def main_figure(
    output: Path,
    stats: dict[str, np.ndarray],
    dimensions: pd.DataFrame,
    ppmi: np.ndarray,
    pairs: pd.DataFrame,
    coarse: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.1))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    coverage = stats["coverage_counts"][4]
    activation = stats["activation_counts"][2.0]
    bins = np.arange(0, max(coverage.max(), activation.max()) + 2) - 0.5
    ax_a.hist(coverage, bins=bins, density=True, histtype="step", linewidth=1.2,
              color="#0072B2", label="Coverage: $n_{patch}\\geq4$")
    ax_a.hist(activation, bins=bins, density=True, histtype="step", linewidth=1.2,
              color="#D55E00", label="Activation: robust $z\\geq2$")
    ax_a.axvline(np.median(coverage), color="#0072B2", linewidth=0.7, linestyle="--")
    ax_a.axvline(np.median(activation), color="#D55E00", linewidth=0.7, linestyle="--")
    ax_a.set(xlabel="Dimensions present per image", ylabel="Density", xlim=(0, 60))
    ax_a.legend(frameon=False, fontsize=5.8)

    palette = categorical_palette(32)
    colours = palette[coarse]
    ax_b.scatter(
        100 * dimensions.city_balanced_prevalence,
        100 * dimensions.mean_patch_share_when_present,
        c=colours,
        s=8 + 35 * dimensions.strong_activation_prevalence,
        alpha=0.78,
        linewidths=0,
    )
    label_indices = np.argsort(dimensions.city_balanced_prevalence.to_numpy())[-10:]
    for index in label_indices:
        row = dimensions.iloc[index]
        ax_b.annotate(
            row.dimension_id,
            (100 * row.city_balanced_prevalence, 100 * row.mean_patch_share_when_present),
            xytext=(2, 2), textcoords="offset points", fontsize=4.8,
        )
    ax_b.set(
        xlabel="Image prevalence (%)",
        ylabel="Mean spatial share when present (%)",
    )

    top = np.argsort(dimensions.city_balanced_prevalence.to_numpy())[-30:]
    ordered = ordered_subset(ppmi, top)
    matrix = ppmi[np.ix_(ordered, ordered)]
    vmax = max(float(np.quantile(matrix[matrix > 0], 0.98)) if np.any(matrix > 0) else 1.0, 0.1)
    image = ax_c.imshow(matrix, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    labels = [f"D{x:03d}" for x in ordered]
    ax_c.set(xticks=np.arange(len(labels)), xticklabels=labels,
             yticks=np.arange(len(labels)), yticklabels=labels)
    ax_c.tick_params(axis="x", rotation=90, labelsize=3.3)
    ax_c.tick_params(axis="y", labelsize=3.3)
    fig.colorbar(image, ax=ax_c, fraction=0.035, pad=0.02, label="PPMI")

    supported = pairs[pairs.cooccurrence_count >= int(len(coverage) * 0.005)].head(15).copy()
    supported = supported.sort_values("ppmi")
    pair_labels = supported.dimension_a + "–" + supported.dimension_b
    sizes = 12 + 90 * supported.city_balanced_joint_probability / max(
        supported.city_balanced_joint_probability.max(), 1e-12
    )
    ax_d.scatter(supported.ppmi, np.arange(len(supported)), s=sizes,
                 color="#0072B2", alpha=0.8, linewidths=0)
    ax_d.set(yticks=np.arange(len(supported)), yticklabels=pair_labels,
             xlabel="Positive pointwise mutual information", ylabel="Dimension pair")
    ax_d.grid(axis="x", color="#DDDDDD", linewidth=0.4)

    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def supplementary_figures(
    directory: Path,
    stats: dict[str, np.ndarray],
    dimensions: pd.DataFrame,
    ppmi: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    # Threshold sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    coverage_thresholds = sorted(stats["coverage_counts"])
    coverage_values = [stats["coverage_counts"][x] for x in coverage_thresholds]
    axes[0].errorbar(
        coverage_thresholds,
        [np.median(x) for x in coverage_values],
        yerr=[
            [np.median(x) - np.quantile(x, 0.25) for x in coverage_values],
            [np.quantile(x, 0.75) - np.median(x) for x in coverage_values],
        ],
        marker="o", color="#0072B2", capsize=2,
    )
    axes[0].set(xlabel="Minimum winner patches", ylabel="Dimensions per image",
                xticks=coverage_thresholds)
    z_thresholds = sorted(stats["activation_counts"])
    z_values = [stats["activation_counts"][x] for x in z_thresholds]
    axes[1].errorbar(
        z_thresholds,
        [np.median(x) for x in z_values],
        yerr=[
            [np.median(x) - np.quantile(x, 0.25) for x in z_values],
            [np.quantile(x, 0.75) - np.median(x) for x in z_values],
        ],
        marker="o", color="#D55E00", capsize=2,
    )
    axes[1].set(xlabel="Robust activation threshold ($z$)", ylabel="Dimensions per image",
                xticks=z_thresholds)
    for label, ax in zip("ab", axes):
        ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    path = directory / "Fig_Image_Dimension_Threshold_Sensitivity.png"
    fig.savefig(path, dpi=400, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    # Full 512-dimensional PPMI structure, ordered by co-occurrence profiles.
    distance = pdist(ppmi, metric="cosine")
    distance[~np.isfinite(distance)] = 1.0
    order = leaves_list(linkage(distance, method="average"))
    ordered = ppmi[np.ix_(order, order)]
    positive = ordered[ordered > 0]
    vmax = max(float(np.quantile(positive, 0.99)) if len(positive) else 1.0, 0.1)
    fig, ax = plt.subplots(figsize=(6.2, 5.5), constrained_layout=True)
    image = ax.imshow(ordered, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set(xlabel="Latent dimensions (hierarchically ordered)",
           ylabel="Latent dimensions (hierarchically ordered)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="PPMI")
    path = directory / "Fig_Image_Dimension_PPMI_512.png"
    fig.savefig(path, dpi=400, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--city-indices", type=Path, default=DEFAULT_CITY)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--main-figures", type=Path, default=Path("paper/figures/main"))
    parser.add_argument(
        "--supplementary-figures", type=Path, default=Path("paper/figures/supplementary")
    )
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--sample-per-city", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_plotting()
    top = np.load(args.top, mmap_mode="r")
    city_ids = np.load(args.city_indices, mmap_mode="r")
    activations = np.load(args.activations, mmap_mode="r")
    with np.load(args.hierarchy) as hierarchy:
        fine = hierarchy["fine_labels"].astype(np.int64)
        coarse = hierarchy["coarse_labels"].astype(np.int64)
    if top.shape != (len(activations), 196) or activations.shape[1] != 512:
        raise ValueError(f"incompatible arrays: top={top.shape}, activations={activations.shape}")

    median, mad, reference_indices = robust_reference(
        activations, city_ids, args.sample_per_city, args.seed
    )
    stats = scan(top, activations, city_ids, median, mad, args.chunk_size)
    minimum_support = max(100, int(np.ceil(len(top) * 0.0005)))
    jaccard, ppmi, marginal = association_matrices(
        stats["cooccurrence_raw"], stats["cooccurrence_weighted"], len(top), minimum_support
    )
    dimensions = dimension_statistics(stats, fine, coarse, len(top))
    pairs = top_pairs(
        stats["cooccurrence_raw"], stats["cooccurrence_weighted"], ppmi, len(top)
    )

    save_csv(dimensions, args.results / "dimension_image_statistics.csv")
    save_csv(threshold_summary(stats), args.results / "image_dimension_count_distribution.csv")
    labels = [f"D{x:03d}" for x in range(512)]
    save_csv(pd.DataFrame(stats["cooccurrence_raw"], index=labels, columns=labels),
             args.results / "dimension_cooccurrence.csv", index=True)
    save_csv(pd.DataFrame(jaccard, index=labels, columns=labels),
             args.results / "dimension_cooccurrence_jaccard.csv", index=True)
    save_csv(pd.DataFrame(ppmi, index=labels, columns=labels),
             args.results / "dimension_ppmi.csv", index=True)
    save_csv(pairs, args.results / "dimension_cooccurrence_pairs.csv")

    within_city_index = np.empty(len(top), dtype=np.int32)
    for city in range(len(CITIES)):
        indices = np.flatnonzero(city_ids == city)
        within_city_index[indices] = np.arange(len(indices), dtype=np.int32)
    image_summary = pd.DataFrame(
        {
            "global_image_index": np.arange(len(top)),
            "city": [CITIES[x].split("/")[-1] for x in city_ids],
            "city_image_index": within_city_index,
            "dimensions_patch_ge4": stats["coverage_counts"][4],
            "dimensions_activation_z_ge2": stats["activation_counts"][2.0],
            "dominant_dimension": [f"D{x:03d}" for x in stats["dominant_dimension"]],
            "dominant_patch_share": stats["dominant_patch_share"],
            "patch_composition_entropy": stats["patch_entropy"],
        }
    )
    image_summary.to_parquet(args.results / "image_dimension_summary.parquet", index=False)

    main_figure(
        args.main_figures / "Fig_Image_Dimension_Coexistence.png",
        stats, dimensions, ppmi, pairs, coarse,
    )
    supplementary_figures(args.supplementary_figures, stats, dimensions, ppmi)

    report = {
        "images": len(top),
        "dimensions": 512,
        "patches_per_image": 196,
        "primary_presence": "dimension wins at least 4 patches (2.04% spatial coverage)",
        "activation_validation": "mean-top-20 image score robust z >= 2",
        "robust_reference_images": len(reference_indices),
        "minimum_pair_support": minimum_support,
        "seed": args.seed,
        "input_top1": str(args.top.resolve()),
        "input_activations": str(args.activations.resolve()),
        "input_hierarchy": str(args.hierarchy.resolve()),
    }
    (args.results / "image_dimension_coexistence_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
