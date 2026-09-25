#!/usr/bin/env python3
"""Measure visually distant but unexpectedly frequent element combinations.

The analysis uses the frozen E+D+P hierarchy and the same primary occurrence
definition as the existing multicity analysis: the eight fine elements with
the largest patch-winner counts in each directional image.

Two families of scores are exported:

1. Requested residual score
       U_ij = max((O_ij - E_ij) / sqrt(E_ij), 0) * D_ij
2. Effect-size score
       VBI_ij = max(log((O_ij + .5) / (E_ij + .5)), 0) * D_ij

For cities, the script exports both the descriptive difference in residuals
and a preferred city-versus-rest difference in log odds ratios.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
from scipy.stats import hypergeom, norm, rankdata, spearmanr


DEFAULT_DATA_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)
DEFAULT_SOURCE_HIERARCHY = DEFAULT_DATA_ROOT / "mae" / "hierarchy_32_64"
DEFAULT_MAIN_HIERARCHY = DEFAULT_DATA_ROOT / "mae" / "hierarchy_edp_32_64"
DEFAULT_RESULTS = Path("results/atypical_cooccurrence")
DEFAULT_FIGURES = Path("figures/atypical_cooccurrence")


def save_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input shape."""
    values = np.asarray(p_values, dtype=np.float64)
    flat = values.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    factors = len(flat) / np.arange(1, len(flat) + 1, dtype=np.float64)
    adjusted_ranked = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted.reshape(values.shape)


def percentile_scale(values: np.ndarray) -> np.ndarray:
    """Map pair values monotonically to [0, 1], averaging ranks for ties."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values)
    ranks = rankdata(values, method="average")
    return (ranks - 1.0) / (len(values) - 1.0)


def load_semantic_names(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {
        row["category_id"]: row.get("name_en", row["category_id"])
        for row in payload.get("categories", [])
    }


def element_distances(
    hierarchy_dir: Path,
    fine_clusters_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build 64-element distances from the frozen E+D+P hierarchy.

    The primary distance is the empirical-percentile distance between fine
    cluster centroids in the 32-D spectral embedding used by Ward clustering.
    Cophenetic and direct fused-affinity distances are retained for sensitivity.
    """
    arrays = np.load(hierarchy_dir / "hierarchy_arrays.npz")
    fine = arrays["fine_labels"].astype(np.int64)
    coarse = arrays["coarse_labels"].astype(np.int64)
    embedding = arrays["spectral_embedding"].astype(np.float64)
    fused = arrays["fused_similarity"].astype(np.float64)
    stability = arrays["fine_stability"].astype(np.float64)
    linkage = np.load(hierarchy_dir / "ward_linkage.npy").astype(np.float64)

    clusters = pd.read_csv(fine_clusters_path)
    clusters["fine_index"] = clusters["cluster_id"].str[1:].astype(int)
    clusters = clusters.sort_values("fine_index")
    expected = np.arange(64)
    if not np.array_equal(clusters["fine_index"].to_numpy(), expected):
        raise ValueError("fine_clusters.csv does not contain exactly F000--F063")
    medoids = clusters["medoid_dimension"].to_numpy(np.int64)

    centroids = np.vstack([embedding[fine == i].mean(axis=0) for i in range(64)])
    centroid_raw = np.sqrt(
        np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    )
    cophenetic_all = squareform(cophenet(linkage))
    cophenetic_raw = cophenetic_all[np.ix_(medoids, medoids)]
    fused_mean_similarity = np.zeros((64, 64), dtype=np.float64)
    for i in range(64):
        left = np.flatnonzero(fine == i)
        for j in range(i + 1, 64):
            right = np.flatnonzero(fine == j)
            value = float(fused[np.ix_(left, right)].mean())
            fused_mean_similarity[i, j] = fused_mean_similarity[j, i] = value

    upper = np.triu_indices(64, 1)
    centroid_distance = percentile_scale(centroid_raw[upper])
    cophenetic_distance = percentile_scale(cophenetic_raw[upper])
    # Low cross-cluster affinity means large visual distance.
    fused_distance = percentile_scale(-fused_mean_similarity[upper])

    parent = np.empty(64, dtype=np.int64)
    element_stability = np.empty(64, dtype=np.float64)
    for i in range(64):
        members = np.flatnonzero(fine == i)
        parents = np.unique(coarse[members])
        if len(parents) != 1:
            raise ValueError(f"fine cluster F{i:03d} is not nested in one coarse cluster")
        parent[i] = parents[0]
        element_stability[i] = stability[members].mean()

    rows = []
    primary_matrix = np.zeros((64, 64), dtype=np.float64)
    for k, (i, j) in enumerate(zip(*upper)):
        primary_matrix[i, j] = primary_matrix[j, i] = centroid_distance[k]
        rows.append(
            {
                "element_a": f"F{i:03d}",
                "element_b": f"F{j:03d}",
                "parent_a": f"C{parent[i]:03d}",
                "parent_b": f"C{parent[j]:03d}",
                "same_coarse_family": bool(parent[i] == parent[j]),
                "centroid_distance_raw": centroid_raw[i, j],
                "visual_distance": centroid_distance[k],
                "visual_similarity": 1.0 - centroid_distance[k],
                "cophenetic_distance_raw": cophenetic_raw[i, j],
                "cophenetic_distance_percentile": cophenetic_distance[k],
                "fused_mean_similarity": fused_mean_similarity[i, j],
                "fused_distance_percentile": fused_distance[k],
                "stability_a": element_stability[i],
                "stability_b": element_stability[j],
                "pair_stability": math.sqrt(element_stability[i] * element_stability[j]),
            }
        )
    return pd.DataFrame(rows), primary_matrix


def accumulate_occurrence(
    top_path: Path,
    city_ids_path: Path,
    fine_labels: np.ndarray,
    city_count: int,
    top_n: int = 8,
    batch_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate global and per-city 64x64 co-occurrence matrices."""
    top = np.load(top_path, mmap_mode="r")
    city_ids = np.load(city_ids_path, mmap_mode="r")
    if len(top) != len(city_ids):
        raise ValueError(f"top/city length mismatch: {len(top)} != {len(city_ids)}")

    city_cooccurrence = np.zeros((city_count, 64, 64), dtype=np.int64)
    city_images = np.bincount(np.asarray(city_ids, dtype=np.int64), minlength=city_count)
    for start in range(0, len(top), batch_size):
        block = np.asarray(top[start : start + batch_size], dtype=np.int64)
        block_city = np.asarray(city_ids[start : start + len(block)], dtype=np.int64)
        labels = fine_labels[block]
        rows = np.repeat(np.arange(len(block), dtype=np.int64), labels.shape[1])
        keys = rows * 64 + labels.ravel()
        counts = np.bincount(keys, minlength=len(block) * 64).reshape(len(block), 64)
        selected = np.argpartition(-counts, min(top_n, 64) - 1, axis=1)[:, :top_n]
        occurrence = np.zeros((len(block), 64), dtype=np.int8)
        row_index = np.arange(len(block))[:, None]
        occurrence[row_index, selected] = (counts[row_index, selected] > 0).astype(np.int8)
        for city in np.unique(block_city):
            sub = occurrence[block_city == city].astype(np.int64, copy=False)
            city_cooccurrence[city] += sub.T @ sub
        if start == 0 or start + len(block) == len(top) or start % 50000 == 0:
            print(f"occurrence {start + len(block):,}/{len(top):,}", flush=True)

    global_cooccurrence = city_cooccurrence.sum(axis=0)
    return global_cooccurrence, city_cooccurrence, city_images


def pair_components(cooccurrence: np.ndarray, images: int) -> dict[str, np.ndarray]:
    support = np.diag(cooccurrence).astype(np.float64)
    observed = cooccurrence.astype(np.float64)
    expected = support[:, None] * support[None, :] / max(images, 1)
    pearson = (observed - expected) / np.sqrt(np.maximum(expected, 1e-12))
    variance = (
        expected
        * (1.0 - support[:, None] / max(images, 1))
        * (1.0 - support[None, :] / max(images, 1))
        * images
        / max(images - 1, 1)
    )
    adjusted = (observed - expected) / np.sqrt(np.maximum(variance, 1e-12))
    log_lift = np.log((observed + 0.5) / (expected + 0.5))
    return {
        "support": support,
        "observed": observed,
        "expected": expected,
        "pearson": pearson,
        "adjusted": adjusted,
        "log_lift": log_lift,
    }


def global_statistics(
    cooccurrence: np.ndarray,
    images: int,
    distances: pd.DataFrame,
    names: dict[str, str],
    minimum_support: int,
) -> pd.DataFrame:
    comp = pair_components(cooccurrence, images)
    upper = np.triu_indices(64, 1)
    p_values = hypergeom.sf(
        comp["observed"][upper] - 1,
        images,
        comp["support"][upper[0]],
        comp["support"][upper[1]],
    )
    q_values = bh_adjust(p_values)
    rows = distances.copy()
    rows.insert(2, "name_a", rows["element_a"].map(names))
    rows.insert(3, "name_b", rows["element_b"].map(names))
    rows["images"] = images
    rows["support_a"] = comp["support"][upper[0]].astype(np.int64)
    rows["support_b"] = comp["support"][upper[1]].astype(np.int64)
    rows["observed"] = comp["observed"][upper].astype(np.int64)
    rows["expected"] = comp["expected"][upper]
    rows["pearson_residual"] = comp["pearson"][upper]
    rows["adjusted_residual"] = comp["adjusted"][upper]
    rows["log_lift"] = comp["log_lift"][upper]
    rows["p_value_upper"] = p_values
    rows["q_value_bh"] = q_values
    rows["passes_support"] = rows["observed"] >= minimum_support
    rows["significant_positive"] = (
        rows["passes_support"] & (rows["q_value_bh"] < 0.05) & (rows["log_lift"] > 0)
    )
    rows["unexpected_score"] = (
        rows["pearson_residual"].clip(lower=0) * rows["visual_distance"]
    )
    rows["unexpected_score_adjusted"] = (
        rows["adjusted_residual"].clip(lower=0) * rows["visual_distance"]
    )
    rows["visual_bridge_index"] = (
        rows["log_lift"].clip(lower=0) * rows["visual_distance"]
    )
    for suffix, column in (
        ("cophenetic", "cophenetic_distance_percentile"),
        ("fused", "fused_distance_percentile"),
    ):
        rows[f"unexpected_score_{suffix}"] = rows["pearson_residual"].clip(lower=0) * rows[column]
        rows[f"visual_bridge_index_{suffix}"] = rows["log_lift"].clip(lower=0) * rows[column]
    return rows


def table_cells(cooccurrence: np.ndarray, images: int, i: int, j: int) -> np.ndarray:
    n11 = float(cooccurrence[i, j])
    n10 = float(cooccurrence[i, i] - n11)
    n01 = float(cooccurrence[j, j] - n11)
    n00 = float(images - n11 - n10 - n01)
    return np.array([n11, n10, n01, n00], dtype=np.float64)


def city_statistics(
    city_cooccurrence: np.ndarray,
    city_images: np.ndarray,
    global_cooccurrence: np.ndarray,
    global_images: int,
    global_frame: pd.DataFrame,
    city_keys: list[str],
    minimum_city_support: int,
) -> pd.DataFrame:
    global_residual = pair_components(global_cooccurrence, global_images)["pearson"]
    upper = np.triu_indices(64, 1)
    rows = []
    for city_index, city_key in enumerate(city_keys):
        city_n = int(city_images[city_index])
        city_cooc = city_cooccurrence[city_index]
        rest_n = global_images - city_n
        rest_cooc = global_cooccurrence - city_cooc
        city_comp = pair_components(city_cooc, city_n)
        p_values = []
        temporary = []
        for pair_index, (i, j) in enumerate(zip(*upper)):
            city_cells = table_cells(city_cooc, city_n, i, j)
            rest_cells = table_cells(rest_cooc, rest_n, i, j)
            city_corrected = city_cells + 0.5
            rest_corrected = rest_cells + 0.5
            city_log_odds = math.log(
                city_corrected[0] * city_corrected[3]
                / (city_corrected[1] * city_corrected[2])
            )
            rest_log_odds = math.log(
                rest_corrected[0] * rest_corrected[3]
                / (rest_corrected[1] * rest_corrected[2])
            )
            delta = city_log_odds - rest_log_odds
            standard_error = math.sqrt(
                np.reciprocal(city_corrected).sum() + np.reciprocal(rest_corrected).sum()
            )
            z_value = delta / standard_error
            p_value = float(norm.sf(z_value))
            p_values.append(p_value)
            base = global_frame.iloc[pair_index]
            residual = float(city_comp["pearson"][i, j])
            delta_residual = residual - float(global_residual[i, j])
            temporary.append(
                {
                    "city_key": city_key,
                    "city": city_key.split("/")[-1],
                    "element_a": base["element_a"],
                    "element_b": base["element_b"],
                    "name_a": base["name_a"],
                    "name_b": base["name_b"],
                    "parent_a": base["parent_a"],
                    "parent_b": base["parent_b"],
                    "same_coarse_family": base["same_coarse_family"],
                    "visual_distance": base["visual_distance"],
                    "pair_stability": base["pair_stability"],
                    "city_images": city_n,
                    "rest_images": rest_n,
                    "city_n11": int(city_cells[0]),
                    "city_n10": int(city_cells[1]),
                    "city_n01": int(city_cells[2]),
                    "city_n00": int(city_cells[3]),
                    "rest_n11": int(rest_cells[0]),
                    "rest_n10": int(rest_cells[1]),
                    "rest_n01": int(rest_cells[2]),
                    "rest_n00": int(rest_cells[3]),
                    "city_expected": city_comp["expected"][i, j],
                    "city_pearson_residual": residual,
                    "global_pearson_residual": global_residual[i, j],
                    "delta_residual_vs_global": delta_residual,
                    "requested_city_score": max(delta_residual, 0.0) * base["visual_distance"],
                    "city_log_odds": city_log_odds,
                    "rest_log_odds": rest_log_odds,
                    "delta_log_odds": delta,
                    "delta_log_odds_se": standard_error,
                    "z_city_vs_rest": z_value,
                    "p_value_upper": p_value,
                    "city_atypicality_index": max(delta, 0.0) * base["visual_distance"],
                }
            )
        within_q = bh_adjust(np.asarray(p_values))
        for row, q_value in zip(temporary, within_q):
            row["q_value_within_city"] = q_value
        rows.extend(temporary)

    frame = pd.DataFrame(rows)
    frame["q_value_global_bh"] = bh_adjust(frame["p_value_upper"].to_numpy())
    frame["passes_city_support"] = frame["city_n11"] >= minimum_city_support
    frame["significant_city_specific"] = (
        frame["passes_city_support"]
        & (frame["q_value_global_bh"] < 0.05)
        & (frame["delta_log_odds"] > 0)
    )
    return frame


def plot_global(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    supported = frame[frame["passes_support"]].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    points = ax.scatter(
        supported["visual_distance"],
        supported["log_lift"],
        c=supported["unexpected_score"],
        s=np.clip(np.sqrt(supported["observed"]) * 3, 8, 100),
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
    )
    top = supported[supported["significant_positive"]].nlargest(12, "unexpected_score")
    for _, row in top.iterrows():
        ax.annotate(
            f"{row.element_a}+{row.element_b}",
            (row.visual_distance, row.log_lift),
            xytext=(3, 3), textcoords="offset points", fontsize=6,
        )
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set(
        xlabel="E+D+P visual distance (pairwise percentile)",
        ylabel="log observed / expected",
        title="Global visually distant co-occurrence",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(points, ax=ax, label="Requested residual × distance score")
    fig.savefig(output / "global_unexpected_cooccurrence.png", dpi=300)
    fig.savefig(output / "global_unexpected_cooccurrence.pdf")
    plt.close(fig)


def plot_city(frame: pd.DataFrame, output: Path) -> None:
    significant = frame[frame["significant_city_specific"]].copy()
    if significant.empty:
        return
    significant["pair"] = significant["element_a"] + "+" + significant["element_b"]
    chosen = (
        significant.groupby("pair")["city_atypicality_index"]
        .max().nlargest(30).index
    )
    matrix = (
        frame[frame["element_a"].add("+").add(frame["element_b"]).isin(chosen)]
        .assign(pair=lambda x: x["element_a"] + "+" + x["element_b"])
        .pivot(index="pair", columns="city", values="city_atypicality_index")
        .reindex(chosen)
    )
    width = max(10, 0.32 * matrix.shape[1])
    height = max(7, 0.23 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index, fontsize=6)
    ax.set_title("City-specific atypical visual combinations")
    fig.colorbar(image, ax=ax, label="City atypicality index", fraction=0.025, pad=0.02)
    fig.savefig(output / "city_atypicality_heatmap.png", dpi=300)
    fig.savefig(output / "city_atypicality_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-hierarchy", type=Path, default=DEFAULT_SOURCE_HIERARCHY)
    parser.add_argument("--main-hierarchy", type=Path, default=DEFAULT_MAIN_HIERARCHY)
    parser.add_argument("--base-results", type=Path, default=Path("results"))
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--minimum-global-support", type=int, default=None)
    parser.add_argument("--minimum-city-support", type=int, default=50)
    args = parser.parse_args()

    report = json.loads((args.source_hierarchy / "scan_report.json").read_text())
    city_keys = list(report["cities"])
    images = int(report["images"])
    minimum_global_support = args.minimum_global_support
    if minimum_global_support is None:
        minimum_global_support = max(100, int(math.ceil(0.0005 * images)))

    print("[1/5] E+D+P element distances", flush=True)
    distances, distance_matrix = element_distances(
        args.main_hierarchy, args.base_results / "fine_clusters.csv"
    )
    names = load_semantic_names(args.base_results / "semantic_labels_fine.json")
    distances.insert(2, "name_a", distances["element_a"].map(names))
    distances.insert(3, "name_b", distances["element_b"].map(names))
    save_csv(distances, args.results / "visual_element_distance.csv")
    # global_statistics inserts names itself; keep one copy of the structural columns.
    distance_input = distances.drop(columns=["name_a", "name_b"])

    print("[2/5] top-8 image occurrence and city counts", flush=True)
    arrays = np.load(args.main_hierarchy / "hierarchy_arrays.npz")
    fine_labels = arrays["fine_labels"].astype(np.int64)
    global_cooc, city_cooc, city_images = accumulate_occurrence(
        args.source_hierarchy / "top1_dimensions.npy",
        args.source_hierarchy / "city_indices.npy",
        fine_labels,
        len(city_keys),
        top_n=args.top_n,
        batch_size=args.batch_size,
    )
    del distance_matrix

    existing_path = args.base_results / "global_element_cooccurrence.csv"
    if existing_path.is_file() and args.top_n == 8:
        existing = pd.read_csv(existing_path, index_col=0).to_numpy(np.int64)
        if not np.array_equal(existing, global_cooc):
            difference = int(np.abs(existing - global_cooc).max())
            raise ValueError(f"recomputed co-occurrence does not match existing result; max diff={difference}")
        print("validated against results/global_element_cooccurrence.csv", flush=True)

    print("[3/5] global residual, lift, and significance", flush=True)
    global_frame = global_statistics(
        global_cooc, images, distance_input, names, minimum_global_support
    )
    save_csv(global_frame, args.results / "global_pair_statistics.csv")
    global_top = (
        global_frame[global_frame["significant_positive"]]
        .sort_values("unexpected_score", ascending=False)
        .head(100)
    )
    save_csv(global_top, args.results / "global_top_pairs.csv")

    print("[4/5] city-versus-rest atypicality", flush=True)
    city_frame = city_statistics(
        city_cooc, city_images, global_cooc, images, global_frame,
        city_keys, args.minimum_city_support,
    )
    save_csv(city_frame, args.results / "city_pair_statistics.csv")
    city_top = (
        city_frame[city_frame["significant_city_specific"]]
        .sort_values(["city", "city_atypicality_index"], ascending=[True, False])
        .groupby("city", sort=False)
        .head(20)
    )
    save_csv(city_top, args.results / "city_top_atypical_pairs.csv")

    print("[5/5] figures and report", flush=True)
    plot_global(global_frame, args.figures)
    plot_city(city_frame, args.figures)
    eligible = global_frame[global_frame["significant_positive"]]
    sensitivity = {}
    for column in ("unexpected_score_cophenetic", "unexpected_score_fused"):
        correlation = spearmanr(
            eligible["unexpected_score"], eligible[column]
        ).statistic if len(eligible) > 1 else float("nan")
        sensitivity[column] = float(correlation)
    report_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": images,
        "cities": len(city_keys),
        "elements": 64,
        "pairs": 2016,
        "occurrence_definition": f"top-{args.top_n} fine elements by patch-winner count per directional image",
        "visual_distance": "percentile-ranked Euclidean distance between fine-element centroids in the frozen E+D+P spectral embedding",
        "minimum_global_support": minimum_global_support,
        "minimum_city_support": args.minimum_city_support,
        "global_significant_positive_pairs": int(global_frame["significant_positive"].sum()),
        "city_significant_records": int(city_frame["significant_city_specific"].sum()),
        "cities_with_significant_pairs": int(city_top["city"].nunique()),
        "distance_sensitivity_spearman": sensitivity,
        "requested_score": "max(Pearson residual, 0) * visual distance",
        "recommended_global_effect_score": "max(log observed/expected, 0) * visual distance",
        "recommended_city_score": "max(log OR city - log OR rest, 0) * visual distance",
        "multiple_testing": "BH FDR; global pairs separately and all city-pair interaction tests jointly",
        "source_hierarchy": str(args.main_hierarchy.resolve()),
    }
    save_json(report_payload, args.results / "report.json")
    print(json.dumps(report_payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
