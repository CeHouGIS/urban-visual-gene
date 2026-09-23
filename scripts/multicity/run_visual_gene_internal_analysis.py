#!/usr/bin/env python3
"""Run the internal-data Urban Visual Gene experiments.

This script deliberately constructs the primary hierarchy from E+D+P only.
City profiles (C) and activation context/co-occurrence (Q) are used as
downstream variables and clustering ablations, never as inputs to the primary
visual vocabulary.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.manifold import MDS, spectral_embedding
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
    r2_score,
)

from scripts.multicity.build_feature_mae_hierarchy import (
    _accumulate_statistics,
    canonical_labels,
    cluster_jaccard,
    ppmi_context,
    rank_fuse,
    row_cosine,
)
from scripts.multicity.config import CITIES, SEED, city_slug


DEFAULT_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)
DEFAULT_SOURCE_HIERARCHY = DEFAULT_ROOT / "mae" / "hierarchy_32_64"
DEFAULT_MAIN_HIERARCHY = DEFAULT_ROOT / "mae" / "hierarchy_edp_32_64"
DEFAULT_RESULTS = Path("results")
DEFAULT_FIGURES = Path("figures")
MAIN_VIEWS = ("E", "D", "P")
VIEW_CONFIGS = (
    ("E",),
    ("E", "D"),
    ("E", "D", "P"),
    ("E", "D", "P", "Q"),
    ("E", "D", "P", "C"),
    ("E", "D", "P", "C", "Q"),
)
RESOLUTIONS = (16, 24, 32, 48, 64, 80, 96, 128)
EARTH_RADIUS_M = 6_378_137.0


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def provenance(args: argparse.Namespace) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "data_root": str(args.data_root.resolve()),
        "source_hierarchy": str(args.source_hierarchy.resolve()),
        "main_hierarchy": str(args.main_hierarchy.resolve()),
        "results_dir": str(args.results.resolve()),
        "figures_dir": str(args.figures.resolve()),
        "model_checkpoint": str((args.data_root / "mae" / "model_best.pt").resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "main_clustering_views": list(MAIN_VIEWS),
        "cooccurrence_primary_definition": "top-8 fine clusters by patch-winner prevalence per image",
        "cooccurrence_min_support": "max(100 images, 0.05% of images)",
        "primary_spatial_grid_m": args.grid_size,
        "primary_min_panoramas": args.min_panoramas,
        "moran_permutations": args.moran_permutations,
    }


def normalized_entropy(labels: np.ndarray) -> float:
    sizes = np.bincount(labels)
    p = sizes[sizes > 0] / sizes.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(sizes)))


def cluster_metrics(embedding: np.ndarray, labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    return {
        "n_clusters": int(len(sizes)),
        "silhouette": float(silhouette_score(embedding, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "normalized_size_entropy": normalized_entropy(labels),
        "minimum_cluster_size": int(sizes.min()),
        "maximum_cluster_size": int(sizes.max()),
        "median_cluster_size": float(np.median(sizes)),
    }


def build_embedding_tree(view_matrices: dict[str, np.ndarray], names: Iterable[str], seed: int):
    names = tuple(names)
    fused = rank_fuse([view_matrices[name] for name in names], neighbours=30)
    graph = fused.copy()
    np.fill_diagonal(graph, 1e-6)
    embedding = spectral_embedding(
        graph, n_components=32, random_state=seed, drop_first=False
    ).astype(np.float32)
    tree = linkage(embedding, method="ward")
    return fused, embedding, tree


def cut_tree(tree: np.ndarray, clusters: int) -> np.ndarray:
    return canonical_labels(fcluster(tree, clusters, criterion="maxclust"))


def medoid(features: np.ndarray, fused: np.ndarray) -> int:
    sub = fused[np.ix_(features, features)]
    return int(features[np.argmax(sub.sum(axis=1))])


def representative_files(data_root: Path, feature: int) -> tuple[str, str]:
    feature_dir = data_root / "heatmaps" / f"feature_{feature:03d}"
    examples = feature_dir / "examples.json"
    files = []
    if examples.is_file():
        for row in json.loads(examples.read_text()):
            files.append(str(feature_dir / row["file"]))
    return json.dumps(files, ensure_ascii=False), str(feature_dir / "contact_sheet.jpg")


def export_vocabulary(
    data_root: Path,
    hierarchy_dir: Path,
    results_dir: Path,
    fused: np.ndarray,
    embedding: np.ndarray,
    tree: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    coarse_stability: np.ndarray,
    fine_stability: np.ndarray,
    support: np.ndarray,
    position: np.ndarray,
) -> None:
    hierarchy_dir.mkdir(parents=True, exist_ok=True)
    membership_rows = []
    fine_rows = []
    coarse_rows = []
    branches = []
    for coarse_id in range(32):
        coarse_features = np.flatnonzero(coarse == coarse_id)
        coarse_medoid = medoid(coarse_features, fused)
        children = []
        child_ids = sorted(np.unique(fine[coarse_features]).tolist())
        for fine_id in child_ids:
            features = np.flatnonzero(fine == fine_id)
            fine_medoid = medoid(features, fused)
            image_files, contact_sheet = representative_files(data_root, fine_medoid)
            fine_rows.append(
                {
                    "cluster_id": f"F{fine_id:03d}",
                    "parent_cluster_id": f"C{coarse_id:03d}",
                    "member_dimensions": json.dumps(features.tolist()),
                    "medoid_dimension": fine_medoid,
                    "cluster_size": len(features),
                    "mean_split_sample_jaccard": float(fine_stability[features].mean()),
                    "representative_images": image_files,
                    "representative_activation_maps": contact_sheet,
                }
            )
            children.append(
                {
                    "id": f"F{fine_id:03d}",
                    "n": len(features),
                    "medoid_feature": fine_medoid,
                    "stability": float(fine_stability[features].mean()),
                    "features": features.tolist(),
                }
            )
        image_files, contact_sheet = representative_files(data_root, coarse_medoid)
        coarse_rows.append(
            {
                "cluster_id": f"C{coarse_id:03d}",
                "parent_cluster_id": "ROOT",
                "member_dimensions": json.dumps(coarse_features.tolist()),
                "medoid_dimension": coarse_medoid,
                "cluster_size": len(coarse_features),
                "fine_children": json.dumps([f"F{x:03d}" for x in child_ids]),
                "mean_split_sample_jaccard": float(coarse_stability[coarse_features].mean()),
                "representative_images": image_files,
                "representative_activation_maps": contact_sheet,
            }
        )
        branches.append(
            {
                "id": f"C{coarse_id:03d}",
                "n": len(coarse_features),
                "medoid_feature": coarse_medoid,
                "stability": float(coarse_stability[coarse_features].mean()),
                "children": children,
            }
        )
    for feature in range(len(coarse)):
        membership_rows.append(
            {
                "feature_id": feature,
                "coarse_cluster": f"C{int(coarse[feature]):03d}",
                "fine_cluster": f"F{int(fine[feature]):03d}",
                "coarse_split_sample_jaccard": float(coarse_stability[feature]),
                "fine_split_sample_jaccard": float(fine_stability[feature]),
                "top1_patch_support": int(position[feature].sum()),
                "image_support": int(support[feature]),
            }
        )
    save_csv(pd.DataFrame(fine_rows), results_dir / "fine_clusters.csv")
    save_csv(pd.DataFrame(coarse_rows), results_dir / "coarse_clusters.csv")
    membership = pd.DataFrame(membership_rows)
    save_csv(membership, results_dir / "cluster_membership.csv")
    save_csv(membership.rename(columns={
        "coarse_split_sample_jaccard": "coarse_stability",
        "fine_split_sample_jaccard": "fine_stability",
    }), hierarchy_dir / "feature_hierarchy.csv")
    atomic_json(hierarchy_dir / "taxonomy.json", {"id": "ROOT", "n": 512, "children": branches})
    np.save(hierarchy_dir / "ward_linkage.npy", tree.astype(np.float32))
    np.savez_compressed(
        hierarchy_dir / "hierarchy_arrays.npz",
        fused_similarity=fused.astype(np.float16),
        spectral_embedding=embedding,
        coarse_labels=coarse,
        fine_labels=fine,
        coarse_stability=coarse_stability,
        fine_stability=fine_stability,
    )


def view_ablation_and_hierarchy(args: argparse.Namespace):
    print("[1/8] accumulating latent-dimension statistics", flush=True)
    source = args.source_hierarchy
    top = np.load(source / "top1_dimensions.npy", mmap_mode="r")
    city_ids = np.load(source / "city_indices.npy", mmap_mode="r")
    report = json.loads((source / "scan_report.json").read_text())
    width = int(report["width"])
    position, city, cooc, support, halves = _accumulate_statistics(
        top, city_ids, width, len(CITIES)
    )
    encoder = np.load(source / "encoder_directions.npy")
    decoder = np.load(source / "decoder_directions.npy")
    full_context = ppmi_context(cooc, support, len(top))
    views = {
        "E": row_cosine(encoder),
        "D": row_cosine(decoder),
        "P": row_cosine(position),
        "C": row_cosine(city),
        "Q": row_cosine(full_context),
    }
    half_views = []
    for half in halves:
        context = ppmi_context(half["cooccurrence"], half["support"], half["images"])
        half_views.append(
            {
                "E": views["E"],
                "D": views["D"],
                "P": row_cosine(half["position"]),
                "C": row_cosine(half["city"]),
                "Q": row_cosine(context),
            }
        )

    ablation_rows = []
    solutions = {}
    main_payload = None
    for config in VIEW_CONFIGS:
        print(f"  clustering views={'+'.join(config)}", flush=True)
        fused, embedding, tree = build_embedding_tree(views, config, args.seed)
        alternates = []
        for parity in range(2):
            _, half_embedding, half_tree = build_embedding_tree(
                half_views[parity], config, args.seed + parity + 1
            )
            alternates.append((half_embedding, half_tree))
        for clusters in (32, 64):
            labels = cut_tree(tree, clusters)
            half_labels = [cut_tree(x[1], clusters) for x in alternates]
            metrics = cluster_metrics(embedding, labels)
            metrics.update(
                {
                    "view_configuration": "+".join(config),
                    "is_primary": config == MAIN_VIEWS,
                    "split_sample_ari": float(np.mean([
                        adjusted_rand_score(labels, value) for value in half_labels
                    ])),
                    "split_sample_nmi": float(np.mean([
                        normalized_mutual_info_score(labels, value) for value in half_labels
                    ])),
                    "split_sample_mean_jaccard": float(np.mean([
                        cluster_jaccard(labels, value).mean() for value in half_labels
                    ])),
                }
            )
            ablation_rows.append(metrics)
            solutions[("+".join(config), clusters)] = labels
        if config == MAIN_VIEWS:
            coarse = solutions[("E+D+P", 32)]
            fine = solutions[("E+D+P", 64)]
            half_coarse = [cut_tree(x[1], 32) for x in alternates]
            half_fine = [cut_tree(x[1], 64) for x in alternates]
            coarse_stability = np.mean(
                [cluster_jaccard(coarse, x) for x in half_coarse], axis=0
            )
            fine_stability = np.mean(
                [cluster_jaccard(fine, x) for x in half_fine], axis=0
            )
            main_payload = (
                fused, embedding, tree, coarse, fine, coarse_stability,
                fine_stability, alternates,
            )

    save_csv(pd.DataFrame(ablation_rows), args.results / "clustering_view_ablation.csv")
    solution_rows = []
    keys = list(solutions)
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            if key_a[1] != key_b[1]:
                continue
            a, b = solutions[key_a], solutions[key_b]
            solution_rows.append(
                {
                    "configuration_a": key_a[0],
                    "configuration_b": key_b[0],
                    "n_clusters": key_a[1],
                    "ari": adjusted_rand_score(a, b),
                    "nmi": normalized_mutual_info_score(a, b),
                }
            )
    save_csv(pd.DataFrame(solution_rows), args.results / "clustering_solution_similarity.csv")

    fused, embedding, tree, coarse, fine, coarse_stability, fine_stability, alternates = main_payload
    export_vocabulary(
        args.data_root, args.main_hierarchy, args.results, fused, embedding, tree,
        coarse, fine, coarse_stability, fine_stability, support, position,
    )
    resolution_rows = []
    for clusters in RESOLUTIONS:
        labels = cut_tree(tree, clusters)
        half_labels = [cut_tree(x[1], clusters) for x in alternates]
        row = cluster_metrics(embedding, labels)
        row.update(
            {
                "view_configuration": "E+D+P",
                "split_sample_ari": float(np.mean([
                    adjusted_rand_score(labels, x) for x in half_labels
                ])),
                "split_sample_nmi": float(np.mean([
                    normalized_mutual_info_score(labels, x) for x in half_labels
                ])),
            }
        )
        resolution_rows.append(row)
    resolution = pd.DataFrame(resolution_rows)
    save_csv(resolution, args.results / "cluster_resolution_sensitivity.csv")
    hierarchy_report = {
        "method": "E+D+P rank fusion + 32D spectral embedding + one Ward tree",
        "views": ["encoder_direction", "linearized_decoder", "patch_position"],
        "excluded_from_primary_clustering": ["city_profile", "PPMI_activation_context"],
        "neighbours_per_view": 30,
        "coarse": cluster_metrics(embedding, coarse),
        "fine": cluster_metrics(embedding, fine),
        "nested": True,
        "split_sample": {
            "coarse_mean_jaccard": float(coarse_stability.mean()),
            "fine_mean_jaccard": float(fine_stability.mean()),
            "coarse_ari": float(np.mean([
                adjusted_rand_score(coarse, cut_tree(x[1], 32)) for x in alternates
            ])),
            "fine_ari": float(np.mean([
                adjusted_rand_score(fine, cut_tree(x[1], 64)) for x in alternates
            ])),
            "coarse_nmi": float(np.mean([
                normalized_mutual_info_score(coarse, cut_tree(x[1], 32)) for x in alternates
            ])),
            "fine_nmi": float(np.mean([
                normalized_mutual_info_score(fine, cut_tree(x[1], 64)) for x in alternates
            ])),
        },
        "seed": args.seed,
        "note": "C and Q are downstream analysis variables and clustering ablations only",
    }
    atomic_json(args.main_hierarchy / "hierarchy_report.json", hierarchy_report)
    (args.main_hierarchy / "scan_report.json").write_text((source / "scan_report.json").read_text())
    return top, city_ids, coarse, fine


def plot_resolution(results: Path, figures: Path) -> None:
    frame = pd.read_csv(results / "cluster_resolution_sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    specs = [
        ("silhouette", "Silhouette"),
        ("calinski_harabasz", "Calinski--Harabasz"),
        ("davies_bouldin", "Davies--Bouldin"),
        ("split_sample_ari", "Split-sample ARI"),
    ]
    for ax, (column, label) in zip(axes.flat, specs):
        ax.plot(frame["n_clusters"], frame[column], marker="o", lw=1, ms=3, color="#0072B2")
        for value in (32, 64):
            ax.axvline(value, color="#777777", lw=0.6, ls="--")
        ax.set(xlabel="Number of clusters", ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "cluster_resolution_metrics.png", dpi=400)
    plt.close(fig)

    membership = pd.read_csv(results / "cluster_membership.csv")
    rows = []
    for level, column in (("Coarse (32)", "coarse_cluster"), ("Fine (64)", "fine_cluster")):
        for cluster, size in membership[column].value_counts().sort_index().items():
            rows.append({"level": level, "cluster": cluster, "size": size})
    sizes = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, (level, part) in zip(axes, sizes.groupby("level", sort=False)):
        ax.bar(np.arange(len(part)), part["size"], color="#56B4E9", width=0.8)
        ax.set(title=level, xlabel="Cluster", ylabel="Latent dimensions")
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "cluster_size_distribution.png", dpi=400)
    plt.close(fig)


def image_cluster_counts(block: np.ndarray, dimension_labels: np.ndarray, clusters: int) -> np.ndarray:
    labels = dimension_labels[np.asarray(block, dtype=np.int64)]
    rows = np.repeat(np.arange(len(labels), dtype=np.int64), labels.shape[1])
    keys = rows * clusters + labels.ravel()
    return np.bincount(keys, minlength=len(labels) * clusters).reshape(len(labels), clusters)


def city_blocks(city_ids: np.ndarray):
    values = np.asarray(city_ids)
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        yield int(values[start]), int(start), int(end)


def safe_cosine_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norm, 1e-12)
    return np.clip(normalized @ normalized.T, -1, 1)


def export_square(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    save_csv(pd.DataFrame(matrix, index=labels, columns=labels), path, index=True)


def city_composition_and_similarity(
    args: argparse.Namespace,
    top: np.ndarray,
    city_ids: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
):
    print("[2/8] city visual composition and inter-city similarity", flush=True)
    city_names = [city.split("/")[-1] for city in CITIES]
    fine_composition = np.zeros((len(CITIES), 64), dtype=np.float64)
    coarse_composition = np.zeros((len(CITIES), 32), dtype=np.float64)
    fine_image_counts = []
    for city_i, start, end in city_blocks(city_ids):
        fine_counts = image_cluster_counts(top[start:end], fine, 64)
        coarse_counts = image_cluster_counts(top[start:end], coarse, 32)
        fine_composition[city_i] = fine_counts.sum(axis=0) / fine_counts.sum()
        coarse_composition[city_i] = coarse_counts.sum(axis=0) / coarse_counts.sum()
        fine_image_counts.append(fine_counts.astype(np.uint16))
    fine_columns = [f"F{x:03d}" for x in range(64)]
    coarse_columns = [f"C{x:03d}" for x in range(32)]
    fine_frame = pd.DataFrame(fine_composition, columns=fine_columns)
    fine_frame.insert(0, "city", city_names)
    coarse_frame = pd.DataFrame(coarse_composition, columns=coarse_columns)
    coarse_frame.insert(0, "city", city_names)
    save_csv(fine_frame, args.results / "city_visual_composition_fine.csv")
    save_csv(coarse_frame, args.results / "city_visual_composition_coarse.csv")
    long = fine_frame.melt(id_vars="city", var_name="element_id", value_name="prevalence")
    save_csv(long, args.results / "visual_element_city_prevalence.csv")
    top_rows = []
    for city_i, city in enumerate(city_names):
        order = np.argsort(-fine_composition[city_i])[:10]
        for rank, element in enumerate(order, 1):
            top_rows.append(
                {
                    "city": city,
                    "rank": rank,
                    "element_id": fine_columns[element],
                    "prevalence": fine_composition[city_i, element],
                }
            )
    save_csv(pd.DataFrame(top_rows), args.results / "top_visual_elements_by_city.csv")
    diversity = []
    for city_i, city in enumerate(city_names):
        p = fine_composition[city_i]
        entropy = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p))
        diversity.append({"city": city, "normalized_entropy": entropy})
    save_csv(pd.DataFrame(diversity), args.results / "city_visual_diversity.csv")

    cosine = safe_cosine_matrix(fine_composition)
    js = np.zeros_like(cosine)
    for i in range(len(city_names)):
        for j in range(i + 1, len(city_names)):
            js[i, j] = js[j, i] = jensenshannon(
                fine_composition[i], fine_composition[j], base=2
            )
    export_square(cosine, city_names, args.results / "city_similarity_cosine.csv")
    export_square(js, city_names, args.results / "city_similarity_js.csv")
    neighbours = []
    for i, city in enumerate(city_names):
        others = np.array([j for j in range(len(city_names)) if j != i])
        similar = others[np.argsort(-cosine[i, others])[:5]]
        different = others[np.argsort(cosine[i, others])[:5]]
        for kind, indices in (("most_similar", similar), ("most_different", different)):
            for rank, j in enumerate(indices, 1):
                neighbours.append(
                    {
                        "city": city,
                        "relationship": kind,
                        "rank": rank,
                        "other_city": city_names[j],
                        "cosine_similarity": cosine[i, j],
                        "jensen_shannon_distance": js[i, j],
                    }
                )
    save_csv(pd.DataFrame(neighbours), args.results / "city_visual_neighbors.csv")

    contributions = []
    for i in range(len(city_names)):
        for j in range(i + 1, len(city_names)):
            sim_contrib = (
                fine_composition[i] * fine_composition[j]
                / max(np.linalg.norm(fine_composition[i]) * np.linalg.norm(fine_composition[j]), 1e-12)
            )
            diff_contrib = np.abs(fine_composition[i] - fine_composition[j])
            for kind, values in (("similarity", sim_contrib), ("difference", diff_contrib)):
                for rank, element in enumerate(np.argsort(-values)[:5], 1):
                    contributions.append(
                        {
                            "city_a": city_names[i],
                            "city_b": city_names[j],
                            "contribution_type": kind,
                            "rank": rank,
                            "element_id": fine_columns[element],
                            "contribution": values[element],
                            "prevalence_a": fine_composition[i, element],
                            "prevalence_b": fine_composition[j, element],
                        }
                    )
    save_csv(pd.DataFrame(contributions), args.results / "city_pair_element_contributions.csv")
    return city_names, fine_composition, fine_image_counts, cosine, js


def plot_city_composition(
    args: argparse.Namespace,
    city_names: list[str],
    composition: np.ndarray,
    cosine: np.ndarray,
    js: np.ndarray,
) -> None:
    args.figures.mkdir(parents=True, exist_ok=True)
    row_tree = linkage(composition, method="ward")
    # A left-oriented scipy dendrogram places its first leaf at the bottom,
    # whereas imshow places row zero at the top.  Reverse the leaf order so
    # every terminal branch is physically aligned with its heat-map row.
    row_order = dendrogram(row_tree, no_plot=True)["leaves"][::-1]
    z = (composition - composition.mean(axis=0)) / (composition.std(axis=0) + 1e-9)
    ordered = z[row_order]

    fig = plt.figure(figsize=(11.5, 6.8), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.35, 8.65),
        left=0.035,
        right=0.91,
        bottom=0.17,
        top=0.97,
        wspace=0.24,
    )
    left_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    dendrogram(
        row_tree,
        ax=left_ax,
        orientation="left",
        no_labels=True,
        color_threshold=0.0,
        above_threshold_color="#555555",
    )
    left_ax.set_ylim(0, 10 * len(row_order))
    left_ax.set_axis_off()
    for collection in left_ax.collections:
        collection.set_linewidth(0.6)

    im = ax.imshow(ordered, aspect="equal", cmap="RdBu_r", vmin=-3.0, vmax=3.0)
    ax.set_yticks(
        np.arange(len(row_order)),
        labels=[city_names[i] for i in row_order],
    )
    ax.set_xticks(
        np.arange(64),
        labels=[f"F{i:03d}" for i in range(64)],
        rotation=90,
    )
    ax.tick_params(axis="x", labelsize=5.0, length=0, pad=1.5)
    ax.tick_params(axis="y", labelsize=7.0, length=0, pad=2.0)
    ax.set(xlabel="Fine visual element", ylabel="")
    colorbar = fig.colorbar(im, ax=ax, label="Standardized prevalence", fraction=0.025, pad=0.02)
    colorbar.set_ticks(np.arange(-3, 4, 1))
    colorbar.ax.tick_params(labelsize=6.5)
    colorbar.set_label("Standardized prevalence", fontsize=7.5)

    # Keep the city dendrogram physically aligned with the final heat-map
    # height after Matplotlib has resolved the axes positions.
    fig.canvas.draw()
    heat_position = ax.get_position()
    left_position = left_ax.get_position()
    left_ax.set_position(
        [left_position.x0, heat_position.y0, left_position.width, heat_position.height]
    )
    composition_path = args.figures / "city_visual_composition_heatmap.png"
    fig.savefig(
        composition_path,
        dpi=400,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        composition_path.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    for matrix, distance, filename, label, cmap in (
        (
            cosine,
            np.clip(1.0 - cosine, 0.0, None),
            "city_similarity_heatmap.png",
            "Cosine similarity",
            "YlGnBu",
        ),
        (
            js,
            js,
            "city_similarity_js_heatmap.png",
            "Jensen--Shannon distance",
            "YlOrRd",
        ),
    ):
        plot_symmetric_clustermap(
            matrix=matrix,
            distance=distance,
            labels=city_names,
            path=args.figures / filename,
            colorbar_label=label,
            cmap=cmap,
        )

    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=args.seed).fit_transform(js)
    save_csv(pd.DataFrame({"city": city_names, "mds_1": coords[:, 0], "mds_2": coords[:, 1]}),
             args.results / "city_similarity_embedding.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    ax.scatter(coords[:, 0], coords[:, 1], s=20, color="#0072B2")
    for i, city in enumerate(city_names):
        ax.annotate(city, coords[i], xytext=(3, 2), textcoords="offset points", fontsize=5.5)
    ax.set(xlabel="MDS 1", ylabel="MDS 2")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.figures / "city_similarity_embedding.png", dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
    dendrogram(linkage(composition, method="ward"), labels=city_names, orientation="right", ax=ax,
               color_threshold=None, above_threshold_color="#555555")
    for collection in ax.collections:
        collection.set_linewidth(0.6)
    ax.set_xlabel("Ward linkage distance")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(args.figures / "city_hierarchical_clustering.png", dpi=400)
    plt.close(fig)


def plot_symmetric_clustermap(
    matrix: np.ndarray,
    distance: np.ndarray,
    labels: list[str],
    path: Path,
    colorbar_label: str,
    cmap: str,
) -> None:
    """Plot a symmetric annotated matrix with matched row/column dendrograms."""
    matrix = np.asarray(matrix, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64).copy()
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(
        squareform(distance, checks=False),
        method="average",
        optimal_ordering=True,
    )
    order = dendrogram(tree, no_plot=True)["leaves"]
    ordered = matrix[np.ix_(order, order)]
    ordered_labels = [labels[index] for index in order]

    off_diagonal = ordered[~np.eye(len(ordered), dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(13.5, 12.5), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.55, 10.45),
        height_ratios=(1.55, 10.45),
        left=0.055,
        right=0.895,
        bottom=0.215,
        top=0.965,
        wspace=0.075,
        hspace=0.01,
    )
    top_ax = fig.add_subplot(grid[0, 1])
    left_ax = fig.add_subplot(grid[1, 0])
    heat_ax = fig.add_subplot(grid[1, 1])

    dendrogram(
        tree,
        ax=top_ax,
        no_labels=True,
        color_threshold=0.0,
        above_threshold_color="#555555",
    )
    dendrogram(
        tree,
        ax=left_ax,
        orientation="left",
        no_labels=True,
        color_threshold=0.0,
        above_threshold_color="#555555",
    )
    # scipy positions dendrogram leaves at 5, 15, ..., 10*n-5.  Fixing
    # these limits makes both trees align exactly with the n matrix cells
    # and prevents automatic margins from pushing branches out of bounds.
    top_ax.set_xlim(0, 10 * len(labels))
    # The first leaf of a left-oriented dendrogram is drawn at the bottom,
    # while matrix row zero is displayed at the top.  Reverse only the
    # dendrogram's y-axis so its leaves match the symmetric matrix order.
    left_ax.set_ylim(10 * len(labels), 0)
    for axis in (top_ax, left_ax):
        axis.set_axis_off()
        for collection in axis.collections:
            collection.set_linewidth(0.7)

    image = heat_ax.imshow(
        ordered,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    positions = np.arange(len(labels))
    heat_ax.set_xticks(positions, labels=ordered_labels, rotation=90)
    heat_ax.set_yticks(positions, labels=ordered_labels)
    heat_ax.tick_params(axis="both", length=0, labelsize=10.5, pad=2.5)
    heat_ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    heat_ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.7)
    heat_ax.tick_params(which="minor", bottom=False, left=False)

    color_map = plt.get_cmap(cmap)
    scale = vmax - vmin
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = ordered[row, column]
            normalized = np.clip((value - vmin) / scale, 0.0, 1.0)
            red, green, blue, _ = color_map(normalized)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            heat_ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="black" if luminance > 0.52 else "white",
            )

    color_ax = fig.add_axes((0.925, 0.245, 0.014, 0.46))
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label(colorbar_label, fontsize=12.5)
    colorbar.ax.tick_params(labelsize=10.0, length=3.0)
    colorbar.outline.set_linewidth(0.4)

    # ``aspect='equal'`` may shrink the heat-map axes inside its GridSpec
    # cell.  Bind the dendrograms to that final physical bounding box so
    # their first and last leaves cannot extend beyond the matrix edges.
    fig.canvas.draw()
    heat_position = heat_ax.get_position()
    top_position = top_ax.get_position()
    left_position = left_ax.get_position()
    top_ax.set_position(
        [heat_position.x0, top_position.y0, heat_position.width, top_position.height]
    )
    left_ax.set_position(
        [left_position.x0, heat_position.y0, left_position.width, heat_position.height]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def occurrence_from_counts(counts: np.ndarray, mode: str, value: float) -> np.ndarray:
    # Cast away from the compact uint storage before negating for top-N selection.
    # Unary minus on uint16 wraps and would otherwise select zero-count elements.
    counts = np.asarray(counts, dtype=np.int32)
    if mode == "topn":
        n = min(int(value), counts.shape[1])
        selected = np.argpartition(-counts, n - 1, axis=1)[:, :n]
        rows = np.arange(len(counts))[:, None]
        occurrence = np.zeros_like(counts, dtype=bool)
        occurrence[rows, selected] = counts[rows, selected] > 0
        return occurrence
    if mode == "fraction":
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) >= value
    raise ValueError(mode)


def cooccurrence_ppmi(occurrence: np.ndarray, min_support: int):
    binary = sparse.csr_matrix(occurrence.astype(np.int32))
    cooc = (binary.T @ binary).toarray().astype(np.int64)
    support = np.diag(cooc).copy()
    n = len(occurrence)
    expected = support[:, None] * support[None, :] / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooc + 0.5) / (expected + 0.5))
    ppmi = np.maximum(pmi, 0)
    ppmi[cooc < min_support] = 0
    np.fill_diagonal(ppmi, 0)
    return cooc, ppmi, support


def cooccurrence_analysis(
    args: argparse.Namespace,
    city_names: list[str],
    fine_image_counts: list[np.ndarray],
    composition_cosine: np.ndarray,
):
    print("[3/8] global and city-specific element co-occurrence", flush=True)
    counts = np.concatenate(fine_image_counts, axis=0)
    definitions = (("topn", 4), ("topn", 8), ("topn", 12),
                   ("fraction", 0.01), ("fraction", 0.02), ("fraction", 0.05))
    sensitivity_rows = []
    primary = None
    primary_occurrence = None
    for mode, value in definitions:
        occurrence = occurrence_from_counts(counts, mode, value)
        threshold = max(100, int(math.ceil(0.0005 * len(occurrence))))
        cooc, ppmi, support = cooccurrence_ppmi(occurrence, threshold)
        positive = ppmi[np.triu_indices_from(ppmi, 1)]
        sensitivity_rows.append(
            {
                "definition": f"{mode}_{value:g}",
                "mean_elements_per_image": occurrence.sum(axis=1).mean(),
                "median_elements_per_image": float(np.median(occurrence.sum(axis=1))),
                "supported_pairs": int((positive > 0).sum()),
                "mean_positive_ppmi": float(positive[positive > 0].mean()) if np.any(positive > 0) else 0,
                "minimum_pair_support": threshold,
            }
        )
        if mode == "topn" and int(value) == 8:
            primary = (cooc, ppmi, support)
            primary_occurrence = occurrence
    save_csv(pd.DataFrame(sensitivity_rows), args.results / "cooccurrence_threshold_sensitivity.csv")
    cooc, ppmi, support = primary
    elements = [f"F{x:03d}" for x in range(64)]
    export_square(cooc, elements, args.results / "global_element_cooccurrence.csv")
    export_square(ppmi, elements, args.results / "global_element_ppmi.csv")

    # Keep the strongest supported neighbours per node to avoid an unreadable dense graph.
    graph = nx.Graph()
    graph.add_nodes_from(range(64))
    for i in range(64):
        candidates = np.argsort(-ppmi[i])[:10]
        for j in candidates:
            if i != j and ppmi[i, j] > 0:
                graph.add_edge(i, int(j), weight=float(ppmi[i, j]), support=int(cooc[i, j]))
    if graph.number_of_edges():
        communities = nx.community.louvain_communities(
            graph, weight="weight", seed=args.seed
        )
    else:
        communities = [{node} for node in graph.nodes]
    community_rows = []
    for community_id, members in enumerate(sorted(communities, key=lambda x: min(x))):
        members = sorted(members)
        edges = sorted(
            (
                (u, v, graph[u][v]["weight"], graph[u][v]["support"])
                for u, v in graph.subgraph(members).edges()
            ),
            key=lambda row: -row[2],
        )[:10]
        scores = primary_occurrence[:, members].sum(axis=1)
        representative = np.argsort(-scores)[:10]
        community_rows.append(
            {
                "community_id": community_id,
                "member_elements": json.dumps([elements[x] for x in members]),
                "community_size": len(members),
                "strongest_internal_edges": json.dumps([
                    {"a": elements[u], "b": elements[v], "ppmi": w, "support": s}
                    for u, v, w, s in edges
                ]),
                "representative_global_image_rows": json.dumps(representative.tolist()),
            }
        )
    save_csv(pd.DataFrame(community_rows), args.results / "cooccurrence_communities.csv")
    community_edge_rows = []
    for community_a in range(len(communities)):
        for community_b in range(community_a + 1, len(communities)):
            members_a = sorted(communities[community_a])
            members_b = sorted(communities[community_b])
            block = ppmi[np.ix_(members_a, members_b)]
            positive = block[block > 0]
            if not len(positive):
                continue
            community_edge_rows.append(
                {
                    "community_a": community_a,
                    "community_b": community_b,
                    "size_a": len(members_a),
                    "size_b": len(members_b),
                    "mean_ppmi_all_pairs": float(block.mean()),
                    "mean_positive_ppmi": float(positive.mean()),
                    "sum_ppmi": float(block.sum()),
                    "positive_pair_count": int(len(positive)),
                    "possible_pair_count": int(block.size),
                }
            )
    save_csv(
        pd.DataFrame(community_edge_rows),
        args.results / "cooccurrence_community_edges.csv",
    )

    city_ppmi = []
    cursor = 0
    city_dir = args.results / "city_cooccurrence"
    city_dir.mkdir(parents=True, exist_ok=True)
    for city, block in zip(city_names, fine_image_counts):
        n = len(block)
        occurrence = primary_occurrence[cursor : cursor + n]
        cursor += n
        city_threshold = max(50, int(math.ceil(0.0005 * n)))
        city_cooc, matrix, _ = cooccurrence_ppmi(occurrence, city_threshold)
        export_square(matrix, elements, city_dir / f"{city}.csv")
        city_ppmi.append(matrix)
    upper = np.triu_indices(64, 1)
    flat = np.stack([matrix[upper] for matrix in city_ppmi])
    co_similarity = safe_cosine_matrix(flat)
    export_square(co_similarity, city_names, args.results / "city_cooccurrence_similarity.csv")

    pair_rows = []
    comp_values = composition_cosine[np.triu_indices(30, 1)]
    co_values = co_similarity[np.triu_indices(30, 1)]
    comp_median, co_median = np.median(comp_values), np.median(co_values)
    for i in range(30):
        for j in range(i + 1, 30):
            if composition_cosine[i, j] >= comp_median and co_similarity[i, j] < co_median:
                pattern = "high_composition_low_cooccurrence"
            elif composition_cosine[i, j] < comp_median and co_similarity[i, j] >= co_median:
                pattern = "low_composition_high_cooccurrence"
            elif composition_cosine[i, j] >= comp_median:
                pattern = "high_both"
            else:
                pattern = "low_both"
            pair_rows.append(
                {
                    "city_a": city_names[i],
                    "city_b": city_names[j],
                    "composition_similarity": composition_cosine[i, j],
                    "cooccurrence_similarity": co_similarity[i, j],
                    "pattern": pattern,
                }
            )
    save_csv(pd.DataFrame(pair_rows), args.results / "composition_vs_cooccurrence_city_pairs.csv")
    plot_cooccurrence(
        args,
        elements,
        ppmi,
        graph,
        communities,
        city_names,
        co_similarity,
        pd.DataFrame(pair_rows),
    )
    return primary_occurrence, ppmi, co_similarity


def plot_cooccurrence(
    args: argparse.Namespace,
    elements: list[str],
    ppmi: np.ndarray,
    graph: nx.Graph,
    communities: list[set[int]],
    city_names: list[str],
    city_similarity: np.ndarray,
    pairs: pd.DataFrame,
) -> None:
    ppmi_similarity = safe_cosine_matrix(ppmi)
    ppmi_distance = np.clip(1.0 - ppmi_similarity, 0.0, 2.0)
    np.fill_diagonal(ppmi_distance, 0.0)
    order = dendrogram(
        linkage(squareform(ppmi_distance, checks=False), method="average"),
        no_plot=True,
    )["leaves"]
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    image = ax.imshow(ppmi[np.ix_(order, order)], cmap="magma", aspect="auto")
    ax.set(xticks=np.arange(0, 64, 4), xticklabels=[elements[order[i]] for i in range(0, 64, 4)],
           yticks=np.arange(0, 64, 4), yticklabels=[elements[order[i]] for i in range(0, 64, 4)])
    ax.tick_params(axis="x", rotation=90)
    fig.colorbar(image, ax=ax, label="PPMI", fraction=0.03, pad=0.02)
    fig.savefig(args.figures / "global_cooccurrence_heatmap.png", dpi=400)
    plt.close(fig)

    # Preserve the dense 64-element view as a supplementary audit figure.
    fig, ax = plt.subplots(figsize=(8.4, 7.4), constrained_layout=True)
    positions = nx.spring_layout(graph, seed=args.seed, weight="weight", iterations=200)
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
    nx.draw_networkx_edges(graph, positions, ax=ax, width=0.2 + weights / max(weights.max(), 1e-9),
                           alpha=0.22, edge_color="#777777")
    membership = {
        member: community_id
        for community_id, members in enumerate(communities)
        for member in members
    }
    palette = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
        "#56B4E9", "#F0E442", "#999999", "#6A3D9A", "#8C564B",
    ]
    node_colours = [palette[membership[node] % len(palette)] for node in graph]
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=34, node_color=node_colours,
                           edgecolors="white", linewidths=0.3)
    nx.draw_networkx_labels(graph, positions, labels={i: elements[i] for i in graph},
                           font_size=4.5, ax=ax)
    ax.axis("off")
    detailed_path = args.figures / "global_visual_element_network_detailed.png"
    fig.savefig(detailed_path, dpi=400, facecolor="white")
    fig.savefig(detailed_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    # Main-text view: aggregate elements into Louvain communities and retain
    # a maximum-spanning-tree backbone plus the three strongest extra links.
    community_graph = nx.Graph()
    for community_id, members in enumerate(communities):
        community_graph.add_node(community_id, size=len(members))
    for community_a in range(len(communities)):
        for community_b in range(community_a + 1, len(communities)):
            block = ppmi[np.ix_(sorted(communities[community_a]), sorted(communities[community_b]))]
            strength = float(block.mean())
            if strength > 0:
                community_graph.add_edge(community_a, community_b, weight=strength)

    connected = [node for node in community_graph if community_graph.degree(node) > 0]
    isolated = [node for node in community_graph if community_graph.degree(node) == 0]
    core_graph = community_graph.subgraph(connected).copy()
    backbone = nx.maximum_spanning_tree(core_graph, weight="weight")
    remaining = sorted(
        (
            (u, v, data["weight"])
            for u, v, data in core_graph.edges(data=True)
            if not backbone.has_edge(u, v)
        ),
        key=lambda item: -item[2],
    )[:3]
    display_edges = list(backbone.edges()) + [(u, v) for u, v, _ in remaining]

    community_positions = nx.spring_layout(
        core_graph,
        seed=args.seed,
        weight="weight",
        iterations=500,
        k=1.15,
        scale=1.0,
    )
    for node in community_positions:
        community_positions[node][1] += 0.45
    if isolated:
        isolate_x = np.linspace(-1.15, 1.15, len(isolated))
        for x_position, node in zip(isolate_x, isolated):
            community_positions[node] = np.array([x_position, -1.35])

    fine_labels = {}
    semantic_path = args.results / "semantic_labels_fine.json"
    if semantic_path.exists():
        payload = json.loads(semantic_path.read_text())
        for category in payload.get("categories", []):
            fine_labels[int(category["cluster_id"][1:])] = category.get(
                "short_label", category["cluster_id"]
            ).replace("_", " ")

    community_labels = {}
    for community_id, members in enumerate(communities):
        members = sorted(members)
        if len(members) > 1:
            subgraph = graph.subgraph(members)
            weighted_degree = dict(subgraph.degree(weight="weight"))
            hub = max(members, key=lambda member: weighted_degree.get(member, 0.0))
            hub_name = fine_labels.get(hub, elements[hub])
            if len(hub_name) > 22:
                hub_name = hub_name[:20] + "..."
            community_labels[community_id] = (
                f"G{community_id + 1}\n{hub_name}\n$n={len(members)}$"
            )
        else:
            community_labels[community_id] = (
                f"G{community_id + 1}\n{elements[members[0]]}"
            )

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    display_weights = np.array(
        [community_graph[u][v]["weight"] for u, v in display_edges],
        dtype=np.float64,
    )
    nx.draw_networkx_edges(
        community_graph,
        community_positions,
        edgelist=display_edges,
        width=0.8 + 4.2 * display_weights / max(display_weights.max(), 1e-12),
        alpha=0.58,
        edge_color="#777777",
        ax=ax,
    )
    community_sizes = np.array(
        [community_graph.nodes[node]["size"] for node in community_graph],
        dtype=np.float64,
    )
    nx.draw_networkx_nodes(
        community_graph,
        community_positions,
        nodelist=list(community_graph.nodes()),
        node_size=260 + 55 * community_sizes,
        node_color=[palette[node % len(palette)] for node in community_graph],
        edgecolors="white",
        linewidths=0.9,
        ax=ax,
    )
    nx.draw_networkx_labels(
        community_graph,
        community_positions,
        labels=community_labels,
        font_size=6.2,
        font_color="black",
        ax=ax,
    )
    edge_labels = {
        (u, v): f"{community_graph[u][v]['weight']:.3f}"
        for u, v in display_edges
    }
    nx.draw_networkx_edge_labels(
        community_graph,
        community_positions,
        edge_labels=edge_labels,
        font_size=4.8,
        rotate=False,
        label_pos=0.5,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8},
        ax=ax,
    )
    ax.text(
        0.5,
        0.015,
        "Node size: number of elements    Edge width: mean inter-community PPMI",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.62, 1.55)
    ax.axis("off")
    network_path = args.figures / "global_visual_element_network.png"
    fig.savefig(network_path, dpi=400, facecolor="white")
    fig.savefig(network_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    city_distance = np.clip(1.0 - city_similarity, 0.0, 2.0)
    np.fill_diagonal(city_distance, 0.0)
    order_city = dendrogram(
        linkage(squareform(city_distance, checks=False), method="average"),
        no_plot=True,
    )["leaves"]
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    image = ax.imshow(city_similarity[np.ix_(order_city, order_city)], cmap="viridis", vmin=0, vmax=1)
    labels = [city_names[i] for i in order_city]
    ax.set(xticks=np.arange(30), xticklabels=labels, yticks=np.arange(30), yticklabels=labels)
    ax.tick_params(axis="x", rotation=90)
    fig.colorbar(image, ax=ax, label="Co-occurrence cosine similarity", fraction=0.03, pad=0.02)
    fig.savefig(args.figures / "city_cooccurrence_similarity_heatmap.png", dpi=400)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    palette = {
        "high_composition_low_cooccurrence": "#D55E00",
        "low_composition_high_cooccurrence": "#0072B2",
        "high_both": "#009E73",
        "low_both": "#999999",
    }
    for pattern, part in pairs.groupby("pattern"):
        ax.scatter(part["composition_similarity"], part["cooccurrence_similarity"],
                   s=13, alpha=0.75, label=pattern.replace("_", " "), color=palette[pattern])
    ax.set(xlabel="Visual-composition similarity", ylabel="Co-occurrence similarity")
    ax.legend(frameon=False, fontsize=5.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.figures / "composition_vs_cooccurrence.png", dpi=400)
    plt.close(fig)


def web_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(np.clip(lat, -85.05112878, 85.05112878))
    return EARTH_RADIUS_M * lon_rad, EARTH_RADIUS_M * np.log(np.tan(np.pi / 4 + lat_rad / 2))


def inverse_web_mercator(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.rad2deg(x / EARTH_RADIUS_M)
    lat = np.rad2deg(2 * np.arctan(np.exp(y / EARTH_RADIUS_M)) - np.pi / 2)
    return lon, lat


def pano_records_for_city(
    data_root: Path, city_key: str, image_counts: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(
        data_root / "filtered_manifests" / f"{city_slug(city_key)}.parquet",
        columns=["panoid", "lat", "lon", "source_image_index"],
    ).reset_index(drop=True)
    if len(frame) != len(image_counts):
        raise ValueError(f"{city_key}: manifest/count mismatch {len(frame)} != {len(image_counts)}")
    metadata = frame.groupby("panoid", sort=False).agg(lat=("lat", "first"), lon=("lon", "first"))
    codes, uniques = pd.factorize(frame["panoid"], sort=False)
    sums = np.zeros((len(uniques), image_counts.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, image_counts)
    return metadata.reset_index(), sums


def make_grid(
    panos: pd.DataFrame,
    counts: np.ndarray,
    grid_size: int,
    min_panoramas: int,
) -> pd.DataFrame:
    x, y = web_mercator(panos["lon"].to_numpy(), panos["lat"].to_numpy())
    gx = np.floor(x / grid_size).astype(np.int64)
    gy = np.floor(y / grid_size).astype(np.int64)
    keys = pd.MultiIndex.from_arrays([gx, gy])
    codes, unique = pd.factorize(keys, sort=True)
    n = len(unique)
    summed = np.zeros((n, counts.shape[1]), dtype=np.float64)
    np.add.at(summed, codes, counts)
    pano_count = np.bincount(codes, minlength=n)
    keep = pano_count >= min_panoramas
    gx_unique = np.array([value[0] for value in unique], dtype=np.int64)[keep]
    gy_unique = np.array([value[1] for value in unique], dtype=np.int64)[keep]
    values = summed[keep]
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1)
    center_x = (gx_unique + 0.5) * grid_size
    center_y = (gy_unique + 0.5) * grid_size
    lon, lat = inverse_web_mercator(center_x, center_y)
    frame = pd.DataFrame(
        {
            "grid_x": gx_unique,
            "grid_y": gy_unique,
            "longitude": lon,
            "latitude": lat,
            "number_of_panoramas": pano_count[keep],
        }
    )
    for feature in range(counts.shape[1]):
        frame[f"F{feature:03d}"] = values[:, feature]
    return frame


def upper_similarity(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.array([], dtype=np.float64)
    return safe_cosine_matrix(values)[np.triu_indices(len(values), 1)]


def grid_edges(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(frame.grid_x, frame.grid_y))}
    left, right = [], []
    for (x, y), i in lookup.items():
        for neighbour in ((x + 1, y), (x, y + 1)):
            if neighbour in lookup:
                left.append(i); right.append(lookup[neighbour])
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def grid_summary(frame: pd.DataFrame, element_columns: list[str]) -> dict:
    values = frame[element_columns].to_numpy(np.float64)
    pairwise = upper_similarity(values)
    left, right = grid_edges(frame)
    if len(left):
        local = np.sum(values[left] * values[right], axis=1) / np.maximum(
            np.linalg.norm(values[left], axis=1) * np.linalg.norm(values[right], axis=1), 1e-12
        )
    else:
        local = np.array([], dtype=np.float64)
    mean_pair = float(pairwise.mean()) if len(pairwise) else np.nan
    return {
        "n_valid_grids": len(frame),
        "n_adjacent_grid_pairs": len(left),
        "mean_pairwise_similarity": mean_pair,
        "median_pairwise_similarity": float(np.median(pairwise)) if len(pairwise) else np.nan,
        "visual_homogeneity": mean_pair,
        "visual_heterogeneity": 1 - mean_pair if np.isfinite(mean_pair) else np.nan,
        "local_visual_continuity": float(local.mean()) if len(local) else np.nan,
    }


def moran_all(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k = values.shape
    if n < 3 or not len(left):
        missing = np.full(k, np.nan)
        return missing, missing, missing, missing
    rows = np.r_[left, right]
    cols = np.r_[right, left]
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    s0 = adjacency.sum()
    centered = values - values.mean(axis=0, keepdims=True)
    denominator = np.sum(centered**2, axis=0)

    def statistic(block: np.ndarray) -> np.ndarray:
        return n / s0 * np.sum(block * (adjacency @ block), axis=0) / np.maximum(denominator, 1e-12)

    observed = statistic(centered)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, k), dtype=np.float32)
    for index in range(permutations):
        null[index] = statistic(centered[rng.permutation(n)])
    expected = np.full(k, -1.0 / (n - 1), dtype=np.float64)
    p = (1 + np.sum(np.abs(null - expected) >= np.abs(observed - expected), axis=0)) / (permutations + 1)
    z = (observed - null.mean(axis=0)) / (null.std(axis=0) + 1e-12)
    return observed, expected, p, z


def spatial_analysis(
    args: argparse.Namespace,
    city_names: list[str],
    fine_image_counts: list[np.ndarray],
):
    print("[4/8] within-city grids, homogeneity, continuity and Moran's I", flush=True)
    element_columns = [f"F{x:03d}" for x in range(64)]
    primary_frames = []
    homogeneity_rows = []
    continuity_rows = []
    structure_rows = []
    moran_rows = []
    sensitivity_rows = []
    cached_panos = []
    for city_key, image_counts in zip(CITIES, fine_image_counts):
        cached_panos.append(pano_records_for_city(args.data_root, city_key, image_counts))
    # The sampled panoramas are spatially sparse in several metropolitan
    # extents. Five panoramas is therefore the primary support threshold;
    # 10/20/30 and the optional 2-km grid are retained as sensitivity checks.
    for grid_size in (500, 1000, 2000):
        for minimum in (5, 10, 20, 30):
            for city_i, (city, (panos, counts)) in enumerate(zip(city_names, cached_panos)):
                grid = make_grid(panos, counts, grid_size, minimum)
                summary = grid_summary(grid, element_columns)
                sensitivity_rows.append(
                    {"city": city, "grid_size_m": grid_size, "minimum_panoramas": minimum, **summary}
                )
                if grid_size == args.grid_size and minimum == args.min_panoramas:
                    grid.insert(0, "grid_id", [f"{city}_{x}_{y}" for x, y in zip(grid.grid_x, grid.grid_y)])
                    grid.insert(0, "city", city)
                    primary_frames.append(grid)
                    base = {"city": city, **summary}
                    homogeneity_rows.append({k: base[k] for k in (
                        "city", "n_valid_grids", "mean_pairwise_similarity",
                        "median_pairwise_similarity", "visual_homogeneity", "visual_heterogeneity"
                    )})
                    continuity_rows.append({k: base[k] for k in (
                        "city", "n_valid_grids", "n_adjacent_grid_pairs", "local_visual_continuity"
                    )})
                    left, right = grid_edges(grid)
                    values = grid[element_columns].to_numpy(np.float64)
                    observed, expected, p, z = moran_all(
                        values, left, right, args.moran_permutations, args.seed + city_i
                    )
                    for element, obs, exp, p_value, z_score in zip(
                        element_columns, observed, expected, p, z
                    ):
                        moran_rows.append(
                            {
                                "city": city,
                                "element_id": element,
                                "morans_i": obs,
                                "expected_i": exp,
                                "p_value": p_value,
                                "z_score": z_score,
                                "grid_size_m": grid_size,
                                "minimum_panoramas": minimum,
                            }
                        )
                    finite_moran = observed[np.isfinite(observed)]
                    structure_rows.append({
                        **base,
                        "mean_morans_i": (
                            float(finite_moran.mean()) if len(finite_moran) else np.nan
                        ),
                    })
    grid_frame = pd.concat(primary_frames, ignore_index=True)
    save_csv(grid_frame, args.results / "grid_visual_composition.csv")
    save_csv(pd.DataFrame(homogeneity_rows), args.results / "city_visual_homogeneity.csv")
    save_csv(pd.DataFrame(continuity_rows), args.results / "city_local_visual_continuity.csv")
    save_csv(pd.DataFrame(structure_rows), args.results / "city_visual_structure_summary.csv")
    save_csv(pd.DataFrame(moran_rows), args.results / "visual_element_morans_i.csv")
    sensitivity = pd.DataFrame(sensitivity_rows)
    primary_h = sensitivity[(sensitivity.grid_size_m == args.grid_size) &
                            (sensitivity.minimum_panoramas == args.min_panoramas)].set_index("city")
    comparison = []
    for (grid_size, minimum), part in sensitivity.groupby(["grid_size_m", "minimum_panoramas"]):
        part = part.set_index("city").reindex(primary_h.index)
        for metric in ("visual_homogeneity", "local_visual_continuity"):
            valid = np.isfinite(part[metric]) & np.isfinite(primary_h[metric])
            rho = spearmanr(part.loc[valid, metric], primary_h.loc[valid, metric]).statistic if valid.sum() >= 3 else np.nan
            comparison.append(
                {
                    "grid_size_m": grid_size,
                    "minimum_panoramas": minimum,
                    "metric": metric,
                    "spearman_vs_primary": rho,
                    "n_cities": int(valid.sum()),
                }
            )
    save_csv(pd.DataFrame(comparison), args.results / "spatial_resolution_sensitivity.csv")
    plot_spatial_summary(args, pd.DataFrame(structure_rows), grid_frame)
    return grid_frame


def plot_spatial_summary(args: argparse.Namespace, structure: pd.DataFrame, grids: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.scatter(structure["visual_homogeneity"], structure["local_visual_continuity"],
               s=22, color="#0072B2")
    for _, row in structure.iterrows():
        ax.annotate(row.city, (row.visual_homogeneity, row.local_visual_continuity),
                    xytext=(3, 2), textcoords="offset points", fontsize=5.2)
    ax.set(xlabel="Urban visual homogeneity", ylabel="Local visual continuity")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.figures / "homogeneity_vs_local_continuity.png", dpi=400)
    plt.close(fig)

    # Convert the 64-dimensional grid compositions into a small, shared set of
    # local visual environments.  This is more stable than colouring each grid
    # by its single most prevalent fine element, especially when the two most
    # prevalent elements have nearly equal shares.
    element_columns = [f"F{x:03d}" for x in range(64)]
    figure_minimum = max(10, int(args.min_panoramas))
    mapped = grids.loc[grids.number_of_panoramas >= figure_minimum].copy()
    composition = mapped[element_columns].to_numpy(np.float64)
    hellinger = np.sqrt(np.clip(composition, 0, None))

    # Temper, rather than completely remove, city-size imbalance. Full inverse
    # weighting would let cities represented by only one or two cells dominate
    # a centroid; inverse-square-root weighting avoids that failure mode.
    city_cell_count = mapped.groupby("city")["city"].transform("size").to_numpy(np.float64)
    sample_weight = 1.0 / np.sqrt(np.maximum(city_cell_count, 1.0))
    sample_weight *= len(sample_weight) / sample_weight.sum()
    n_environments = 10
    model = KMeans(n_clusters=n_environments, n_init=50, random_state=args.seed)
    raw_labels = model.fit_predict(hellinger, sample_weight=sample_weight)

    # Give otherwise arbitrary K-means labels a deterministic order based on
    # their dominant visual element, then centroid strength.
    raw_centres = model.cluster_centers_
    label_order = np.lexsort(
        (np.arange(n_environments), -raw_centres.max(axis=1), raw_centres.argmax(axis=1))
    )
    remap = np.empty(n_environments, dtype=np.int64)
    remap[label_order] = np.arange(n_environments)
    labels = remap[raw_labels]
    centres = raw_centres[label_order]

    distances = model.transform(hellinger)[:, label_order]
    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    nearest_two.sort(axis=1)
    confidence = (nearest_two[:, 1] - nearest_two[:, 0]) / np.maximum(nearest_two[:, 1], 1e-12)
    confidence_threshold = 0.10
    low_confidence = confidence < confidence_threshold
    mapped["environment_id"] = [f"E{x + 1:02d}" for x in labels]
    mapped["assignment_confidence"] = confidence
    mapped["low_confidence"] = low_confidence

    assignment_columns = [
        "city", "grid_id", "longitude", "latitude", "number_of_panoramas",
        "environment_id", "assignment_confidence", "low_confidence",
    ]
    save_csv(mapped[assignment_columns], args.results / "grid_visual_environment_assignments.csv")

    profile_rows = []
    for environment in range(n_environments):
        members = labels == environment
        profile = composition[members].mean(axis=0)
        profile /= max(profile.sum(), 1e-12)
        row = {
            "environment_id": f"E{environment + 1:02d}",
            "n_grid_cells": int(members.sum()),
            "dominant_element": element_columns[int(profile.argmax())],
            "mean_assignment_confidence": float(confidence[members].mean()),
        }
        row.update({name: float(value) for name, value in zip(element_columns, profile)})
        profile_rows.append(row)
    save_csv(pd.DataFrame(profile_rows), args.results / "local_visual_environment_profiles.csv")

    # Select well-supported cities spanning the recalculated homogeneity range.
    supported_rows = []
    for city, part in mapped.groupby("city", sort=False):
        if len(part) < 100:
            continue
        summary = grid_summary(part, element_columns)
        supported_rows.append({"city": city, "minimum_panoramas": figure_minimum, **summary})
    supported = pd.DataFrame(supported_rows).sort_values("visual_homogeneity").reset_index(drop=True)
    order = supported.city.tolist()
    selected = list(dict.fromkeys([
        order[0], order[len(order) // 4], order[len(order) // 2],
        order[3 * len(order) // 4], order[-1], "HongKong",
    ]))[:6]
    selected_summary = supported.set_index("city").loc[selected].reset_index()
    save_csv(selected_summary, args.results / "within_city_visual_structure_figure_cities.csv")

    palette = np.asarray([
        "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
        "#56B4E9", "#C49A00", "#6A3D9A", "#8C564B", "#17BECF",
    ])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.25), constrained_layout=True)
    for panel, (ax, city) in enumerate(zip(axes.flat, selected)):
        part = mapped[mapped.city == city]
        label_index = part.environment_id.str[1:].astype(int).to_numpy() - 1
        point_colours = np.zeros((len(part), 4), dtype=np.float64)
        point_colours[:, :3] = np.asarray([
            matplotlib.colors.to_rgb(palette[index]) for index in label_index
        ])
        point_confidence = part.assignment_confidence.to_numpy(np.float64)
        point_colours[:, 3] = 0.42 + 0.58 * np.clip(
            (point_confidence - confidence_threshold) / 0.30, 0, 1
        )
        uncertain = part.low_confidence.to_numpy(bool)
        point_colours[uncertain, :3] = matplotlib.colors.to_rgb("#B8B8B8")
        point_colours[uncertain, 3] = 0.55
        grid_x = part.grid_x.to_numpy(np.float64)
        grid_y = part.grid_y.to_numpy(np.float64)
        vertices = np.stack(
            [
                np.column_stack([grid_x, grid_y]),
                np.column_stack([grid_x + 1, grid_y]),
                np.column_stack([grid_x + 1, grid_y + 1]),
                np.column_stack([grid_x, grid_y + 1]),
            ],
            axis=1,
        )
        cells = PolyCollection(
            vertices,
            facecolors=point_colours,
            edgecolors="white",
            linewidths=0.12,
            antialiased=False,
            rasterized=True,
        )
        ax.add_collection(cells)
        ax.update_datalim(np.column_stack([grid_x, grid_y]))
        ax.update_datalim(np.column_stack([grid_x + 1, grid_y + 1]))
        ax.autoscale_view()
        ax.set_title(f"{chr(97 + panel)}  {city}  ($n={len(part)}$)", loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=4.5,
                   markerfacecolor=palette[index], markeredgewidth=0,
                   label=f"E{index + 1:02d}")
        for index in range(n_environments)
    ]
    handles.append(
        plt.Line2D([], [], marker="s", linestyle="none", markersize=4.5,
                   markerfacecolor="#B8B8B8", markeredgewidth=0, label="Uncertain")
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=11, frameon=False,
        bbox_to_anchor=(0.5, -0.012), columnspacing=0.75, handletextpad=0.25,
        title="Local visual environment",
    )
    figure_path = args.figures / "within_city_visual_structure.png"
    fig.savefig(figure_path, dpi=400, facecolor="white")
    fig.savefig(figure_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def sampling_sensitivity(
    args: argparse.Namespace,
    city_names: list[str],
    fine_image_counts: list[np.ndarray],
    full_composition: np.ndarray,
    full_ppmi: np.ndarray,
):
    print("[5/8] sampling-fraction sensitivity", flush=True)
    rng = np.random.default_rng(args.seed)
    full_similarity = safe_cosine_matrix(full_composition)
    upper_city = np.triu_indices(len(city_names), 1)
    upper_element = np.triu_indices(64, 1)
    rows = []
    for fraction in (0.25, 0.50, 0.75, 1.00):
        blocks = []
        occurrences = []
        for counts in fine_image_counts:
            take = max(1, int(round(len(counts) * fraction)))
            indices = np.arange(len(counts)) if fraction == 1 else rng.choice(len(counts), take, replace=False)
            block = counts[indices]
            blocks.append(block.sum(axis=0) / block.sum())
            occurrences.append(occurrence_from_counts(block, "topn", 8))
        composition = np.stack(blocks)
        similarity = safe_cosine_matrix(composition)
        occurrence = np.concatenate(occurrences)
        threshold = max(100, int(math.ceil(0.0005 * len(occurrence))))
        _, ppmi, _ = cooccurrence_ppmi(occurrence, threshold)
        rows.append(
            {
                "sample_fraction": fraction,
                "images": len(occurrence),
                "prevalence_correlation_with_full": np.corrcoef(
                    composition.ravel(), full_composition.ravel()
                )[0, 1],
                "city_similarity_correlation_with_full": np.corrcoef(
                    similarity[upper_city], full_similarity[upper_city]
                )[0, 1],
                "cooccurrence_ppmi_correlation_with_full": np.corrcoef(
                    ppmi[upper_element], full_ppmi[upper_element]
                )[0, 1],
            }
        )
    save_csv(pd.DataFrame(rows), args.results / "sampling_sensitivity.csv")


def nuisance_sensitivity(
    args: argparse.Namespace,
    fine_image_counts: list[np.ndarray],
) -> None:
    """Estimate held-out linear variance explained by available capture nuisances.

    Outcomes and continuous predictors are centred within city before fitting so
    that geographic composition differences are not misattributed to metadata.
    """
    print("[6/8] nuisance sensitivity", flush=True)
    predictors = []
    outcomes = []
    city_index = []
    for city_i, (city, counts) in enumerate(zip(CITIES, fine_image_counts)):
        frame = pd.read_parquet(
            args.data_root / "filtered_manifests" / f"{city_slug(city)}.parquet",
            columns=["heading", "year", "month", "artifact_bad_probability"],
        )
        heading = np.deg2rad(frame["heading"].to_numpy(np.float64))
        month = frame["month"].to_numpy(np.float64)
        month_angle = 2 * np.pi * np.maximum(month - 1, 0) / 12
        year = frame["year"].to_numpy(np.float64)
        valid_year = year > 1900
        year[~valid_year] = np.nanmedian(year[valid_year]) if valid_year.any() else 0
        artifact = frame["artifact_bad_probability"].to_numpy(np.float64)
        x = np.column_stack(
            (
                np.sin(heading), np.cos(heading),
                np.sin(month_angle), np.cos(month_angle),
                year, valid_year.astype(np.float64), artifact,
            )
        )
        y = counts.astype(np.float64) / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean(axis=0, keepdims=True)
        predictors.append(x); outcomes.append(y)
        city_index.append(np.full(len(frame), city_i, dtype=np.int16))
    x = np.concatenate(predictors)
    y = np.concatenate(outcomes)
    cities = np.concatenate(city_index)
    rng = np.random.default_rng(args.seed)
    validation = np.zeros(len(x), dtype=bool)
    for city_i in range(len(CITIES)):
        indices = np.flatnonzero(cities == city_i)
        validation[rng.choice(indices, size=max(1, len(indices) // 5), replace=False)] = True
    mean = x[~validation].mean(axis=0)
    scale = x[~validation].std(axis=0) + 1e-9
    model = Ridge(alpha=10.0).fit((x[~validation] - mean) / scale, y[~validation])
    prediction = model.predict((x[validation] - mean) / scale)
    scores = r2_score(y[validation], prediction, multioutput="raw_values")
    summary = {
        "median_r2": float(np.median(scores)),
        "p90_r2": float(np.quantile(scores, 0.9)),
        "maximum_r2": float(np.max(scores)),
        "strong_r2_threshold": 0.2,
        "n_strongly_nuisance_sensitive": int(np.sum(scores >= 0.2)),
        "n_images": int(len(x)),
        "validation_images": int(validation.sum()),
    }
    rows = []
    for element, score in enumerate(scores):
        rows.append(
            {
                "element_id": f"F{element:03d}",
                "held_out_r2": float(score),
                **summary,
                "predictors": "heading_sincos;month_sincos;year;year_available;artifact_probability",
                "city_centered": True,
            }
        )
    save_csv(pd.DataFrame(rows), args.results / "nuisance_sensitivity.csv")
    atomic_json(args.results / "nuisance_sensitivity_summary.json", summary)


def export_semantics_if_available(args: argparse.Namespace) -> None:
    semantic_root = args.main_hierarchy / "semantic_labels_qwen"
    labels_path = semantic_root / "category_labels.json"
    if not labels_path.is_file():
        return
    payload = json.loads(labels_path.read_text())
    categories = payload["categories"]
    fields = (
        "category_id", "name_en", "semantic_type", "summary_zh", "visual_cues_zh",
        "urban_meaning_zh", "confidence", "artifact_probability",
    )
    for level, prefix, filename in (
        ("fine", "F", "semantic_labels_fine.json"),
        ("coarse", "C", "semantic_labels_coarse.json"),
    ):
        records = []
        for key in sorted(k for k in categories if k.startswith(prefix)):
            source = categories[key]
            row = {field: source.get(field) for field in fields}
            row["cluster_id"] = key
            row["short_label"] = source.get("name_en")
            row["summary"] = source.get("summary_zh")
            row["visual_cues"] = source.get("visual_cues_zh")
            row["urban_interpretation"] = source.get("urban_meaning_zh")
            records.append(row)
        atomic_json(args.results / filename, {"level": level, "categories": records})


def model_comparison(args: argparse.Namespace) -> None:
    print("[7/8] available-model comparison table", flush=True)
    mae_report = json.loads((args.data_root / "mae" / "training_report.json").read_text())
    topk_root = args.data_root.parent / "patch_sae_topk32_w512_n30x12800"
    topk_report_path = topk_root / "patch_sae" / "training_report.json"
    rows = [
        {
            "model": "Raw DINOv3 ViT-B/16",
            "latent_dim": 768,
            "reconstruction_score": 1.0,
            "reconstruction_protocol": "identity reference; not a learned reconstruction",
            "cluster_ari": np.nan,
            "cluster_nmi": np.nan,
            "semantic_coherence": np.nan,
            "spatial_interpretability": "native 14x14 patch features",
            "notes": "Frozen backbone reference",
        },
        {
            "model": "DINOv3 + PCA",
            "latent_dim": 512,
            "reconstruction_score": np.nan,
            "reconstruction_protocol": "pending city-balanced PCA fit",
            "cluster_ari": np.nan,
            "cluster_nmi": np.nan,
            "semantic_coherence": np.nan,
            "spatial_interpretability": "linear patch scores",
            "notes": "Baseline pipeline prepared; fit required",
        },
    ]
    if topk_report_path.is_file():
        topk = json.loads(topk_report_path.read_text())
        rows.append(
            {
                "model": "Top-K sparse autoencoder",
                "latent_dim": topk["width"],
                "reconstruction_score": 1 - topk["best_metrics"]["val_loss"],
                "reconstruction_protocol": "unmasked patch-token cosine reconstruction",
                "cluster_ari": np.nan,
                "cluster_nmi": np.nan,
                "semantic_coherence": np.nan,
                "spatial_interpretability": "non-negative Top-K patch activations",
                "notes": f"K={topk['topk']}; dead features={topk['best_metrics']['dead_features']}",
            }
        )
    main_report_path = args.main_hierarchy / "hierarchy_report.json"
    hierarchy = json.loads(main_report_path.read_text()) if main_report_path.is_file() else {}
    rows.append(
        {
            "model": "Feature-MAE",
            "latent_dim": mae_report["latent_width"],
            "reconstruction_score": 1 - mae_report["best_metrics"]["validation_loss"],
            "reconstruction_protocol": "75% masked patch-token cosine reconstruction",
            "cluster_ari": hierarchy.get("split_sample", {}).get("fine_ari"),
            "cluster_nmi": hierarchy.get("split_sample", {}).get("fine_nmi"),
            "semantic_coherence": np.nan,
            "spatial_interpretability": "signed contextual 14x14 latent maps",
            "notes": "Primary model; E+D+P hierarchy",
        }
    )
    save_csv(pd.DataFrame(rows), args.results / "model_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-hierarchy", type=Path, default=DEFAULT_SOURCE_HIERARCHY)
    parser.add_argument("--main-hierarchy", type=Path, default=DEFAULT_MAIN_HIERARCHY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--min-panoramas", type=int, default=5)
    parser.add_argument("--moran-permutations", type=int, default=999)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    atomic_json(args.results / "analysis_provenance.json", provenance(args))
    top, city_ids, coarse, fine = view_ablation_and_hierarchy(args)
    plot_resolution(args.results, args.figures)
    city_names, composition, image_counts, cosine, js = city_composition_and_similarity(
        args, top, city_ids, coarse, fine
    )
    plot_city_composition(args, city_names, composition, cosine, js)
    occurrence, ppmi, co_similarity = cooccurrence_analysis(
        args, city_names, image_counts, cosine
    )
    del occurrence, co_similarity
    spatial_analysis(args, city_names, image_counts)
    sampling_sensitivity(args, city_names, image_counts, composition, ppmi)
    nuisance_sensitivity(args, image_counts)
    model_comparison(args)
    export_semantics_if_available(args)
    print("[8/8] complete", flush=True)
    print(f"results={args.results.resolve()}", flush=True)
    print(f"figures={args.figures.resolve()}", flush=True)


if __name__ == "__main__":
    main()
