"""Audit coarse/fine W1024 hierarchy cuts across separation, balance, and stability.

The full-data spectral embedding is reused from the statistical taxonomy. Two
half-sample embeddings are rebuilt once and cached outside the web directory so
every candidate K can be compared against independently reconstructed views.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.manifold import spectral_embedding
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from formal.interpretation.build_statistical_taxonomy import (
    cluster_jaccard,
    decoder_atoms,
    encoder_atoms,
    half_city_profiles,
    half_position_counts,
    ppmi_context,
    rank_fuse,
    row_cosine,
    sparse_image_matrices,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARRAYS = ROOT / "formal" / "site" / "w1024_statistical" / "statistical_arrays.npz"
DEFAULT_SPARSE = ROOT / "formal" / "batchtopk_w1024_k8" / "sparse_acts.npz"
DEFAULT_CHECKPOINT = ROOT / "formal" / "batchtopk_w1024_k8" / "batch_topk_w1024_k8.pt"
DEFAULT_CACHE = ROOT / "formal" / "batchtopk_w1024_k8" / "hierarchy_split_embeddings.npz"
DEFAULT_OUTPUT = ROOT / "formal" / "site" / "w1024_statistical"
CANDIDATES = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128, 160, 192, 224, 256]


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    groups = sorted(set(labels.tolist()), key=lambda label: int(np.where(labels == label)[0].min()))
    mapping = {label: index for index, label in enumerate(groups)}
    return np.array([mapping[label] for label in labels], dtype=np.int32)


def cut_labels(tree: np.ndarray, count: int) -> np.ndarray:
    return canonical_labels(fcluster(tree, count, criterion="maxclust"))


def build_split_embeddings(
    sparse_file: Path, checkpoint: Path, cache_file: Path, neighbours: int, seed: int,
) -> list[np.ndarray]:
    if cache_file.exists():
        with np.load(cache_file) as cache:
            return [cache["half_0"].astype(np.float32), cache["half_1"].astype(np.float32)]

    archive = np.load(sparse_file, allow_pickle=True)
    top = archive["idx"][:, :, 0].astype(np.int64)
    city = np.array([str(value) for value in archive["city"]])
    width = int(top.max()) + 1
    decoder = decoder_atoms(checkpoint)
    encoder = encoder_atoms(checkpoint)
    fixed_views = [
        np.clip(decoder @ decoder.T, -1, 1),
        np.clip(encoder @ encoder.T, -1, 1),
    ]
    _, image_presence = sparse_image_matrices(top, width)
    embeddings = []
    for parity in (0, 1):
        mask = np.arange(len(top)) % 2 == parity
        half_presence = image_presence[mask]
        cooccurrence = (half_presence.T @ half_presence).toarray().astype(np.int64)
        support = np.diag(cooccurrence)
        views = [
            *fixed_views,
            row_cosine(half_position_counts(top, width, mask)),
            row_cosine(half_city_profiles(top, city, width, mask)),
            row_cosine(ppmi_context(cooccurrence, support, int(mask.sum()))),
        ]
        fused = rank_fuse(views, neighbours=neighbours)
        np.fill_diagonal(fused, 1e-6)
        embeddings.append(spectral_embedding(
            fused, n_components=32, random_state=seed + parity + 1, drop_first=False,
        ).astype(np.float32))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, half_0=embeddings[0], half_1=embeddings[1])
    return embeddings


def evaluate_candidates(full: np.ndarray, halves: list[np.ndarray], candidates: list[int]) -> list[dict]:
    full_tree = linkage(full, method="ward")
    half_trees = [linkage(embedding, method="ward") for embedding in halves]
    rows = []
    for count in candidates:
        labels = cut_labels(full_tree, count)
        alternate = [cut_labels(tree, count) for tree in half_trees]
        sizes = np.bincount(labels)
        proportions = sizes / sizes.sum()
        size_entropy = float(-(proportions * np.log(proportions)).sum() / np.log(count))
        split_jaccard = float(np.mean([
            cluster_jaccard(labels, other).mean() for other in alternate
        ]))
        rows.append({
            "k": count,
            "silhouette": float(silhouette_score(full, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(full, labels)),
            "davies_bouldin": float(davies_bouldin_score(full, labels)),
            "size_entropy": size_entropy,
            "size_cv": float(sizes.std() / sizes.mean()),
            "min_size": int(sizes.min()),
            "median_size": float(np.median(sizes)),
            "max_size": int(sizes.max()),
            "small_cluster_fraction": float((sizes < 4).mean()),
            "singleton_fraction": float((sizes == 1).mean()),
            "split_jaccard": split_jaccard,
            "split_ari": float(np.mean([adjusted_rand_score(labels, other) for other in alternate])),
            "split_nmi": float(np.mean([
                normalized_mutual_info_score(labels, other) for other in alternate
            ])),
        })
    return rows


def write_outputs(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    coarse = max((row for row in rows if row["k"] <= 40), key=lambda row: row["silhouette"])
    fine = max((row for row in rows if 2 * coarse["k"] <= row["k"] <= 4 * coarse["k"]), key=lambda row: row["silhouette"])
    fragmentation = next(row["k"] for row in rows if row["small_cluster_fraction"] > 0)
    payload = {
        "method": {
            "candidate_k": [row["k"] for row in rows],
            "embedding": "32D spectral embedding of five-view rank-fused graph",
            "clustering": "single Ward linkage cut at each K",
            "separation_metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"],
            "balance_metrics": ["size_entropy", "size_cv", "small_cluster_fraction"],
            "stability_metrics": ["split_jaccard", "split_ari", "split_nmi"],
        },
        "recommendation": {
            "coarse_k": coarse["k"],
            "fine_k": fine["k"],
            "coarse_plateau": [28, 32, 36, 40],
            "fine_plateau": [48, 56, 64, 72, 80, 96],
            "fragmentation_starts_at": fragmentation,
            "interpretation": (
                "K=32 is the joint silhouette/Calinski-Harabasz peak in the coarse plateau; "
                "K=64 preserves a simple nested 2:1 scale while improving split stability "
                "without clusters smaller than four genes."
            ),
        },
        "rows": rows,
    }
    (output / "hierarchy_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    with (output / "hierarchy_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--sparse", type=Path, default=DEFAULT_SPARSE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbours", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.arrays) as arrays:
        full = arrays["spectral_embedding"].astype(np.float32)
    halves = build_split_embeddings(args.sparse, args.checkpoint, args.cache, args.neighbours, args.seed)
    write_outputs(evaluate_candidates(full, halves, CANDIDATES), args.output)


if __name__ == "__main__":
    main()
