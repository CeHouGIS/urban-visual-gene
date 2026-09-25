#!/usr/bin/env python3
"""Build the 201D composition-corrected spectral graph encoding."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.image_graph_archetypes.utils import (
    EDGE_COUNT,
    NODES,
    SEED,
    assert_safe_affinity,
    balanced_training_indices,
    edge_index_frame,
    save_csv,
    save_json,
    save_parquet,
)
from scripts.image_graph_compact_archetypes.utils import (
    ALPHA,
    COMPACT_DIM,
    EPSILON,
    OUTPUT_ROOT,
    SOURCE_ROOT,
    SPECTRAL_FEATURES,
    SPECTRAL_RANK,
    compact_encode,
    fit_global_spectral_basis,
)


def make_qa_figure(features: np.ndarray, eigenvalues: np.ndarray, path: Path) -> None:
    """Summarize all three feature blocks without re-reading source imagery."""
    node = features[:, :NODES]
    density = features[:, NODES]
    topology = features[:, NODES + 1 :]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(eigenvalues, marker=".", markersize=3, linewidth=1)
    axes[0, 0].axvline(SPECTRAL_RANK, color="#b23a48", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Global normalized-Laplacian spectrum", xlabel="Eigenvalue index", ylabel="Eigenvalue")
    axes[0, 1].hist(density, bins=35, color="#3973ac")
    axes[0, 1].set(title="Cross-category boundary density", xlabel="rho", ylabel="Images")
    axes[1, 0].plot(np.mean(node, axis=0), color="#2a9d8f")
    axes[1, 0].set(title="Mean square-root node composition", xlabel="F category", ylabel="Mean value")
    axes[1, 1].plot(np.std(topology, axis=0), color="#e76f51")
    axes[1, 1].set(title="Spectral topology feature variability", xlabel="Upper-triangle coordinate", ylabel="Standard deviation")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Compact image-graph encoding QA (201D)", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp" + path.suffix)
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    temporary.replace(path)


def block_variance(features: np.ndarray) -> dict[str, float]:
    values = np.asarray(features, dtype=np.float32)
    variances = values.var(axis=0, dtype=np.float64)
    return {
        "node_variance_sum": float(variances[:NODES].sum()),
        "boundary_density_variance": float(variances[NODES]),
        "spectral_topology_variance_sum": float(variances[NODES + 1 :].sum()),
        "total_variance_sum": float(variances.sum()),
    }


def build(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    source_path = args.source / "image_graph_features.npy"
    metadata_path = args.source / "image_graph_metadata.parquet"
    source = np.load(source_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path)
    n = min(len(source), len(metadata))
    if args.limit:
        n = min(n, args.limit)
    if source.shape[1] != NODES + EDGE_COUNT or len(metadata) != len(source):
        raise ValueError("source V1 feature/metadata contract mismatch")
    metadata = metadata.iloc[:n].reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    save_parquet(metadata, args.output / "image_graph_metadata.parquet")
    save_csv(edge_index_frame(), args.output / "edge_index.csv")

    training_indices = balanced_training_indices(metadata, args.sample_per_city, args.seed)
    training_frame = metadata.iloc[training_indices][
        ["image_id", "panorama_id", "city", "image_path"]
    ].copy()
    training_frame.insert(0, "feature_row", training_indices)
    if training_frame.duplicated(["city", "panorama_id"]).any():
        raise ValueError("training sample contains duplicate panoramas within a city")
    save_csv(training_frame, args.output / "training_sample_ids.csv")

    basis, eigenvalues, mean_edges, global_prior = fit_global_spectral_basis(
        source, training_indices, rank=args.rank, batch_size=args.batch_size
    )
    zero_modes = int(np.count_nonzero(eigenvalues < 1e-7))
    if zero_modes != 1:
        raise ValueError(f"global mean graph is not connected: {zero_modes} zero modes")
    np.save(args.output / "global_graph_spectral_basis.npy", basis)
    np.save(args.output / "global_laplacian_eigenvalues.npy", eigenvalues)
    np.save(args.output / "global_mean_edge_weights.npy", mean_edges.astype(np.float32))
    np.save(args.output / "global_edge_prior.npy", global_prior.astype(np.float32))

    destination_path = args.output / "image_graph_compact_features.npy"
    temporary_path = args.output / "image_graph_compact_features.tmp.npy"
    destination = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.float32, shape=(n, COMPACT_DIM)
    )
    finite = True
    max_node_error = 0.0
    max_rho_error = 0.0
    q0_error = 0.0
    smoothed_error = 0.0
    for start in range(0, n, args.batch_size):
        stop = min(start + args.batch_size, n)
        original = np.asarray(source[start:stop], dtype=np.float32)
        compact, diagnostics = compact_encode(
            original, basis, alpha=args.alpha, epsilon=args.epsilon
        )
        destination[start:stop] = compact
        finite = finite and bool(np.isfinite(compact).all())
        max_node_error = max(
            max_node_error,
            float(np.abs(np.sum(compact[:, :NODES] ** 2, axis=1) - 1).max(initial=0)),
        )
        max_rho_error = max(
            max_rho_error,
            float(np.abs(compact[:, NODES] - original[:, NODES:].sum(axis=1)).max(initial=0)),
        )
        valid = diagnostics["q0_sum"] > 0
        if valid.any():
            q0_error = max(q0_error, float(np.abs(diagnostics["q0_sum"][valid] - 1).max()))
            smoothed_error = max(
                smoothed_error,
                float(np.abs(diagnostics["smoothed_sum"][valid] - 1).max()),
            )
        if start == 0 or stop == n or stop % 50000 < args.batch_size:
            print(f"encoded {stop:,}/{n:,}", flush=True)
    destination.flush()
    del destination
    os.replace(temporary_path, destination_path)
    encoded = np.load(destination_path, mmap_mode="r")
    if not finite or not np.isfinite(encoded).all():
        raise ValueError("compact encoding contains NaN or Inf")
    if max_node_error >= 1e-6 or max_rho_error >= 1e-6:
        raise ValueError("compact encoding QA tolerance failed")

    rng = np.random.default_rng(args.seed)
    qa_indices = np.sort(rng.choice(n, size=min(20, n), replace=False))
    rebuilt, _ = compact_encode(
        np.asarray(source[qa_indices], dtype=np.float32), basis,
        alpha=args.alpha, epsilon=args.epsilon,
    )
    exact_rebuild_error = float(
        np.max(np.abs(rebuilt - np.asarray(encoded[qa_indices], dtype=np.float32)))
    )
    training_compact = np.asarray(encoded[training_indices], dtype=np.float32)
    variance = block_variance(training_compact)
    make_qa_figure(training_compact, eigenvalues, args.qa_figure)

    parameters = {
        "encoding": "sqrt(node_area) + boundary_density + vech(U.T @ residual_adjacency @ U)",
        "dimensions": {
            "node_composition": NODES,
            "boundary_density": 1,
            "spectral_topology": SPECTRAL_FEATURES,
            "total": COMPACT_DIM,
        },
        "spectral_rank": int(args.rank),
        "edge_null": "independent mixing conditional on unlike-category contact",
        "edge_smoothing": "(C_ij + alpha*q0_ij) / (C_cross + alpha)",
        "alpha": float(args.alpha),
        "epsilon": float(args.epsilon),
        "basis": "non-constant eigenvectors of training-mean normalized Laplacian",
        "basis_city_labels_used": False,
        "semantic_labels_used": False,
    }
    save_json(parameters, args.output / "compact_encoding_parameters.json")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_feature_file": str(source_path),
        "images": int(n),
        "cities": int(metadata["city"].nunique()),
        "training_images": int(len(training_indices)),
        "training_panorama_deduplicated_within_city": True,
        "shape": [int(n), COMPACT_DIM],
        "dtype": "float32",
        "finite": finite,
        "maximum_squared_node_sum_error": max_node_error,
        "maximum_boundary_density_reconstruction_error": max_rho_error,
        "maximum_q0_sum_error": q0_error,
        "maximum_smoothed_edge_sum_error": smoothed_error,
        "random_20_exact_rebuild_max_abs_error": exact_rebuild_error,
        "global_laplacian_zero_modes": zero_modes,
        "spectral_basis_orthonormal_max_abs_error": float(
            np.abs(basis.T @ basis - np.eye(args.rank)).max()
        ),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        **variance,
    }
    save_json(report, args.output / "compact_feature_report.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--qa-figure", type=Path, default=OUTPUT_ROOT / "Fig_Compact_Graph_QA.png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-per-city", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=SPECTRAL_RANK)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
