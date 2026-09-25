"""Compact graph encoding shared utilities."""
from __future__ import annotations

import scripts._env  # noqa: F401

from pathlib import Path

import numpy as np

from scripts.image_graph_archetypes.utils import (
    ADJACENCIES,
    EDGE_COUNT,
    NODES,
    edge_definition,
)


SPECTRAL_RANK = 16
SPECTRAL_FEATURES = SPECTRAL_RANK * (SPECTRAL_RANK + 1) // 2
COMPACT_DIM = NODES + 1 + SPECTRAL_FEATURES
ALPHA = 20.0
EPSILON = 1e-6
OUTPUT_ROOT = Path("paper/data/image_graph_compact_archetypes")
FIGURE_ROOT = Path("paper/figures/main")
SOURCE_ROOT = Path("paper/data/image_graph_archetypes")


def fit_global_spectral_basis(
    graph_features: np.ndarray,
    training_indices: np.ndarray,
    rank: int = SPECTRAL_RANK,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one normalized-Laplacian basis from training mean adjacency."""
    edge_sum = np.zeros(EDGE_COUNT, dtype=np.float64)
    for start in range(0, len(training_indices), batch_size):
        indices = training_indices[start : start + batch_size]
        edge_sum += np.asarray(
            graph_features[indices, NODES:], dtype=np.float32
        ).sum(axis=0, dtype=np.float64)
    mean_edges = edge_sum / len(training_indices)
    total = edge_sum.sum()
    global_edge_prior = edge_sum / total if total > 0 else np.zeros_like(edge_sum)
    left, right, _ = edge_definition()
    adjacency = np.zeros((NODES, NODES), dtype=np.float64)
    adjacency[left, right] = mean_edges
    adjacency[right, left] = mean_edges
    degrees = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    normalized = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    laplacian = np.eye(NODES, dtype=np.float64) - normalized
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Skip only the constant connected-component mode. The training mean graph
    # is checked to be connected in the build report.
    basis = eigenvectors[:, 1 : rank + 1].astype(np.float32)
    if basis.shape != (NODES, rank):
        raise ValueError(f"spectral basis shape mismatch: {basis.shape}")
    if not np.allclose(basis.T @ basis, np.eye(rank), atol=1e-5):
        raise ValueError("spectral basis is not orthonormal")
    return basis, eigenvalues.astype(np.float32), mean_edges, global_edge_prior


def compact_encode(
    graph_block: np.ndarray,
    basis: np.ndarray,
    alpha: float = ALPHA,
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Encode node composition, boundary density, and spectral edge residuals.

    Edge smoothing shrinks each image toward its own independent-mixing null:

        q_tilde = (C + alpha*q0) / (C_cross + alpha)

    This avoids assigning prior contacts to F categories absent from the image.
    """
    values = np.asarray(graph_block, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != NODES + EDGE_COUNT:
        raise ValueError(f"expected N x 2080 graph block, got {values.shape}")
    area = values[:, :NODES]
    edges = values[:, NODES:]
    if not np.allclose(area.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("node areas do not sum to one")
    counts = edges * ADJACENCIES
    cross_contacts = counts.sum(axis=1)
    rho = cross_contacts / ADJACENCIES
    left, right, _ = edge_definition()
    denominator = 1.0 - np.sum(area * area, axis=1)
    q0 = np.zeros_like(edges, dtype=np.float32)
    valid = denominator > epsilon
    if valid.any():
        q0[valid] = (
            2.0 * area[valid][:, left] * area[valid][:, right]
            / denominator[valid, None]
        )
        # Float32 rounding can move the theoretical sum very slightly.
        q0_sum = q0[valid].sum(axis=1)
        q0[valid] /= np.maximum(q0_sum[:, None], epsilon)
    smoothed = (counts + alpha * q0) / (cross_contacts[:, None] + alpha)
    residual_edges = (smoothed - q0) / np.sqrt(q0 + epsilon)
    residual_edges[~np.isfinite(residual_edges)] = 0.0
    n = len(values)
    residual_matrix = np.zeros((n, NODES, NODES), dtype=np.float32)
    residual_matrix[:, left, right] = residual_edges
    residual_matrix[:, right, left] = residual_edges
    projected = np.matmul(residual_matrix, basis)
    spectral = np.einsum("ia,nib->nab", basis, projected, optimize=True)
    spectral = 0.5 * (spectral + spectral.transpose(0, 2, 1))
    upper = np.triu_indices(basis.shape[1])
    compact = np.empty((n, NODES + 1 + len(upper[0])), dtype=np.float32)
    compact[:, :NODES] = np.sqrt(np.maximum(area, 0.0))
    compact[:, NODES] = rho
    compact[:, NODES + 1 :] = spectral[:, upper[0], upper[1]]
    diagnostics = {
        "cross_contacts": cross_contacts,
        "rho": rho,
        "q0_sum": q0.sum(axis=1),
        "smoothed_sum": smoothed.sum(axis=1),
        "spectral_frobenius": np.sqrt(np.sum(spectral * spectral, axis=(1, 2))),
        "residual_frobenius": np.sqrt(
            2.0 * np.sum(residual_edges * residual_edges, axis=1)
        ),
    }
    return compact, diagnostics
