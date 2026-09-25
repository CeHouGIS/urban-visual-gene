"""Shared utilities for the image graph archetype experiment."""
from __future__ import annotations

import scripts._env  # noqa: F401

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA


GRID = 14
PATCHES = GRID * GRID
ADJACENCIES = GRID * (GRID - 1) + (GRID - 1) * GRID
NODES = 64
EDGE_COUNT = NODES * (NODES - 1) // 2
DESCRIPTOR_DIM = NODES + EDGE_COUNT
SEED = 42

DATA_ROOT = Path("outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc")
SOURCE_HIERARCHY = DATA_ROOT / "mae" / "hierarchy_32_64"
MAIN_HIERARCHY = DATA_ROOT / "mae" / "hierarchy_edp_32_64"
TOP_PATH = SOURCE_HIERARCHY / "top1_dimensions.npy"
METADATA_PATH = Path("results/image_dimension_pixel_proportions.parquet")
OUTPUT_ROOT = Path("paper/data/image_graph_archetypes")
FIGURE_ROOT = Path("paper/figures/main")


def assert_safe_affinity() -> None:
    """Fail before computation if forbidden logical CPUs are available to us."""
    if hasattr(os, "sched_getaffinity"):
        active = set(os.sched_getaffinity(0))
        forbidden = active.intersection({8, 9})
        if forbidden:
            raise RuntimeError(
                f"unsafe CPU affinity includes forbidden CPUs: {sorted(forbidden)}; "
                "run with taskset -c 0-7,10-15"
            )


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def edge_definition() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(NODES, 1)
    lookup = np.full((NODES, NODES), -1, dtype=np.int16)
    ids = np.arange(EDGE_COUNT, dtype=np.int16)
    lookup[left, right] = ids
    lookup[right, left] = ids
    return left.astype(np.int16), right.astype(np.int16), lookup


def edge_index_frame() -> pd.DataFrame:
    left, right, _ = edge_definition()
    return pd.DataFrame(
        {
            "edge_id": np.arange(EDGE_COUNT, dtype=np.int32),
            "F_i": [f"F{x:03d}" for x in left],
            "F_j": [f"F{x:03d}" for x in right],
            "node_i": left,
            "node_j": right,
        }
    )


def graph_descriptors(
    top_dimensions: np.ndarray,
    fine_labels: np.ndarray,
    edge_lookup: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct node areas and cross-category 4-neighbour edge weights."""
    top = np.asarray(top_dimensions, dtype=np.int64)
    if top.ndim != 2 or top.shape[1] != PATCHES:
        raise ValueError(f"expected N x {PATCHES} top dimensions, got {top.shape}")
    labels = fine_labels[top]
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= NODES:
        raise ValueError("invalid F category ID")
    n = len(labels)
    rows = np.repeat(np.arange(n, dtype=np.int64), PATCHES)
    node_keys = rows * NODES + labels.ravel()
    node_counts = np.bincount(node_keys, minlength=n * NODES).reshape(n, NODES)

    maps = labels.reshape(n, GRID, GRID)
    first = np.concatenate(
        (maps[:, :, :-1].reshape(n, -1), maps[:, :-1, :].reshape(n, -1)),
        axis=1,
    )
    second = np.concatenate(
        (maps[:, :, 1:].reshape(n, -1), maps[:, 1:, :].reshape(n, -1)),
        axis=1,
    )
    if first.shape[1] != ADJACENCIES or second.shape[1] != ADJACENCIES:
        raise ValueError("4-neighbour pair count is not 364")
    different = first != second
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    edge_ids = edge_lookup[low, high]
    valid_rows, valid_columns = np.nonzero(different)
    valid_edges = edge_ids[valid_rows, valid_columns]
    if len(valid_edges) and valid_edges.min() < 0:
        raise ValueError("invalid unordered edge lookup")
    edge_keys = valid_rows.astype(np.int64) * EDGE_COUNT + valid_edges.astype(np.int64)
    edge_counts = np.bincount(
        edge_keys, minlength=n * EDGE_COUNT
    ).reshape(n, EDGE_COUNT)

    output = np.empty((n, DESCRIPTOR_DIM), dtype=np.float32)
    output[:, :NODES] = node_counts / PATCHES
    output[:, NODES:] = edge_counts / ADJACENCIES
    cross_boundary_counts = different.sum(axis=1).astype(np.int16)
    return output, cross_boundary_counts


def validate_descriptor_block(
    features: np.ndarray, cross_boundary_counts: np.ndarray
) -> dict[str, float | int]:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != DESCRIPTOR_DIM:
        raise ValueError(f"invalid descriptor shape: {values.shape}")
    node_error = np.abs(values[:, :NODES].sum(axis=1) - 1.0)
    if node_error.max(initial=0.0) >= 1e-6:
        raise ValueError(f"node areas do not sum to one: max error={node_error.max()}")
    if not np.isfinite(values).all():
        raise ValueError("graph descriptor contains NaN or Inf")
    if (values < 0).any():
        raise ValueError("graph descriptor contains negative weights")
    edge_contacts = values[:, NODES:].sum(axis=1) * ADJACENCIES
    if not np.allclose(edge_contacts, cross_boundary_counts, atol=1e-5):
        raise ValueError("normalized edge weights do not reconstruct boundary counts")
    return {
        "images": int(len(values)),
        "maximum_node_sum_error": float(node_error.max(initial=0.0)),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "minimum_cross_category_boundaries": int(cross_boundary_counts.min(initial=0)),
        "maximum_cross_category_boundaries": int(cross_boundary_counts.max(initial=0)),
        "adjacency_pairs_per_image": ADJACENCIES,
    }


def read_metadata(path: Path = METADATA_PATH, limit: int = 0) -> pd.DataFrame:
    columns = [
        "global_image_index", "city_key", "image_index", "source_image_index",
        "pano_index", "panoid", "direction_index", "heading", "lat", "lon",
        "year", "month", "tar_path", "jpg_offset", "jpg_size",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if limit:
        frame = frame.iloc[:limit].copy()
    frame = frame.reset_index(drop=True)
    expected = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["global_image_index"].to_numpy(), expected):
        raise ValueError("metadata is not aligned with top1_dimensions.npy")
    output = pd.DataFrame(
        {
            "image_id": frame["global_image_index"].astype(np.int64),
            "panorama_id": frame["panoid"].astype(str),
            "city": frame["city_key"].astype(str),
            # Source imagery is stored inside TAR archives. image_path names the
            # archive; offset and size identify the exact JPEG payload.
            "image_path": frame["tar_path"].astype(str),
            "jpg_offset": frame["jpg_offset"].astype(np.int64),
            "jpg_size": frame["jpg_size"].astype(np.int64),
            "direction_index": frame["direction_index"],
            "heading": frame["heading"],
            "lat": frame["lat"],
            "lon": frame["lon"],
            "year": frame["year"],
            "month": frame["month"],
            "source_image_index": frame["source_image_index"],
        }
    )
    return output


def load_fine_labels(path: Path = MAIN_HIERARCHY) -> np.ndarray:
    with np.load(path / "hierarchy_arrays.npz") as arrays:
        labels = arrays["fine_labels"].astype(np.int64)
    if labels.shape != (512,) or labels.min() != 0 or labels.max() != 63:
        raise ValueError("frozen hierarchy must map 512 dimensions to F000--F063")
    return labels


def read_original(row: pd.Series) -> Image.Image:
    fd = os.open(str(row["image_path"]), os.O_RDONLY)
    try:
        payload = os.pread(fd, int(row["jpg_size"]), int(row["jpg_offset"]))
    finally:
        os.close(fd)
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB")


def balanced_training_indices(
    metadata: pd.DataFrame, per_city: int, seed: int = SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    panoids = metadata["panorama_id"].to_numpy()
    selected: list[int] = []
    for city, group in metadata.groupby("city", sort=True):
        order = rng.permutation(group.index.to_numpy(np.int64))
        seen: set[str] = set()
        rows: list[int] = []
        for index in order:
            panoid = str(panoids[index])
            if panoid in seen:
                continue
            seen.add(panoid)
            rows.append(int(index))
            if len(rows) == per_city:
                break
        if not rows:
            raise ValueError(f"no training images for {city}")
        if len(rows) < per_city:
            print(
                f"warning: {city} provides {len(rows):,} unique panoramas, "
                f"below requested {per_city:,}", flush=True,
            )
        selected.extend(rows)
    return np.asarray(sorted(selected), dtype=np.int64)


@dataclass
class FixedPCAModel:
    """Small serializable PCA transform containing only retained components."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    n_features_in_: int

    @property
    def n_components_(self) -> int:
        return int(len(self.components_))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return ((array - self.mean_) @ self.components_.T).astype(np.float32)


def fit_variance_pca(
    training: np.ndarray,
    target: float = 0.90,
    seed: int = SEED,
    initial_components: int = 256,
) -> FixedPCAModel:
    """Fit randomized PCA, increasing rank until target variance is reached."""
    values = np.asarray(training, dtype=np.float32)
    maximum = min(values.shape[0] - 1, values.shape[1] - 1)
    if maximum < 2:
        raise ValueError("not enough samples/features for PCA")
    components = min(initial_components, maximum)
    fitted: PCA | None = None
    while True:
        print(f"PCA trial with {components} components", flush=True)
        fitted = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=seed,
            iterated_power=4,
        ).fit(values)
        cumulative = np.cumsum(fitted.explained_variance_ratio_)
        if cumulative[-1] >= target or components == maximum:
            break
        next_components = min(max(components + 128, int(components * 1.5)), maximum)
        if next_components == components:
            break
        components = next_components
    if fitted is None:
        raise RuntimeError("PCA fitting did not run")
    cumulative = np.cumsum(fitted.explained_variance_ratio_)
    retained = int(np.searchsorted(cumulative, target) + 1)
    if retained > len(cumulative) or cumulative[-1] < target:
        raise ValueError(
            f"PCA failed to retain {target:.1%} variance; reached {cumulative[-1]:.3%}"
        )
    model = FixedPCAModel(
        mean_=fitted.mean_.astype(np.float32),
        components_=fitted.components_[:retained].astype(np.float32),
        explained_variance_=fitted.explained_variance_[:retained].astype(np.float32),
        explained_variance_ratio_=fitted.explained_variance_ratio_[:retained].astype(np.float32),
        singular_values_=fitted.singular_values_[:retained].astype(np.float32),
        n_features_in_=values.shape[1],
    )
    print(
        f"retained {retained} PCA components, "
        f"variance={model.explained_variance_ratio_.sum():.3%}", flush=True,
    )
    return model


def dump_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)


def transform_batches(
    features: np.ndarray,
    model: FixedPCAModel,
    indices: np.ndarray | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    if indices is None:
        n = len(features)
    else:
        n = len(indices)
    output = np.empty((n, model.n_components_), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        if indices is None:
            block = np.asarray(features[start:stop], dtype=np.float32)
        else:
            block = np.asarray(features[indices[start:stop]], dtype=np.float32)
        output[start:stop] = model.transform(block)
    return output
