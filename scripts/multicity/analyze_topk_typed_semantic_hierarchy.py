#!/usr/bin/env python3
"""Discover a macro-scale typed hierarchy from per-patch Top-X winners.

The experiment deliberately discards activation magnitudes after selecting the
Top-X Feature-MAE dimensions of every 14x56 panorama patch.  It streams the
30-city panorama corpus, stores only panorama-level winner counts, and derives
two kinds of dimension relations:

* AND / composition: dimensions co-occur across panoramas more often than a
  city-conditioned independence model predicts;
* OR / substitution: dimensions avoid one another but have similar relations
  to the remaining dimensions and similar spatial-position profiles.

Mapillary profiles are used only to validate the Top-X hyperparameter and to
name/audit the discovered groups; they never enter the discovery fingerprint,
hierarchical linkage, or AND/OR relation scores.
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (must precede numpy / scipy / sklearn)

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


STORAGE_ROOT = Path("/workplace/urban_visual_gene")
FEATURE_ROOT = (
    STORAGE_ROOT
    / "outputs/experiments/dinov3_multicity/rectangular_panorama_frozen_mae_n30"
)
SEMANTIC_ROOT = (
    STORAGE_ROOT
    / "paper/data/semantic_alignment/feature_mae_mapillary_alignment_n60000"
)
CACHE_ROOT = (
    STORAGE_ROOT
    / "outputs/experiments/dinov3_multicity/feature_mae_top4_typed_hierarchy"
)
PAPER_DATA = STORAGE_ROOT / "paper/data/feature_mae_top4_typed_hierarchy"
PAPER_FIGURES = (
    STORAGE_ROOT
    / "paper/figures/supplementary/feature_mae_top4_typed_hierarchy"
)

ROWS = 14
COLS = 56
PATCHES = ROWS * COLS
LOCAL_POSITIONS = ROWS * 14
DIMENSIONS = 512
TOP_X_CANDIDATES = (1, 2, 3, 4, 5, 8, 10)
DEFAULT_TOP_X = 4
DEFAULT_PRESENCE_PATCHES = 16


def assert_safe_affinity() -> None:
    """Fail fast rather than accidentally using the forbidden CPU cores."""
    if hasattr(os, "sched_getaffinity"):
        forbidden = set(os.sched_getaffinity(0)).intersection({8, 9})
        if forbidden:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(forbidden)}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def l2_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def panorama_topk_counts(block: np.ndarray, top_x: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-panorama dimension counts and unsorted Top-X patch indices."""
    values = np.asarray(block)
    if values.ndim != 4 or values.shape[1:] != (ROWS, COLS, DIMENSIONS):
        raise ValueError(f"expected (N,{ROWS},{COLS},{DIMENSIONS}), got {values.shape}")
    if not 0 < top_x <= DIMENSIONS:
        raise ValueError(f"top_x must be in [1,{DIMENSIONS}], got {top_x}")
    top = np.argpartition(values, DIMENSIONS - top_x, axis=-1)[..., -top_x:]
    flat = top.reshape(len(values), PATCHES * top_x)
    row_ids = np.repeat(np.arange(len(values), dtype=np.int64), flat.shape[1])
    keys = row_ids * DIMENSIONS + flat.ravel()
    counts = np.bincount(
        keys, minlength=len(values) * DIMENSIONS
    ).reshape(len(values), DIMENSIONS)
    if not np.all(counts.sum(axis=1) == PATCHES * top_x):
        raise ValueError("Top-X panorama counts do not match patch assignment total")
    return counts.astype(np.uint16), top.astype(np.uint16)


def patch_statistics(top: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate dimension, local-position, and same-patch pair counts."""
    if top.ndim != 4 or top.shape[1:3] != (ROWS, COLS):
        raise ValueError(f"unexpected Top-X tensor {top.shape}")
    top_x = top.shape[-1]
    flat = top.reshape(-1, top_x).astype(np.int64)
    dimension_counts = np.bincount(flat.ravel(), minlength=DIMENSIONS).astype(np.uint64)

    row = np.repeat(np.arange(ROWS), COLS)
    col = np.tile(np.arange(COLS), ROWS) % 14
    local_position = row * 14 + col
    positions = np.tile(np.repeat(local_position, top_x), len(top))
    position_keys = flat.ravel() * LOCAL_POSITIONS + positions
    position_counts = np.bincount(
        position_keys, minlength=DIMENSIONS * LOCAL_POSITIONS
    ).reshape(DIMENSIONS, LOCAL_POSITIONS).astype(np.uint64)

    pair_counts = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.uint64)
    for left in range(top_x):
        for right in range(left + 1, top_x):
            first = flat[:, left]
            second = flat[:, right]
            low = np.minimum(first, second)
            high = np.maximum(first, second)
            keys = low * DIMENSIONS + high
            pair_counts += np.bincount(
                keys, minlength=DIMENSIONS * DIMENSIONS
            ).reshape(DIMENSIONS, DIMENSIONS).astype(np.uint64)
    return dimension_counts, position_counts, pair_counts


def choose_top_x(frame: pd.DataFrame) -> int:
    """Choose the smallest informative X using predeclared macro-scale rules."""
    required = {
        "top_x",
        "dimension_coverage",
        "city_marginal_cosine_mean",
        "same_mapillary_parent_pair_share",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Top-X diagnostics missing {sorted(missing)}")
    baseline_rows = frame.loc[frame.top_x.eq(2)]
    if baseline_rows.empty:
        raise ValueError("Top-X diagnostics require X=2 as semantic-purity baseline")
    purity_floor = 0.90 * float(baseline_rows.iloc[0].same_mapillary_parent_pair_share)
    eligible = frame.loc[
        frame.top_x.ge(2)
        & frame.dimension_coverage.ge(0.95)
        & frame.city_marginal_cosine_mean.ge(0.85)
        & frame.same_mapillary_parent_pair_share.ge(purity_floor)
    ].sort_values("top_x")
    if eligible.empty:
        raise ValueError("no Top-X candidate satisfies coverage/stability/purity rules")
    return int(eligible.iloc[0].top_x)


def city_slug(path: Path) -> str:
    return path.parent.parent.name


def prediction_paths(feature_root: Path) -> list[Path]:
    paths = sorted(
        (feature_root / "predictions").glob(
            "*/full/panorama_feature_mae_latent.f16.npy"
        )
    )
    if len(paths) != 30:
        raise ValueError(f"expected 30 complete city predictions, found {len(paths)}")
    return paths


def city_table(feature_root: Path, max_panoramas_per_city: int = 0) -> pd.DataFrame:
    rows = []
    offset = 0
    for city_id, path in enumerate(prediction_paths(feature_root)):
        latent = np.load(path, mmap_mode="r")
        if latent.shape[1:] != (ROWS, COLS, DIMENSIONS):
            raise ValueError(f"invalid latent shape at {path}: {latent.shape}")
        panoramas = len(latent)
        if max_panoramas_per_city > 0:
            panoramas = min(panoramas, max_panoramas_per_city)
        rows.append(
            {
                "city_id": city_id,
                "city_slug": city_slug(path),
                "city": city_slug(path).replace("__", "/"),
                "panoramas": panoramas,
                "start": offset,
                "end": offset + panoramas,
                "latent_path": str(path),
            }
        )
        offset += panoramas
    return pd.DataFrame(rows)


def run_top_x_selection(
    feature_root: Path,
    semantic_root: Path,
    output: Path,
    panoramas_per_city: int,
) -> pd.DataFrame:
    """Evaluate candidate X values on an equal-city panorama sample."""
    metrics = pd.read_csv(semantic_root / "dimension_semantic_profiles.csv").sort_values(
        "dimension_id"
    )
    if metrics.dimension_id.tolist() != list(range(DIMENSIONS)):
        raise ValueError("semantic profiles must contain D000--D511 exactly once")
    dominant = metrics.top1_class_id.to_numpy(np.int16)
    fine = metrics.fine_id.to_numpy(np.int16)
    maximum = max(TOP_X_CANDIDATES) + 1
    rank_sum = np.zeros(maximum, dtype=np.float64)
    rank_square_sum = np.zeros(maximum, dtype=np.float64)
    observations = 0
    mass_sum = {x: 0.0 for x in TOP_X_CANDIDATES}
    positive = {x: 0 for x in TOP_X_CANDIDATES}
    same_semantic = {x: 0 for x in TOP_X_CANDIDATES}
    same_fine = {x: 0 for x in TOP_X_CANDIDATES}
    pair_total = {x: 0 for x in TOP_X_CANDIDATES}
    frequencies = {
        x: np.zeros(DIMENSIONS, dtype=np.int64) for x in TOP_X_CANDIDATES
    }
    city_frequencies = {x: [] for x in TOP_X_CANDIDATES}

    for city_index, path in enumerate(prediction_paths(feature_root), start=1):
        latent = np.load(path, mmap_mode="r")
        ids = np.linspace(
            0, len(latent) - 1, min(panoramas_per_city, len(latent)), dtype=np.int64
        )
        local = {x: np.zeros(DIMENSIONS, dtype=np.int64) for x in TOP_X_CANDIDATES}
        for start in range(0, len(ids), 4):
            values = np.asarray(latent[ids[start : start + 4]], dtype=np.float32).reshape(
                -1, DIMENSIONS
            )
            partition = np.argpartition(values, -maximum, axis=1)[:, -maximum:]
            top_values = np.take_along_axis(values, partition, axis=1)
            order = np.argsort(top_values, axis=1)[:, ::-1]
            indices = np.take_along_axis(partition, order, axis=1)
            top_values = np.take_along_axis(top_values, order, axis=1)
            rank_sum += top_values.sum(axis=0, dtype=np.float64)
            rank_square_sum += np.square(top_values, dtype=np.float64).sum(axis=0)
            observations += len(values)
            maxima = values.max(axis=1, keepdims=True)
            denominator = np.exp(values - maxima, dtype=np.float32).sum(axis=1)
            cumulative = np.cumsum(
                np.exp(top_values[:, : max(TOP_X_CANDIDATES)] - maxima, dtype=np.float32),
                axis=1,
            )
            for top_x in TOP_X_CANDIDATES:
                mass_sum[top_x] += float((cumulative[:, top_x - 1] / denominator).sum())
                positive[top_x] += int((top_values[:, top_x - 1] > 0).sum())
                chosen = indices[:, :top_x]
                counts = np.bincount(chosen.ravel(), minlength=DIMENSIONS)
                frequencies[top_x] += counts
                local[top_x] += counts
                for left in range(top_x):
                    for right in range(left + 1, top_x):
                        first = chosen[:, left]
                        second = chosen[:, right]
                        same_semantic[top_x] += int(
                            (dominant[first] == dominant[second]).sum()
                        )
                        same_fine[top_x] += int((fine[first] == fine[second]).sum())
                        pair_total[top_x] += len(first)
        for top_x in TOP_X_CANDIDATES:
            city_frequencies[top_x].append(local[top_x] / local[top_x].sum())
        print(
            f"Top-X [{city_index:02d}/30] {city_slug(path)}: {len(ids):,} panoramas",
            flush=True,
        )

    rank_mean = rank_sum / observations
    rank_sd = np.sqrt(rank_square_sum / observations - np.square(rank_mean))
    records = []
    for top_x in TOP_X_CANDIDATES:
        cities = l2_rows(np.stack(city_frequencies[top_x]))
        similarities = cities @ cities.T
        upper = similarities[np.triu_indices(len(cities), 1)]
        record = {
            "top_x": top_x,
            "sample_panoramas": 30 * panoramas_per_city,
            "sample_patches": observations,
            "softmax_mass_mean": mass_sum[top_x] / observations,
            "xth_score_mean": rank_mean[top_x - 1],
            "xth_score_sd": rank_sd[top_x - 1],
            "xth_positive_fraction": positive[top_x] / observations,
            "next_score_gap_mean": rank_mean[top_x - 1] - rank_mean[top_x],
            "city_marginal_cosine_mean": float(upper.mean()),
            "dimensions_used": int((frequencies[top_x] > 0).sum()),
            "dimension_coverage": float((frequencies[top_x] > 0).mean()),
            "same_mapillary_parent_pair_share": np.nan,
            "same_f64_pair_share": np.nan,
        }
        if top_x > 1:
            record["same_mapillary_parent_pair_share"] = (
                same_semantic[top_x] / pair_total[top_x]
            )
            record["same_f64_pair_share"] = same_fine[top_x] / pair_total[top_x]
        records.append(record)
    frame = pd.DataFrame(records)
    selected = choose_top_x(frame)
    frame["selected"] = frame.top_x.eq(selected)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "top_x_selection.csv", index=False)
    save_json(
        output / "top_x_selection_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selected_top_x": selected,
            "selection_rule": (
                "smallest X>=2 with >=95% dimension coverage, >=0.85 mean "
                "cross-city marginal cosine, and >=90% of Top-2 Mapillary-parent purity"
            ),
            "sample_cities": 30,
            "sample_panoramas_per_city": panoramas_per_city,
            "sample_patches": observations,
        },
    )
    return frame


def manifest_panoramas(feature_root: Path, city: str, expected: int) -> pd.DataFrame:
    manifest = pd.read_parquet(
        feature_root / "manifests" / f"{city}.parquet",
        columns=["panoid", "city_key", "lat", "lon", "year", "month", "pano_index"],
    )
    frame = (
        manifest.sort_values(["pano_index"])
        .drop_duplicates("pano_index")
        .reset_index(drop=True)
    )
    if len(frame) < expected:
        raise ValueError(f"manifest contains fewer than {expected} panoramas for {city}")
    frame = frame.iloc[:expected].copy()
    if frame.pano_index.tolist() != list(range(expected)):
        raise ValueError(f"manifest/latent alignment failed for {city}")
    return frame.rename(columns={"panoid": "panorama_id", "city_key": "city"})


def scan_full_corpus(
    feature_root: Path,
    cache_root: Path,
    top_x: int,
    chunk_panoramas: int,
    max_panoramas_per_city: int = 0,
) -> pd.DataFrame:
    """Stream all dense latents once and cache only Top-X sufficient statistics."""
    table = city_table(feature_root, max_panoramas_per_city)
    total = int(table.end.iloc[-1])
    cache_root.mkdir(parents=True, exist_ok=True)
    stats_root = cache_root / "city_stats"
    stats_root.mkdir(exist_ok=True)
    counts_path = cache_root / f"panorama_top{top_x}_counts.u16.npy"
    if counts_path.exists():
        counts = np.load(counts_path, mmap_mode="r+")
        if counts.shape != (total, DIMENSIONS) or counts.dtype != np.uint16:
            raise ValueError(f"incompatible existing count cache {counts.shape}/{counts.dtype}")
    else:
        counts = np.lib.format.open_memmap(
            counts_path, mode="w+", dtype=np.uint16, shape=(total, DIMENSIONS)
        )

    metadata_path = cache_root / "panorama_metadata.parquet"
    if not metadata_path.exists():
        metadata_parts = []
        for row in table.itertuples(index=False):
            part = manifest_panoramas(feature_root, row.city_slug, row.panoramas)
            part.insert(0, "global_panorama_index", np.arange(row.start, row.end))
            metadata_parts.append(part)
        pd.concat(metadata_parts, ignore_index=True).to_parquet(metadata_path, index=False)
    table.to_csv(cache_root / "city_index.csv", index=False)

    for city_number, row in enumerate(table.itertuples(index=False), start=1):
        stats_path = stats_root / f"{row.city_slug}.npz"
        if stats_path.exists():
            print(f"scan [{city_number:02d}/30] {row.city}: cached", flush=True)
            continue
        latent = np.load(row.latent_path, mmap_mode="r")
        dimension_counts = np.zeros(DIMENSIONS, dtype=np.uint64)
        position_counts = np.zeros((DIMENSIONS, LOCAL_POSITIONS), dtype=np.uint64)
        pair_counts = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.uint64)
        for start in range(0, row.panoramas, chunk_panoramas):
            stop = min(start + chunk_panoramas, row.panoramas)
            block = np.asarray(latent[start:stop])
            panorama_counts, top = panorama_topk_counts(block, top_x)
            counts[row.start + start : row.start + stop] = panorama_counts
            local_dimension, local_position, local_pairs = patch_statistics(top)
            dimension_counts += local_dimension
            position_counts += local_position
            pair_counts += local_pairs
            if start == 0 or stop == row.panoramas or (start // chunk_panoramas) % 100 == 0:
                print(
                    f"scan [{city_number:02d}/30] {row.city}: {stop:,}/{row.panoramas:,}",
                    flush=True,
                )
        counts.flush()
        expected_assignments = row.panoramas * PATCHES * top_x
        if int(dimension_counts.sum()) != expected_assignments:
            raise ValueError(f"assignment QA failed for {row.city}")
        if int(pair_counts.sum()) != row.panoramas * PATCHES * math.comb(top_x, 2):
            raise ValueError(f"same-patch pair QA failed for {row.city}")
        np.savez_compressed(
            stats_path,
            dimension_counts=dimension_counts,
            position_counts=position_counts,
            pair_counts=pair_counts,
            panoramas=np.asarray([row.panoramas], dtype=np.int64),
        )
    save_json(
        cache_root / "scan_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "top_x": top_x,
            "cities": len(table),
            "panoramas": total,
            "patches": total * PATCHES,
            "winner_assignments": total * PATCHES * top_x,
            "counts_shape": [total, DIMENSIONS],
            "counts_dtype": "uint16",
            "complete": True,
        },
    )
    return table


def city_conditioned_pair_statistics(
    occurrence: Iterable[np.ndarray], sizes: Iterable[int]
) -> dict[str, np.ndarray]:
    """Aggregate observed/expected matrices while conditioning on city."""
    observed_total = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.float64)
    expected_total = np.zeros_like(observed_total)
    positive = np.zeros_like(observed_total)
    negative = np.zeros_like(observed_total)
    evaluated = np.zeros_like(observed_total)
    support_total = np.zeros(DIMENSIONS, dtype=np.float64)
    for observed, size in zip(occurrence, sizes):
        observed = np.asarray(observed, dtype=np.float64)
        support = np.diag(observed).copy()
        expected = np.outer(support, support) / float(size)
        lift = np.log2((observed + 0.5) / (expected + 0.5))
        valid = expected >= 5
        observed_total += observed
        expected_total += expected
        positive += valid & (lift > 0)
        negative += valid & (lift < 0)
        evaluated += valid
        support_total += support
    np.fill_diagonal(expected_total, support_total)
    return {
        "observed": observed_total,
        "expected": expected_total,
        "support": support_total,
        "city_positive_fraction": positive / np.maximum(evaluated, 1),
        "city_negative_fraction": negative / np.maximum(evaluated, 1),
        "city_evaluated": evaluated,
    }


def panorama_cooccurrence(
    counts: np.ndarray,
    table: pd.DataFrame,
    presence_patches: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    matrices = []
    city_prevalence = []
    for number, row in enumerate(table.itertuples(index=False), start=1):
        binary = np.asarray(counts[row.start : row.end] >= presence_patches)
        design = sparse.csr_matrix(binary, dtype=np.float32)
        cooccurrence = (design.T @ design).toarray().astype(np.float64)
        support = binary.sum(axis=0, dtype=np.int64)
        np.fill_diagonal(cooccurrence, support)
        matrices.append(cooccurrence)
        city_prevalence.append(support / len(binary))
        print(
            f"cooccurrence [{number:02d}/30] {row.city}: "
            f"mean {binary.sum(axis=1).mean():.1f} active dimensions",
            flush=True,
        )
    return matrices, np.stack(city_prevalence)


def classify_pair_relations(
    conditioned_log_lift: np.ndarray,
    residual: np.ndarray,
    role_similarity: np.ndarray,
    position_similarity: np.ndarray,
    city_positive: np.ndarray,
    city_negative: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    reliability = 1.0 - np.exp(-expected / 100.0)
    and_score = (
        np.maximum(conditioned_log_lift, 0)
        * city_positive
        * reliability
    )
    or_score = (
        np.maximum(-conditioned_log_lift, 0)
        * role_similarity
        * np.sqrt(np.clip(position_similarity, 0, 1))
        * city_negative
        * reliability
    )
    np.fill_diagonal(and_score, 0)
    np.fill_diagonal(or_score, 0)
    upper = np.triu_indices(DIMENSIONS, 1)
    and_candidates = (
        (conditioned_log_lift[upper] > 0)
        & (residual[upper] > 5)
        & (city_positive[upper] >= 0.60)
        & (expected[upper] >= 25)
    )
    role_floor = float(np.quantile(role_similarity[upper], 0.75))
    or_candidates = (
        (conditioned_log_lift[upper] < 0)
        & (residual[upper] < -5)
        & (city_negative[upper] >= 0.60)
        & (role_similarity[upper] >= role_floor)
        & (position_similarity[upper] >= 0.75)
        & (expected[upper] >= 25)
    )
    and_floor = (
        float(np.quantile(and_score[upper][and_candidates], 0.95))
        if and_candidates.any()
        else math.inf
    )
    # Substitution is intrinsically rarer than positive co-occurrence. The
    # candidate gate above is already deliberately strict, so retain its top
    # quartile rather than applying the AND relation's top-five-percent rule.
    or_floor = (
        float(np.quantile(or_score[upper][or_candidates], 0.75))
        if or_candidates.any()
        else math.inf
    )
    relation = np.full((DIMENSIONS, DIMENSIONS), "UNRESOLVED", dtype=object)
    and_mask = np.zeros_like(and_score, dtype=bool)
    or_mask = np.zeros_like(or_score, dtype=bool)
    and_mask[upper] = and_candidates & (and_score[upper] >= and_floor)
    or_mask[upper] = or_candidates & (or_score[upper] >= or_floor)
    and_mask |= and_mask.T
    or_mask |= or_mask.T
    relation[and_mask] = "AND"
    relation[or_mask & ~and_mask] = "OR"
    return relation, and_score, or_score, {
        "and_score_floor": and_floor if math.isfinite(and_floor) else None,
        "or_score_floor": or_floor if math.isfinite(or_floor) else None,
        "role_similarity_floor": role_floor,
        "and_candidate_pairs": int(and_candidates.sum()),
        "or_candidate_pairs": int(or_candidates.sum()),
        "and_score_quantile": 0.95,
        "or_score_quantile": 0.75,
        "and_pairs": int(and_mask[upper].sum()),
        "or_pairs": int((or_mask & ~and_mask)[upper].sum()),
    }


def semantic_matrix(semantic_root: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    metrics = pd.read_csv(semantic_root / "dimension_semantic_profiles.csv").sort_values(
        "dimension_id"
    )
    distribution = pd.read_csv(semantic_root / "dimension_semantic_distribution_65.csv")
    profiles = (
        distribution.pivot(
            index="dimension_id", columns="class_id", values="semantic_fraction"
        )
        .reindex(index=range(DIMENSIONS), columns=range(65))
        .to_numpy(dtype=np.float64)
    )
    names = (
        distribution.drop_duplicates("class_id")
        .sort_values("class_id")
        .class_name.astype(str)
        .tolist()
    )
    return metrics, profiles, names


def select_macro_cut(linkage_matrix: np.ndarray, distance: np.ndarray) -> tuple[int, np.ndarray, pd.DataFrame]:
    records = []
    candidates = []
    item_count = len(linkage_matrix) + 1
    for requested in (8, 12, 16, 24, 32, 48, 64):
        labels = fcluster(linkage_matrix, requested, criterion="maxclust")
        counts = np.bincount(labels)[1:]
        if len(counts) < 2:
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        singleton_share = float((counts == 1).sum() / len(counts))
        largest_cluster_share = float(counts.max() / item_count)
        adjusted = score - 0.10 * singleton_share
        records.append(
            {
                "requested_clusters": requested,
                "actual_clusters": len(counts),
                "silhouette_cosine": score,
                "singleton_cluster_share": singleton_share,
                "largest_cluster_size": int(counts.max()),
                "largest_cluster_share": largest_cluster_share,
                "selection_score": adjusted,
            }
        )
        candidates.append((requested, largest_cluster_share, adjusted, labels))
    if not candidates:
        raise ValueError("unable to select macro hierarchy cut")
    # Raw silhouette preferred an eight-way cut in the full corpus, but that
    # placed 442/512 dimensions in one uninformative catch-all parent. Select
    # the coarsest evaluated cut whose largest parent contains at most 25% of
    # the vocabulary. Silhouette remains in the audit table as a diagnostic.
    balanced = [candidate for candidate in candidates if candidate[1] <= 0.25]
    chosen = min(balanced or candidates, key=lambda candidate: candidate[0])
    labels = chosen[3]
    return int(len(np.unique(labels))), labels, pd.DataFrame(records)


def hierarchy_node_table(
    linkage_matrix: np.ndarray,
    relation: np.ndarray,
    conditioned_lift: np.ndarray,
    role_similarity: np.ndarray,
    and_score: np.ndarray,
    or_score: np.ndarray,
    profiles: np.ndarray,
    semantic_names: list[str],
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    leaves: dict[int, list[int]] = {index: [index] for index in range(DIMENSIONS)}
    records = []
    for merge_index, row in enumerate(linkage_matrix):
        node_id = DIMENSIONS + merge_index
        left_id, right_id = int(row[0]), int(row[1])
        left = leaves[left_id]
        right = leaves[right_id]
        first = np.repeat(left, len(right))
        second = np.tile(right, len(left))
        cross_relation = relation[first, second]
        and_fraction = float(np.mean(cross_relation == "AND"))
        or_fraction = float(np.mean(cross_relation == "OR"))
        if and_fraction > or_fraction and and_fraction >= 0.02:
            relation_type = "AND"
        elif or_fraction > and_fraction and or_fraction >= 0.02:
            relation_type = "OR"
        elif and_fraction + or_fraction >= 0.02:
            relation_type = "MIXED"
        else:
            relation_type = "ASSOCIATIVE"
        node_leaves = left + right
        leaves[node_id] = node_leaves
        semantic = profiles[node_leaves].mean(axis=0)
        top = np.argsort(semantic)[-3:][::-1]
        records.append(
            {
                "node_id": node_id,
                "left_child": left_id,
                "right_child": right_id,
                "leaf_count": len(node_leaves),
                "distance": float(row[2]),
                "relation_type": relation_type,
                "cross_and_pair_fraction": and_fraction,
                "cross_or_pair_fraction": or_fraction,
                "mean_cross_conditioned_log2_lift": float(
                    conditioned_lift[first, second].mean()
                ),
                "mean_cross_role_similarity": float(role_similarity[first, second].mean()),
                "mean_cross_and_score": float(and_score[first, second].mean()),
                "mean_cross_or_score": float(or_score[first, second].mean()),
                "top_semantic_1": semantic_names[top[0]],
                "top_semantic_2": semantic_names[top[1]],
                "top_semantic_3": semantic_names[top[2]],
            }
        )
    return pd.DataFrame(records), leaves


def nested_hierarchy_json(
    node_id: int,
    nodes: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict:
    if node_id < DIMENSIONS:
        row = metrics.iloc[node_id]
        return {
            "id": f"D{node_id:03d}",
            "type": "dimension",
            "mapillary_semantic": str(row.top1_class),
            "f64_category": str(row.fine_category),
            "f64_label": str(row.fine_label_en),
        }
    row = nodes.loc[node_id]
    return {
        "id": f"N{node_id:03d}",
        "type": "latent_parent",
        "relation": str(row.relation_type),
        "leaf_count": int(row.leaf_count),
        "top_semantics": [
            str(row.top_semantic_1),
            str(row.top_semantic_2),
            str(row.top_semantic_3),
        ],
        "children": [
            nested_hierarchy_json(int(row.left_child), nodes, metrics),
            nested_hierarchy_json(int(row.right_child), nodes, metrics),
        ],
    }


def analyze_hierarchy(
    cache_root: Path,
    semantic_root: Path,
    output: Path,
    top_x: int,
    presence_patches: int,
) -> dict:
    table = pd.read_csv(cache_root / "city_index.csv")
    counts = np.load(cache_root / f"panorama_top{top_x}_counts.u16.npy", mmap_mode="r")
    if counts.shape != (int(table.end.iloc[-1]), DIMENSIONS):
        raise ValueError("panorama count cache and city index disagree")
    city_occurrence, city_prevalence = panorama_cooccurrence(
        counts, table, presence_patches
    )
    conditioned = city_conditioned_pair_statistics(
        city_occurrence, table.panoramas.astype(int).tolist()
    )
    observed = conditioned["observed"]
    expected = conditioned["expected"]
    conditioned_lift = np.log2((observed + 0.5) / (expected + 0.5))
    residual = (observed - expected) / np.sqrt(expected + 0.5)
    np.fill_diagonal(conditioned_lift, 0)
    np.fill_diagonal(residual, 0)

    total_panoramas = int(table.panoramas.sum())
    support = conditioned["support"]
    global_expected = np.outer(support, support) / total_panoramas
    ppmi = np.maximum(np.log2((observed + 0.5) / (global_expected + 0.5)), 0)
    ppmi *= observed / (observed + 50.0)
    np.fill_diagonal(ppmi, 0)
    normalized_context = l2_rows(ppmi)
    role_similarity = np.clip(normalized_context @ normalized_context.T, 0, 1)

    dimension_patch = np.zeros(DIMENSIONS, dtype=np.uint64)
    position = np.zeros((DIMENSIONS, LOCAL_POSITIONS), dtype=np.uint64)
    same_patch = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.uint64)
    for row in table.itertuples(index=False):
        with np.load(cache_root / "city_stats" / f"{row.city_slug}.npz") as stats:
            dimension_patch += stats["dimension_counts"]
            position += stats["position_counts"]
            same_patch += stats["pair_counts"]
    same_patch = same_patch.astype(np.float64)
    same_patch += same_patch.T
    total_patches = total_panoramas * PATCHES
    patch_expected = 0.75 * np.outer(dimension_patch, dimension_patch) / total_patches
    patch_lift = np.log2((same_patch + 0.5) / (patch_expected + 0.5))
    np.fill_diagonal(patch_lift, 0)
    patch_ppmi = np.maximum(patch_lift, 0) * (same_patch / (same_patch + 50.0))
    normalized_patch = l2_rows(patch_ppmi)

    normalized_position = l2_rows(position)
    position_similarity = np.clip(normalized_position @ normalized_position.T, 0, 1)
    relation, and_score, or_score, thresholds = classify_pair_relations(
        conditioned_lift,
        residual,
        role_similarity,
        position_similarity,
        conditioned["city_positive_fraction"],
        conditioned["city_negative_fraction"],
        expected,
    )

    metrics, semantic_profiles, semantic_names = semantic_matrix(semantic_root)
    semantic_similarity = l2_rows(semantic_profiles) @ l2_rows(semantic_profiles).T

    fingerprints = np.concatenate(
        [0.65 * normalized_context, 0.20 * normalized_patch, 0.15 * normalized_position],
        axis=1,
    )
    fingerprint_similarity = np.clip(l2_rows(fingerprints) @ l2_rows(fingerprints).T, 0, 1)
    fingerprint_distance = np.clip(1.0 - fingerprint_similarity, 0, 1)
    np.fill_diagonal(fingerprint_distance, 0)
    linkage_matrix = linkage(squareform(fingerprint_distance, checks=False), method="average")
    macro_k, raw_macro, cut_selection = select_macro_cut(
        linkage_matrix, fingerprint_distance
    )
    raw_counts = pd.Series(raw_macro).value_counts()
    ordered_raw = raw_counts.index.tolist()
    macro_map = {raw: index + 1 for index, raw in enumerate(ordered_raw)}
    macro = np.asarray([macro_map[value] for value in raw_macro], dtype=np.int16)

    output.mkdir(parents=True, exist_ok=True)
    upper = np.triu_indices(DIMENSIONS, 1)
    pair_frame = pd.DataFrame(
        {
            "dimension_i": [f"D{x:03d}" for x in upper[0]],
            "dimension_j": [f"D{x:03d}" for x in upper[1]],
            "relation": relation[upper],
            "and_score": and_score[upper],
            "or_score": or_score[upper],
            "city_conditioned_log2_lift": conditioned_lift[upper],
            "city_conditioned_residual": residual[upper],
            "city_positive_fraction": conditioned["city_positive_fraction"][upper],
            "city_negative_fraction": conditioned["city_negative_fraction"][upper],
            "expected_panorama_cooccurrence": expected[upper],
            "observed_panorama_cooccurrence": observed[upper].astype(np.int64),
            "same_patch_log2_lift": patch_lift[upper],
            "role_similarity": role_similarity[upper],
            "position_similarity": position_similarity[upper],
            "semantic_similarity_posthoc": semantic_similarity[upper],
            "same_mapillary_parent_posthoc": (
                metrics.top1_class_id.to_numpy()[upper[0]]
                == metrics.top1_class_id.to_numpy()[upper[1]]
            ),
            "same_macro_parent": macro[upper[0]] == macro[upper[1]],
        }
    )
    pair_frame.to_csv(output / "dimension_pair_relations.csv", index=False)

    nodes, leaves = hierarchy_node_table(
        linkage_matrix,
        relation,
        conditioned_lift,
        role_similarity,
        and_score,
        or_score,
        semantic_profiles,
        semantic_names,
    )
    nodes = nodes.set_index("node_id", drop=False)
    nodes.to_csv(output / "hierarchy_nodes.csv", index=False)
    np.save(output / "hierarchy_linkage.npy", linkage_matrix.astype(np.float64))
    cut_selection.to_csv(output / "macro_parent_cut_selection.csv", index=False)

    dimension_rows = []
    macro_rows = []
    for parent in range(1, macro_k + 1):
        members = np.flatnonzero(macro == parent)
        weights = dimension_patch[members].astype(np.float64)
        if weights.sum() > 0:
            semantic = np.average(semantic_profiles[members], axis=0, weights=weights)
        else:
            # A tiny smoke sample (and potentially a genuinely dead dimension)
            # can yield a parent with no Top-X support.  Semantic profiles are
            # post-hoc labels only, so an equal-weight fallback is appropriate.
            semantic = semantic_profiles[members].mean(axis=0)
        top_semantic = np.argsort(semantic)[-3:][::-1]
        within = np.ix_(members, members)
        local_relations = relation[within]
        local_upper = np.triu_indices(len(members), 1)
        and_pairs = int((local_relations[local_upper] == "AND").sum())
        or_pairs = int((local_relations[local_upper] == "OR").sum())
        label = " / ".join(semantic_names[index] for index in top_semantic[:2])
        macro_rows.append(
            {
                "macro_parent_id": f"P{parent:02d}",
                "posthoc_semantic_label": label,
                "dimension_count": len(members),
                "global_top4_patch_assignments": int(dimension_patch[members].sum()),
                "and_pairs": and_pairs,
                "or_pairs": or_pairs,
                "top_semantic_1": semantic_names[top_semantic[0]],
                "top_semantic_1_share": semantic[top_semantic[0]],
                "top_semantic_2": semantic_names[top_semantic[1]],
                "top_semantic_2_share": semantic[top_semantic[1]],
                "top_semantic_3": semantic_names[top_semantic[2]],
                "top_semantic_3_share": semantic[top_semantic[2]],
            }
        )
        for dimension in members:
            dimension_rows.append(
                {
                    "dimension_id": dimension,
                    "dimension": f"D{dimension:03d}",
                    "macro_parent_id": f"P{parent:02d}",
                    "macro_parent_label": label,
                    "global_top4_patch_assignments": int(dimension_patch[dimension]),
                    "panorama_presence_count": int(support[dimension]),
                    "panorama_presence_share": support[dimension] / total_panoramas,
                    "mapillary_parent_posthoc": metrics.iloc[dimension].top1_class,
                    "f64_category_posthoc": metrics.iloc[dimension].fine_category,
                    "f64_label_posthoc": metrics.iloc[dimension].fine_label_en,
                }
            )
    pd.DataFrame(dimension_rows).sort_values(
        ["macro_parent_id", "global_top4_patch_assignments"], ascending=[True, False]
    ).to_csv(output / "dimension_macro_parent.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(output / "macro_parent_summary.csv", index=False)

    city_frame = []
    for city_index, row in table.iterrows():
        for dimension in range(DIMENSIONS):
            city_frame.append(
                {
                    "city": row.city,
                    "dimension": f"D{dimension:03d}",
                    "panorama_presence_share": city_prevalence[city_index, dimension],
                }
            )
    pd.DataFrame(city_frame).to_csv(output / "city_dimension_prevalence.csv", index=False)
    root_id = DIMENSIONS + len(linkage_matrix) - 1
    save_json(
        output / "typed_hierarchy.json",
        nested_hierarchy_json(root_id, nodes, metrics),
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "complete four-direction panorama",
        "cities": len(table),
        "panoramas": total_panoramas,
        "patches": total_patches,
        "dimensions": DIMENSIONS,
        "top_x": top_x,
        "activation_magnitudes_used_after_top_x_selection": False,
        "winner_assignments": total_patches * top_x,
        "panorama_presence_threshold_patches": presence_patches,
        "mean_active_dimensions_per_panorama": float(
            sum(np.diag(matrix).sum() for matrix in city_occurrence) / total_panoramas
        ),
        "macro_parent_count": macro_k,
        "relation_thresholds": thresholds,
        "mapillary_usage": (
            "Top-X validation and post-hoc naming/audit only; excluded from "
            "relation scores, hierarchy fingerprints, and hierarchical linkage"
        ),
        "hierarchy_fingerprint": {
            "panorama_context_ppmi_weight": 0.65,
            "same_patch_context_ppmi_weight": 0.20,
            "14x14_local_position_weight": 0.15,
        },
    }
    save_json(output / "analysis_summary.json", summary)
    return {
        "summary": summary,
        "linkage": linkage_matrix,
        "nodes": nodes,
        "pair_frame": pair_frame,
        "macro": macro,
        "semantic_profiles": semantic_profiles,
        "semantic_names": semantic_names,
        "cut_selection": cut_selection,
    }


def plot_outputs(output: Path, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(output / "top_x_selection.csv")
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].plot(selection.top_x, selection.dimension_coverage, marker="o")
    axes[0].axhline(0.95, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Top-X per patch", ylabel="Dimension coverage", ylim=(0.75, 1.01))
    axes[1].plot(
        selection.top_x,
        selection.same_mapillary_parent_pair_share,
        marker="o",
        color="#2a9d8f",
    )
    axes[1].set(xlabel="Top-X per patch", ylabel="Same-parent pair share")
    axes[2].plot(
        selection.top_x,
        selection.city_marginal_cosine_mean,
        marker="o",
        color="#e76f51",
    )
    axes[2].set(xlabel="Top-X per patch", ylabel="Cross-city marginal cosine")
    selected = selection.loc[selection.selected].iloc[0]
    for axis in axes:
        axis.axvline(selected.top_x, color="#6a4c93", linestyle="--", linewidth=1.4)
        axis.grid(alpha=0.2)
    figure.suptitle("Per-patch Top-X selection: coverage, semantic purity, and city stability")
    figure.savefig(figures / "Fig_TopX_Selection.png", dpi=220)
    plt.close(figure)

    pairs = pd.read_csv(output / "dimension_pair_relations.csv")
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    unresolved = pairs.relation.eq("UNRESOLVED")
    axis.scatter(
        pairs.loc[unresolved, "role_similarity"],
        pairs.loc[unresolved, "city_conditioned_log2_lift"],
        s=3,
        alpha=0.08,
        color="#7f8c8d",
        rasterized=True,
        label="Unresolved",
    )
    colours = {"AND": "#277da1", "OR": "#f8961e"}
    for relation_type, colour in colours.items():
        selected_pairs = pairs.relation.eq(relation_type)
        axis.scatter(
            pairs.loc[selected_pairs, "role_similarity"],
            pairs.loc[selected_pairs, "city_conditioned_log2_lift"],
            s=13,
            alpha=0.75,
            color=colour,
            label=relation_type,
        )
    axis.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    axis.set(
        xlabel="Context-role similarity",
        ylabel="City-conditioned panorama co-occurrence log2 lift",
        title="Typed pair relations from Top-4 patch winners",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.15)
    figure.savefig(figures / "Fig_AND_OR_Relation_Map.png", dpi=220)
    plt.close(figure)

    nodes = pd.read_csv(output / "hierarchy_nodes.csv").set_index("node_id")
    linkage_matrix = np.load(output / "hierarchy_linkage.npy").astype(np.float64)
    link_colours = {
        int(node_id): {
            "AND": "#277da1",
            "OR": "#f8961e",
            "MIXED": "#9b5de5",
            "ASSOCIATIVE": "#8d99ae",
        }[row.relation_type]
        for node_id, row in nodes.iterrows()
    }
    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=0,
        link_color_func=lambda node: link_colours.get(int(node), "#8d99ae"),
        ax=axis,
    )
    axis.set(
        ylabel="Top-4 relation-fingerprint cosine distance",
        title="Feature-MAE typed semantic hierarchy (blue=AND, orange=OR, purple=mixed)",
    )
    figure.savefig(figures / "Fig_Typed_Semantic_Hierarchy.png", dpi=220)
    plt.close(figure)

    macro = pd.read_csv(output / "dimension_macro_parent.csv")
    summary = pd.read_csv(output / "macro_parent_summary.csv")
    distribution = pd.read_csv(
        SEMANTIC_ROOT / "dimension_semantic_distribution_65.csv"
    )
    profiles = distribution.pivot(
        index="dimension", columns="class_name", values="semantic_fraction"
    )
    matrix = []
    for parent in summary.macro_parent_id:
        members = macro.loc[macro.macro_parent_id.eq(parent), "dimension"]
        matrix.append(profiles.loc[members].mean(axis=0))
    matrix = pd.DataFrame(matrix, index=summary.macro_parent_id)
    keep = matrix.mean(axis=0).sort_values(ascending=False).head(20).index
    figure, axis = plt.subplots(
        figsize=(12, max(5, 0.28 * len(matrix))), constrained_layout=True
    )
    image = axis.imshow(matrix[keep], aspect="auto", cmap="magma", vmin=0)
    axis.set_xticks(range(len(keep)), keep, rotation=65, ha="right", fontsize=8)
    axis.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    axis.set(
        xlabel="Post-hoc Mapillary semantics (not used for discovery)",
        ylabel="Macro parent",
        title="Semantic audit of Top-4-derived macro parents",
    )
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015).set_label(
        "Mean semantic fraction"
    )
    figure.savefig(figures / "Fig_Macro_Parent_Semantic_Audit.png", dpi=220)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(output: Path, figures: Path) -> None:
    summary = json.loads((output / "analysis_summary.json").read_text())
    selection = pd.read_csv(output / "top_x_selection.csv")
    selected = selection.loc[selection.selected].iloc[0]
    parents = pd.read_csv(output / "macro_parent_summary.csv")
    pairs = pd.read_csv(output / "dimension_pair_relations.csv")
    top_and = pairs.loc[pairs.relation.eq("AND")].nlargest(10, "and_score")
    top_or = pairs.loc[pairs.relation.eq("OR")].nlargest(10, "or_score")
    lines = [
        "# Top-4 Feature-MAE Typed Semantic Hierarchy",
        "",
        "## Scope",
        "",
        f"- Cities: `{summary['cities']}`",
        f"- Complete four-direction panoramas: `{summary['panoramas']:,}`",
        f"- Panorama patches: `{summary['patches']:,}`",
        f"- Winner assignments: `{summary['winner_assignments']:,}`",
        "- Discovery input: per-patch Top-4 dimension identities only; activation magnitudes are discarded after ranking.",
        "- Mapillary semantics validate Top-X and provide post-hoc names/audits; they do not enter relation scores or hierarchical linkage.",
        "",
        "## Why Top-4",
        "",
        f"Top-4 is the smallest candidate reaching `{selected.dimension_coverage:.1%}` dimension coverage while preserving `{selected.same_mapillary_parent_pair_share:.1%}` same-parent pair purity and `{selected.city_marginal_cosine_mean:.3f}` cross-city marginal cosine stability.",
        "",
        "## Typed hierarchy",
        "",
        f"- Macro parents selected from the hierarchy: `{summary['macro_parent_count']}`",
        f"- High-confidence AND pairs: `{summary['relation_thresholds']['and_pairs']}`",
        f"- High-confidence OR pairs: `{summary['relation_thresholds']['or_pairs']}`",
        f"- Panorama presence threshold: `{summary['panorama_presence_threshold_patches']}` of 784 patches",
        "- The reported macro cut is the coarsest evaluated cut whose largest parent contains no more than 25% of the 512 dimensions; the full dendrogram is retained.",
        "",
        "AND means positive city-conditioned panorama co-occurrence. OR requires negative city-conditioned co-occurrence together with similar context-role and spatial-position signatures. Unresolved pairs are not forced into either class.",
        "",
        "OR is a sparse lateral relation: an OR pair may connect dimensions assigned to different macro parents. It should not be read as requiring an orange dendrogram branch. Macro-parent labels are post-hoc semantic audits, not semantic supervision or ground-truth classes.",
        "",
        "## Macro-cut audit",
        "",
        markdown_table(pd.read_csv(output / "macro_parent_cut_selection.csv")),
        "",
        "## Largest macro parents",
        "",
        markdown_table(parents.nlargest(12, "dimension_count")),
        "",
        "## Strongest AND pairs",
        "",
        markdown_table(top_and[["dimension_i", "dimension_j", "and_score", "city_conditioned_log2_lift", "role_similarity"]]),
        "",
        "## Strongest OR pairs",
        "",
        markdown_table(top_or[["dimension_i", "dimension_j", "or_score", "city_conditioned_log2_lift", "role_similarity", "position_similarity"]]),
        "",
        "## Figures",
        "",
        f"- `{figures / 'Fig_TopX_Selection.png'}`",
        f"- `{figures / 'Fig_AND_OR_Relation_Map.png'}`",
        f"- `{figures / 'Fig_Typed_Semantic_Hierarchy.png'}`",
        f"- `{figures / 'Fig_Macro_Parent_Semantic_Audit.png'}`",
        "",
    ]
    (output / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("select", "scan", "analyze", "figures", "all"), default="all"
    )
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=SEMANTIC_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-data", type=Path, default=PAPER_DATA)
    parser.add_argument("--output-figures", type=Path, default=PAPER_FIGURES)
    parser.add_argument("--top-x", type=int, default=DEFAULT_TOP_X)
    parser.add_argument("--presence-patches", type=int, default=DEFAULT_PRESENCE_PATCHES)
    parser.add_argument("--selection-panoramas-per-city", type=int, default=32)
    parser.add_argument("--chunk-panoramas", type=int, default=4)
    parser.add_argument(
        "--max-panoramas-per-city",
        type=int,
        default=0,
        help="debug/smoke limit; zero scans every available panorama",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_safe_affinity()
    if args.stage in {"select", "all"}:
        selection = run_top_x_selection(
            args.feature_root,
            args.semantic_root,
            args.output_data,
            args.selection_panoramas_per_city,
        )
        selected = choose_top_x(selection)
        if args.top_x == DEFAULT_TOP_X and selected != DEFAULT_TOP_X:
            raise ValueError(f"empirical Top-X selection changed: expected 4, found {selected}")
    if args.stage in {"scan", "all"}:
        scan_full_corpus(
            args.feature_root,
            args.cache_root,
            args.top_x,
            args.chunk_panoramas,
            args.max_panoramas_per_city,
        )
    if args.stage in {"analyze", "all"}:
        analyze_hierarchy(
            args.cache_root,
            args.semantic_root,
            args.output_data,
            args.top_x,
            args.presence_patches,
        )
    if args.stage in {"figures", "all"}:
        plot_outputs(args.output_data, args.output_figures)
        write_report(args.output_data, args.output_figures)
    if (args.output_data / "analysis_summary.json").exists():
        print((args.output_data / "analysis_summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
