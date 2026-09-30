#!/usr/bin/env python3
"""Build a semantic-parent / latent-child hierarchy for 512 Feature-MAE dimensions.

This is the second version of the typed hierarchy experiment.  It reuses the
completed Top-4 panorama cache and activation-weighted Mapillary profiles; it
does not read image pixels, rerun a neural network, or rescan latent tensors.

The resulting object is deliberately not a dendrogram.  It consists of:

* a semantic parent layer derived from the 65 Mapillary Vistas classes;
* data-driven child concepts discovered independently inside each parent;
* lateral AND (composition) and OR (substitution) relations between children.

Semantic profiles determine the parent and contribute to within-parent child
similarity.  Relation labels are based only on Top-4 co-occurrence evidence.
"""
from __future__ import annotations

import scripts._env  # noqa: F401  (must precede numpy/scipy/sklearn)

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.metrics import silhouette_score


STORAGE_ROOT = Path("/workplace/urban_visual_gene")
TOP4_CACHE = STORAGE_ROOT / "outputs/experiments/dinov3_multicity/feature_mae_top4_typed_hierarchy"
TOP4_PAPER = STORAGE_ROOT / "paper/data/feature_mae_top4_typed_hierarchy"
SEMANTIC_ROOT = STORAGE_ROOT / "paper/data/semantic_alignment/feature_mae_mapillary_alignment_n60000"
PREVALENCE_FILE = STORAGE_ROOT / "paper/data/semantic_alignment/mask2former_swin_l_mapillary_n2000_per_city/class_prevalence_65.csv"
OUTPUT = STORAGE_ROOT / "paper/data/feature_mae_semantic_parent_child_v2"
FIGURES = STORAGE_ROOT / "paper/figures/supplementary/feature_mae_semantic_parent_child_v2"

DIMENSIONS = 512
PATCHES = 14 * 56
LOCAL_POSITIONS = 14 * 14
PRESENCE_PATCHES = 16
BOOTSTRAPS = 20
SEED = 42

# Mutually exclusive high-level parents.  Rare classes remain represented in
# semantic profiles even when no latent dimension selects them as its primary
# parent.  Class IDs follow Mapillary Vistas v2.0.
PARENT_CLASSES: dict[str, list[int]] = {
    "Human / Animal": [0, 1, 19, 20, 21, 22],
    "Road / Path": [2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23, 24, 36, 41, 43],
    "Built / Boundary": [3, 4, 5, 6, 16, 17, 18],
    "Terrain / Water": [25, 26, 28, 29, 31],
    "Sky": [27],
    "Vegetation": [30],
    "Street Furniture": [32, 33, 34, 35, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51],
    "Vehicle": [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
}

PARENT_CODES = {
    "Human / Animal": "HUM",
    "Road / Path": "RDP",
    "Built / Boundary": "BLT",
    "Terrain / Water": "TRN",
    "Sky": "SKY",
    "Vegetation": "VEG",
    "Street Furniture": "FUR",
    "Vehicle": "VEH",
}


def assert_safe_affinity() -> None:
    """Refuse to run if forbidden CPU 8 or 9 is available to this process."""
    if hasattr(os, "sched_getaffinity"):
        unsafe = set(os.sched_getaffinity(0)).intersection({8, 9})
        if unsafe:
            raise RuntimeError(f"unsafe CPU affinity includes {sorted(unsafe)}")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def l2_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def cosine_rows(values: np.ndarray) -> np.ndarray:
    normalized = l2_rows(values)
    return np.clip(normalized @ normalized.T, 0.0, 1.0)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    if pvalues.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def load_semantics() -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    metrics = pd.read_csv(SEMANTIC_ROOT / "dimension_semantic_profiles.csv").sort_values("dimension_id")
    if metrics.dimension_id.tolist() != list(range(DIMENSIONS)):
        raise ValueError("semantic profile must contain D000--D511 exactly once")
    long = pd.read_csv(SEMANTIC_ROOT / "dimension_semantic_distribution_65.csv")
    profiles = (
        long.pivot(index="dimension_id", columns="class_id", values="semantic_fraction")
        .reindex(index=range(DIMENSIONS), columns=range(65))
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    names = (
        long.drop_duplicates("class_id").sort_values("class_id").class_name.astype(str).tolist()
    )
    prevalence = (
        pd.read_csv(PREVALENCE_FILE).set_index("class_id").reindex(range(65)).pixel_share.to_numpy(np.float64)
    )
    if not np.allclose(profiles.sum(axis=1), 1, atol=2e-4):
        raise ValueError("semantic distributions do not sum to one")
    return metrics, profiles, names, prevalence


def assign_semantic_parents(
    profiles: np.ndarray, prevalence: np.ndarray, lift_power: float = 0.5
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Assign one primary and optional secondary semantic parent per dimension.

    A square-root lift correction prevents globally common classes from
    swallowing rare but specific street-object or vehicle features.  Primary
    assignment remains dominated by activation mass rather than lift alone.
    """
    parent_names = list(PARENT_CLASSES)
    shares = np.stack([profiles[:, PARENT_CLASSES[name]].sum(axis=1) for name in parent_names], axis=1)
    global_share = np.asarray([prevalence[PARENT_CLASSES[name]].sum() for name in parent_names])
    lift = (shares + 1e-6) / (global_share[None, :] + 1e-6)
    score = shares * np.maximum(lift, 1.0) ** lift_power
    score[shares < 0.01] = -1
    primary = score.argmax(axis=1)
    rows = []
    for dimension in range(DIMENSIONS):
        primary_score = score[dimension, primary[dimension]]
        for parent_id, name in enumerate(parent_names):
            is_primary = parent_id == primary[dimension]
            is_secondary = (
                not is_primary
                and shares[dimension, parent_id] >= 0.12
                and score[dimension, parent_id] >= 0.65 * primary_score
            )
            if is_primary or is_secondary:
                rows.append(
                    {
                        "dimension_id": dimension,
                        "dimension": f"D{dimension:03d}",
                        "parent_id": PARENT_CODES[name],
                        "parent_label": name,
                        "membership": "primary" if is_primary else "secondary",
                        "semantic_share": shares[dimension, parent_id],
                        "lift_vs_global": lift[dimension, parent_id],
                        "membership_score": score[dimension, parent_id],
                    }
                )
    return pd.DataFrame(rows), primary, shares


def load_context_evidence() -> dict[str, object]:
    """Load or reconstruct all macro-scale Top-4 evidence from compact caches."""
    table = pd.read_csv(TOP4_CACHE / "city_index.csv")
    counts = np.load(TOP4_CACHE / "panorama_top4_counts.u16.npy", mmap_mode="r")
    if counts.shape != (int(table.end.iloc[-1]), DIMENSIONS):
        raise ValueError("Top-4 panorama count cache and city index disagree")

    city_occurrence: list[np.ndarray] = []
    city_prevalence = []
    for number, row in enumerate(table.itertuples(index=False), 1):
        binary = np.asarray(counts[row.start : row.end] >= PRESENCE_PATCHES)
        design = sparse.csr_matrix(binary, dtype=np.float32)
        cooccurrence = (design.T @ design).toarray().astype(np.float64)
        support = binary.sum(axis=0, dtype=np.int64)
        np.fill_diagonal(cooccurrence, support)
        city_occurrence.append(cooccurrence)
        city_prevalence.append(support / len(binary))
        print(f"context [{number:02d}/{len(table)}] {row.city}", flush=True)

    observed = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.float64)
    expected = np.zeros_like(observed)
    evaluated = np.zeros_like(observed)
    positive = np.zeros_like(observed)
    negative = np.zeros_like(observed)
    support_total = np.zeros(DIMENSIONS, dtype=np.float64)
    for matrix, size in zip(city_occurrence, table.panoramas.astype(int)):
        support = np.diag(matrix)
        local_expected = np.outer(support, support) / float(size)
        local_lift = np.log2((matrix + 0.5) / (local_expected + 0.5))
        valid = local_expected >= 5
        observed += matrix
        expected += local_expected
        positive += valid & (local_lift > 0)
        negative += valid & (local_lift < 0)
        evaluated += valid
        support_total += support
    np.fill_diagonal(expected, support_total)
    lift = np.log2((observed + 0.5) / (expected + 0.5))
    residual = (observed - expected) / np.sqrt(expected + 0.5)
    np.fill_diagonal(lift, 0)
    np.fill_diagonal(residual, 0)

    total_panoramas = int(table.panoramas.sum())
    global_expected = np.outer(support_total, support_total) / total_panoramas
    ppmi = np.maximum(np.log2((observed + 0.5) / (global_expected + 0.5)), 0)
    ppmi *= observed / (observed + 50)
    np.fill_diagonal(ppmi, 0)
    role_similarity = cosine_rows(ppmi)

    dimension_patch = np.zeros(DIMENSIONS, dtype=np.uint64)
    position = np.zeros((DIMENSIONS, LOCAL_POSITIONS), dtype=np.uint64)
    same_patch = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.uint64)
    patch_positive = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.float64)
    patch_negative = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.float64)
    patch_evaluated = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.float64)
    for row in table.itertuples(index=False):
        with np.load(TOP4_CACHE / "city_stats" / f"{row.city_slug}.npz") as stats:
            local_dimension = stats["dimension_counts"].astype(np.float64)
            local_pair = stats["pair_counts"].astype(np.float64)
            local_pair += local_pair.T
            local_expected = 0.75 * np.outer(local_dimension, local_dimension) / (row.panoramas * PATCHES)
            local_lift = np.log2((local_pair + 0.5) / (local_expected + 0.5))
            local_valid = local_expected >= 5
            patch_positive += local_valid & (local_lift > 0)
            patch_negative += local_valid & (local_lift < 0)
            patch_evaluated += local_valid
            dimension_patch += local_dimension.astype(np.uint64)
            position += stats["position_counts"]
            same_patch += stats["pair_counts"]
    same_patch = same_patch.astype(np.float64)
    same_patch += same_patch.T
    total_patches = total_panoramas * PATCHES
    patch_expected = 0.75 * np.outer(dimension_patch, dimension_patch) / total_patches
    patch_lift = np.log2((same_patch + 0.5) / (patch_expected + 0.5))
    patch_residual = (same_patch - patch_expected) / np.sqrt(patch_expected + 0.5)
    np.fill_diagonal(patch_lift, 0)
    np.fill_diagonal(patch_residual, 0)
    patch_ppmi = np.maximum(patch_lift, 0) * same_patch / (same_patch + 50)

    position_similarity = cosine_rows(position)
    city_matrix = np.stack(city_prevalence, axis=1)
    centered = city_matrix - city_matrix.mean(axis=1, keepdims=True)
    city_correlation = l2_rows(centered) @ l2_rows(centered).T
    city_similarity = np.clip((city_correlation + 1) / 2, 0, 1)
    return {
        "table": table,
        "city_occurrence": city_occurrence,
        "city_prevalence": np.stack(city_prevalence),
        "observed": observed,
        "expected": expected,
        "lift": lift,
        "residual": residual,
        "city_positive": positive / np.maximum(evaluated, 1),
        "city_negative": negative / np.maximum(evaluated, 1),
        "role_similarity": role_similarity,
        "patch_lift": patch_lift,
        "patch_expected": patch_expected,
        "patch_residual": patch_residual,
        "patch_city_positive": patch_positive / np.maximum(patch_evaluated, 1),
        "patch_city_negative": patch_negative / np.maximum(patch_evaluated, 1),
        "patch_context_similarity": cosine_rows(patch_ppmi),
        "position_similarity": position_similarity,
        "city_similarity": city_similarity,
        "dimension_patch": dimension_patch,
        "support": support_total,
        "total_panoramas": total_panoramas,
        "total_patches": total_patches,
    }


def combined_similarity(evidence: dict[str, object], semantic_profiles: np.ndarray) -> np.ndarray:
    semantic_similarity = cosine_rows(semantic_profiles)
    result = (
        0.30 * np.asarray(evidence["role_similarity"])
        + 0.20 * np.asarray(evidence["patch_context_similarity"])
        + 0.30 * semantic_similarity
        + 0.10 * np.asarray(evidence["position_similarity"])
        + 0.10 * np.asarray(evidence["city_similarity"])
    )
    result = np.clip(result, 0, 1)
    np.fill_diagonal(result, 1)
    return result


def choose_child_cut(distance: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Select an interpretable within-parent cut with bounded child size."""
    n = len(distance)
    if n < 4:
        return np.ones(n, dtype=np.int16), pd.DataFrame([{
            "requested_children": 1, "actual_children": 1, "silhouette": np.nan,
            "largest_child": n, "largest_child_share": 1.0, "selected": True,
        }])
    tree = linkage(squareform(distance, checks=False), method="complete")
    max_k = min(24, n - 1)
    records = []
    candidates = []
    size_cap = max(12, math.ceil(0.30 * n))
    for requested in range(2, max_k + 1):
        labels = fcluster(tree, requested, criterion="maxclust")
        counts = np.bincount(labels)[1:]
        actual = len(counts)
        if actual < 2:
            continue
        silhouette = float(silhouette_score(distance, labels, metric="precomputed"))
        singleton_share = float(np.mean(counts == 1))
        record = {
            "requested_children": requested,
            "actual_children": actual,
            "silhouette": silhouette,
            "largest_child": int(counts.max()),
            "largest_child_share": float(counts.max() / n),
            "singleton_share": singleton_share,
            "selected": False,
        }
        records.append(record)
        candidates.append((labels, record))
    balanced = [item for item in candidates if item[1]["largest_child"] <= size_cap]
    # Use the coarsest cut that prevents an uninformative catch-all child.  If
    # the cap cannot be reached, use the cut with the smallest largest child.
    labels, chosen = min(
        balanced or candidates,
        key=lambda item: (
            item[1]["actual_children"] if balanced else item[1]["largest_child"],
            -item[1]["silhouette"],
        ),
    )
    chosen["selected"] = True
    return labels.astype(np.int16), pd.DataFrame(records)


def discover_children(
    primary: np.ndarray,
    similarity: np.ndarray,
    metrics: pd.DataFrame,
    profiles: np.ndarray,
    semantic_names: list[str],
    evidence: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], np.ndarray]:
    parent_names = list(PARENT_CLASSES)
    child_ids = np.full(DIMENSIONS, "", dtype=object)
    rows = []
    cuts = []
    chosen_k: dict[str, int] = {}
    for parent_index, parent_name in enumerate(parent_names):
        members = np.flatnonzero(primary == parent_index)
        if len(members) == 0:
            chosen_k[PARENT_CODES[parent_name]] = 0
            continue
        local_similarity = similarity[np.ix_(members, members)]
        distance = np.clip(1 - local_similarity, 0, 1)
        np.fill_diagonal(distance, 0)
        raw_labels, selection = choose_child_cut(distance)
        counts = pd.Series(raw_labels).value_counts()
        raw_order = counts.index.tolist()
        ordered = {raw: number + 1 for number, raw in enumerate(raw_order)}
        labels = np.asarray([ordered[x] for x in raw_labels])
        code = PARENT_CODES[parent_name]
        chosen_k[code] = len(ordered)
        selection.insert(0, "parent_id", code)
        selection.insert(1, "parent_label", parent_name)
        cuts.append(selection)
        for local_child in range(1, len(ordered) + 1):
            child_members = members[labels == local_child]
            child_id = f"{code}-C{local_child:02d}"
            child_ids[child_members] = child_id
            weights = np.asarray(evidence["dimension_patch"])[child_members].astype(np.float64)
            semantic = np.average(profiles[child_members], axis=0, weights=np.maximum(weights, 1))
            top = np.argsort(semantic)[-3:][::-1]
            label = " / ".join(semantic_names[x] for x in top[:2])
            for dimension in child_members:
                row = metrics.iloc[dimension]
                rows.append({
                    "dimension_id": dimension,
                    "dimension": f"D{dimension:03d}",
                    "parent_id": code,
                    "parent_label": parent_name,
                    "child_id": child_id,
                    "child_label": label,
                    "child_size": len(child_members),
                    "mapillary_top1": row.top1_class,
                    "f64_category_posthoc": row.fine_category,
                    "f64_label_posthoc": row.fine_label_en,
                    "global_top4_patch_assignments": int(np.asarray(evidence["dimension_patch"])[dimension]),
                    "panorama_presence_count": int(np.asarray(evidence["support"])[dimension]),
                })
    if np.any(child_ids == ""):
        raise ValueError("not all dimensions received a child concept")
    return pd.DataFrame(rows), pd.concat(cuts, ignore_index=True), chosen_k, child_ids


def bootstrap_child_stability(
    primary: np.ndarray,
    child_ids: np.ndarray,
    chosen_k: dict[str, int],
    base_similarity: np.ndarray,
    city_occurrence: list[np.ndarray],
    city_sizes: np.ndarray,
    bootstraps: int,
) -> np.ndarray:
    """City bootstrap co-clustering stability for each dimension."""
    rng = np.random.default_rng(SEED)
    same_counts = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.uint16)
    parent_names = list(PARENT_CLASSES)
    for bootstrap in range(bootstraps):
        sample = rng.integers(0, len(city_occurrence), len(city_occurrence))
        observed = sum(city_occurrence[index] for index in sample)
        support = np.diag(observed)
        total = float(city_sizes[sample].sum())
        expected = np.outer(support, support) / max(total, 1)
        ppmi = np.maximum(np.log2((observed + 0.5) / (expected + 0.5)), 0)
        ppmi *= observed / (observed + 50)
        np.fill_diagonal(ppmi, 0)
        boot_role = cosine_rows(ppmi)
        similarity = np.clip(base_similarity + 0.30 * boot_role, 0, 1)
        # base_similarity passed here excludes the original 0.30 role component.
        np.fill_diagonal(similarity, 1)
        for parent_index, parent_name in enumerate(parent_names):
            members = np.flatnonzero(primary == parent_index)
            k = chosen_k[PARENT_CODES[parent_name]]
            if len(members) < 2 or k <= 1:
                same_counts[np.ix_(members, members)] += 1
                continue
            distance = np.clip(1 - similarity[np.ix_(members, members)], 0, 1)
            np.fill_diagonal(distance, 0)
            tree = linkage(squareform(distance, checks=False), method="complete")
            labels = fcluster(tree, k, criterion="maxclust")
            same_counts[np.ix_(members, members)] += labels[:, None] == labels[None, :]
        print(f"bootstrap [{bootstrap + 1:02d}/{bootstraps}]", flush=True)
    co_cluster = same_counts.astype(np.float64) / bootstraps
    stability = np.zeros(DIMENSIONS, dtype=np.float64)
    for dimension in range(DIMENSIONS):
        peers = np.flatnonzero(child_ids == child_ids[dimension])
        peers = peers[peers != dimension]
        stability[dimension] = 1.0 if len(peers) == 0 else co_cluster[dimension, peers].mean()
    return stability


def type_dimension_relations(
    primary: np.ndarray, child_ids: np.ndarray, evidence: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lift = np.asarray(evidence["lift"])
    residual = np.asarray(evidence["residual"])
    expected = np.asarray(evidence["expected"])
    positive = np.asarray(evidence["city_positive"])
    negative = np.asarray(evidence["city_negative"])
    role = np.asarray(evidence["role_similarity"])
    position = np.asarray(evidence["position_similarity"])
    patch_lift = np.asarray(evidence["patch_lift"])
    patch_expected = np.asarray(evidence["patch_expected"])
    patch_residual = np.asarray(evidence["patch_residual"])
    patch_positive = np.asarray(evidence["patch_city_positive"])
    patch_negative = np.asarray(evidence["patch_city_negative"])
    upper = np.triu_indices(DIMENSIONS, 1)
    same_parent = primary[upper[0]] == primary[upper[1]]
    pvalue = np.minimum(1, 2 * norm.sf(np.abs(residual[upper])))
    qvalue = np.ones_like(pvalue)
    qvalue[same_parent] = bh_qvalues(pvalue[same_parent])
    patch_pvalue = np.minimum(1, 2 * norm.sf(np.abs(patch_residual[upper])))
    patch_qvalue = np.ones_like(pvalue)
    patch_qvalue[same_parent] = bh_qvalues(patch_pvalue[same_parent])
    reliability = 1 - np.exp(-expected[upper] / 100)
    patch_reliability = 1 - np.exp(-patch_expected[upper] / 100)
    and_score = np.maximum(lift[upper], 0) * np.sqrt(np.maximum(patch_lift[upper], 0)) * np.sqrt(positive[upper] * patch_positive[upper]) * reliability
    # OR is a local substitutability relation.  Two dimensions may coexist in
    # one panorama while avoiding the same patch (e.g. alternative tree types).
    or_score = np.maximum(-patch_lift[upper], 0) * role[upper] * np.sqrt(position[upper]) * patch_negative[upper] * patch_reliability
    role_floor = float(np.quantile(role[upper][same_parent], 0.75))
    and_candidate = same_parent & (residual[upper] > 5) & (positive[upper] >= 0.60) & (expected[upper] >= 25) & (patch_lift[upper] > 0) & (patch_positive[upper] >= 0.60) & (qvalue < 0.01) & (patch_qvalue < 0.01)
    or_candidate = same_parent & (patch_residual[upper] < -5) & (patch_negative[upper] >= 0.60) & (patch_expected[upper] >= 25) & (patch_lift[upper] < 0) & (role[upper] >= role_floor) & (position[upper] >= 0.65) & (patch_qvalue < 0.01)
    and_floor = float(np.quantile(and_score[and_candidate], 0.90)) if and_candidate.any() else math.inf
    # Substitution is rarer than composition; the candidate gate is already
    # strict, so retain the upper quartile for a small auditable OR set.
    or_floor = float(np.quantile(or_score[or_candidate], 0.75)) if or_candidate.any() else math.inf
    relation = np.full(len(pvalue), "UNRESOLVED", dtype=object)
    relation[and_candidate & (and_score >= and_floor)] = "AND"
    relation[or_candidate & (or_score >= or_floor)] = "OR"
    frame = pd.DataFrame({
        "dimension_i": [f"D{x:03d}" for x in upper[0]],
        "dimension_j": [f"D{x:03d}" for x in upper[1]],
        "parent_id": [PARENT_CODES[list(PARENT_CLASSES)[x]] if same else "CROSS_PARENT" for x, same in zip(primary[upper[0]], same_parent)],
        "child_i": child_ids[upper[0]], "child_j": child_ids[upper[1]],
        "relation": relation, "and_score": and_score, "or_score": or_score,
        "city_conditioned_log2_lift": lift[upper], "city_conditioned_residual": residual[upper],
        "q_value_bh": qvalue, "same_patch_q_value_bh": patch_qvalue,
        "city_positive_fraction": positive[upper], "city_negative_fraction": negative[upper],
        "same_patch_city_positive_fraction": patch_positive[upper], "same_patch_city_negative_fraction": patch_negative[upper],
        "expected_panorama_cooccurrence": expected[upper], "same_patch_log2_lift": patch_lift[upper],
        "role_similarity": role[upper], "position_similarity": position[upper],
    })
    within = frame.loc[frame.parent_id.ne("CROSS_PARENT")].copy()

    selected = within.loc[within.relation.ne("UNRESOLVED") & within.child_i.ne(within.child_j)].copy()
    if not selected.empty:
        # Child order must be canonical: dimension order does not imply child order.
        canonical = np.sort(selected[["child_i", "child_j"]].to_numpy(str), axis=1)
        selected.loc[:, "child_i"] = canonical[:, 0]
        selected.loc[:, "child_j"] = canonical[:, 1]
    child_rows = []
    for (parent_id, child_i, child_j), group in selected.groupby(["parent_id", "child_i", "child_j"]):
        counts = group.relation.value_counts()
        relation_type = "AND" if counts.get("AND", 0) >= counts.get("OR", 0) else "OR"
        relevant = group.loc[group.relation.eq(relation_type)]
        score_column = "and_score" if relation_type == "AND" else "or_score"
        child_rows.append({
            "parent_id": parent_id, "child_i": child_i, "child_j": child_j,
            "relation": relation_type, "supporting_dimension_pairs": len(relevant),
            "relation_score": float(relevant[score_column].nlargest(min(5, len(relevant))).mean()),
            "mean_city_sign_fraction": float(relevant["city_positive_fraction" if relation_type == "AND" else "same_patch_city_negative_fraction"].mean()),
            "best_q_value": float(relevant["q_value_bh" if relation_type == "AND" else "same_patch_q_value_bh"].min()),
            "example_dimension_pairs": "; ".join((relevant.dimension_i + "+" + relevant.dimension_j).head(5)),
        })
    child_frame = pd.DataFrame(child_rows)
    if not child_frame.empty:
        child_frame = child_frame.sort_values(["relation", "relation_score"], ascending=[True, False])
    return within, child_frame


def child_summary(
    membership: pd.DataFrame,
    stability: np.ndarray,
    profiles: np.ndarray,
    semantic_names: list[str],
) -> pd.DataFrame:
    result = []
    for child_id, group in membership.groupby("child_id", sort=False):
        ids = group.dimension_id.to_numpy(int)
        semantic = profiles[ids].mean(axis=0)
        top = np.argsort(semantic)[-3:][::-1]
        exemplars = group.sort_values("global_top4_patch_assignments", ascending=False).dimension.head(6)
        result.append({
            "parent_id": group.parent_id.iloc[0], "parent_label": group.parent_label.iloc[0],
            "child_id": child_id, "child_label": group.child_label.iloc[0], "dimension_count": len(group),
            "mean_bootstrap_stability": float(stability[ids].mean()),
            "top_semantic_1": semantic_names[top[0]], "top_semantic_1_share": semantic[top[0]],
            "top_semantic_2": semantic_names[top[1]], "top_semantic_2_share": semantic[top[1]],
            "top_semantic_3": semantic_names[top[2]], "top_semantic_3_share": semantic[top[2]],
            "representative_dimensions": "; ".join(exemplars),
        })
    return pd.DataFrame(result)


def build_json(summary: pd.DataFrame, membership: pd.DataFrame, relations: pd.DataFrame) -> dict:
    parents = []
    for parent_id, children in summary.groupby("parent_id", sort=False):
        label = children.parent_label.iloc[0]
        child_records = []
        for row in children.itertuples(index=False):
            dimensions = membership.loc[membership.child_id.eq(row.child_id), "dimension"].tolist()
            child_records.append({
                "id": row.child_id, "label": row.child_label, "dimension_count": row.dimension_count,
                "bootstrap_stability": row.mean_bootstrap_stability, "dimensions": dimensions,
                "top_semantics": [row.top_semantic_1, row.top_semantic_2, row.top_semantic_3],
            })
        local_relations = [] if relations.empty else relations.loc[relations.parent_id.eq(parent_id)].to_dict("records")
        parents.append({"id": parent_id, "label": label, "children": child_records, "relations": local_relations})
    return {"schema": "semantic-parent/latent-child/typed-lateral-relations-v2", "parents": parents}


def plot_outputs(
    output: Path, figures: Path, profiles: np.ndarray, semantic_names: list[str]
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    children = pd.read_csv(output / "child_concept_summary.csv")
    membership = pd.read_csv(output / "child_concept_membership.csv")
    relations = pd.read_csv(output / "child_relation_edges.csv") if (output / "child_relation_edges.csv").stat().st_size > 1 else pd.DataFrame()
    dim_rel = pd.read_csv(output / "dimension_relation_evidence.csv")

    nonempty = children.parent_id.drop_duplicates().tolist()
    figure, axes = plt.subplots(math.ceil(len(nonempty) / 2), 2, figsize=(16, 3.8 * math.ceil(len(nonempty) / 2)), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    palette = {"AND": "#2878B5", "OR": "#E07B39"}
    for axis, parent_id in zip(axes, nonempty):
        local = children.loc[children.parent_id.eq(parent_id)].reset_index(drop=True)
        n = len(local)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = {row.child_id: np.asarray([np.cos(angle), np.sin(angle)]) for angle, row in zip(angles, local.itertuples())}
        if not relations.empty:
            for edge in relations.loc[relations.parent_id.eq(parent_id)].itertuples():
                a, b = xy[edge.child_i], xy[edge.child_j]
                axis.plot([a[0], b[0]], [a[1], b[1]], color=palette[edge.relation], linewidth=0.7 + 2.5 * min(edge.relation_score, 1), alpha=0.65, linestyle="-" if edge.relation == "AND" else "--", zorder=1)
        sizes = 100 + 65 * local.dimension_count.to_numpy()
        axis.scatter(np.cos(angles), np.sin(angles), s=sizes, color="#74A9CF", edgecolor="white", linewidth=1.2, zorder=2)
        for angle, row in zip(angles, local.itertuples()):
            label = f"{row.child_id}\n{row.top_semantic_1}\nn={row.dimension_count}"
            axis.text(1.18 * np.cos(angle), 1.18 * np.sin(angle), label, ha="center", va="center", fontsize=7)
        axis.set_title(f"{parent_id}  {local.parent_label.iloc[0]} — {n} child concepts", loc="left", fontweight="bold")
        axis.set_xlim(-1.55, 1.55); axis.set_ylim(-1.4, 1.4); axis.axis("off")
    for axis in axes[len(nonempty):]: axis.axis("off")
    figure.suptitle("Semantic-parent / latent-child hierarchy\nBlue solid = AND (composition); orange dashed = OR (substitution)", fontsize=14)
    figure.savefig(figures / "Fig_Semantic_Parent_Child_Hierarchy.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    order = children.sort_values(["parent_id", "top_semantic_1"]).child_id.tolist()
    matrix = []
    for child in order:
        ids = membership.loc[membership.child_id.eq(child), "dimension_id"].to_numpy(int)
        matrix.append(profiles[ids].mean(axis=0))
    matrix = np.asarray(matrix)
    top_classes = np.argsort(matrix.sum(axis=0))[-24:][::-1]
    figure, axis = plt.subplots(figsize=(13, max(7, len(order) * 0.18)), constrained_layout=True)
    image = axis.imshow(matrix[:, top_classes], aspect="auto", cmap="magma", vmin=0, vmax=np.quantile(matrix, 0.98))
    axis.set_xticks(range(len(top_classes)), [semantic_names[x] for x in top_classes], rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(len(order)), order, fontsize=6)
    axis.set(xlabel="Mapillary semantic class", ylabel="Latent child concept", title="Activation-weighted semantic profile of child concepts")
    figure.colorbar(image, ax=axis, label="Mean semantic fraction", shrink=0.65)
    figure.savefig(figures / "Fig_Parent_Child_Semantic_Heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    subset = dim_rel.loc[dim_rel.relation.ne("UNRESOLVED")]
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    background = dim_rel.sample(min(16000, len(dim_rel)), random_state=SEED)
    axis.scatter(background.role_similarity, background.city_conditioned_log2_lift, s=4, c="#BBBBBB", alpha=0.18, label="Unresolved")
    for relation, local in subset.groupby("relation"):
        axis.scatter(local.role_similarity, local.city_conditioned_log2_lift, s=24, c=palette[relation], alpha=0.8, label=f"{relation} (n={len(local)})")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Context-role similarity", ylabel="City-conditioned panorama log2 lift", title="Evidence separating composition (AND) from substitution (OR)")
    axis.legend(frameon=False)
    figure.savefig(figures / "Fig_Child_Relation_Evidence.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_summary(output: Path, figures: Path, summary: dict, children: pd.DataFrame, relations: pd.DataFrame) -> None:
    lines = [
        "# 512D Semantic Parent–Child Typed Hierarchy (v2)", "",
        "## Scope", "",
        f"- Analysis unit: complete four-direction panorama", f"- Cities: {summary['cities']}",
        f"- Panoramas: {summary['panoramas']:,}", f"- Patches: {summary['patches']:,}",
        f"- Patch winners: Top-{summary['top_x']} identities only (activation magnitudes discarded after ranking)",
        f"- Semantic parents with assigned dimensions: {summary['nonempty_parents']}",
        f"- Latent child concepts: {summary['child_concepts']}",
        f"- Selected AND dimension pairs: {summary['and_dimension_pairs']}",
        f"- Selected OR dimension pairs: {summary['or_dimension_pairs']}",
        f"- Child-level typed edges: {summary['child_relation_edges']}", "",
        "## Interpretation", "",
        "The hierarchy is a tree only from semantic parent to latent child. AND and OR are lateral, typed relations between children under the same parent. AND means stable positive co-composition; OR means avoidance despite similar context and spatial role, consistent with substitution. UNRESOLVED pairs are deliberately not forced into either type.", "",
        "## Child concepts", "",
    ]
    for row in children.itertuples(index=False):
        lines.append(f"- **{row.child_id} — {row.child_label}**: {row.dimension_count} dimensions; bootstrap stability {row.mean_bootstrap_stability:.3f}; representatives {row.representative_dimensions}.")
    lines += ["", "## Typed child relations", ""]
    if relations.empty:
        lines.append("No cross-child relation passed the predeclared reliability gates.")
    else:
        for row in relations.sort_values("relation_score", ascending=False).head(30).itertuples(index=False):
            lines.append(f"- **{row.relation}** {row.child_i} ↔ {row.child_j}: score {row.relation_score:.3f}, {row.supporting_dimension_pairs} supporting pair(s), city sign fraction {row.mean_city_sign_fraction:.3f}.")
    lines += ["", "## Files", "", f"- Data: `{output}`", f"- Figures: `{figures}`", "- Machine-readable hierarchy: `typed_visual_hierarchy.json`", ""]
    (output / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines))


def run(output: Path = OUTPUT, figures: Path = FIGURES, bootstraps: int = BOOTSTRAPS) -> dict:
    assert_safe_affinity()
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics, profiles, semantic_names, prevalence = load_semantics()
    parent_membership, primary, parent_shares = assign_semantic_parents(profiles, prevalence)
    evidence = load_context_evidence()
    similarity = combined_similarity(evidence, profiles)
    membership, cuts, chosen_k, child_ids = discover_children(primary, similarity, metrics, profiles, semantic_names, evidence)

    # Hold all non-role components fixed and resample cities for the role term.
    similarity_without_role = similarity - 0.30 * np.asarray(evidence["role_similarity"])
    stability = bootstrap_child_stability(
        primary, child_ids, chosen_k, similarity_without_role,
        evidence["city_occurrence"], evidence["table"].panoramas.to_numpy(int), bootstraps,
    )
    membership["bootstrap_stability"] = membership.dimension_id.map(dict(enumerate(stability)))
    children = child_summary(membership, stability, profiles, semantic_names)
    dimension_relations, child_relations = type_dimension_relations(primary, child_ids, evidence)

    parent_membership.to_csv(output / "dimension_parent_membership.csv", index=False)
    membership.sort_values(["parent_id", "child_id", "global_top4_patch_assignments"], ascending=[True, True, False]).to_csv(output / "child_concept_membership.csv", index=False)
    children.to_csv(output / "child_concept_summary.csv", index=False)
    cuts.to_csv(output / "child_cut_selection.csv", index=False)
    dimension_relations.to_csv(output / "dimension_relation_evidence.csv", index=False)
    child_relations.to_csv(output / "child_relation_edges.csv", index=False)
    np.save(output / "dimension_combined_similarity.f32.npy", similarity.astype(np.float32))
    hierarchy = build_json(children, membership, child_relations)
    save_json(output / "typed_visual_hierarchy.json", hierarchy)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "complete four-direction panorama",
        "cities": len(evidence["table"]), "panoramas": evidence["total_panoramas"],
        "patches": evidence["total_patches"], "dimensions": DIMENSIONS, "top_x": 4,
        "presence_threshold_patches": PRESENCE_PATCHES,
        "activation_magnitudes_used_for_cooccurrence": False,
        "activation_weighted_semantic_profiles_used_for_parent_assignment": True,
        "nonempty_parents": int((pd.Series(primary).value_counts() > 0).sum()),
        "child_concepts": len(children), "city_bootstraps": bootstraps,
        "mean_dimension_bootstrap_stability": float(stability.mean()),
        "and_dimension_pairs": int(dimension_relations.relation.eq("AND").sum()),
        "or_dimension_pairs": int(dimension_relations.relation.eq("OR").sum()),
        "child_relation_edges": len(child_relations),
        "similarity_weights": {"panorama_role": 0.30, "same_patch_context": 0.20, "semantic_profile": 0.30, "spatial_position": 0.10, "city_profile": 0.10},
        "parent_assignment": "semantic mass multiplied by square-root positive lift vs global prevalence; primary plus optional secondary audit membership",
        "relation_null": "city-conditioned independence at panorama and same-patch scales; normal residual p-values with within-parent Benjamini-Hochberg q<0.01",
        "limitations": ["Local evidence uses same-patch Top-4 co-activation, not a 3x3 neighborhood.", "OR indicates substitutability evidence, not proof of semantic synonymy.", "Semantic segmentation covers the existing 60,000-panorama alignment sample."],
    }
    save_json(output / "analysis_summary.json", summary)
    plot_outputs(output, figures, profiles, semantic_names)
    write_summary(output, figures, summary, children, child_relations)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output, args.figures, args.bootstraps)
