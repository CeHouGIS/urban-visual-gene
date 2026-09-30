"""Discover a language-free statistical taxonomy for BatchTopK W1024/K8.

The pipeline uses five views of every gene (encoder, decoder, position, city,
and activation context), fuses their nearest-neighbour ranks, and builds a
spectral/Ward hierarchy. Statistical roles and relations are inferred from
split-sample stability, image-quality association, spatial structure,
activation containment, and same-patch co-expression.

No natural-language description, manual annotation, or VLM is used. Cluster
names are stable identifiers such as C003/F017 rather than semantic claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import chi2, f as f_distribution, hypergeom
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import spectral_embedding
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from formal.interpretation.build_w1024_audit import (
    CITY_ORDER,
    ROOT,
    RUN,
    decoder_atoms,
    normalized_entropy,
    position_diagnostics,
    prevalence_group,
)


DEFAULT_OUT = ROOT / "formal" / "site" / "w1024_statistical"
QC_ROOT = ROOT / "formal" / "qc_full"


def log(*args: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def row_cosine(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)
    return np.clip(values @ values.T, -1, 1)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input shape."""
    values = np.asarray(p_values, dtype=np.float64)
    flat = values.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.reshape(values.shape)


def rank_fuse(views: list[np.ndarray], neighbours: int = 30) -> np.ndarray:
    """Fuse heterogeneous similarities by symmetric within-view neighbour rank."""
    if not views:
        raise ValueError("at least one similarity view is required")
    width = views[0].shape[0]
    neighbours = min(neighbours, width - 1)
    fused = np.zeros((width, width), dtype=np.float32)
    for view in views:
        if view.shape != (width, width):
            raise ValueError("all similarity views must have the same square shape")
        work = np.asarray(view, dtype=np.float32).copy()
        np.fill_diagonal(work, -np.inf)
        candidates = np.argpartition(-work, neighbours - 1, axis=1)[:, :neighbours]
        for gene in range(width):
            ordered = candidates[gene][np.argsort(-work[gene, candidates[gene]])]
            weights = 1.0 - np.arange(neighbours, dtype=np.float32) / neighbours
            fused[gene, ordered] += weights
    fused /= len(views)
    fused = (fused + fused.T) / 2
    np.fill_diagonal(fused, 0)
    scale = fused.max()
    return fused / scale if scale > 0 else fused


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    groups = sorted(set(labels.tolist()), key=lambda lab: int(np.where(labels == lab)[0].min()))
    mapping = {lab: i for i, lab in enumerate(groups)}
    return np.array([mapping[lab] for lab in labels], dtype=int)


def choose_cut(embedding: np.ndarray, candidates: Iterable[int]) -> tuple[int, np.ndarray, dict[int, float]]:
    """Select a hierarchy cut by silhouette, penalising singleton-heavy cuts."""
    scores: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for count in sorted(set(int(x) for x in candidates)):
        if count < 2 or count >= len(embedding):
            continue
        labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(embedding)
        sizes = np.bincount(labels)
        singleton_fraction = float((sizes == 1).sum() / count)
        score = float(silhouette_score(embedding, labels) * (1 - singleton_fraction))
        scores[count] = score
        labels_by_k[count] = labels
    if not scores:
        raise ValueError("no valid hierarchy cut candidates")
    best = max(scores, key=scores.get)
    return best, canonical_labels(labels_by_k[best]), scores


def sparse_image_matrices(top: np.ndarray, width: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    n_img, n_patch = top.shape
    rows = np.repeat(np.arange(n_img, dtype=np.int32), n_patch)
    counts = sparse.csr_matrix(
        (np.ones(top.size, dtype=np.int16), (rows, top.ravel())),
        shape=(n_img, width),
        dtype=np.int32,
    )
    counts.sum_duplicates()
    presence = counts.copy()
    presence.data[:] = 1
    return counts, presence


def ppmi_context(cooccurrence: np.ndarray, support: np.ndarray, n_img: int) -> np.ndarray:
    expected = support[:, None] * support[None, :] / max(n_img, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log((cooccurrence + 0.5) / (expected + 0.5))
    ppmi[ppmi < 0] = 0
    np.fill_diagonal(ppmi, 0)
    return ppmi.astype(np.float32)


def half_position_counts(top: np.ndarray, width: int, mask: np.ndarray) -> np.ndarray:
    part = top[mask]
    n_patch = part.shape[1]
    keys = part.ravel() * n_patch + np.tile(np.arange(n_patch, dtype=np.int64), len(part))
    return np.bincount(keys, minlength=width * n_patch).reshape(width, n_patch).astype(np.float32)


def half_city_profiles(top: np.ndarray, city: np.ndarray, width: int, mask: np.ndarray) -> np.ndarray:
    profiles = np.zeros((width, len(CITY_ORDER)), dtype=np.float32)
    for city_id, name in enumerate(CITY_ORDER):
        ids = top[mask & (city == name)].ravel()
        profiles[:, city_id] = np.bincount(ids, minlength=width)
    return profiles


def encoder_atoms(checkpoint: Path) -> np.ndarray:
    meta = torch.load(checkpoint, map_location="cpu")
    weight = meta["state"]["encoder.weight"].detach().float().cpu().numpy()
    return weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-9)


def adjacency_counts(top: np.ndarray, width: int) -> np.ndarray:
    grid = int(round(math.sqrt(top.shape[1])))
    maps = top.reshape(-1, grid, grid)
    counts = np.zeros((width, width), dtype=np.int64)
    for start in range(0, len(maps), 250):
        chunk = maps[start:start + 250]
        for left, right in (
            (chunk[:, :, :-1], chunk[:, :, 1:]),
            (chunk[:, :-1, :], chunk[:, 1:, :]),
            (chunk[:, :-1, :-1], chunk[:, 1:, 1:]),
            (chunk[:, :-1, 1:], chunk[:, 1:, :-1]),
        ):
            keys = left.ravel().astype(np.int64) * width + right.ravel()
            counts += np.bincount(keys, minlength=width * width).reshape(width, width)
    return counts + counts.T


def patch_coexpression(idx: np.ndarray, val: np.ndarray, width: int) -> np.ndarray:
    matrix = np.zeros((width, width), dtype=np.int64)
    for start in range(0, len(idx), 250):
        ii = idx[start:start + 250]
        vv = val[start:start + 250]
        active = (vv > 0) & (vv >= 0.25 * vv.max(2, keepdims=True))
        genes = ii.reshape(-1, ii.shape[2])
        mask = active.reshape(-1, active.shape[2])
        rows = np.repeat(np.arange(len(genes), dtype=np.int32), genes.shape[1])[mask.ravel()]
        cols = genes[mask]
        patch_gene = sparse.csr_matrix(
            (np.ones(len(cols), dtype=np.int8), (rows, cols)),
            shape=(len(genes), width),
            dtype=np.int32,
        )
        patch_gene.data[:] = 1
        matrix += (patch_gene.T @ patch_gene).toarray()
    return matrix


def morans_i(maps: np.ndarray) -> np.ndarray:
    _width, grid, _ = maps.shape
    centered = maps - maps.mean((1, 2), keepdims=True)
    denominator = (centered * centered).sum((1, 2)) + 1e-9
    numerator = (
        (centered[:, :, :-1] * centered[:, :, 1:]).sum((1, 2))
        + (centered[:, :-1, :] * centered[:, 1:, :]).sum((1, 2))
    ) * 2
    n_edges_directed = 4 * grid * (grid - 1)
    return grid * grid / n_edges_directed * numerator / denominator


def load_qc_features(city: np.ndarray, pano: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    columns = ["p_good", "p_dark", "p_blur", "p_tunnel"]
    values = np.full((len(city), len(columns)), np.nan, dtype=np.float32)
    available = np.zeros(len(city), dtype=bool)
    for name in CITY_ORDER:
        rows = np.where(city == name)[0]
        if not len(rows):
            continue
        file = QC_ROOT / f"{name}.parquet"
        if not file.exists():
            continue
        wanted = set(str(x) for x in pano[rows])
        table = pq.read_table(file, columns=["pano_id", *columns])
        ids = table["pano_id"].to_pylist()
        lookup: dict[str, tuple[float, ...]] = {}
        arrays = [table[col].to_numpy(zero_copy_only=False) for col in columns]
        for table_row, pano_id in enumerate(ids):
            key = str(pano_id)
            if key in wanted:
                lookup[key] = tuple(float(array[table_row]) for array in arrays)
        for row in rows:
            match = lookup.get(str(pano[row]))
            if match is not None:
                values[row] = match
                available[row] = True
    return values, available


def quality_association(
    image_counts: np.ndarray, quality: np.ndarray, available: np.ndarray, city: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = ["good", "dark", "blur", "tunnel"]
    use = np.where(available)[0]
    x = quality[use].astype(np.float64)
    y = image_counts[use].astype(np.float64) / image_counts.shape[1]
    used_city = city[use]
    for name in CITY_ORDER:
        mask = used_city == name
        if mask.any():
            x[mask] -= x[mask].mean(0)
            y[mask] -= y[mask].mean(0)
    x = StandardScaler().fit_transform(x)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ beta
    ss_total = (y * y).sum(0) + 1e-12
    r2 = np.clip((predicted * predicted).sum(0) / ss_total, 0, 1)
    residual = ((y - predicted) ** 2).sum(0)
    df_model = x.shape[1]
    df_resid = max(len(x) - df_model - 1, 1)
    f_stat = (r2 / df_model) / np.maximum((1 - r2) / df_resid, 1e-12)
    p_values = f_distribution.sf(f_stat, df_model, df_resid)
    strongest = np.argmax(np.abs(beta), axis=0)
    return r2.astype(np.float32), p_values, [names[i] for i in strongest]


def chi_square_effect(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected = counts.sum(1, keepdims=True) / counts.shape[1]
    statistic = ((counts - expected) ** 2 / np.maximum(expected, 1e-9)).sum(1)
    p = chi2.sf(statistic, counts.shape[1] - 1)
    cramers_v = np.sqrt(statistic / np.maximum(counts.sum(1) * (counts.shape[1] - 1), 1))
    return cramers_v, p


def mixture_posterior(values: np.ndarray, prefer_high: bool, seed: int) -> tuple[np.ndarray, int]:
    """Choose 1-3 Gaussian components by BIC and return extreme-component posterior."""
    raw = np.asarray(values, dtype=np.float64)
    transformed = np.log1p(np.maximum(raw, 0)).reshape(-1, 1)
    models = []
    for components in (1, 2, 3):
        model = GaussianMixture(n_components=components, random_state=seed, n_init=5).fit(transformed)
        models.append((model.bic(transformed), model))
    model = min(models, key=lambda item: item[0])[1]
    means = model.means_.ravel()
    target = int(np.argmax(means) if prefer_high else np.argmin(means))
    return model.predict_proba(transformed)[:, target], model.n_components


def spectral_hierarchy(fused: np.ndarray, seed: int) -> dict[str, Any]:
    graph = fused.copy()
    np.fill_diagonal(graph, 1e-6)
    embedding = spectral_embedding(graph, n_components=32, random_state=seed, drop_first=False)
    coarse_k, coarse, coarse_scores = choose_cut(embedding, [8, 12, 16, 20, 24, 32])
    fine_candidates = [coarse_k * x for x in (2, 3, 4, 5)]
    fine_k, fine, fine_scores = choose_cut(embedding, fine_candidates)
    z = linkage(embedding, method="ward")
    # Use cuts from one linkage so fine branches are guaranteed to nest.
    coarse = canonical_labels(fcluster(z, coarse_k, criterion="maxclust"))
    fine = canonical_labels(fcluster(z, fine_k, criterion="maxclust"))
    return {
        "embedding": embedding,
        "linkage": z,
        "coarse_k": coarse_k,
        "fine_k": fine_k,
        "coarse": coarse,
        "fine": fine,
        "coarse_scores": coarse_scores,
        "fine_scores": fine_scores,
    }


def cluster_jaccard(full: np.ndarray, alternate: np.ndarray) -> np.ndarray:
    result = np.zeros(len(full), dtype=np.float32)
    for gene in range(len(full)):
        a = full == full[gene]
        b = alternate == alternate[gene]
        result[gene] = (a & b).sum() / max((a | b).sum(), 1)
    return result


def canonical_cluster_ids(labels: np.ndarray, prefix: str) -> dict[int, str]:
    groups = sorted(set(labels.tolist()), key=lambda lab: int(np.where(labels == lab)[0].min()))
    return {lab: f"{prefix}{rank:03d}" for rank, lab in enumerate(groups)}


def build_taxonomy(
    coarse: np.ndarray, fine: np.ndarray, fused: np.ndarray, stability: np.ndarray,
    roles: list[str], granularity: np.ndarray,
) -> dict[str, Any]:
    coarse_ids = canonical_cluster_ids(coarse, "C")
    fine_ids = canonical_cluster_ids(fine, "F")
    branches = []
    for coarse_label in sorted(set(coarse.tolist()), key=lambda x: coarse_ids[x]):
        coarse_genes = np.where(coarse == coarse_label)[0]
        children = []
        for fine_label in sorted(set(fine[coarse_genes].tolist()), key=lambda x: fine_ids[x]):
            genes = np.where((coarse == coarse_label) & (fine == fine_label))[0]
            sub = fused[np.ix_(genes, genes)]
            medoid = int(genes[np.argmax(sub.sum(1))])
            children.append({
                "id": fine_ids[fine_label],
                "n": int(len(genes)),
                "medoid_gene": medoid,
                "stability": round(float(stability[genes].mean()), 6),
                "granularity_distribution": {
                    f"L{level}": int((granularity[genes] == level).sum())
                    for level in sorted(set(granularity[genes].tolist()))
                },
                "role_distribution": {
                    role: sum(roles[g] == role for g in genes) for role in sorted(set(roles[g] for g in genes))
                },
                "genes": [int(g) for g in genes],
            })
        sub = fused[np.ix_(coarse_genes, coarse_genes)]
        branches.append({
            "id": coarse_ids[coarse_label],
            "n": int(len(coarse_genes)),
            "medoid_gene": int(coarse_genes[np.argmax(sub.sum(1))]),
            "stability": round(float(stability[coarse_genes].mean()), 6),
            "children": children,
        })
    return {"id": "ROOT", "n": len(coarse), "children": branches}


def relation_candidates(
    fused: np.ndarray, fine: np.ndarray, granularity: np.ndarray,
    image_cooc: np.ndarray, image_support: np.ndarray, decoder_sim: np.ndarray,
    patch_cooc: np.ndarray, n_images: int, n_patches: int,
) -> list[dict[str, Any]]:
    width = len(fine)
    upper_i, upper_j = np.triu_indices(width, 1)
    overlap = image_cooc[upper_i, upper_j].astype(np.float64)
    union = image_support[upper_i] + image_support[upper_j] - overlap
    jaccard = overlap / np.maximum(union, 1)
    background_p = hypergeom.sf(
        overlap - 1, n_images, image_support[upper_i], image_support[upper_j],
    )
    background_q = bh_fdr(background_p)

    fused_pair = fused[upper_i, upper_j]
    decoder_pair = np.maximum(decoder_sim[upper_i, upper_j], 0)
    equivalent_score = np.cbrt(np.maximum(fused_pair * decoder_pair * jaccard, 0))
    # "Equivalent" is intentionally strict: every view must independently be
    # extreme and the pair must be reciprocal top-5 neighbours in all views.
    def reciprocal_top(values: np.ndarray, count: int = 5) -> np.ndarray:
        work = values.copy()
        np.fill_diagonal(work, -np.inf)
        top = np.argpartition(-work, count - 1, axis=1)[:, :count]
        directed = np.zeros_like(work, dtype=bool)
        directed[np.arange(width)[:, None], top] = True
        return directed & directed.T

    jaccard_matrix = image_cooc / np.maximum(
        image_support[:, None] + image_support[None, :] - image_cooc, 1,
    )
    reciprocal = reciprocal_top(fused) & reciprocal_top(decoder_sim) & reciprocal_top(jaccard_matrix)
    fused_cut = np.quantile(fused_pair, 0.99)
    decoder_cut = np.quantile(decoder_pair, 0.99)
    jaccard_cut = np.quantile(jaccard, 0.99)
    equivalent_mask = (
        reciprocal[upper_i, upper_j]
        & (fused_pair >= fused_cut) & (decoder_pair >= decoder_cut) & (jaccard >= jaccard_cut)
        & (background_q <= 0.01)
        & (np.minimum(image_support[upper_i], image_support[upper_j])
           / np.maximum(image_support[upper_i], image_support[upper_j]) >= 0.5)
    )
    relations: list[dict[str, Any]] = []
    equivalent_pairs: set[tuple[int, int]] = set()
    for index in np.where(equivalent_mask)[0]:
        a, b = int(upper_i[index]), int(upper_j[index])
        equivalent_pairs.add((a, b))
        relations.append({
            "source": a, "target": b, "relation": "equivalent_to",
            "score": round(float(equivalent_score[index]), 6),
            "support": int(overlap[index]), "q_value": float(background_q[index]),
        })

    same_fine = fine[upper_i] == fine[upper_j]
    parent = np.where(image_support[upper_i] >= image_support[upper_j], upper_i, upper_j)
    child = np.where(parent == upper_i, upper_j, upper_i)
    conditional = overlap / np.maximum(image_support[child], 1)
    reverse = overlap / np.maximum(image_support[parent], 1)
    asymmetry = conditional - reverse
    pool = same_fine & (overlap > 0)
    containment_cut = np.quantile(conditional[pool], 0.95) if pool.any() else 1.0
    asymmetry_cut = np.quantile(asymmetry[pool & (asymmetry > 0)], 0.75) if (pool & (asymmetry > 0)).any() else 1.0
    specialize = pool & (conditional >= containment_cut) & (asymmetry >= asymmetry_cut) & (background_q <= 0.01)
    specialization_pairs: set[tuple[int, int]] = set()
    for index in np.where(specialize)[0]:
        p, c = int(parent[index]), int(child[index])
        key = tuple(sorted((p, c)))
        if key in equivalent_pairs:
            continue
        specialization_pairs.add(key)
        relations.append({
            "source": c, "target": p, "relation": "specializes",
            "score": round(float(conditional[index] - reverse[index]), 6),
            "support": int(overlap[index]), "q_value": float(background_q[index]),
        })

    # Same-patch co-expression is kept separate from similarity and hierarchy.
    patch_support = np.diag(patch_cooc).astype(np.float64)
    patch_overlap = patch_cooc[upper_i, upper_j].astype(np.float64)
    patch_lift = patch_overlap * n_patches / np.maximum(patch_support[upper_i] * patch_support[upper_j], 1)
    patch_p = hypergeom.sf(patch_overlap - 1, n_patches, patch_support[upper_i], patch_support[upper_j])
    patch_q = bh_fdr(patch_p)
    supported = patch_overlap > 0
    lift_cut = np.quantile(patch_lift[supported], 0.995) if supported.any() else np.inf
    coactive = np.where(supported & (patch_lift >= lift_cut) & (patch_q <= 0.01))[0]
    coactive = coactive[np.argsort(-patch_lift[coactive])[:2000]]
    for index in coactive:
        relations.append({
            "source": int(upper_i[index]), "target": int(upper_j[index]),
            "relation": "coactivates_with", "score": round(float(patch_lift[index]), 6),
            "support": int(patch_overlap[index]), "q_value": float(patch_q[index]),
        })

    # Siblings are the strongest fused neighbours in the same fine branch and scale.
    sibling_pairs: set[tuple[int, int]] = set()
    for gene in range(width):
        candidates = np.where((fine == fine[gene]) & (granularity == granularity[gene]))[0]
        candidates = candidates[candidates != gene]
        candidates = candidates[fused[gene, candidates] > 0]
        for other in candidates[np.argsort(-fused[gene, candidates])[:3]]:
            key = tuple(sorted((gene, int(other))))
            if key in equivalent_pairs or key in specialization_pairs or key in sibling_pairs:
                continue
            sibling_pairs.add(key)
            relations.append({
                "source": key[0], "target": key[1], "relation": "sibling_of",
                "score": round(float(fused[key]), 6), "support": int(image_cooc[key]),
                "q_value": None,
            })
    return relations


def write_csv(rows: list[dict[str, Any]], file: Path) -> None:
    fields = list(rows[0])
    with file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log(f"load {args.sparse}")
    archive = np.load(args.sparse, allow_pickle=True)
    idx = archive["idx"].astype(np.int64)
    val = archive["val"].astype(np.float32)
    city = np.array([str(x) for x in archive["city"]])
    pano = np.array([str(x) for x in archive["pano"]])
    n_img, n_patch, _ = idx.shape
    decoder = decoder_atoms(Path(args.checkpoint))
    encoder = encoder_atoms(Path(args.checkpoint))
    width = len(decoder)
    top = idx[:, :, 0]

    log("build image presence/context matrices")
    image_counts_sparse, image_presence = sparse_image_matrices(top, width)
    image_counts = image_counts_sparse.toarray().astype(np.float32)
    image_support = np.asarray(image_presence.sum(0)).ravel().astype(np.int64)
    image_cooc = (image_presence.T @ image_presence).toarray().astype(np.int64)
    context = ppmi_context(image_cooc, image_support, n_img)

    log("compute position, adjacency, and same-patch co-expression")
    pos = position_diagnostics(top, width)
    adjacent = adjacency_counts(top, width)
    patch_cooc = patch_coexpression(idx, val, width)
    patch_support = np.diag(patch_cooc).astype(np.int64)

    city_counts = np.zeros((width, len(CITY_ORDER)), dtype=np.float32)
    for city_id, name in enumerate(CITY_ORDER):
        city_counts[:, city_id] = np.bincount(top[city == name].ravel(), minlength=width)

    log("load QC probabilities and fit within-city quality associations")
    quality, quality_available = load_qc_features(city, pano)
    quality_r2, quality_p, quality_driver = quality_association(
        image_counts, quality, quality_available, city,
    )

    decoder_sim = np.clip(decoder @ decoder.T, -1, 1)
    views = [
        decoder_sim,
        np.clip(encoder @ encoder.T, -1, 1),
        row_cosine(pos["maps"].reshape(width, -1)),
        row_cosine(city_counts),
        row_cosine(context),
    ]
    log("rank-fuse five statistical views")
    fused = rank_fuse(views, neighbours=args.neighbours)
    graph = nx.from_numpy_array((fused > 0).astype(np.int8))
    components = nx.number_connected_components(graph)
    log(f"fused graph components={components}, edges={graph.number_of_edges():,}")
    cache_file = output / "statistical_arrays.npz"
    if cache_file.exists() and not args.force_hierarchy:
        log(f"reuse hierarchy/stability from {cache_file}")
        cache = np.load(cache_file)
        coarse = cache["coarse_labels"].astype(int)
        fine = cache["fine_labels"].astype(int)
        split_stability = cache["split_stability"].astype(np.float32)
        old_summary_file = output / "summary.json"
        old_summary = json.loads(old_summary_file.read_text()) if old_summary_file.exists() else {}
        hierarchy = {
            "embedding": cache["spectral_embedding"].astype(np.float32),
            "coarse": coarse,
            "fine": fine,
            "coarse_k": int(coarse.max() + 1),
            "fine_k": int(fine.max() + 1),
            "coarse_scores": old_summary.get("cut_scores", {}).get("coarse", {}),
            "fine_scores": old_summary.get("cut_scores", {}).get("fine", {}),
        }
    else:
        hierarchy = spectral_hierarchy(fused, args.seed)
        coarse = hierarchy["coarse"]
        fine = hierarchy["fine"]
        log(f"selected hierarchy coarse={hierarchy['coarse_k']} fine={hierarchy['fine_k']}")

        log("split-sample stability")
        half_labels = []
        for parity in (0, 1):
            mask = np.arange(n_img) % 2 == parity
            half_presence = image_presence[mask]
            half_cooc = (half_presence.T @ half_presence).toarray().astype(np.int64)
            half_support = np.diag(half_cooc)
            half_context = ppmi_context(half_cooc, half_support, int(mask.sum()))
            half_views = [
                views[0], views[1],
                row_cosine(half_position_counts(top, width, mask)),
                row_cosine(half_city_profiles(top, city, width, mask)),
                row_cosine(half_context),
            ]
            half_fused = rank_fuse(half_views, neighbours=args.neighbours)
            half_embedding = spectral_embedding(
                half_fused + np.eye(width, dtype=np.float32) * 1e-6,
                n_components=32, random_state=args.seed + parity + 1, drop_first=False,
            )
            half_z = linkage(half_embedding, method="ward")
            half_labels.append(canonical_labels(fcluster(half_z, hierarchy["fine_k"], criterion="maxclust")))
        split_stability = (cluster_jaccard(fine, half_labels[0]) + cluster_jaccard(fine, half_labels[1])) / 2

    patches_per_image = np.bincount(top.ravel(), minlength=width) / np.maximum(image_support, 1)
    self_adjacency = np.diag(adjacent) / np.maximum(np.bincount(top.ravel(), minlength=width) * 8, 1)
    spatial_entropy = np.array([normalized_entropy(row) for row in pos["maps"].reshape(width, -1)])
    granularity_features = StandardScaler().fit_transform(np.column_stack([
        np.log1p(patches_per_image), self_adjacency, spatial_entropy,
    ]))
    granularity_models = []
    for components_count in range(2, 7):
        model = GaussianMixture(n_components=components_count, random_state=args.seed, n_init=10).fit(granularity_features)
        granularity_models.append((model.bic(granularity_features), model))
    granularity_model = min(granularity_models, key=lambda x: x[0])[1]
    raw_granularity = granularity_model.predict(granularity_features)
    order = sorted(set(raw_granularity), key=lambda label: float(patches_per_image[raw_granularity == label].mean()))
    granularity_map = {label: rank for rank, label in enumerate(order)}
    granularity = np.array([granularity_map[label] for label in raw_granularity], dtype=int)

    city_effect, city_p = chi_square_effect(city_counts)
    position_effect, position_p = chi_square_effect(pos["maps"].reshape(width, -1))
    city_q = bh_fdr(city_p)
    position_q = bh_fdr(position_p)
    quality_q = bh_fdr(quality_p)
    position_posterior, position_components = mixture_posterior(pos["r2"], True, args.seed)
    quality_posterior, quality_components = mixture_posterior(quality_r2, True, args.seed)
    rare_posterior, support_components = mixture_posterior(image_support, False, args.seed)
    unstable_posterior, stability_components = mixture_posterior(1 - split_stability, True, args.seed)
    city_specific_posterior, city_components = mixture_posterior(1 - np.array([
        normalized_entropy(row) for row in city_counts
    ]), True, args.seed)

    nearest_fused = np.partition(fused, -2, axis=1)[:, -2]
    preliminary_roles = []
    for gene in range(width):
        if position_posterior[gene] >= 0.8:
            role = "position_component"
        elif quality_posterior[gene] >= 0.8:
            role = "quality_component"
        elif unstable_posterior[gene] >= 0.8:
            role = "unstable_component"
        elif city_specific_posterior[gene] >= 0.8:
            role = "city_specific_component"
        elif rare_posterior[gene] >= 0.8 and split_stability[gene] >= np.median(split_stability):
            role = "rare_stable_component"
        elif split_stability[gene] >= np.median(split_stability) and nearest_fused[gene] >= np.median(nearest_fused):
            role = "stable_structured"
        else:
            role = "weakly_structured"
        preliminary_roles.append(role)

    log("infer empirical gene relations")
    relations = relation_candidates(
        fused, fine, granularity, image_cooc, image_support, decoder_sim,
        patch_cooc, n_img, n_img * n_patch,
    )
    redundant = {int(r[key]) for r in relations if r["relation"] == "equivalent_to" for key in ("source", "target")}
    roles = preliminary_roles

    coarse_ids = canonical_cluster_ids(coarse, "C")
    fine_ids = canonical_cluster_ids(fine, "F")
    metrics = []
    top1_support = np.bincount(top.ravel(), minlength=width)
    for gene in range(width):
        metrics.append({
            "gene_id": gene,
            "statistical_role": roles[gene],
            "granularity_level": f"L{granularity[gene]}",
            "coarse_cluster": coarse_ids[int(coarse[gene])],
            "fine_cluster": fine_ids[int(fine[gene])],
            "split_stability": round(float(split_stability[gene]), 6),
            "nearest_fused_similarity": round(float(nearest_fused[gene]), 6),
            "top1_patch_support": int(top1_support[gene]),
            "image_support": int(image_support[gene]),
            "patches_per_active_image": round(float(patches_per_image[gene]), 6),
            "self_adjacency_rate": round(float(self_adjacency[gene]), 6),
            "spatial_morans_i": round(float(morans_i(pos["maps"])[gene]), 6),
            "spatial_entropy": round(float(spatial_entropy[gene]), 6),
            "position_r2": round(float(pos["r2"][gene]), 6),
            "position_effect": round(float(position_effect[gene]), 6),
            "position_q": float(position_q[gene]),
            "position_component_probability": round(float(position_posterior[gene]), 6),
            "quality_r2": round(float(quality_r2[gene]), 6),
            "quality_driver": quality_driver[gene],
            "quality_q": float(quality_q[gene]),
            "quality_component_probability": round(float(quality_posterior[gene]), 6),
            "city_effect": round(float(city_effect[gene]), 6),
            "city_q": float(city_q[gene]),
            "city_entropy": round(float(normalized_entropy(city_counts[gene])), 6),
            "city_specific_probability": round(float(city_specific_posterior[gene]), 6),
            "prevalence_group": prevalence_group(int((city_counts[gene] / np.maximum(city_counts.sum(0), 1) >= 5e-4).sum())),
            "rare_probability": round(float(rare_posterior[gene]), 6),
            "unstable_probability": round(float(unstable_posterior[gene]), 6),
            "has_equivalent_relation": gene in redundant,
        })

    taxonomy = build_taxonomy(coarse, fine, fused, split_stability, roles, granularity)
    summary = {
        "dataset": "BatchTopK W1024/K8",
        "method": "language-free multiview rank fusion + spectral/Ward hierarchy",
        "n_genes": width,
        "n_images": n_img,
        "n_patches": n_img * n_patch,
        "views": ["decoder", "encoder", "position", "city", "activation_context"],
        "fused_graph_components": components,
        "coarse_clusters": hierarchy["coarse_k"],
        "fine_clusters": hierarchy["fine_k"],
        "granularity_levels": int(len(set(granularity.tolist()))),
        "role_counts": {role: roles.count(role) for role in sorted(set(roles))},
        "relation_counts": {
            relation: sum(r["relation"] == relation for r in relations)
            for relation in sorted(set(r["relation"] for r in relations))
        },
        "mixture_components": {
            "position": position_components,
            "quality": quality_components,
            "support": support_components,
            "stability": stability_components,
            "city_specificity": city_components,
        },
        "cut_scores": {
            "coarse": hierarchy["coarse_scores"],
            "fine": hierarchy["fine_scores"],
        },
    }
    write_csv(metrics, output / "gene_metrics.csv")
    write_csv(relations, output / "gene_relations.csv")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output / "taxonomy.json").write_text(json.dumps(taxonomy, ensure_ascii=False))
    (output / "gene_metrics.json").write_text(json.dumps({str(r["gene_id"]): r for r in metrics}, ensure_ascii=False))
    (output / "gene_relations.json").write_text(json.dumps(relations, ensure_ascii=False))
    np.savez_compressed(
        output / "statistical_arrays.npz",
        fused_similarity=fused.astype(np.float16),
        spectral_embedding=hierarchy["embedding"].astype(np.float32),
        coarse_labels=coarse.astype(np.int16),
        fine_labels=fine.astype(np.int16),
        granularity=granularity.astype(np.int8),
        split_stability=split_stability.astype(np.float32),
    )
    log(f"[done] statistical taxonomy -> {output}")
    log(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, default=RUN / "sparse_acts.npz")
    parser.add_argument("--checkpoint", type=Path, default=RUN / "batch_topk_w1024_k8.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--neighbours", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-hierarchy", action="store_true", help="recompute spectral hierarchy and split stability")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
