#!/usr/bin/env python3
"""Build a nested 32/64 statistical hierarchy for Feature-MAE dimensions.

Torch feature scanning and scipy/sklearn clustering run in separate processes
to respect the repository's OpenMP isolation rule.
"""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.multicity.config import CITIES, city_slug
from scripts.multicity.patch_config import OUTPUT_ROOT as TOKEN_ROOT, PATCHES_PER_IMAGE, token_path


DEFAULT_ROOT = TOKEN_ROOT.parent / "feature_mae_n30x12800_qc"


def row_cosine(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-9
    return np.clip(values @ values.T, -1, 1)


def rank_fuse(views: list[np.ndarray], neighbours: int = 30) -> np.ndarray:
    width = views[0].shape[0]
    neighbours = min(neighbours, width - 1)
    fused = np.zeros((width, width), dtype=np.float32)
    weights = 1.0 - np.arange(neighbours, dtype=np.float32) / neighbours
    for view in views:
        work = np.asarray(view, dtype=np.float32).copy()
        np.fill_diagonal(work, -np.inf)
        candidates = np.argpartition(-work, neighbours - 1, axis=1)[:, :neighbours]
        for feature in range(width):
            ordered = candidates[feature][np.argsort(-work[feature, candidates[feature]])]
            fused[feature, ordered] += weights
    fused /= len(views)
    fused = (fused + fused.T) / 2
    np.fill_diagonal(fused, 0)
    return fused / fused.max() if fused.max() > 0 else fused


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    groups = sorted(set(labels.tolist()), key=lambda value: int(np.where(labels == value)[0].min()))
    mapping = {value: index for index, value in enumerate(groups)}
    return np.asarray([mapping[value] for value in labels], dtype=np.int16)


def cluster_jaccard(full: np.ndarray, alternate: np.ndarray) -> np.ndarray:
    result = np.zeros(len(full), dtype=np.float32)
    for feature in range(len(full)):
        left = full == full[feature]
        right = alternate == alternate[feature]
        result[feature] = (left & right).sum() / max((left | right).sum(), 1)
    return result


def ppmi_context(cooccurrence: np.ndarray, support: np.ndarray, images: int) -> np.ndarray:
    expected = support[:, None] * support[None, :] / max(images, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log((cooccurrence + 0.5) / (expected + 0.5))
    value[value < 0] = 0
    np.fill_diagonal(value, 0)
    return value.astype(np.float32)


def scan(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch

    from scripts.multicity.train_feature_mae import FeatureMAE

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        args.data_root / "mae" / "model_best.pt", map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model = FeatureMAE(
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_width=int(config["decoder_width"]),
        decoder_layers=int(config["decoder_layers"]),
        mask_ratio=float(config["mask_ratio"]),
    )
    model.load_state_dict(checkpoint["model"])
    model = model.eval().requires_grad_(False).to("cuda")
    encoder = model.input_projection.weight.detach().float().cpu().numpy()
    decoder = (
        model.output_projection.weight.detach().float().cpu().numpy()
        @ model.decoder_projection.weight.detach().float().cpu().numpy()
    ).T
    np.save(output / "encoder_directions.npy", encoder.astype(np.float32))
    np.save(output / "decoder_directions.npy", decoder.astype(np.float32))

    counts = []
    for city in args.cities:
        frame = pd.read_parquet(
            args.data_root / "filtered_manifests" / f"{city_slug(city)}.parquet",
            columns=["source_image_index"],
        )
        counts.append(len(frame))
    total = sum(counts)
    top_path = output / "top1_dimensions.npy"
    city_path = output / "city_indices.npy"
    top = np.lib.format.open_memmap(
        top_path, mode="w+", dtype=np.uint16, shape=(total, PATCHES_PER_IMAGE)
    )
    city_ids = np.lib.format.open_memmap(city_path, mode="w+", dtype=np.uint8, shape=(total,))
    cursor = 0
    started = time.time()
    with torch.inference_mode():
        for city_i, (city, count) in enumerate(zip(args.cities, counts)):
            frame = pd.read_parquet(
                args.data_root / "filtered_manifests" / f"{city_slug(city)}.parquet",
                columns=["source_image_index"],
            )
            ids = frame["source_image_index"].to_numpy(np.int64)
            tokens = np.load(token_path(city, args.token_root), mmap_mode="r")
            for start in range(0, len(ids), args.batch):
                block_ids = ids[start : start + args.batch]
                block = np.asarray(tokens[block_ids], dtype=np.float32)
                batch = torch.from_numpy(block).to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent = model.encode_full(batch)
                winners = latent.argmax(dim=-1).to(torch.int16).cpu().numpy().astype(np.uint16)
                top[cursor + start : cursor + start + len(winners)] = winners
            city_ids[cursor : cursor + count] = city_i
            cursor += count
            top.flush(); city_ids.flush()
            print(f"scan [{city_i+1:02d}/{len(args.cities)}] {city}: {count:,}", flush=True)
    report = {
        "images": total,
        "patches_per_image": PATCHES_PER_IMAGE,
        "width": int(model.width),
        "cities": args.cities,
        "city_image_counts": dict(zip(args.cities, counts)),
        "elapsed_seconds": time.time() - started,
        "encoder_direction": "FeatureMAE input_projection rows",
        "decoder_direction": "linearized output_projection @ decoder_projection columns",
    }
    (output / "scan_report.json").write_text(json.dumps(report, indent=2) + "\n")


def _accumulate_statistics(top: np.ndarray, city_ids: np.ndarray, width: int, cities: int):
    from scipy import sparse

    position = np.zeros((width, PATCHES_PER_IMAGE), dtype=np.int64)
    city = np.zeros((width, cities), dtype=np.int64)
    cooccurrence = np.zeros((width, width), dtype=np.int64)
    support = np.zeros(width, dtype=np.int64)
    halves = [
        {
            "position": np.zeros_like(position), "city": np.zeros_like(city),
            "cooccurrence": np.zeros_like(cooccurrence), "support": np.zeros_like(support),
            "images": 0,
        }
        for _ in range(2)
    ]
    patch_position = np.arange(PATCHES_PER_IMAGE, dtype=np.int64)
    for start in range(0, len(top), 5000):
        block = np.asarray(top[start : start + 5000], dtype=np.int64)
        block_cities = np.asarray(city_ids[start : start + len(block)], dtype=np.int64)
        keys = block.ravel() * PATCHES_PER_IMAGE + np.tile(patch_position, len(block))
        position += np.bincount(keys, minlength=width * PATCHES_PER_IMAGE).reshape(
            width, PATCHES_PER_IMAGE
        )
        city_keys = block.ravel() * cities + np.repeat(block_cities, PATCHES_PER_IMAGE)
        city += np.bincount(city_keys, minlength=width * cities).reshape(width, cities)
        rows = np.repeat(np.arange(len(block), dtype=np.int32), PATCHES_PER_IMAGE)
        presence = sparse.csr_matrix(
            (np.ones(block.size, dtype=np.int8), (rows, block.ravel())),
            shape=(len(block), width), dtype=np.int32,
        )
        presence.data[:] = 1
        cooccurrence += (presence.T @ presence).toarray()
        support += np.asarray(presence.sum(0)).ravel()
        global_rows = np.arange(start, start + len(block))
        for parity in (0, 1):
            mask = global_rows % 2 == parity
            half = halves[parity]
            half["images"] += int(mask.sum())
            part = block[mask]
            part_cities = block_cities[mask]
            part_keys = part.ravel() * PATCHES_PER_IMAGE + np.tile(patch_position, len(part))
            half["position"] += np.bincount(
                part_keys, minlength=width * PATCHES_PER_IMAGE
            ).reshape(width, PATCHES_PER_IMAGE)
            part_city_keys = part.ravel() * cities + np.repeat(part_cities, PATCHES_PER_IMAGE)
            half["city"] += np.bincount(
                part_city_keys, minlength=width * cities
            ).reshape(width, cities)
            sub = presence[mask]
            half["cooccurrence"] += (sub.T @ sub).toarray()
            half["support"] += np.asarray(sub.sum(0)).ravel()
        if start % 50000 == 0:
            print(f"statistics {min(start+len(block),len(top)):,}/{len(top):,}", flush=True)
    return position, city, cooccurrence, support, halves


def cluster(args: argparse.Namespace) -> None:
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.manifold import spectral_embedding
    from sklearn.metrics import (
        adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
        normalized_mutual_info_score, silhouette_score,
    )

    output = args.output
    report = json.loads((output / "scan_report.json").read_text())
    width = int(report["width"])
    top = np.load(output / "top1_dimensions.npy", mmap_mode="r")
    city_ids = np.load(output / "city_indices.npy", mmap_mode="r")
    position, city, cooc, support, halves = _accumulate_statistics(
        top, city_ids, width, len(args.cities)
    )
    encoder = np.load(output / "encoder_directions.npy")
    decoder = np.load(output / "decoder_directions.npy")
    fixed = [row_cosine(decoder), row_cosine(encoder)]
    context = ppmi_context(cooc, support, len(top))
    views = [*fixed, row_cosine(position), row_cosine(city), row_cosine(context)]
    fused = rank_fuse(views, args.neighbours)
    graph = fused.copy(); np.fill_diagonal(graph, 1e-6)
    embedding = spectral_embedding(
        graph, n_components=32, random_state=args.seed, drop_first=False
    ).astype(np.float32)
    tree = linkage(embedding, method="ward")
    coarse = canonical_labels(fcluster(tree, 32, criterion="maxclust"))
    fine = canonical_labels(fcluster(tree, 64, criterion="maxclust"))
    for fine_id in range(64):
        parents = np.unique(coarse[fine == fine_id])
        if len(parents) != 1:
            raise RuntimeError(f"fine cluster {fine_id} is not nested: {parents}")

    alternate = []
    for parity, half in enumerate(halves):
        half_context = ppmi_context(
            half["cooccurrence"], half["support"], int(half["images"])
        )
        half_views = [
            *fixed, row_cosine(half["position"]), row_cosine(half["city"]),
            row_cosine(half_context),
        ]
        half_fused = rank_fuse(half_views, args.neighbours)
        half_graph = half_fused.copy(); np.fill_diagonal(half_graph, 1e-6)
        half_embedding = spectral_embedding(
            half_graph, n_components=32, random_state=args.seed + parity + 1,
            drop_first=False,
        )
        half_tree = linkage(half_embedding, method="ward")
        alternate.append(
            (
                canonical_labels(fcluster(half_tree, 32, criterion="maxclust")),
                canonical_labels(fcluster(half_tree, 64, criterion="maxclust")),
            )
        )
    coarse_stability = np.mean(
        [cluster_jaccard(coarse, value[0]) for value in alternate], axis=0
    )
    fine_stability = np.mean(
        [cluster_jaccard(fine, value[1]) for value in alternate], axis=0
    )

    rows = []
    for feature in range(width):
        rows.append(
            {
                "feature_id": feature,
                "coarse_cluster": f"C{int(coarse[feature]):03d}",
                "fine_cluster": f"F{int(fine[feature]):03d}",
                "coarse_stability": float(coarse_stability[feature]),
                "fine_stability": float(fine_stability[feature]),
                "top1_patch_support": int(position[feature].sum()),
                "image_support": int(support[feature]),
            }
        )
    with (output / "feature_hierarchy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    branches = []
    for coarse_id in range(32):
        coarse_features = np.where(coarse == coarse_id)[0]
        children = []
        for fine_id in sorted(np.unique(fine[coarse_features]).tolist()):
            features = np.where((coarse == coarse_id) & (fine == fine_id))[0]
            sub = fused[np.ix_(features, features)]
            children.append(
                {
                    "id": f"F{fine_id:03d}", "n": len(features),
                    "medoid_feature": int(features[np.argmax(sub.sum(1))]),
                    "stability": float(fine_stability[features].mean()),
                    "features": features.tolist(),
                }
            )
        sub = fused[np.ix_(coarse_features, coarse_features)]
        branches.append(
            {
                "id": f"C{coarse_id:03d}", "n": len(coarse_features),
                "medoid_feature": int(coarse_features[np.argmax(sub.sum(1))]),
                "stability": float(coarse_stability[coarse_features].mean()),
                "children": children,
            }
        )
    taxonomy = {"id": "ROOT", "n": width, "children": branches}
    (output / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2) + "\n")

    def metrics(labels: np.ndarray) -> dict:
        sizes = np.bincount(labels)
        p = sizes / sizes.sum()
        return {
            "clusters": int(len(sizes)),
            "silhouette": float(silhouette_score(embedding, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
            "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
            "size_entropy": float(-(p * np.log(p)).sum() / np.log(len(sizes))),
            "size_min": int(sizes.min()), "size_median": float(np.median(sizes)),
            "size_max": int(sizes.max()),
        }
    hierarchy_report = {
        "method": "five-view rank fusion + 32D spectral embedding + one Ward tree",
        "views": [
            "linearized_decoder", "input_projection_encoder", "patch_position",
            "city_profile", "PPMI_activation_context",
        ],
        "neighbours_per_view": args.neighbours,
        "coarse": metrics(coarse), "fine": metrics(fine),
        "nested": True,
        "split_sample": {
            "coarse_mean_jaccard": float(coarse_stability.mean()),
            "fine_mean_jaccard": float(fine_stability.mean()),
            "coarse_ari": float(np.mean([adjusted_rand_score(coarse, x[0]) for x in alternate])),
            "fine_ari": float(np.mean([adjusted_rand_score(fine, x[1]) for x in alternate])),
            "coarse_nmi": float(np.mean([normalized_mutual_info_score(coarse, x[0]) for x in alternate])),
            "fine_nmi": float(np.mean([normalized_mutual_info_score(fine, x[1]) for x in alternate])),
        },
        "note": "semantic names are intentionally not used to construct the hierarchy",
    }
    (output / "hierarchy_report.json").write_text(
        json.dumps(hierarchy_report, indent=2) + "\n"
    )
    np.save(output / "ward_linkage.npy", tree.astype(np.float32))
    np.savez_compressed(
        output / "hierarchy_arrays.npz", fused_similarity=fused.astype(np.float16),
        spectral_embedding=embedding, coarse_labels=coarse, fine_labels=fine,
        coarse_stability=coarse_stability, fine_stability=fine_stability,
    )
    print(json.dumps(hierarchy_report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "scan", "cluster"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cities", nargs="*", default=list(CITIES))
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--neighbours", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output = args.output or (args.data_root / "mae" / "hierarchy_32_64")
    if args.stage == "scan":
        scan(args)
    elif args.stage == "cluster":
        cluster(args)
    else:
        base = [
            sys.executable, "-m", "scripts.multicity.build_feature_mae_hierarchy",
            "--data-root", str(args.data_root), "--token-root", str(args.token_root),
            "--output", str(args.output), "--batch", str(args.batch),
            "--neighbours", str(args.neighbours), "--seed", str(args.seed),
            "--cities", *args.cities,
        ]
        subprocess.run([*base, "--stage", "scan"], check=True)
        subprocess.run([*base, "--stage", "cluster"], check=True)


if __name__ == "__main__":
    main()
