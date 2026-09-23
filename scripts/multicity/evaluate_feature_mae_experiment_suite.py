#!/usr/bin/env python3
"""Evaluate completed Feature-MAE controlled runs and seed reproducibility."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import spectral_embedding
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from scripts.multicity.build_feature_mae_hierarchy import (
    canonical_labels,
    rank_fuse,
    row_cosine,
)
from scripts.multicity.train_feature_mae import FeatureMAE


DEFAULT_SUITE = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_controlled_suite"
)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def directions(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FeatureMAE(
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_width=int(config["decoder_width"]),
        decoder_layers=int(config["decoder_layers"]),
        mask_ratio=float(config["mask_ratio"]),
    )
    model.load_state_dict(checkpoint["model"])
    encoder = model.input_projection.weight.detach().float().numpy()
    decoder = (
        model.output_projection.weight.detach().float().numpy()
        @ model.decoder_projection.weight.detach().float().numpy()
    ).T
    return encoder, decoder, config


def embedding_and_labels(encoder: np.ndarray, decoder: np.ndarray, seed: int):
    fused = rank_fuse([row_cosine(encoder), row_cosine(decoder)], neighbours=30)
    graph = fused.copy(); np.fill_diagonal(graph, 1e-6)
    dimensions = min(32, len(encoder) - 2)
    embedding = spectral_embedding(
        graph, n_components=dimensions, random_state=seed, drop_first=False
    )
    tree = linkage(embedding, method="ward")
    coarse = canonical_labels(fcluster(tree, min(32, len(encoder) // 2), criterion="maxclust"))
    fine = canonical_labels(fcluster(tree, min(64, len(encoder) // 2), criterion="maxclust"))
    return embedding, coarse, fine


def entropy(labels: np.ndarray) -> float:
    sizes = np.bincount(labels)
    p = sizes[sizes > 0] / len(labels)
    return float(-(p * np.log(p)).sum() / np.log(len(sizes)))


def matched_cluster_accuracy(reference: np.ndarray, target: np.ndarray) -> float:
    n_reference = int(reference.max()) + 1
    n_target = int(target.max()) + 1
    confusion = np.zeros((n_reference, n_target), dtype=np.int64)
    np.add.at(confusion, (reference, target), 1)
    rows, columns = linear_sum_assignment(-confusion)
    return float(confusion[rows, columns].sum() / len(reference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    manifest = json.loads((args.suite_root / "suite_manifest.json").read_text())
    records = []
    cached = {}
    for configuration in manifest["configurations"]:
        experiment_id = configuration["experiment_id"]
        run = args.suite_root / experiment_id
        report_path = run / "training_report.json"
        if not report_path.is_file():
            print(f"skip incomplete {experiment_id}", flush=True)
            continue
        report = json.loads(report_path.read_text())
        history = pd.read_csv(run / "history.csv")
        encoder, decoder, config = directions(run / "model_best.pt")
        embedding, coarse, fine = embedding_and_labels(
            encoder, decoder, int(configuration["seed"])
        )
        cached[experiment_id] = {
            "encoder": encoder, "decoder": decoder,
            "coarse": coarse, "fine": fine,
            "configuration": configuration,
        }
        records.append(
            {
                "experiment_id": experiment_id,
                "mask_ratio": configuration["mask_ratio"],
                "latent_width": configuration["width"],
                "encoder_depth": configuration["depth"],
                "seed": configuration["seed"],
                "train_loss": float(history.iloc[-1]["train_loss"]),
                "val_loss": float(history.iloc[-1]["validation_loss"]),
                "best_val_loss": float(history["validation_loss"].min()),
                "reconstruction_cosine": 1.0 - float(history["validation_loss"].min()),
                "silhouette": float(silhouette_score(embedding, fine)),
                "calinski_harabasz": float(calinski_harabasz_score(embedding, fine)),
                "davies_bouldin": float(davies_bouldin_score(embedding, fine)),
                "cluster_entropy": entropy(fine),
                "best_epoch": report["best_epoch"],
                "peak_gpu_gib": report["peak_gpu_gib"],
                "train_images_per_city_per_epoch": manifest["images_per_city_per_epoch"],
                "validation_images_per_city": manifest["validation_images_per_city"],
                "clustering_view": "encoder_direction+linearized_decoder_direction",
            }
        )
    save_csv(pd.DataFrame(records), args.results / "feature_mae_sensitivity.csv")

    seed_ids = [
        key for key, value in cached.items()
        if value["configuration"]["mask_ratio"] == 0.75
        and value["configuration"]["width"] == 512
        and value["configuration"]["depth"] == 4
    ]
    matching_rows = []
    stability_rows = []
    for left_id, right_id in combinations(sorted(seed_ids), 2):
        left, right = cached[left_id], cached[right_id]
        # Cross-run cosines require separate normalization, rather than row_cosine.
        e_left = left["encoder"] / (np.linalg.norm(left["encoder"], axis=1, keepdims=True) + 1e-9)
        e_right = right["encoder"] / (np.linalg.norm(right["encoder"], axis=1, keepdims=True) + 1e-9)
        d_left = left["decoder"] / (np.linalg.norm(left["decoder"], axis=1, keepdims=True) + 1e-9)
        d_right = right["decoder"] / (np.linalg.norm(right["decoder"], axis=1, keepdims=True) + 1e-9)
        cross_encoder = e_left @ e_right.T
        cross_decoder = d_left @ d_right.T
        combined = (cross_encoder + cross_decoder) / 2
        rows, columns = linear_sum_assignment(-combined)
        order = np.empty(len(columns), dtype=np.int64)
        order[rows] = columns
        matched = combined[rows, columns]
        target_coarse = right["coarse"][order]
        target_fine = right["fine"][order]
        stability_rows.append(
            {
                "run_a": left_id,
                "run_b": right_id,
                "seed_a": left["configuration"]["seed"],
                "seed_b": right["configuration"]["seed"],
                "mean_matched_cosine_similarity": float(matched.mean()),
                "median_matched_cosine_similarity": float(np.median(matched)),
                "coarse_cluster_ari": adjusted_rand_score(left["coarse"], target_coarse),
                "coarse_cluster_nmi": normalized_mutual_info_score(left["coarse"], target_coarse),
                "fine_cluster_ari": adjusted_rand_score(left["fine"], target_fine),
                "fine_cluster_nmi": normalized_mutual_info_score(left["fine"], target_fine),
                "coarse_cluster_agreement": matched_cluster_accuracy(left["coarse"], target_coarse),
                "fine_cluster_agreement": matched_cluster_accuracy(left["fine"], target_fine),
                "matching_method": "Hungarian maximum of mean encoder/decoder directional cosine",
            }
        )
        for reference, target in zip(rows, columns):
            matching_rows.append(
                {
                    "run_a": left_id,
                    "run_b": right_id,
                    "dimension_a": int(reference),
                    "dimension_b": int(target),
                    "encoder_cosine": float(cross_encoder[reference, target]),
                    "decoder_cosine": float(cross_decoder[reference, target]),
                    "combined_cosine": float(combined[reference, target]),
                }
            )
    save_csv(pd.DataFrame(stability_rows), args.results / "seed_stability.csv")
    save_csv(pd.DataFrame(matching_rows), args.results / "seed_dimension_matching.csv")
    print(f"evaluated {len(records)} runs and {len(stability_rows)} seed pairs", flush=True)


if __name__ == "__main__":
    main()
