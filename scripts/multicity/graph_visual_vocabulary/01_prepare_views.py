#!/usr/bin/env python3
"""Prepare E, D and corrected 14x56 panorama spatial view P."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, save_json, should_skip


def aggregate_panorama_views(config: dict, force: bool = False) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    cache = config["paths"]["cache_root"]
    source = config["paths"]["source_hierarchy"]
    panorama_root = config["paths"]["panorama_root"]
    views = cache / "views"
    targets = [views / "encoder_directions.f32.npy", views / "decoder_directions.f32.npy", views / "spatial_profile_14x56.u64.npy", views / "dimension_top_panoramas.csv"]
    if should_skip(targets, force):
        return {"status": "existing"}

    encoder = np.load(source / "encoder_directions.npy")
    decoder = np.load(source / "decoder_directions.npy")
    if encoder.shape != (512, 768) or decoder.shape != (512, 768):
        raise ValueError(f"unexpected direction shapes E={encoder.shape}, D={decoder.shape}")
    np.save(targets[0], encoder.astype(np.float32))
    np.save(targets[1], decoder.astype(np.float32))

    rows = int(config["panorama"]["rows"]); columns = int(config["panorama"]["columns"])
    dimensions = int(config["panorama"]["dimensions"]); patches = rows * columns
    spatial = np.zeros((dimensions, patches), dtype=np.uint64)
    city_counts = []
    top_n = int(config["panorama"]["exemplar_panoramas_per_dimension"])
    best_scores = np.full((dimensions, top_n), -np.inf, dtype=np.float32)
    best_city = np.full((dimensions, top_n), "", dtype=object)
    best_pano = np.full((dimensions, top_n), -1, dtype=np.int32)
    total_panoramas = 0
    city_rows = []
    prediction_dirs = sorted((panorama_root / "predictions").glob("*/full"))
    if len(prediction_dirs) != 30:
        raise ValueError(f"expected 30 complete city prediction directories, found {len(prediction_dirs)}")
    position = np.arange(patches, dtype=np.int64)
    for city_index, prediction in enumerate(prediction_dirs, 1):
        city_slug = prediction.parent.name
        winners = np.load(prediction / "top1_dimensions.u16.npy", mmap_mode="r")
        activation = np.load(prediction / "activation_top20.f16.npy", mmap_mode="r")
        if winners.shape[1:] != (rows, columns) or activation.shape != (len(winners), dimensions):
            raise ValueError(f"invalid rectangular panorama arrays for {city_slug}")
        local_counts = np.zeros(dimensions, dtype=np.uint64)
        for start in range(0, len(winners), 512):
            block = np.asarray(winners[start : start + 512], dtype=np.int64).reshape(-1, patches)
            keys = block.ravel() * patches + np.tile(position, len(block))
            spatial += np.bincount(keys, minlength=dimensions * patches).reshape(dimensions, patches).astype(np.uint64)
            local_counts += np.bincount(block.ravel(), minlength=dimensions).astype(np.uint64)
        local_activation = np.asarray(activation, dtype=np.float32)
        candidate_n = min(top_n, len(local_activation))
        indices = np.argpartition(local_activation, -candidate_n, axis=0)[-candidate_n:]
        values = np.take_along_axis(local_activation, indices, axis=0)
        for dimension in range(dimensions):
            scores = np.r_[best_scores[dimension], values[:, dimension]]
            cities = np.r_[best_city[dimension], np.repeat(city_slug, candidate_n)]
            panos = np.r_[best_pano[dimension], indices[:, dimension]]
            keep = np.argsort(-scores)[:top_n]
            best_scores[dimension] = scores[keep]
            best_city[dimension] = cities[keep]
            best_pano[dimension] = panos[keep]
        total_panoramas += len(winners)
        city_counts.append(local_counts)
        city_rows.append({"city_slug": city_slug, "city": city_slug.replace("__", "/"), "panoramas": len(winners), "patches": len(winners) * patches})
        print(f"prepare [{city_index:02d}/30] {city_slug}: {len(winners):,} panoramas", flush=True)
    if int(spatial.sum()) != total_panoramas * patches:
        raise ValueError("spatial profile total does not equal panoramas x 784")
    dead_spatial_dimensions = np.flatnonzero(spatial.sum(axis=1) == 0)
    np.save(targets[2], spatial)
    np.save(views / "city_dimension_patch_counts.u64.npy", np.stack(city_counts))
    pd.DataFrame(city_rows).to_csv(views / "city_index.csv", index=False)
    exemplar_rows = []
    for dimension in range(dimensions):
        order = np.argsort(-best_scores[dimension])
        for rank, slot in enumerate(order, 1):
            exemplar_rows.append({"dimension_id": dimension, "dimension": f"D{dimension:03d}", "rank": rank, "city_slug": best_city[dimension, slot], "pano_index": int(best_pano[dimension, slot]), "activation_top20": float(best_scores[dimension, slot])})
    pd.DataFrame(exemplar_rows).to_csv(targets[3], index=False)
    audit = {
        "analysis_unit": "complete four-direction panorama",
        "input_construction": "four 640x640 views -> 2560x640 -> 896x224",
        "latent_grid": [rows, columns], "spatial_profile_shape": [dimensions, patches],
        "panoramas": total_panoramas, "patches": total_panoramas * patches,
        "cities": len(prediction_dirs), "encoder_shape": list(encoder.shape), "decoder_shape": list(decoder.shape),
        "dimensions_with_zero_top1_spatial_support": dead_spatial_dimensions.tolist(),
        "zero_spatial_profile_policy": "Retain the node. Its normalized P vector and P-view similarities are zero; E and D still determine graph relations.",
        "correction_to_original_plan": "The current frozen inference unit is a four-direction panorama with a 14x56 latent map, not an independent 14x14 direction image. P therefore retains all 784 circular panorama positions.",
    }
    save_json(views / "input_audit.json", audit)
    paper = config["paths"]["paper_data_root"]
    shutil.copy2(views / "input_audit.json", paper / "input_audit.json")
    pd.DataFrame(city_rows).to_csv(paper / "panorama_city_counts.csv", index=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); print(aggregate_panorama_views(load_config(args.config), args.force))


if __name__ == "__main__": main()
