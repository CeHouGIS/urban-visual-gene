#!/usr/bin/env python3
"""Complete four Feature-MAE directions for ten representative panoramas."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.image_graph_archetypes.utils import assert_safe_affinity
from scripts.multicity.config import MODEL_DIR
from scripts.multicity.dinov3_vit_backport import load_dinov3_vit


DEFAULT_DATA_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)
DEFAULT_GRAPH_ROOT = Path("paper/data/image_graph_compact_archetypes")
DEFAULT_OUTPUT = DEFAULT_GRAPH_ROOT / "multi_area_four_directions"
DEFAULT_SOURCE_ROOT = Path("/nas_data_24T/GSV/packages_main/cities")

# One representative for every A01--A08, then two additional regions chosen
# for geographic breadth. All ten cities and panorama IDs are unique.
AREA_REQUESTS = (
    ("M01", "A01", 1, "UnitedStates/NewYorkCity", "BFlilbUSNYggO3YcnRY3kg"),
    ("M02", "A02", 1, "Indonesia/Jakarta", "Ow2PN8qSvoYmbOVF2BS6Yg"),
    ("M03", "A03", 1, "Singapore/Singapore", "yyoWUgFTg_X_I71bkwbjXQ"),
    ("M04", "A04", 1, "Japan/Osaka", "1FuSNLk-chbGRLtqokanmg"),
    ("M05", "A05", 1, "Philippines/Manila", "yR4HaQ5uu8VvUqcHLm4yFA"),
    ("M06", "A06", 2, "Thailand/Bangkok", "ukeJHe-PyDE7ualz5td5VA"),
    ("M07", "A07", 1, "Mexico/MexicoCity", "CyGzAR_d_Mi7OujGYCmDmQ"),
    ("M08", "A08", 1, "Colombia/Bogota", "fooe9wu9hS1VqmJGXcpbcw"),
    ("M09", "A01", 2, "Turkey/Istanbul", "B5LhM2iZOdu-wn8oNo_rGg"),
    ("M10", "A08", 8, "France/Paris", "4nvSfB0WY-sdJMKBYWBO3A"),
)


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-area four-direction completion")
    single = importlib.import_module(
        "scripts.image_graph_compact_archetypes.07_infer_four_direction_mae"
    )
    checkpoint_path = args.data_root / "mae" / "model_best.pt"
    dino_model = load_dinov3_vit(args.model_dir).eval().requires_grad_(False).to("cuda")
    mae_model, config = single.load_feature_mae(checkpoint_path)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    metadata_blocks = []
    token_blocks = []
    latent_blocks = []
    for area_id, source_archetype, rank, city, panoid in AREA_REQUESTS:
        metadata = single.locate_panorama_rows(
            city, panoid, args.source_root, args.graph_root
        )
        metadata.insert(0, "area_id", area_id)
        metadata.insert(1, "source_archetype_id", source_archetype)
        metadata.insert(2, "source_representative_rank", rank)
        tokens, latent = single.infer(metadata, dino_model, mae_model)
        metadata_blocks.append(metadata)
        token_blocks.append(tokens)
        latent_blocks.append(latent)
        print(f"completed {area_id}: {city} / {panoid}", flush=True)

    metadata = pd.concat(metadata_blocks, ignore_index=True)
    tokens = np.concatenate(token_blocks, axis=0)
    latent = np.concatenate(latent_blocks, axis=0)
    winners = latent.astype(np.float32).argmax(axis=-1).astype(np.uint16)
    raw_activation = np.sort(latent.astype(np.float32), axis=1)[:, -20:].mean(axis=1)
    if tokens.shape != (40, 196, 768) or latent.shape != (40, 196, 512):
        raise ValueError("unexpected multi-area inference shape")

    cached_top = np.load(
        args.data_root / "mae" / "hierarchy_32_64" / "top1_dimensions.npy",
        mmap_mode="r",
    )
    match_fraction = []
    for index, row in metadata.iterrows():
        cached_id = row["cached_image_id"]
        if pd.isna(cached_id):
            match_fraction.append(np.nan)
        else:
            reference = np.asarray(cached_top[int(cached_id)], dtype=np.uint16)
            match_fraction.append(float(np.mean(reference == winners[index])))
    metadata["cached_top1_match_fraction"] = match_fraction
    metadata.insert(0, "result_row", np.arange(len(metadata), dtype=np.int32))

    args.output.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output / "multi_area_direction_metadata.csv", index=False)
    np.save(args.output / "dino_patch_tokens.npy", tokens)
    np.save(args.output / "feature_mae_latent.npy", latent)
    np.save(args.output / "top1_dimensions.npy", winners)
    np.save(args.output / "activation_raw.npy", raw_activation.astype(np.float32))

    finite_matches = np.asarray(match_fraction, dtype=np.float64)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "areas": len(AREA_REQUESTS),
        "cities": metadata["city"].nunique(),
        "images": len(metadata),
        "headings_per_area": [0, 90, 180, 270],
        "source_archetypes_covered": sorted(metadata["source_archetype_id"].unique()),
        "directions_previously_cached": int(metadata["was_in_analysis_cache"].sum()),
        "directions_completed_from_source": int((~metadata["was_in_analysis_cache"]).sum()),
        "all_directions_recomputed": True,
        "cached_top1_match_mean": float(np.nanmean(finite_matches)),
        "cached_top1_match_minimum": float(np.nanmin(finite_matches)),
        "dino_token_shape": list(tokens.shape),
        "feature_mae_latent_shape": list(latent.shape),
        "feature_mae_checkpoint": str(checkpoint_path),
        "feature_mae_config": config,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
