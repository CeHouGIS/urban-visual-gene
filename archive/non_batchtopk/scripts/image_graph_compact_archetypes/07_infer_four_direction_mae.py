#!/usr/bin/env python3
"""Run DINOv3 + the frozen Feature-MAE for four views of one panorama."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from scripts.image_graph_archetypes.utils import assert_safe_affinity, read_original
from scripts.multicity.config import MODEL_DIR
from scripts.multicity.extract_dinov3_features import MEAN, STD
from scripts.multicity.dinov3_vit_backport import load_dinov3_vit
from scripts.multicity.train_feature_mae import FeatureMAE


HEADINGS = (0, 90, 180, 270)
DEFAULT_CITY = "UnitedStates/NewYorkCity"
DEFAULT_PANOID = "BFlilbUSNYggO3YcnRY3kg"
DEFAULT_SOURCE_ROOT = Path("/nas_data_24T/GSV/packages_main/cities")
DEFAULT_DATA_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_n30x12800_qc"
)
DEFAULT_GRAPH_ROOT = Path("paper/data/image_graph_compact_archetypes")
DEFAULT_OUTPUT = DEFAULT_GRAPH_ROOT / "four_direction_area"


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def locate_panorama_rows(
    city: str,
    panoid: str,
    source_root: Path,
    graph_root: Path,
) -> pd.DataFrame:
    """Locate the source shard via any cached direction, then recover all four."""
    cached = pd.read_parquet(graph_root / "image_graph_metadata.parquet")
    match = cached[
        cached["panorama_id"].astype(str).eq(panoid)
        & cached["city"].astype(str).eq(city)
    ]
    sidecars: list[Path] = []
    if not match.empty:
        sidecars.extend(
            Path(path).with_suffix(".parquet")
            for path in match["image_path"].astype(str).drop_duplicates()
        )
    if not sidecars:
        sidecars = sorted((source_root / city).glob("shard-*.parquet"))

    found = []
    for sidecar in sidecars:
        if not sidecar.is_file():
            continue
        frame = pd.read_parquet(sidecar)
        part = frame[
            frame["panoid"].astype(str).eq(panoid)
            & frame["heading"].isin(HEADINGS)
        ].copy()
        if not part.empty:
            part["tar_path"] = str(sidecar.with_suffix(".tar"))
            found.append(part)
        if found and pd.concat(found)["heading"].nunique() == 4:
            break
    if not found:
        raise ValueError(f"panorama {panoid!r} was not found for {city}")
    rows = pd.concat(found, ignore_index=True)
    rows = rows.drop_duplicates("heading", keep="first").sort_values("heading")
    observed = tuple(rows["heading"].astype(int))
    if observed != HEADINGS:
        raise ValueError(f"expected headings {HEADINGS}, found {observed}")

    output = pd.DataFrame(
        {
            "direction_id": [f"H{heading:03d}" for heading in HEADINGS],
            "panorama_id": rows["panoid"].astype(str).to_numpy(),
            "city": city,
            "heading": rows["heading"].astype(int).to_numpy(),
            "direction_index": np.arange(4, dtype=np.int8),
            "image_path": rows["tar_path"].astype(str).to_numpy(),
            "jpg_offset": rows["jpg_offset"].astype(np.int64).to_numpy(),
            "jpg_size": rows["jpg_size"].astype(np.int64).to_numpy(),
            "lat": rows["lat"].astype(float).to_numpy(),
            "lon": rows["lon"].astype(float).to_numpy(),
            "year": rows["year"].astype(int).to_numpy(),
            "month": rows["month"].astype(int).to_numpy(),
            "sample_key": rows["sample_key"].astype(str).to_numpy(),
        }
    )
    cached_keys = {
        (str(row.panorama_id), int(row.heading)): int(row.image_id)
        for row in cached.itertuples()
        if str(row.panorama_id) == panoid and str(row.city) == city
    }
    output["was_in_analysis_cache"] = [
        (panoid, heading) in cached_keys for heading in HEADINGS
    ]
    output["cached_image_id"] = [
        cached_keys.get((panoid, heading), pd.NA) for heading in HEADINGS
    ]
    return output


def load_feature_mae(checkpoint_path: Path) -> tuple[FeatureMAE, dict]:
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
    return model.eval().requires_grad_(False).to("cuda"), config


@torch.inference_mode()
def infer(
    metadata: pd.DataFrame,
    dino_model: torch.nn.Module,
    mae_model: FeatureMAE,
) -> tuple[np.ndarray, np.ndarray]:
    transform = v2.Compose(
        [
            v2.Resize(
                (224, 224), interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )
    pixels = torch.stack([transform(read_original(row)) for _, row in metadata.iterrows()])
    pixels = pixels.to("cuda", non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sequence = dino_model(pixel_values=pixels).last_hidden_state
    start = 1 + int(dino_model.config.num_register_tokens)
    tokens = sequence[:, start:].float()
    if tokens.shape != (4, 196, 768):
        raise ValueError(f"unexpected DINO patch-token shape {tuple(tokens.shape)}")
    tokens = torch.nn.functional.normalize(tokens, dim=-1)
    # The main cache stores normalized tokens as float16 before MAE inference.
    tokens = tokens.half().float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent = mae_model.encode_full(tokens)
    return (
        tokens.cpu().numpy().astype(np.float16),
        latent.float().cpu().numpy().astype(np.float16),
    )


def run(args: argparse.Namespace) -> dict:
    assert_safe_affinity()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for four-direction completion")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = locate_panorama_rows(
        args.city, args.panoid, args.source_root, args.graph_root
    )
    checkpoint_path = args.data_root / "mae" / "model_best.pt"
    dino_model = load_dinov3_vit(args.model_dir)
    dino_model = dino_model.eval().requires_grad_(False).to("cuda")
    mae_model, config = load_feature_mae(checkpoint_path)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    tokens, latent = infer(metadata, dino_model, mae_model)
    winners = latent.astype(np.float32).argmax(axis=-1).astype(np.uint16)
    raw_activation = np.sort(latent.astype(np.float32), axis=1)[:, -20:].mean(axis=1)

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
    metadata.to_csv(args.output / "four_direction_metadata.csv", index=False)
    np.save(args.output / "dino_patch_tokens.npy", tokens)
    np.save(args.output / "feature_mae_latent.npy", latent)
    np.save(args.output / "top1_dimensions.npy", winners)
    np.save(args.output / "activation_raw.npy", raw_activation.astype(np.float32))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city": args.city,
        "panorama_id": args.panoid,
        "coordinates": [float(metadata.iloc[0]["lat"]), float(metadata.iloc[0]["lon"])],
        "headings": metadata["heading"].astype(int).tolist(),
        "directions_previously_cached": int(metadata["was_in_analysis_cache"].sum()),
        "directions_completed_from_source": int((~metadata["was_in_analysis_cache"]).sum()),
        "all_four_directions_recomputed": True,
        "dino_model": str(args.model_dir),
        "feature_mae_checkpoint": str(checkpoint_path),
        "feature_mae_checkpoint_sha256": sha256(checkpoint_path),
        "feature_mae_config": config,
        "dino_token_shape": list(tokens.shape),
        "feature_mae_latent_shape": list(latent.shape),
        "top1_shape": list(winners.shape),
        "activation_score": "mean of strongest 20 of 196 patch responses per dimension",
        "cached_direction_top1_match_fraction": [
            None if np.isnan(value) else value for value in match_fraction
        ],
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
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--panoid", default=DEFAULT_PANOID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
