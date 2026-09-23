"""Configuration for patch-level urban visual gene discovery."""
from __future__ import annotations

from pathlib import Path

from scripts.multicity.config import CITIES, MODEL_DIR, SEED, city_slug

SOURCE_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "global_sae_topk32_w512_n30x10000"
)
OUTPUT_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "patch_sae_topk32_w512_n30x12800"
)
IMAGES_PER_CITY = 12_800
PATCH_GRID = 14
PATCHES_PER_IMAGE = PATCH_GRID * PATCH_GRID
INPUT_DIM = 768
WIDTH = 512
TOPK = 32


def image_manifest_path(city_key: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / "manifests" / f"{city_slug(city_key)}.parquet"


def token_path(city_key: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / "patch_tokens" / city_slug(city_key) / "tokens.f16.npy"
