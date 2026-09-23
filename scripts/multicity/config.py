"""Fixed configuration for the 30-city DINOv3-B/16 experiment."""
from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/nas_data_24T/GSV/packages_main/cities")
OUTPUT_ROOT = Path(
    "/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/"
    "global_sae_topk32_w512_n30x10000"
)
MODEL_DIR = Path(
    "/workplace/models/modelscope/models/"
    "facebook--dinov3-vitb16-pretrain-lvd1689m/snapshots/master"
)

PANOS_PER_CITY = 10_000
HEADINGS = (0, 90, 180, 270)
FEATURES_PER_HEADING = 768
FEATURE_DIM = len(HEADINGS) * FEATURES_PER_HEADING
WIDTH = 512
TOPK = 32
SEED = 42

CITIES = (
    "Argentina/BuenosAires",
    "Australia/Sydney",
    "Austria/Vienna",
    "Bangladesh/Dhaka",
    "Brazil/SaoPaulo",
    "Canada/Toronto",
    "China/HongKong",
    "Colombia/Bogota",
    "France/Paris",
    "India/Mumbai",
    "India/NewDelhi",
    "Indonesia/Jakarta",
    "Japan/Osaka",
    "Malaysia/KualaLumpur",
    "Mexico/MexicoCity",
    "Netherlands/Amsterdam",
    "Nigeria/Lagos",
    "Peru/Lima",
    "Philippines/Manila",
    "Russia/Moscow",
    "Singapore/Singapore",
    "SouthAfrica/CapeTown",
    "SouthAfrica/Johannesburg",
    "SouthKorea/Seoul",
    "Taiwan/Taipei",
    "Thailand/Bangkok",
    "Turkey/Istanbul",
    "UnitedKingdom/London",
    "UnitedStates/LosAngeles",
    "UnitedStates/NewYorkCity",
)


def city_slug(city_key: str) -> str:
    """Return a filesystem-safe, unambiguous city identifier."""
    return city_key.replace("/", "__")


def manifest_path(city_key: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / "manifests" / f"{city_slug(city_key)}.parquet"


def feature_path(city_key: str, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / "features" / city_slug(city_key) / "pano_features.f16.npy"
