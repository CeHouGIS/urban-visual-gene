from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


def write_extraction_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def read_extraction_config(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_cache_config(cache_config: dict, expected: dict, features: torch.Tensor | None = None) -> None:
    for key in ("model_name", "feature_layer", "token_type"):
        if cache_config.get(key) != expected.get(key):
            raise ValueError(f"cache {key}={cache_config.get(key)!r} != expected {expected.get(key)!r}")
    if features is not None and "embed_dim" in cache_config:
        actual = int(features.shape[-1])
        if int(cache_config["embed_dim"]) != actual:
            raise ValueError(f"cache embed_dim={cache_config['embed_dim']} != features.shape[-1]={actual}")


def save_feature_cache(
    root: Path,
    features: torch.Tensor,
    metadata: pd.DataFrame,
    extraction_config: dict,
    *,
    shard_id: int | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    feature_name = "features.pt" if shard_id is None else f"features_shard_{shard_id:03d}.pt"
    feature_path = root / feature_name
    torch.save(features.cpu(), feature_path)
    metadata.to_parquet(root / "metadata.parquet", index=False)
    write_extraction_config(root / "extraction_config.yaml", extraction_config)
    return feature_path


def load_feature_cache(root: Path, expected_config: dict | None = None):
    cfg = read_extraction_config(root / "extraction_config.yaml")
    feature_files = sorted(root.glob("features_shard_*.pt")) or [root / "features.pt"]
    features = torch.cat([torch.load(p, map_location="cpu") for p in feature_files], dim=0)
    if expected_config is not None:
        validate_cache_config(cfg, expected_config, features)
    metadata = pd.read_parquet(root / "metadata.parquet")
    return features, metadata, cfg

