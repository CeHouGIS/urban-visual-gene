from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from sae_experiments.backbones.dinov3_feature_extractor import (
    DINOv3Backbone,
    DINOv3ExtractorConfig,
    extract_multiview_dinov3_features,
)
from sae_experiments.feature_extraction.feature_cache import load_feature_cache, save_feature_cache
from sae_experiments.models.base_sae import build_sae


class _Output:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1]


class FakeDINOv3(nn.Module):
    def __init__(self, embed_dim=12, num_blocks=4, prefix_tokens=5, patch_count=16):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=embed_dim,
            num_hidden_layers=num_blocks,
            num_prefix_tokens=prefix_tokens,
        )
        self.weight = nn.Parameter(torch.ones(1))
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.prefix_tokens = prefix_tokens
        self.patch_count = patch_count

    def forward(self, images, output_hidden_states=False):
        b = images.shape[0]
        t = self.prefix_tokens + self.patch_count
        base = torch.arange(b * t * self.embed_dim, dtype=images.dtype, device=images.device).reshape(
            b, t, self.embed_dim
        )
        hidden = tuple(base + i for i in range(self.num_blocks + 1))
        return _Output(hidden)


def _config(token_type="patch"):
    return DINOv3ExtractorConfig(
        model_name="fake",
        input_size=64,
        patch_size=16,
        feature_layer=-2,
        token_type=token_type,
        remove_prefix_tokens=True,
    )


def test_model_is_frozen():
    backbone = DINOv3Backbone(_config(), model=FakeDINOv3())
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


def test_cls_feature_shape():
    backbone = DINOv3Backbone(_config("cls"), model=FakeDINOv3())
    images = torch.zeros(2, 3, 64, 64)
    features, meta = backbone.extract_features(images, layer_index=-2, token_type="cls")
    assert features.ndim == 2
    assert features.shape == (images.shape[0], backbone.embed_dim)
    assert meta.token_type == "cls"


def test_patch_tokens_remove_prefix():
    backbone = DINOv3Backbone(_config("patch"), model=FakeDINOv3(prefix_tokens=5, patch_count=16))
    images = torch.zeros(2, 3, 64, 64)
    patch_features, meta = backbone.extract_features(images, layer_index=-2, token_type="patch")
    assert patch_features.ndim == 3
    assert patch_features.shape == (images.shape[0], 16, backbone.embed_dim)
    assert meta.prefix_tokens_removed == 5
    assert meta.patch_grid_size == (4, 4)


def test_mean_patch_feature_shape():
    backbone = DINOv3Backbone(_config("mean_patch"), model=FakeDINOv3())
    images = torch.zeros(2, 3, 64, 64)
    features, meta = backbone.extract_features(images, layer_index=-2, token_type="mean_patch")
    assert features.shape == (2, backbone.embed_dim)
    assert meta.token_type == "mean_patch"


def test_multiview_concat_shape():
    backbone = DINOv3Backbone(_config("concat_views"), model=FakeDINOv3())
    images_by_direction = {d: torch.zeros(2, 3, 64, 64) for d in (0, 90, 180, 270)}
    features, meta = extract_multiview_dinov3_features(backbone, images_by_direction)
    assert features.shape[-1] == backbone.embed_dim * 4
    assert meta["used_directions"] == [0, 90, 180, 270]


def test_multiview_missing_view_skip():
    backbone = DINOv3Backbone(_config("concat_views"), model=FakeDINOv3())
    with pytest.raises(ValueError, match="missing required view direction"):
        extract_multiview_dinov3_features(backbone, {0: torch.zeros(2, 3, 64, 64)})


def test_feature_cache_consistency(tmp_path):
    features = torch.randn(3, 5)
    metadata = pd.DataFrame({"image_id": ["a", "b", "c"], "city": ["x", "x", "y"]})
    cfg = {
        "model_name": "fake-dinov3",
        "feature_layer": -2,
        "token_type": "cls",
        "embed_dim": 5,
    }
    save_feature_cache(tmp_path, features, metadata, cfg)
    cached_features, cached_metadata, cached_cfg = load_feature_cache(tmp_path, cfg)
    assert torch.allclose(features, cached_features)
    assert cached_metadata["image_id"].tolist() == ["a", "b", "c"]
    assert cached_cfg["embed_dim"] == 5


def test_sae_dynamic_input_dim():
    features = torch.randn(4, 7)
    sae = build_sae(
        input_dim=features.shape[-1],
        config={"type": "batch_topk", "expansion_factor": 2, "latent_dim": "auto", "k": 3},
    )
    assert sae.input_dim == features.shape[-1]
    assert sae.latent_dim == 14
