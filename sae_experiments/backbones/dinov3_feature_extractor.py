from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .base_backbone import BaseVisualBackbone
from .dinov3_loader import (
    infer_dinov3_embed_dim,
    infer_dinov3_num_blocks,
    load_dinov3_model,
)
from .token_utils import (
    TokenMetadata,
    expected_patch_count,
    infer_prefix_tokens,
    patch_grid_from_config,
    remove_prefix_tokens,
    resolve_layer_index,
)


@dataclass
class DINOv3ExtractorConfig:
    model_name: str | None = None
    checkpoint_path: str | None = None
    input_size: int = 224
    patch_size: int = 16
    feature_layer: int = -2
    token_type: str = "patch"
    remove_prefix_tokens: bool = True
    use_register_tokens: bool = False
    view_directions: tuple[int, ...] = (0, 90, 180, 270)
    missing_view_strategy: str = "skip"


def _output_tokens(output, resolved_layer: int, num_blocks: int) -> torch.Tensor:
    hidden = getattr(output, "hidden_states", None)
    if hidden is None and isinstance(output, dict):
        hidden = output.get("hidden_states")
    if hidden is not None:
        # HF hidden_states often includes embeddings at index 0, so block i maps to i+1.
        idx = resolved_layer + 1 if len(hidden) == num_blocks + 1 else resolved_layer
        return hidden[idx]
    if resolved_layer != num_blocks - 1:
        raise ValueError("model output does not expose hidden_states; only layer -1 is available")
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("unsupported DINOv3 forward output format")


def extract_dinov3_layer_features(
    model: nn.Module,
    images: torch.Tensor,
    layer_index: int,
    token_type: str,
    *,
    remove_prefix: bool = True,
    config=None,
) -> tuple[torch.Tensor, TokenMetadata]:
    """Extract CLS, mean-patch, or patch features from a DINOv3-like model."""
    num_blocks = infer_dinov3_num_blocks(model)
    resolved = resolve_layer_index(num_blocks, layer_index)
    output = model(images, output_hidden_states=True)
    tokens = _output_tokens(output, resolved, num_blocks)
    if tokens.ndim != 3:
        raise ValueError(f"expected DINOv3 tokens [B,T,D], got {tuple(tokens.shape)}")

    exp_patches = expected_patch_count(config) if config is not None else None
    prefix = infer_prefix_tokens(model, tokens.shape[1], exp_patches) if remove_prefix else 0
    patch_tokens = remove_prefix_tokens(tokens, prefix) if remove_prefix else tokens
    grid = patch_grid_from_config(config) if config is not None else None

    if token_type == "cls":
        return tokens[:, 0, :], TokenMetadata("cls", 0, None, None)
    if token_type == "mean_patch":
        return patch_tokens.mean(dim=1), TokenMetadata("mean_patch", prefix, grid, None)
    if token_type == "patch":
        idx = torch.arange(patch_tokens.shape[1], device=patch_tokens.device)
        return patch_tokens, TokenMetadata("patch", prefix, grid, idx)
    raise ValueError(f"unsupported token_type={token_type!r}")


class DINOv3Backbone(BaseVisualBackbone):
    """Frozen DINOv3 adapter used by feature extraction, not by SAE trainers."""

    def __init__(self, config: DINOv3ExtractorConfig, model: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.model = model if model is not None else load_dinov3_model(
            model_name=config.model_name,
            checkpoint_path=config.checkpoint_path,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return infer_dinov3_embed_dim(self.model)

    @property
    def num_blocks(self) -> int:
        return infer_dinov3_num_blocks(self.model)

    @torch.no_grad()
    def extract_features(self, images, layer_index: int | None = None, token_type: str | None = None):
        return extract_dinov3_layer_features(
            self.model,
            images,
            self.config.feature_layer if layer_index is None else layer_index,
            self.config.token_type if token_type is None else token_type,
            remove_prefix=self.config.remove_prefix_tokens,
            config=self.config,
        )


class DINOv3FeatureExtractor(nn.Module):
    """Thin extractor wrapper so SAE code only consumes extracted activations."""

    def __init__(
        self,
        model: nn.Module,
        feature_layer: int,
        token_type: str,
        remove_prefix_tokens: bool = True,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.feature_layer = feature_layer
        self.token_type = token_type
        self.remove_prefix_tokens = remove_prefix_tokens
        self.config = config
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, images):
        return extract_dinov3_layer_features(
            self.model,
            images,
            self.feature_layer,
            self.token_type,
            remove_prefix=self.remove_prefix_tokens,
            config=self.config,
        )


@torch.no_grad()
def extract_multiview_dinov3_features(
    backbone: DINOv3Backbone,
    images_by_direction: dict[int, torch.Tensor],
    view_directions: list[int] | tuple[int, ...] | None = None,
    missing_view_strategy: str | None = None,
) -> tuple[torch.Tensor, dict]:
    """Extract and concatenate one global DINOv3 feature per street-view direction."""
    directions = tuple(view_directions or backbone.config.view_directions)
    strategy = missing_view_strategy or backbone.config.missing_view_strategy
    chunks = []
    used = []
    missing = []
    for direction in directions:
        images = images_by_direction.get(direction)
        if images is None:
            missing.append(direction)
            if strategy == "skip":
                raise ValueError(f"missing required view direction {direction}")
            if strategy == "zero":
                if not chunks:
                    raise ValueError("zero missing-view strategy needs at least one valid view first")
                chunks.append(torch.zeros_like(chunks[-1]))
                continue
            raise ValueError(f"unsupported missing_view_strategy={strategy!r}")
        feat, _meta = backbone.extract_features(images, token_type="cls")
        chunks.append(feat)
        used.append(direction)
    return torch.cat(chunks, dim=-1), {"used_directions": used, "missing_directions": missing}

