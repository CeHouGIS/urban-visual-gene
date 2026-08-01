from __future__ import annotations

from dataclasses import dataclass

import torch


PREFIX_TOKEN_NAMES = ("cls", "register", "prefix")


@dataclass(frozen=True)
class TokenMetadata:
    token_type: str
    prefix_tokens_removed: int
    patch_grid_size: tuple[int, int] | None = None
    patch_indices: torch.Tensor | None = None


def resolve_layer_index(num_blocks: int, layer_index: int) -> int:
    """Resolve negative layer indices against the actual model depth."""
    resolved = num_blocks + layer_index if layer_index < 0 else layer_index
    if resolved < 0 or resolved >= num_blocks:
        raise IndexError(f"layer_index={layer_index} resolves to {resolved}, outside 0..{num_blocks - 1}")
    return resolved


def infer_prefix_tokens(model, total_tokens: int, expected_patch_count: int | None = None) -> int:
    """Infer how many CLS/register/prefix tokens precede patch tokens.

    Prefer explicit model/config metadata. Fall back to total_tokens - expected_patch_count
    when image size and patch size are known.
    """
    for obj in (model, getattr(model, "config", None)):
        if obj is None:
            continue
        for name in ("num_prefix_tokens", "prefix_tokens", "num_prefix"):
            val = getattr(obj, name, None)
            if val is not None:
                return int(val)
        nreg = getattr(obj, "num_register_tokens", None)
        if nreg is not None:
            return 1 + int(nreg)
    if expected_patch_count is not None:
        prefix = total_tokens - expected_patch_count
        if prefix < 0:
            raise ValueError(
                f"expected_patch_count={expected_patch_count} exceeds total_tokens={total_tokens}"
            )
        return int(prefix)
    return 1


def patch_grid_from_config(config) -> tuple[int, int] | None:
    input_size = getattr(config, "input_size", None)
    patch_size = getattr(config, "patch_size", None)
    if input_size is None:
        input_size = getattr(config, "image_size", None)
    if patch_size is None and hasattr(config, "model"):
        patch_size = getattr(config.model, "patch_size", None)
    if input_size is None or patch_size is None:
        return None
    if isinstance(input_size, (list, tuple)):
        h, w = int(input_size[0]), int(input_size[1])
    else:
        h = w = int(input_size)
    if isinstance(patch_size, (list, tuple)):
        ph, pw = int(patch_size[0]), int(patch_size[1])
    else:
        ph = pw = int(patch_size)
    return h // ph, w // pw


def expected_patch_count(config) -> int | None:
    grid = patch_grid_from_config(config)
    if grid is None:
        return None
    return grid[0] * grid[1]


def remove_prefix_tokens(tokens: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"expected token tensor [B,T,D], got shape {tuple(tokens.shape)}")
    if prefix_tokens < 0 or prefix_tokens > tokens.shape[1]:
        raise ValueError(f"invalid prefix token count {prefix_tokens} for T={tokens.shape[1]}")
    return tokens[:, prefix_tokens:, :]

