from __future__ import annotations

import torch
import torch.nn.functional as F


PANORAMA_DIRECTIONS = 4
PANORAMA_CONTEXT_WEIGHT = 0.25


def mix_panorama_context(
    features: torch.Tensor,
    *,
    grid_size: int = 28,
    context_weight: float = PANORAMA_CONTEXT_WEIGHT,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mix each patch with its 8-neighborhood on a horizontally circular panorama.

    ``features`` has shape ``[..., 4, grid_size**2, D]``. Directions are laid
    out left-to-right in yaw order, so the last patch column wraps to the first.
    ``valid_mask`` optionally marks patches that may contribute as neighbors.
    """
    if not 0.0 <= context_weight <= 1.0:
        raise ValueError(f"context_weight must be in [0, 1], got {context_weight}")
    expected_patches = grid_size * grid_size
    if features.ndim < 3 or tuple(features.shape[-3:-1]) != (PANORAMA_DIRECTIONS, expected_patches):
        raise ValueError(
            "expected features [..., 4, grid_size**2, D], "
            f"got {tuple(features.shape)} for grid_size={grid_size}"
        )

    leading_shape = features.shape[:-3]
    dim = features.shape[-1]
    n = 1
    for size in leading_shape:
        n *= size
    views = features.reshape(n, PANORAMA_DIRECTIONS, grid_size, grid_size, dim)
    panorama = views.permute(0, 4, 2, 1, 3).reshape(n, dim, grid_size, PANORAMA_DIRECTIONS * grid_size)

    if valid_mask is None:
        mask = torch.ones(
            (n, 1, grid_size, PANORAMA_DIRECTIONS * grid_size),
            dtype=panorama.dtype,
            device=panorama.device,
        )
    else:
        if tuple(valid_mask.shape) != tuple(features.shape[:-1]):
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                f"features {tuple(features.shape[:-1])}"
            )
        mask = valid_mask.reshape(n, PANORAMA_DIRECTIONS, grid_size, grid_size)
        mask = mask.permute(0, 1, 2, 3).reshape(n, 1, grid_size, PANORAMA_DIRECTIONS * grid_size)
        mask = mask.to(device=panorama.device, dtype=panorama.dtype)

    def padded(value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (1, 1, 0, 0), mode="circular")
        # The panorama closes only around yaw. The top and bottom image rows
        # have fewer neighbors, so use zero padding there and count neighbors
        # explicitly below instead of duplicating boundary patches.
        return F.pad(value, (0, 0, 1, 1), mode="constant", value=0.0)

    neighborhood_sum = F.avg_pool2d(padded(panorama * mask), kernel_size=3, stride=1) * 9.0
    neighborhood_count = F.avg_pool2d(padded(mask), kernel_size=3, stride=1) * 9.0
    neighbor_sum = neighborhood_sum - panorama * mask
    neighbor_count = neighborhood_count - mask
    neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0)
    mixed = (1.0 - context_weight) * panorama + context_weight * neighbor_mean
    mixed = F.normalize(mixed, dim=1)

    result = mixed.reshape(n, dim, grid_size, PANORAMA_DIRECTIONS, grid_size)
    result = result.permute(0, 3, 2, 4, 1).reshape(*leading_shape, PANORAMA_DIRECTIONS, expected_patches, dim)
    return result
