from __future__ import annotations

import pytest
import torch

from sae_experiments.models.panorama_context import mix_panorama_context


def test_panorama_context_preserves_shape_and_normalizes_tokens():
    x = torch.randn(2, 4, 28 * 28, 8)
    y = mix_panorama_context(x, grid_size=28, context_weight=0.25)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=-1), torch.ones_like(y[..., 0]), atol=1e-5)


def test_panorama_context_wraps_last_view_to_first_view():
    x = torch.zeros(1, 4, 28 * 28, 4)
    x[0, 3, 27, 0] = 1.0
    y = mix_panorama_context(x, grid_size=28, context_weight=1.0)
    assert y[0, 0, 0, 0] > 0


def test_panorama_context_uses_valid_neighbor_mask():
    x = torch.randn(1, 4, 28 * 28, 8)
    valid = torch.ones(1, 4, 28 * 28, dtype=torch.bool)
    valid[0, 1, 0] = False
    y = mix_panorama_context(x, grid_size=28, context_weight=0.5, valid_mask=valid)
    assert torch.isfinite(y).all()


def test_panorama_context_rejects_non_four_view_input():
    with pytest.raises(ValueError, match="expected features"):
        mix_panorama_context(torch.randn(1, 3, 28 * 28, 8), grid_size=28)
