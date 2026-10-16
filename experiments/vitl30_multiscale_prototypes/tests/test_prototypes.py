import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.vitl30_multiscale_prototypes.code import data
from experiments.vitl30_multiscale_prototypes.code.model import (
    TemplateBank, PrototypeAutoencoder, balanced_loss, spatial_shuffle, feature_shuffle,
)
from sae_experiments.models.base_sae import BatchTopKSAE


def test_factorized_decoder_is_exact_nonnegative_unit_template_convolution():
    torch.manual_seed(3)
    bank = TemplateBank(7, 3, 5, 2)
    coefficients = torch.rand(2, 3, 9, 9)
    templates = bank.templates()
    torch.testing.assert_close(templates.flatten(1).norm(dim=1), torch.ones(3))
    assert templates.min() >= 0
    expected = F.conv_transpose2d(coefficients, templates, padding=2)
    torch.testing.assert_close(bank(coefficients), expected, rtol=2e-5, atol=2e-6)


def test_sae_batching_preserves_independent_panorama_budget(monkeypatch):
    monkeypatch.setattr(data, 'DIM', 3)
    monkeypatch.setattr(data, 'TOKENS', 8)
    model = BatchTopKSAE(3, 5, 2)
    x = torch.randn(3, 8, 3)
    values, indices = data.sparse_encode(model, x)
    restored = torch.zeros(3, 8 * 5).scatter_(1, indices, values).reshape(3, 8, 5)
    for i in range(3):
        torch.testing.assert_close(restored[i], model.encode(x[i]))
    v, d, ptr = data.csr_pack(values.numpy(), indices.numpy(), width=5, tokens=8)
    assert len(ptr) == 3 * 8 + 1
    dense = np.zeros((24, 5), np.float32)
    dense[np.repeat(np.arange(24), np.diff(ptr)), d] = v
    np.testing.assert_allclose(dense.reshape(3, 8, 5), restored.numpy(), rtol=1e-3, atol=1e-4)


def test_spatial_split_keeps_same_block_together():
    blocks, splits = data.block_split('test', [20., 20., 20.1], [30., 30., 30.1])
    assert blocks[0] == blocks[1] and splits[0] == splits[1]


def test_shuffles_preserve_the_intended_marginals():
    x = torch.arange(5 * 7 * 6 * 6).reshape(5, 7, 6, 6).float()
    generator = torch.Generator().manual_seed(3)
    shuffled = spatial_shuffle(x, generator)
    torch.testing.assert_close(x.flatten(2).sort(2).values, shuffled.flatten(2).sort(2).values)
    # All dimensions use the same pixel permutation within each view.
    torch.testing.assert_close(shuffled[:, 1] - shuffled[:, 0], torch.full((5, 6, 6), 36.))
    shuffled = feature_shuffle(x, generator)
    torch.testing.assert_close(x.sort(0).values, shuffled.sort(0).values)
    assert not torch.equal(x, shuffled)


def test_mask_channel_and_loss_have_finite_useful_gradients():
    model = PrototypeAutoencoder(dimensions=7, scales=(3, 5), per_scale=3, rank=2, hidden=8)
    x = torch.rand(2, 7, 9, 9)
    x[x < .8] = 0
    hidden = torch.zeros(2, 1, 9, 9, dtype=torch.bool)
    hidden[:, :, 3:6, 3:6] = True
    prediction, maps = model(x, hidden)
    loss, metrics = balanced_loss(prediction, x, maps, hidden)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.banks[0].channels.grad.abs().sum() > 0
    assert maps.shape == (2, 6, 9, 9)
    assert metrics['masked_positive_mse'] > 0


@pytest.mark.parametrize('density', [0., .02, 1.])
@pytest.mark.parametrize('masked', [False, True])
def test_memory_safe_loss_matches_selected_element_objective_and_gradients(density, masked):
    torch.manual_seed(19)
    target = torch.rand(3, 11, 7, 7)
    target *= (torch.rand_like(target) < density)
    hidden = torch.rand(3, 1, 7, 7) < .2 if masked else None
    prediction = torch.rand_like(target, requires_grad=True)
    coefficients = torch.rand(3, 6, 7, 7, requires_grad=True)
    actual, _ = balanced_loss(prediction, target, coefficients, hidden)
    error = (prediction - target).square()
    positive = target > 0
    expected = error[positive].sum() / positive.sum().clamp_min(1)
    expected += 32 * error[~positive].sum() / (~positive).sum().clamp_min(1)
    expected += .01 * coefficients.mean()
    if hidden is not None:
        selected = positive & hidden
        expected += error[selected].sum() / selected.sum().clamp_min(1)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (prediction, coefficients), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (prediction, coefficients))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
