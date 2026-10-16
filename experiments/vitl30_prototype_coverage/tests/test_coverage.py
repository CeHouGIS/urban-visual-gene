import numpy as np
import pandas as pd
import pytest
import torch

from experiments.vitl30_prototype_coverage.code.protocol import spatial_split, panel, csr_training_moments, rms
from experiments.vitl30_prototype_coverage.code.trial import refit_responses, reconstruction_curve, runtime
from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder


def test_three_way_split_is_block_disjoint_city_balanced_and_order_independent():
    frame = pd.DataFrame([{'city': city, 'panoid': f'{i}-{j}', 'spatial_block': f'{city}-{i}',
                           'prototype_split': 'train'}
                          for city in ['a', 'b'] for i in range(100) for j in range(2)])
    split = spatial_split(frame)
    reordered = spatial_split(frame.sample(frac=1., random_state=1)).sort_index()
    pd.testing.assert_series_equal(split.prototype_split, reordered.prototype_split)
    assert split.groupby('spatial_block').prototype_split.nunique().max() == 1
    assert split.groupby(['city', 'prototype_split']).size().to_dict() == {
        ('a', 'train'): 140, ('a', 'dev'): 30, ('a', 'test'): 30,
        ('b', 'train'): 140, ('b', 'dev'): 30, ('b', 'test'): 30}
    ids = panel(split, per_city=2)
    assert len(ids) == 4 and (split.loc[ids].prototype_split == 'dev').all()
    with pytest.raises(ValueError, match='sealed'):
        panel(split, split='test')


def test_normalization_excludes_dev_and_test_values_and_unseen_dimensions():
    # Three panoramas, two patch rows each. Only the first panorama is training.
    ptr = np.array([0, 1, 2, 3, 3, 4, 5])
    indices = np.array([0, 1, 0, 0, 2])
    values = np.array([2., 4., 1000., 9000., 12000.])
    count, square = csr_training_moments(values, indices, ptr, [True, False, False], dimensions=3, tokens=2)
    np.testing.assert_array_equal(count, [1, 1, 0])
    np.testing.assert_allclose(rms(count, square), [2., 4., 1.])


def test_four_direction_rule_does_not_confuse_any_direction_with_panorama_pass():
    score = np.array([[.1, .1, .1, .1], [.1, .1, .1, .95], [np.nan, .1, .1, .1]])
    result = reconstruction_curve(score, thresholds=[.8])[0]
    assert result['all_four_fraction'] == pytest.approx(1 / 3)
    assert result['any_direction_fraction'] == 1.
    assert result['direction_fraction'] == pytest.approx(10 / 12)


def test_response_refit_keeps_best_iterate_and_frozen_dictionary():
    # Tiny synthetic optimization tests CPU math only; real dataset jobs reject CPU.
    torch.set_num_threads(1)
    torch.manual_seed(6)
    model = PrototypeAutoencoder(dimensions=4, scales=(3,), per_scale=2, rank=1, hidden=4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    target = torch.rand(1, 4, 5, 5)
    initial, best, _, maps = refit_responses(model, target, steps=20, lr=.1)
    assert torch.all(best <= initial)
    assert torch.all(maps >= 0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
        assert not value.requires_grad
    with pytest.raises(ValueError, match='CUDA'):
        runtime('cpu')
