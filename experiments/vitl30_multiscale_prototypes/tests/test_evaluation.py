from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import pytest

from experiments.vitl30_multiscale_prototypes.code import evaluate as ev
from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder, TemplateBank


def test_exact_statistics_sum_before_ratio_and_zero_baseline():
    target = torch.tensor([[[[2., 0.]]], [[[1., 0.]]]])
    prediction = torch.tensor([[[[1., 1.]]], [[[0., 2.]]]])
    stats = ev.sufficient_statistics(prediction, target).numpy()
    derived = ev.derived_statistics(stats.sum(0))
    assert derived['nmse'] == 7 / 5
    assert derived['positive_mse'] == 1
    assert derived['zero_mse'] == 2.5
    assert derived['balanced_error'] == 81
    assert derived['zero_prediction_balanced_error'] == 2.5


def test_permutation_tail_includes_plus_one_and_ties():
    result = ev.permutation_test(1., [2., 3., 1.])
    assert result['p_monte_carlo'] == .5
    assert result['p_resolution'] == .25


def test_native_footprints_clip_at_image_boundaries():
    assert ev.patch_box(0, 27, 7) == [24, 0, 28, 4]
    model = PrototypeAutoencoder(dimensions=3, scales=(7,), per_scale=2, rank=2, hidden=4)
    target = torch.rand(1, 3, 28, 28)
    heat, record = ev.exemplar_evidence(model, target, 0, 0, 27)
    pred, maps = model(target)
    coefficient = torch.zeros_like(maps)
    coefficient[0, 0, 0, 27] = maps[0, 0, 0, 27]
    component = model.banks[0](coefficient)
    delta = ((target - pred + component).square() - (target - pred).square()).sum()
    np.testing.assert_allclose(record['removing_response_delta_sse'], delta.detach().numpy(), rtol=2e-4, atol=1e-6)
    assert heat[4:, :].sum() == 0
    assert heat[:, :24].sum() == 0


def test_missing_original_is_distinguished_from_output_failure(tmp_path):
    with pytest.raises(ev.SourceImageError):
        ev.native_evidence(tmp_path / 'missing.jpg', np.ones((28, 28)), [0, 0, 3, 3], tmp_path / 'out')


def test_local_peak_nms_keeps_spatially_separate_candidates_and_suppresses_plateaus():
    maps = torch.zeros(1, 2, 9, 9)
    maps[0, 0, 2, 2] = 4
    maps[0, 0, 2, 3] = 3
    maps[0, 0, 6, 6] = 2.5
    maps[0, 1] = 1
    values, positions = ev.local_peaks(maps, scales=[3], per_scale=2)
    assert values[0, 0].tolist() == [4, 2.5]
    assert positions[0, 0].tolist() == [20, 60]
    # A constant plateau yields a single deterministic peak, not every pixel.
    assert (values[0, 1] > 0).sum() == 1


def test_seed_cosine_factorization_matches_materialized_templates():
    a, b = TemplateBank(5, 3, 7, 2), TemplateBank(5, 3, 7, 2)
    channels = torch.einsum('ird,jsd->ijrs', a.channels, b.channels)
    spatial = torch.einsum('iruv,jsuv->ijrs', a.layout, b.layout)
    torch.testing.assert_close((channels * spatial).sum((2, 3)),
                               a.templates().flatten(1) @ b.templates().flatten(1).T)


def test_instance_selection_never_duplicates_panorama_and_prefers_city_diversity():
    frame = pd.DataFrame({'city': ['a', 'a', 'a', 'b', 'c']})
    rows = [(20, 0, 0, 1, 1), (19, 0, 1, 1, 1), (18, 1, 0, 1, 1),
            (17, 2, 0, 1, 1), (16, 3, 0, 1, 1), (15, 4, 0, 1, 1)]
    selected = ev.select_diverse(rows, frame, count=3, max_per_city=1)
    assert [r[1] for r in selected] == [0, 3, 4]


def test_null_panel_excludes_all_epoch_monitoring_panoramas():
    import hashlib
    frame = pd.DataFrame({'city': ['test'] * 50, 'prototype_split': ['validation'] * 50})
    seed = int.from_bytes(hashlib.sha256(b'test').digest()[:4], 'little')
    monitor = set(np.random.default_rng(seed).choice(frame.index.to_numpy(), 16, replace=False))
    panel = ev.panel_ids(frame)
    assert len(panel) == 16 and not (set(panel) & monitor)


def test_nmf_input_exactly_matches_mean_of_normalized_clipped_sparse_map(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, 'TOKENS', 8)
    values = np.array([1., 20., 4., 2.], np.float16)
    dims = np.array([0, 1, 2, 0], np.uint16)
    counts = np.zeros(16, np.uint32)
    counts[[0, 3, 8, 15]] = 1
    ptr = np.r_[np.zeros(1, np.uint32), counts.cumsum(dtype=np.uint32)]
    data = SimpleNamespace(out=tmp_path, scale=torch.tensor([1., 2., 2.]),
        frame=pd.DataFrame({'city': ['a', 'a']}), spec={'chunk_panos': 2},
        chunk=lambda city, start: ((values, dims, ptr), 0))
    composition = ev.scaled_composition(data)
    np.testing.assert_allclose(composition, [[1 / 8, 1, 0], [2 / 8, 0, 2 / 8]])
