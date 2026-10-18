"""Scientific invariants for geometry uncertainty and spatial aggregation."""
import numpy as np
import pandas as pd

from experiments.vitl30_city_patterns.code.road_sides import (
    axis_estimate, bearings, classify, choose_examples)
from experiments.vitl30_city_patterns.code.prototype_city_distribution import (
    cell_statistics, spatial_interval)
from experiments.vitl30_city_patterns.code.chapter5_figures import pair_similarity


def test_linear_pair_similarity_matches_explicit_pairs():
    x = np.array([[1., 0., 0.], [1., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-20)
    expected = np.mean([z[i] @ z[j] for i in range(len(x)) for j in range(i + 1, len(x))])
    assert np.isclose(pair_similarity(x), expected)


def test_perspective_bearings_and_heading_rotation():
    angles = bearings()
    assert np.allclose(angles[0], -angles[0, ::-1])
    assert np.allclose(angles[1] - angles[0], np.pi / 2)
    assert np.max(np.abs(angles[0])) < np.pi / 4


def test_road_axis_needs_antipodal_evidence_and_rejects_intersection():
    road = np.zeros((4, 28, 28))
    road[0, 13:19, 10:18] = 1
    assert not axis_estimate(road)['valid']
    road[2, 13:19, 10:18] = 1
    assert axis_estimate(road)['valid']
    road[1, 13:19, 10:18] = 1
    road[3, 13:19, 10:18] = 1
    assert not axis_estimate(road)['valid']


def test_strong_geometric_hypothesis_does_not_bypass_calibration():
    road = np.zeros((4, 28, 28))
    road[0, 13:19, 10:18] = road[2, 13:19, 10:18] = 1
    a, b = np.zeros_like(road), np.zeros_like(road)
    a[1, 10:20, 10:18] = 1
    b[3, 10:20, 10:18] = 1
    result = classify({'a': a, 'b': b}, road, 'a', 'b')
    assert result['tentative_side'] == 'opposite_side'
    assert result['side_label'] == 'undetermined'
    assert not result['calibration_verified']


def test_cell_balance_is_not_equal_panorama_weight():
    frame = pd.DataFrame({'city': ['A'] * 11, 'grid_x': [0] * 10 + [5],
                          'grid_y': [0] * 11, 'present_a': [True] * 10 + [False],
                          'present_b': [True] * 11})
    cells = cell_statistics(frame, 'a', 'b')
    assert cells.prevalence.mean() == .5
    assert cells.positive_panoramas.sum() / cells.sampled_panoramas.sum() != .5
    assert spatial_interval(cells, draws=99)[2] == 2
    assert np.isnan(spatial_interval(cells.iloc[:1], draws=99)[0])


def test_audit_sampling_respects_city_and_cell_limits():
    frame = pd.DataFrame({'city': ['A'] * 20 + ['B'] * 20,
        'panoid': [str(i) for i in range(40)], 'grid_x': list(range(20)) * 2,
        'grid_y': [0] * 40, 'present_a': [True] * 40, 'present_b': [True] * 40})
    selected = choose_examples(frame, ['a+b'], 12)
    assert len(selected) == 6  # never silently violate 3/city to fill quota
    assert len({(r['city'], r['grid_x'], r['grid_y']) for r in selected}) == 6
