"""Check sampling, discovery-only calibration and true cross-object mining."""
import numpy as np
import pandas as pd

from experiments.vitl30_city_patterns.code.composition_patterns import (
    assign_units, calibrate, canonical, mine, packed_mask, pattern_mask,
    landscape_exclusion, sampling_design, score_identity, summarise_pattern,
    unpack_mask, weighted_quantile,
)
from experiments.vitl30_city_patterns.code.candidate_combinations import ELEMENTS


def test_object_parts_and_tree_related_labels_merge():
    assert canonical('building', '窗户') == canonical('building', '建筑墙面') == '建筑'
    assert len({canonical('tree', label) for label in
                ['树', '树木', '树冠', '树干', '棕榈树', '椰子树', '松树', '竹林']}) == 1


def test_landscape_scope_excludes_mobile_objects_but_keeps_fixed_fabric():
    for family, label in [('person', '行人'), ('vehicle', '自行车'),
                          ('construction', '集装箱'), ('construction', '起重机'),
                          ('sign', '交通锥'), ('furniture', '移动厕所')]:
        assert landscape_exclusion(family, label)
    for family, label in [('construction', '储罐'), ('building', '建筑'),
                          ('tree', '乔木'), ('utility', '路灯'), ('furniture', '座椅')]:
        assert landscape_exclusion(family, label) is None


def test_excluded_child_mass_cannot_leak_through_parent_fallback():
    units = [
        {'id': 0, 'family': 'construction', 'fallback': False, 'landscape_eligible': True},
        {'id': 1, 'family': 'construction', 'fallback': False, 'landscape_eligible': False},
        {'id': 2, 'family': 'vehicle', 'fallback': False, 'landscape_eligible': False},
    ]
    units += [{'id': i+3, 'family': f, 'fallback': True} for i, f in enumerate(ELEMENTS)]
    # A tiny allowed signal must not borrow the large container/vehicle response.
    scores = np.array([[.0001, .4, .4], [.01, .3, .3], [.02, 0, 0]])
    tau, pt, parents = calibrate(scores, units, np.ones(3), np.ones(3, bool))
    ci, vi = ELEMENTS.index('construction'), ELEMENTS.index('vehicle')
    np.testing.assert_allclose(parents[:, ci], scores[:, 0])
    assert (parents[:, vi] == 0).all()
    assignment = assign_units(scores, units, tau, pt, parents)
    assert assignment[0, ci] == -1
    assert assignment[2, ci] == 0
    assert (assignment[:, vi] == -1).all()
    assert not np.isin(assignment, [1, 2]).any()


def test_raw_cache_identity_ignores_filter_but_tracks_actual_dimensions():
    unit = {'id': 0, 'family': 'vehicle', 'dimensions': [1, 2], 'fallback': False}
    original = score_identity('run', 'hierarchy', [unit])
    filtered = {**unit, 'landscape_eligible': False, 'exclusion_reason': 'excluded'}
    assert score_identity('run', 'hierarchy', [filtered]) == original
    assert score_identity('run', 'hierarchy', [{**filtered, 'dimensions': [1, 3]}]) != original


def test_calibration_ignores_evaluation_and_does_not_create_city_thresholds():
    units = [{'id': i, 'family': f, 'fallback': False} for i, f in enumerate(ELEMENTS)]
    units += [{'id': i+15, 'family': f, 'fallback': True} for i, f in enumerate(ELEMENTS)]
    scores = np.array([[.01]*15, [.02]*15, [.03]*15, [.9]*15], dtype=np.float32)
    discovery = np.array([True, True, True, False])
    tau, pt, parents = calibrate(scores, units, np.ones(4), discovery, .6, .0001)
    scores[3] = 0
    tau2, pt2, _ = calibrate(scores, units, np.ones(4), discovery, .6, .0001)
    np.testing.assert_equal(tau, tau2)
    np.testing.assert_equal(pt, pt2)
    assert np.allclose(tau, .02)
    assigned = assign_units(scores, units, tau, pt, parents)
    assert (assigned[0] == -1).all()
    assert weighted_quantile(np.zeros(4), np.ones(4), .6) == 1


def test_cell_balancing_and_retrieval_coverage_unit():
    frame = pd.DataFrame({'city': ['A']*11+['B'], 'grid_x': [1]*10+[2,1], 'grid_y': [1]*12})
    cell_id, city_id, cities, cells, w, disc = sampling_design(frame)
    assert np.isclose(w[:10].sum(), .5) and w[10] == .5 and w[11] == 1
    mask = np.arange(12) < 10
    for c in cities:
        assert np.isclose(w[frame.city == c].sum(), 1)
    assert np.isclose(w[mask].sum()/2, .25)
    assert mask.mean() != w[mask].sum()/2


def test_mass_floor_suppresses_weak_background_even_above_quantile():
    units = [{'id': i, 'family': f, 'fallback': False} for i, f in enumerate(ELEMENTS)]
    units += [{'id': i+15, 'family': f, 'fallback': True} for i, f in enumerate(ELEMENTS)]
    scores = np.array([[.0001]*15, [.0003]*15, [.0008]*15, [.006]*15], dtype=np.float32)
    tau, pt, parents = calibrate(scores, units, np.ones(4), np.ones(4, bool))
    assignment = assign_units(scores, units, tau, pt, parents)
    assert np.all(tau >= .001) and np.all(pt >= .005)
    assert (assignment[:3] == -1).all()
    assert (assignment[3] >= 0).all()


def test_bitset_padding_and_cross_parent_higher_order_mining():
    mask = np.array([True, False]*50+[True])
    np.testing.assert_equal(unpack_mask(packed_mask(mask), len(mask)), mask)
    units = [{'id': 0, 'family': 'building'}, {'id': 1, 'family': 'tree'},
             {'id': 2, 'family': 'road'}, {'id': 3, 'family': 'building'}]
    assignment = np.full((120, 15), -1, dtype=np.int16)
    assignment[:90, 0] = 0
    assignment[90:, 0] = 3
    assignment[:80, 1] = 1
    assignment[:70, 3] = 2  # road index is 3 in ELEMENTS
    cities = np.repeat(np.arange(4), 30)
    result = mine(assignment, units, np.full(120, 1/30), np.ones(120, bool), cities,
                  np.arange(120)//2, min_rate=.005, max_order=4)
    assert any(r['units'] == [0,1,2] for r in result)
    assert not any(0 in r['units'] and 3 in r['units'] for r in result)
    assert pattern_mask(assignment, units, [0,1,2]).sum() == 70
