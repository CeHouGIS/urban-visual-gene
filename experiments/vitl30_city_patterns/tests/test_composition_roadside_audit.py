"""The roadside screen must not convert uncertain responses into prevalence."""
import numpy as np

from experiments.vitl30_city_patterns.code.composition_roadside_audit import (
    classify_pattern, lateral_units, summarise)


def test_road_and_cross_road_units_do_not_count_as_lateral_elements():
    units = [
        {'family': 'road', 'key': 'road:道路', 'label': '道路'},
        {'family': 'building', 'key': 'building:建筑', 'label': '建筑'},
        {'family': 'pedestrian', 'key': 'pedestrian:人行横道', 'label': '人行横道'},
        {'family': 'tree', 'key': 'tree:乔木', 'label': '乔木'},
        {'family': 'utility', 'key': 'utility:电线', 'label': '电线'},
    ]
    assert lateral_units({'units': [0, 1, 2, 3, 4]}, units) == [1, 3]


def test_geometry_hypotheses_remain_unconfirmed_and_require_all_units():
    road = np.zeros((4, 28, 28))
    road[0, 13:19, 10:18] = road[2, 13:19, 10:18] = 1
    a, b, c = np.zeros_like(road), np.zeros_like(road), np.zeros_like(road)
    a[1, 8:24, 10:18] = 1
    b[3, 8:24, 10:18] = 1
    c[1, 8:24, 9:18] = 1
    opposite = classify_pattern({1: a, 2: b, 3: c}, road, [1, 2, 3])
    assert opposite['tentative_side'] == 'opposite_sides'
    assert opposite['final_side'] == 'undetermined'
    assert not opposite['calibration_verified']
    same = classify_pattern({1: a, 3: c}, road, [1, 3])
    assert same['tentative_side'] == 'same_side'
    assert classify_pattern({1: a, 2: np.zeros_like(road)}, road, [1, 2])['tentative_side'] == 'undetermined'


def test_summary_keeps_unclassifiable_patterns_and_no_confirmed_count():
    units = [{'family': 'road', 'key': 'road:道路', 'label': '道路'},
             {'family': 'tree', 'key': 'tree:乔木', 'label': '乔木'},
             {'family': 'building', 'key': 'building:建筑', 'label': '建筑'}]
    patterns = [{'pattern_id': 'a', 'label': '道路＋乔木', 'units': [0, 1], 'positive_panoramas': 10},
                {'pattern_id': 'b', 'label': '建筑＋乔木', 'units': [1, 2], 'positive_panoramas': 8}]
    rows = [{'pattern_id': 'b', 'city': 'A', 'panoid': 'one', 'tentative_side': 'same_side'},
            {'pattern_id': 'b', 'city': 'B', 'panoid': 'two', 'tentative_side': 'undetermined'}]
    table, summary = summarise(rows, patterns, units, 6)
    assert not table[0]['eligible_for_side_audit'] and table[0]['sampled_panoramas'] == 0
    assert table[1]['same_side_hypotheses'] == 1 and table[1]['final_undetermined'] == 2
    assert summary['final'] == {'same_side': 0, 'opposite_sides': 0, 'undetermined': 2}
    assert not summary['corpus_roadside_prevalence_estimable']
