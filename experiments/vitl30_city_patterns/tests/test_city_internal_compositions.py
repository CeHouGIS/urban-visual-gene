"""Check city-local mining and cross-city overlap semantics."""
import numpy as np
import pandas as pd

from experiments.vitl30_city_patterns.code.candidate_combinations import ELEMENTS
from experiments.vitl30_city_patterns.code.city_internal_compositions import (
    compare_cities, cross_city_rates, mine_city, parent_overlap, split_blocks,
)


def test_local_holdout_rejects_discovery_only_composition():
    frame = pd.DataFrame({'city': ['A'] * 80,
                          'grid_x': np.repeat(np.arange(20), 4) * 5,
                          'grid_y': [0] * 80})
    holdout = np.repeat(np.arange(20) < 5, 4)
    units = [
        {'id': 0, 'family': 'building', 'label': 'building',
         'key': 'building:building', 'fallback': False, 'landscape_eligible': True},
        {'id': 1, 'family': 'tree', 'label': 'tree',
         'key': 'tree:tree', 'fallback': False, 'landscape_eligible': True},
        {'id': 2, 'family': 'road', 'label': 'road',
         'key': 'road:road', 'fallback': False, 'landscape_eligible': True},
    ]
    assignment = np.full((80, len(ELEMENTS)), -1, dtype=np.int16)
    assignment[:, ELEMENTS.index('building')] = 0
    assignment[:, ELEMENTS.index('tree')] = 1
    assignment[~holdout, ELEMENTS.index('road')] = 2
    found = mine_city(assignment, frame, units, holdout, min_panoramas=6,
                      min_cells=4, max_order=3)
    by_label = {row['label']: row for row in found}
    assert by_label['building ＋ tree']['replicated']
    assert not by_label['building ＋ road']['replicated']
    assert not by_label['building ＋ tree ＋ road']['replicated']
    assert by_label['building ＋ tree']['discovery_cells'] == 15
    assert by_label['building ＋ tree']['holdout_cells'] == 5


def test_exact_city_overlap_does_not_promote_failed_holdout():
    rows = pd.DataFrame([
        {'city': 'A', 'pattern_id': 'x', 'family_signature': 'building+tree', 'replicated': True},
        {'city': 'B', 'pattern_id': 'x', 'family_signature': 'building+tree', 'replicated': True},
        {'city': 'A', 'pattern_id': 'y', 'family_signature': 'road+tree', 'replicated': True},
        {'city': 'B', 'pattern_id': 'z', 'family_signature': 'road+tree', 'replicated': False},
    ])
    result, kept, overlap = compare_cities(rows, ['A', 'B'])
    assert len(kept) == 3
    assert np.isclose(overlap.loc['A', 'B'], .5)
    assert set(result.loc[result.pattern_id == 'x', 'cross_city_status']) == {'shared'}
    assert result.loc[result.pattern_id == 'y', 'cross_city_status'].iloc[0] == 'single_city_detected'
    assert result.loc[result.pattern_id == 'z', 'cross_city_status'].iloc[0] == 'unreplicated'
    assert np.isclose(parent_overlap(kept, ['A', 'B']).loc['A', 'B'], .5)


def test_rates_expose_below_cutoff_response_in_other_city():
    frame = pd.DataFrame({'city': ['A'] * 4 + ['B'] * 4,
                          'grid_x': [0, 0, 1, 1] * 2, 'grid_y': [0] * 8})
    units = [{'id': 0, 'family': 'building', 'label': 'building'},
             {'id': 1, 'family': 'tree', 'label': 'tree'}]
    assignment = np.full((8, len(ELEMENTS)), -1, dtype=np.int16)
    assignment[:4, ELEMENTS.index('building')] = 0
    assignment[[0, 1, 4], ELEMENTS.index('tree')] = 1
    assignment[4, ELEMENTS.index('building')] = 0
    kept = pd.DataFrame([{'pattern_id': 'bt', 'label': 'building + tree',
                          'unit_ids': [0, 1]}])
    rates = cross_city_rates(assignment, frame, units, kept).set_index('city')
    assert np.isclose(rates.loc['A', 'cell_balanced_rate'], .5)
    assert np.isclose(rates.loc['B', 'cell_balanced_rate'], .25)


def test_spatial_block_split_keeps_same_block_together():
    frame = pd.DataFrame({'city': ['A'] * 4 + ['B'] * 2,
                          'grid_x': [10, 11, 14, 15, 10, 11],
                          'grid_y': [20, 21, 24, 25, 20, 21]})
    split = split_blocks(frame)
    assert split[:3].tolist() == [bool(split[0])] * 3
    assert split.dtype == bool
