from experiments.vitl30_panorama_w4096_k32.code.panorama_vitl30_dimension_review import select_examples
from experiments.vitl30_panorama_w4096_k32.code.panorama_vitl30_dimension_review_v3 import review_quality
from experiments.vitl30_panorama_w4096_k32.code.panorama_vitl30_dimension_review_audit import assess


def test_select_examples_covers_rank_bands_and_cities():
    rows = [{'feature': 0, 'rank': rank, 'pano_index': rank, 'activation': 1 / rank,
             'city': f'city-{rank % 8}'} for rank in range(1, 101)]
    selected = select_examples(rows)
    assert len(selected) == 40
    assert len({row['pano_index'] for row in selected}) == 40
    assert [sum(low <= row['rank'] <= high for row in selected)
            for low, high in ((1, 10), (11, 25), (26, 50), (51, 100))] == [10] * 4
    assert len({row['city'] for row in selected}) == 8


def test_select_examples_keeps_all_sparse_evidence():
    rows = [{'feature': 2, 'rank': rank, 'pano_index': rank, 'activation': 1.,
             'city': 'city'} for rank in range(1, 24)]
    assert len(select_examples(rows)) == 23


def test_review_requires_four_valid_visual_groups():
    value = {'label': '天空', 'type': 'color_light', 'consistency': 'mostly_consistent',
             'confidence': 'high'}
    groups = [{'interpretation': value} for _ in range(4)]
    assert review_quality(groups, value) == 'candidate_needs_validation'
    groups[2] = {'interpretation': None}
    assert review_quality(groups, value) == 'incomplete_visual_evidence'


def test_review_audit_rejects_cross_group_semantic_type_conflict():
    sun = {'label': '太阳', 'type': 'color_light', 'consistency': 'mostly_consistent',
           'confidence': 'high'}
    lamp = {**sun, 'label': '路灯', 'type': 'built_structure'}
    record = {'parts': [{'interpretation': value} for value in [sun, sun, lamp, lamp]],
              'interpretation': lamp}
    assert assess(record) == 'cross_group_type_conflict'
    record['parts'] = [{'interpretation': sun} for _ in range(4)]
    record['interpretation'] = sun
    assert assess(record) == 'candidate_needs_independent_validation'
