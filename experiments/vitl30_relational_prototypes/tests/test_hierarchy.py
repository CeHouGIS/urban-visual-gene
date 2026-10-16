"""Guard semantic consolidation, evidence identity, and sealed pilot partitions."""
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.vitl30_relational_prototypes.code.hierarchy import combine, label_family, WIDTH
from experiments.vitl30_relational_prototypes.code.prepare import select

TAXONOMY = json.loads((Path(__file__).parents[1] / 'taxonomy.json').read_text())


def row(feature, label='', count=0, status='unobserved_in_sample', **kwargs):
    return {'feature': feature, 'label_zh': label, 'crop_count': count, 'status': status,
            'consistency': 'mostly_consistent', 'confidence_self_reported': 'high', **kwargs}


def test_parts_do_not_inflate_element_diversity_and_ambiguous_ground_is_retained():
    assert label_family('窗户', TAXONOMY) == label_family('建筑墙面', TAXONOMY) == 'building'
    assert label_family('树冠', TAXONOMY) == label_family('树干', TAXONOMY) == 'tree'
    assert label_family('车轮及轮胎', TAXONOMY) == label_family('车窗', TAXONOMY) == 'vehicle'
    assert label_family('地面', TAXONOMY) == 'ground_unspecified'
    assert label_family('模糊的建筑', TAXONOMY) == 'unresolved'


def test_conflicting_and_sparse_evidence_cannot_silently_enter_graph():
    a, b, c = [[row(i) for i in range(WIDTH)] for _ in range(3)]
    a[0] = row(0, '窗户', 10, 'interpreted')
    c[0] = row(0, '建筑墙面', 100, 'interpreted')
    a[1] = row(1, '车辆', 10, 'interpreted')
    c[1] = row(1, '道路', 100, 'interpreted')
    b[2] = row(2, '树木', 1, 'interpreted')
    a[3] = row(3, '天空', 10, 'interpreted')
    b[4] = row(4, '树冠', 10, 'interpreted')
    out = combine(a, b, c, TAXONOMY)
    assert out[0]['provisional_graph_candidate'] and not out[0]['family_disagreement']
    assert out[1]['status'] == 'needs_review' and not out[1]['provisional_graph_candidate']
    assert out[2]['status'] == 'insufficient_evidence'
    assert out[3]['status'] == 'context_or_material'
    assert out[4]['source'] == 'vlm_dimensions_top10_expanded'
    assert all(not r['independently_validated'] for r in out)
    b[0] = row(0, '窗户', 10, 'interpreted')
    with pytest.raises(ValueError, match='overwrites'):
        combine(a, b, c, TAXONOMY)


def test_city_balanced_selection_never_opens_test_or_splits_panorama():
    frame = pd.DataFrame([{'city': city, 'panoid': f'{split}-{i}', 'prototype_split': split,
        'spatial_block': f'{city}/{split}/{i // 2}', 'cache_pano_index': i}
        for city in ['a', 'b'] for split in ['train', 'dev', 'test'] for i in range(12)])
    result = select(frame, train_per_city=4, dev_per_city=2)
    assert len(result) == 12 and 'test' not in set(result.prototype_split)
    assert result.groupby(['city', 'prototype_split']).size().to_dict() == {
        ('a', 'dev'): 2, ('a', 'train'): 4, ('b', 'dev'): 2, ('b', 'train'): 4}
    pd.testing.assert_frame_equal(result, select(frame, train_per_city=4, dev_per_city=2))
    frame.loc[0, 'spatial_block'] = frame.loc[12, 'spatial_block']
    with pytest.raises(ValueError, match='crosses'):
        select(frame, 4, 2)
