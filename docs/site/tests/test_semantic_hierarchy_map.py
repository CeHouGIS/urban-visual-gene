"""The overview must keep provenance and not fabricate zero-activation images."""
import json

from docs.site import semantic_hierarchy_map


def test_two_level_map_merges_weak_and_zero_under_one_parent():
    graph = {
        'identity': 'source', 'dimension_screening_identity': 'screen',
        'total': 25, 'family_count': 21, 'families': [],
    }
    rows = [{'feature': index, 'status': 'candidate_ready',
             'screening': {'positive_panoramas': 30}} for index in range(25)]
    rows[-3]['status'] = 'insufficient_evidence'
    rows[-3]['screening']['positive_panoramas'] = 2
    rows[-2]['status'] = 'insufficient_evidence'
    rows[-2]['screening']['positive_panoramas'] = 1
    rows[-1]['status'] = 'unobserved'
    evidence = [{'crops': []} for _ in rows]
    for index in (22, 23):
        evidence[index]['crops'] = [{
            'activation': 1, 'image_id': index * 4, 'city': str(index),
            'pano': str(index), 'heading': 0, 'heatmap_png': 'png',
        }]
    for index in range(20):
        graph['families'].append({
            'id': str(index), 'label': str(index), 'role': 'element', 'count': 1,
            'labels': [{'label': '子类', 'count': 1, 'features': [index],
                        'eligible': True, 'examples': [
                {'feature': index, 'city': str(j), 'pano': str(j), 'heading': 0,
                 'views': [f'{j}.jpg'] * 4, 'heat': 'png'} for j in range(4)]}],
        })
    graph['families'].append({'id': 'unresolved', 'label': '未确定', 'role': 'unresolved',
                              'count': 5, 'labels': [{'label': '待复核', 'count': 5,
                                                      'features': list(range(20, 25))}]})
    # This fixture has only two weak examples; the four-image gate should fail.
    import pytest
    with pytest.raises(ValueError, match='four real activation images'):
        semantic_hierarchy_map.build_map(graph, rows, evidence)
    for index in (20, 21):
        rows[index]['status'] = 'insufficient_evidence'
        evidence[index]['crops'] = [{
            'activation': 1, 'image_id': index * 4, 'city': str(index),
            'pano': str(index), 'heading': 0, 'heatmap_png': 'png',
        }]
    result = semantic_hierarchy_map.build_map(graph, rows, evidence)
    unresolved = result['families'][-1]
    assert result['family_count'] == 21
    assert result['leaf_count'] == 25
    assert result['weak_count'] == 4 and result['zero_count'] == 1
    assert unresolved['children'] == [
        {'label': '待复核的未归类响应', 'count': 0, 'features': []},
        {'label': '证据不足／无正激活', 'count': 5, 'features': list(range(20, 25))},
    ]
    assert len(unresolved['examples']) == 4
    assert all(example['label'] == '证据不足' for example in unresolved['examples'])
    assert all(example['feature'] != 24 for example in unresolved['examples'])
    assert len({example['feature'] for example in unresolved['examples']}) == 4
