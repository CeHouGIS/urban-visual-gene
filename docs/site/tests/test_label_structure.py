"""Qwen text relations must keep subtypes, parts, and peer groups distinct."""
import json

from docs.site.element_hierarchy import build_category_graph, public_screened_row
from docs.site.label_structure import RULES, build_structure
from docs.site.feature_quality_review import HIERARCHY, SCREENING


def test_main_experiment_label_structure():
    source = json.loads(HIERARCHY.read_text())
    screening = json.loads(SCREENING.read_text())['dimensions']
    rows = [public_screened_row(row, screening[row['feature']]) for row in source['dimensions']]
    graph = build_category_graph({**source, 'dimensions': rows})
    structure = build_structure(graph, json.loads(RULES.read_text()))
    counts = structure['counts']
    assert counts == {'eligible_labels': 242, 'linked_labels': 80, 'unlinked_labels': 162,
                      'hierarchy_groups': 8, 'hierarchy_edges': 29, 'parallel_groups': 14}
    edges = {(group['parent']['label'], child['label'], group['relation'])
             for group in structure['hierarchy'] for child in group['children']}
    assert ('建筑', '教堂', 'is_a') in edges
    assert ('建筑', '窗户', 'part_of') not in edges
    assert ('车辆', '公交车', 'is_a') in edges
    assert not any(relation == 'part_of' for _, _, relation in edges)
    assert any(group['parent'] == '建筑' and group['relation'] == 'is_a' and
               {'教堂', '工业建筑'} <= {item['label'] for item in group['labels']}
               for group in structure['parallel'])
    assert not any({'建筑', '窗户'} <= {item['label'] for item in group['labels']}
                   for group in structure['parallel'])
    assert any(row['family'] == 'ground_unspecified' and row['label'] == '地面'
               for row in structure['unlinked'])


def test_duplicate_and_self_edges_are_rejected():
    graph = {'identity': 'x', 'families': [{'id': 'building', 'label': '建筑',
        'labels': [{'label': label, 'eligible': True, 'count': 1, 'features': [i]}
                   for i, label in enumerate(['建筑', '门'])]}]}
    rules = {'version': 1, 'hierarchy': [
        {'family': 'building', 'parent': '建筑', 'relation': 'part_of', 'children': ['门', '门']}],
        'parallel': []}
    import pytest
    with pytest.raises(ValueError, match='Duplicate'):
        build_structure(graph, rules)
