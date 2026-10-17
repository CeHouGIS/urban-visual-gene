import numpy as np

from experiments.vitl30_local_compositions.run import Node
from experiments.vitl30_local_compositions.structured_motifs import (
    TaggedNode, instances, label_index, separate_objects, tag_nodes)


def node(family, heading, x, dim=1):
    mask = np.zeros((28, 28), dtype=bool)
    mask[11:15, int(x):int(x) + 4] = True
    return Node(family, heading, 12.5, float(x), int(mask.sum()), 10.,
                np.packbits(mask.reshape(-1)).tobytes().hex(), (dim,))


def test_component_is_collapsed_but_subtype_retains_its_name():
    hierarchy = {'dimensions': [
        {'feature': 1, 'family': 'building', 'label_zh': '窗户',
         'provisional_graph_candidate': True},
        {'feature': 2, 'family': 'tree', 'label_zh': '松树',
         'provisional_graph_candidate': True}]}
    rules = {'hierarchy': [
        {'family': 'building', 'parent': '建筑', 'relation': 'part_of', 'children': ['窗户']},
        {'family': 'tree', 'parent': '树木', 'relation': 'is_a', 'children': ['松树']}],
        'parallel': []}
    mapping = label_index(hierarchy, rules)
    assert mapping == {1: ('窗户', '建筑'), 2: ('松树', '松树')}
    assert tag_nodes([node('building', 0, 1, 1)], mapping)[0].entity_label == '建筑'


def test_building_components_cannot_form_multiple_objects():
    regions = [TaggedNode(node('building', 0, 1), '窗户', '建筑'),
               TaggedNode(node('building', 1, 10), '建筑墙面', '建筑'),
               TaggedNode(node('tree', 2, 10), '树冠', '树木')]
    assert not separate_objects(regions[0], regions[1])
    assert instances(regions) == {}


def test_three_distinct_panorama_regions_are_detected():
    regions = [TaggedNode(node('building', 0, 10), '房屋', '房屋'),
               TaggedNode(node('tree', 1, 10), '树冠', '树木'),
               TaggedNode(node('tree', 3, 10), '松树', '松树')]
    assert len(instances(regions)) == 1
