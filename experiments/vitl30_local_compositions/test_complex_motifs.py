import numpy as np

from experiments.vitl30_local_compositions.complex_motifs import (
    select_gallery_patterns, triplet_instances)
from experiments.vitl30_local_compositions.run import Node


def node(family, heading, top, left):
    mask = np.zeros((28, 28), dtype=bool)
    mask[top:top + 4, left:left + 4] = True
    return Node(family, heading, top + 1.5, left + 1.5, 16, 10.,
                np.packbits(mask.reshape(-1)).tobytes().hex(), ())


def test_triplet_is_rotation_invariant_and_requires_three_families_in_one_view():
    nodes = [node('building', 0, 2, 2), node('tree', 0, 12, 12),
             node('barrier', 0, 22, 22)]
    found = triplet_instances(nodes)
    assert len(found) == 1
    key = next(iter(found))
    assert key[0] == ('barrier', 'building', 'tree')
    shifted = [node(n.family, 2, t, t) for n, t in zip(nodes, (2, 12, 22))]
    assert set(triplet_instances(shifted)) == set(found)
    assert not triplet_instances(nodes[:2] + [node('tree', 0, 22, 22)])
    assert not triplet_instances(nodes[:2] + [node('barrier', 1, 22, 22)])


def test_gallery_shows_distinct_family_combinations():
    patterns = [{'families': ['building', 'road', 'tree'], 'id': i} for i in (1, 2)]
    patterns.append({'families': ['building', 'barrier', 'tree'], 'id': 3})
    assert [p['id'] for p in select_gallery_patterns(patterns, 2)] == [1, 3]
