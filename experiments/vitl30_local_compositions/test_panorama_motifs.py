import numpy as np

from experiments.vitl30_local_compositions.panorama_motifs import (
    circular_distance, motif_key, panorama_angle, triplet_instances_panorama)
from experiments.vitl30_local_compositions.run import Node


def node(family, heading, y, x):
    mask = np.zeros((28, 28), dtype=bool)
    iy, ix = int(round(y)), int(round(x))
    mask[max(0, iy - 1):min(28, iy + 2), max(0, ix - 1):min(28, ix + 2)] = True
    return Node(family, heading, float(y), float(x), int(mask.sum()), 10.0,
                np.packbits(mask.reshape(-1)).tobytes().hex(), ())


def test_circular_bearing_wraps_270_and_zero_as_neighbors():
    a = node('tree', 3, 12, 27)
    b = node('building', 0, 12, 0)
    assert circular_distance(panorama_angle(a), panorama_angle(b)) < 15


def test_cross_heading_triplet_and_repeated_family_are_allowed():
    nodes = [node('building', 3, 11, 23), node('tree', 0, 14, 5),
             node('tree', 1, 10, 8)]
    found = triplet_instances_panorama(nodes)
    assert found
    key = next(iter(found))
    assert key[0] == ('building', 'tree', 'tree')


def test_rotation_keeps_motif_key():
    nodes = [node('building', 0, 11, 15), node('tree', 1, 15, 8),
             node('tree', 3, 10, 18)]
    shifted = [node(n.family, (n.heading + 1) % 4, n.y, n.x)
               for n in nodes]
    original = motif_key(nodes, (0, 1, 2))
    rotated = motif_key(shifted, (0, 1, 2))
    assert original == rotated


def test_same_family_edge_duplicate_is_not_counted_twice():
    nodes = [node('building', 3, 12, 27), node('tree', 3, 14, 25),
             node('tree', 0, 14, 0)]
    assert not triplet_instances_panorama(nodes)
