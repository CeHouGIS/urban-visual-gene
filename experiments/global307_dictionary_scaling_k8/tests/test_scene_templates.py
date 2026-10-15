import numpy as np

from formal.analysis.discover_scene_templates import (
    aligned_similarity,
    filtered_to_source,
    spatial_null,
)


def test_filtered_to_source_handles_adjacent_qc_removals():
    assert [filtered_to_source(i, [1, 2, 5]) for i in range(5)] == [0, 3, 4, 6, 7]


def test_alignment_is_invariant_to_yaw_rotation():
    centers = np.zeros((2, 3, 4, 8), dtype=np.float32)
    centers[0, 0, 0, 1] = 1
    centers[1, 2, 3, 5] = 1
    maps = np.stack([np.roll(centers[0], 3, axis=-1), np.roll(centers[1], 6, axis=-1)])
    labels, _, scores = aligned_similarity(maps, centers)
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_allclose(scores, 1)


def test_spatial_null_preserves_composition_and_height():
    maps = np.arange(2 * 3 * 4 * 8).reshape(2, 3, 4, 8)
    shuffled = spatial_null(maps, 1)
    np.testing.assert_array_equal(maps.sum(axis=-1), shuffled.sum(axis=-1))
    assert not np.array_equal(maps, shuffled)
