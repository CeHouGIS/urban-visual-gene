"""The compact inputs preserve direction, spatial layout and parent grouping."""
import numpy as np

from experiments.vitl30_relational_prototypes.code.regions import family_map, pooled_cells, region_boxes


def test_region_boxes_are_direction_local_and_cover_three_scales():
    boxes = region_boxes()
    assert [sum(b[0] == size for b in boxes) for size in (14, 21, 28)] == [9, 4, 1]
    assert all(0 <= top < bottom <= 28 and 0 <= left < right <= 28
               for _, top, left, bottom, right in boxes)


def test_family_grouping_and_spatial_cells_preserve_distinct_objects():
    # Two SAE dimensions for the same parent and one for another parent.
    lookup = np.array([0, 0, 1], np.int16)
    scale = np.ones(3, np.float32)
    counts = np.array([2, 1])
    # One positive activation per patch: upper-left building, lower-right tree.
    values = np.array([2., 3., 4.], np.float16)
    indices = np.array([0, 1, 2], np.uint16)
    ptr = np.zeros(4 * 28 * 28 + 1, np.uint32)
    ptr[1:] += 2
    ptr[27 * 28 + 28:] += 1
    grid = family_map(values, indices, ptr, 0, 0, lookup, scale, counts)
    assert np.isclose(grid[0, 0, 0], 5 / np.sqrt(2))
    assert grid[1, 27, 27] == 4
    pooled = pooled_cells(grid)
    assert pooled.shape == (14, 2, 16)
    assert pooled[0, 0, 0] > 0 and pooled[0, 1].sum() == 0
    assert pooled[8, 1, -1] > 0 and pooled[8, 0].sum() == 0
