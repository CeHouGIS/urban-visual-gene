"""Locality requires compact responses in the same directional image."""
import numpy as np

from experiments.vitl30_city_patterns.code.local_composition_audit import compact_match


def test_two_nearby_regions_are_local():
    maps = np.zeros((2, 4, 28, 28), dtype=np.float32)
    maps[0, 1, 8:12, 8:12] = 1
    maps[1, 1, 12:16, 13:17] = 1
    result = compact_match(maps)
    assert result['heading'] == 1
    assert result['box'] == (8, 8, 17, 16)
    assert result['component_areas'] == [16, 16]


def test_cross_direction_or_spread_regions_are_not_local():
    maps = np.zeros((2, 4, 28, 28), dtype=np.float32)
    maps[0, 0, 8:12, 8:12] = 1
    maps[1, 1, 12:16, 13:17] = 1
    assert compact_match(maps) is None
    maps[1, 1] = 0
    maps[1, 0, 23:27, 23:27] = 1
    assert compact_match(maps) is None


def test_three_region_chain_can_form_one_compact_local_arrangement():
    maps = np.zeros((3, 4, 28, 28), dtype=np.float32)
    maps[0, 2, 3:7, 3:7] = 1
    maps[1, 2, 8:12, 8:12] = 1
    maps[2, 2, 13:17, 13:17] = 1
    result = compact_match(maps)
    assert result is not None and result['heading'] == 2
    assert result['box'] == (3, 3, 17, 17)
