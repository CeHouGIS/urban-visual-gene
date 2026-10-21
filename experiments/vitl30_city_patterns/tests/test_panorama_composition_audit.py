"""Cross-view relations use the circular four-heading patch order."""
import numpy as np

from experiments.vitl30_city_patterns.code.panorama_composition_audit import (
    directional_match, seam_match,
)


def test_adjacent_heading_seam_match():
    maps = np.zeros((2, 4, 28, 28), dtype=np.float32)
    maps[0, 0, 10:14, 23:27] = 1
    maps[1, 1, 11:15, 1:5] = 1
    result = seam_match(maps)
    assert result is not None
    assert result['headings'] == [0, 1]
    assert result['seam'] == (0, 1)


def test_last_heading_wraps_to_first():
    maps = np.zeros((2, 4, 28, 28), dtype=np.float32)
    maps[0, 3, 10:14, 23:27] = 1
    maps[1, 0, 11:15, 1:5] = 1
    result = seam_match(maps)
    assert result is not None and result['seam'] == (3, 0)


def test_distant_or_same_heading_responses_do_not_cross_seam():
    maps = np.zeros((2, 4, 28, 28), dtype=np.float32)
    maps[0, 0, 10:14, 3:7] = 1
    maps[1, 1, 11:15, 20:24] = 1
    assert seam_match(maps) is None
    maps[1, 1] = 0
    maps[1, 0, 10:14, 23:27] = 1
    assert seam_match(maps) is None


def test_directional_layout_requires_concentrated_distinct_headings():
    maps = np.zeros((3, 4, 28, 28), dtype=np.float32)
    maps[0, 0, 10:14, 10:14] = 1
    maps[1, 2, 10:14, 10:14] = 1
    maps[2, 3, 10:14, 10:14] = 1
    result = directional_match(maps)
    assert result is not None
    assert result['signature'] == [0, 2, 3]
    maps[1] = 1
    assert directional_match(maps) is None
