import numpy as np
import pandas as pd

from scripts.multicity.analyze_topk_typed_semantic_hierarchy import (
    DIMENSIONS,
    PATCHES,
    choose_top_x,
    city_conditioned_pair_statistics,
    panorama_topk_counts,
    patch_statistics,
)


def test_panorama_topk_counts_and_patch_statistics():
    rng = np.random.default_rng(42)
    block = rng.normal(size=(2, 14, 56, DIMENSIONS)).astype(np.float32)
    counts, top = panorama_topk_counts(block, 4)
    assert counts.shape == (2, DIMENSIONS)
    assert top.shape == (2, 14, 56, 4)
    assert np.all(counts.sum(axis=1) == PATCHES * 4)
    dimension, position, pairs = patch_statistics(top)
    assert dimension.sum() == 2 * PATCHES * 4
    assert position.sum() == dimension.sum()
    assert pairs.sum() == 2 * PATCHES * 6
    assert np.all(np.tril(pairs) == 0)


def test_choose_top_x_uses_coverage_stability_and_purity():
    frame = pd.DataFrame(
        {
            "top_x": [1, 2, 3, 4, 5],
            "dimension_coverage": [0.80, 0.89, 0.94, 0.96, 0.98],
            "city_marginal_cosine_mean": [0.80, 0.86, 0.88, 0.89, 0.90],
            "same_mapillary_parent_pair_share": [np.nan, 0.43, 0.42, 0.40, 0.37],
        }
    )
    assert choose_top_x(frame) == 4


def test_city_conditioning_removes_between_city_confounding():
    first = np.zeros((DIMENSIONS, DIMENSIONS), dtype=np.float64)
    second = np.zeros_like(first)
    # Dimensions 0 and 1 are both common in city one and rare in city two,
    # but conditionally independent inside each city.
    for matrix, support, cooccur in ((first, 80, 64), (second, 20, 4)):
        matrix[0, 0] = support
        matrix[1, 1] = support
        matrix[0, 1] = matrix[1, 0] = cooccur
    result = city_conditioned_pair_statistics([first, second], [100, 100])
    assert np.isclose(result["observed"][0, 1], result["expected"][0, 1])
    assert result["city_positive_fraction"][0, 1] == 0
    assert result["city_negative_fraction"][0, 1] == 0
