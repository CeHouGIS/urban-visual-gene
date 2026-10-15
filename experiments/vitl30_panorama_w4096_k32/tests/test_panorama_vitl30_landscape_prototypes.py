"""Check that prototype validation preserves city and feature marginals."""
import numpy as np

from sae_experiments.panorama_vitl30_landscape_prototypes import (
    city_folds, component_similarity, exemplar_ids, shuffled_within_city,
)


def test_city_folds_are_disjoint():
    rows = [{'city': city} for city in ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']]
    visited = []
    for train, test, held in city_folds(rows, 2, 42):
        assert set(rows[i]['city'] for i in train).isdisjoint(held)
        assert set(rows[i]['city'] for i in test) == set(held)
        visited.extend(test)
    assert sorted(visited) == list(range(len(rows)))


def test_city_shuffling_keeps_each_dimension_marginal():
    matrix = np.arange(24, dtype=np.float32).reshape(8, 3)
    cities = np.array(['A'] * 4 + ['B'] * 4)
    shuffled = shuffled_within_city(matrix, cities, np.random.default_rng(3))
    for city in ['A', 'B']:
        for j in range(3):
            np.testing.assert_array_equal(np.sort(matrix[cities == city, j]),
                                          np.sort(shuffled[cities == city, j]))
    assert not np.array_equal(matrix, shuffled)


def test_component_similarity_and_city_distinct_examples():
    components = np.eye(3, dtype=np.float32)
    np.testing.assert_allclose(component_similarity(components, components[[2, 0, 1]]), 1)
    weights = np.array([[9, 1], [8, 1], [7, 1], [1, 9], [1, 8]], np.float32)
    rows = [{'city': city} for city in ['A', 'A', 'B', 'A', 'B']]
    examples, labels, share = exemplar_ids(weights, rows, count=3)
    assert examples == [[0, 2], [3, 4]]
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1])
    np.testing.assert_allclose(share.sum(1), 1)
