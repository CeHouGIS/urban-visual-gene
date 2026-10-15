"""Check the null hypotheses and discovery/confirmation boundaries."""
import numpy as np

from sae_experiments.panorama_vitl30_unit_motifs import (
    bh_adjust, conditional_independence_null, cross_city_agreement,
    discovery_split, fit_presence, pair_agreement, relation_codes,
    spatial_test, unit_rows,
)


def test_cosine_agreement_excludes_diagonal_and_within_city_pairs():
    x = np.array([[1., 0.], [1., 0.], [0., 1.]])
    assert np.isclose(pair_agreement(x), 1 / 3)
    assert np.isclose(cross_city_agreement(x, np.array(['A', 'A', 'B'])), 0)
    assert np.isnan(cross_city_agreement(x, np.array(['A', 'A', 'A'])))
    batched = np.stack([x, x])
    np.testing.assert_allclose(pair_agreement(batched), [1 / 3, 1 / 3])
    np.testing.assert_allclose(cross_city_agreement(batched, np.array(['A', 'A', 'B'])), [0, 0])


def test_discovery_cities_disjoint_and_threshold_does_not_use_confirmation():
    rows = [{'city': c} for c in ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']]
    train, test = discovery_split(rows, np.arange(8), 42)
    assert set(rows[i]['city'] for i in train).isdisjoint(rows[i]['city'] for i in test)
    peaks = np.arange(1, 31, dtype=np.float32)[:, None]
    presence, threshold = fit_presence(peaks, np.arange(20))
    changed = peaks.copy()
    changed[20:] *= 1000
    _, unchanged = fit_presence(changed, np.arange(20))
    np.testing.assert_array_equal(threshold, unchanged)
    assert not presence[:10].any()


def test_city_conditioned_null_handles_perfect_city_confounding():
    cities = np.array(['A'] * 50 + ['B'] * 50)
    presence = np.zeros((100, 2), bool)
    presence[:50] = True
    # Globally this looks associated; within each city the overlap is forced.
    null = conditional_independence_null(presence, np.arange(100), cities, [0, 1],
                                         1000, np.random.default_rng(42))
    np.testing.assert_array_equal(null, np.full(1000, 50))


def test_random_intersection_null_matches_known_expectation_for_triples():
    cities = np.array(['A'] * 100)
    presence = np.zeros((100, 3), bool)
    presence[:50, 0] = True
    presence[:40, 1] = True
    presence[:30, 2] = True
    null = conditional_independence_null(presence, np.arange(100), cities, [0, 1, 2],
                                         20000, np.random.default_rng(42))
    assert abs(null.mean() - 100 * .5 * .4 * .3) < .1


def test_relation_is_invariant_to_rotation_and_wraps_heading_seam():
    first = np.array([10 * 28 + 5, 3 * 784 + 10 * 28 + 5])
    second = np.array([20 * 28 + 20, 10 * 28 + 5])
    original = relation_codes(first, second)
    for shift in range(4):
        rotate_first = ((first // 784 + shift) % 4) * 784 + first % 784
        rotate_second = ((second // 784 + shift) % 4) * 784 + second % 784
        np.testing.assert_array_equal(relation_codes(rotate_first, rotate_second), original)
    assert original[1] == 9  # 270 -> 0 is the next heading.


def test_spatial_null_keeps_sky_above_ground_bias():
    patches = np.tile([3 * 28 + 14, 24 * 28 + 14], (60, 1))
    presence = np.ones((60, 2), bool)
    cities = np.array(['train'] * 30 + ['test'] * 30)
    result = spatial_test(patches, presence, np.arange(30), np.arange(30, 60), cities,
                          [0, 1], 199, np.random.default_rng(42))
    assert result['confirmation_template_fraction'] == 1
    assert result['spatial_null_fraction'] == 1
    assert result['spatial_p'] == 1
    assert result['spatial_excess'] == 0


def test_spatial_test_detects_shared_direction_beyond_marginal_bias():
    headings = np.tile(np.arange(4), 30)
    first = headings * 784 + 10 * 28 + 10
    second = headings * 784 + 20 * 28 + 10
    patches = np.stack([first, second], axis=1)
    presence = np.ones((120, 2), bool)
    cities = np.array(['train'] * 60 + ['test'] * 60)
    result = spatial_test(patches, presence, np.arange(60), np.arange(60, 120), cities,
                          [0, 1], 999, np.random.default_rng(42))
    assert result['spatial_p'] <= .005
    assert result['spatial_excess'] > .6


def test_multiple_testing_is_monotone_and_handles_empty_family():
    q = bh_adjust([.01, .04, .03, 1.])
    np.testing.assert_allclose(q, [.04, .053333333, .053333333, 1.])
    assert not len(bh_adjust([]))
