from __future__ import annotations

import numpy as np

from formal.gpu_run import (
    FilteredFeatureArray,
    InterleavedFeatureBank,
    interleave_city_features,
    merge_feature_shards,
    stage_balanced_train_cache,
)


def test_filtered_feature_array_removes_whole_pano_blocks(tmp_path):
    source = np.arange(20 * 2, dtype=np.float16).reshape(20, 2)
    path = tmp_path / "city.npy"
    np.save(path, source)
    view = FilteredFeatureArray(path, [1, 3], rows_per_pano=4)
    expected = np.delete(source, np.r_[4:8, 12:16], axis=0)
    assert view.shape == expected.shape
    np.testing.assert_array_equal(view[:], expected)
    np.testing.assert_array_equal(view[2:9], expected[2:9])


def test_merge_feature_shards_is_memory_mapped_and_preserves_rows(tmp_path):
    first = np.arange(12, dtype=np.float16).reshape(3, 4)
    second = (np.arange(8, dtype=np.float16) + 20).reshape(2, 4)
    paths = []
    for name, value in (("a", first), ("b", second)):
        path = tmp_path / f"{name}.npy"
        np.save(path, value)
        paths.append(path)

    merged = merge_feature_shards(paths, tmp_path / "merged.npy", width=4)

    assert isinstance(merged, np.memmap)
    np.testing.assert_array_equal(merged, np.concatenate((first, second)))


def test_interleave_city_features_keeps_equal_city_stripes(tmp_path):
    cities = []
    for name, start, count in (("a", 0, 5), ("b", 10, 3)):
        values = np.full((count, 2), start, dtype=np.float16)
        values[:, 1] += np.arange(count)
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        cities.append((name, path))

    mixed = interleave_city_features(cities, tmp_path / "mixed.npy", width=2,
                                     stripe_rows=2)

    assert isinstance(mixed, np.memmap)
    np.testing.assert_array_equal(
        mixed,
        np.array([
            [0, 0], [0, 1], [10, 10], [10, 11],
            [0, 2], [0, 3], [10, 12],
            [0, 4],
        ], dtype=np.float16),
    )


def test_local_train_cache_balances_cities_and_reuses_completed_matrix(tmp_path):
    cities = []
    for name, value in (("a", 1), ("b", 2)):
        path = tmp_path / f"{name}.npy"
        np.save(path, np.full((8, 2), value, dtype=np.float16))
        cities.append((name, path))
    bank = InterleavedFeatureBank(cities, width=2)
    cache = tmp_path / "cache"

    mixed = stage_balanced_train_cache(bank, max_rows=8, cache_root=cache,
                                       stripe_rows=2)

    assert isinstance(mixed, np.memmap)
    assert mixed.shape == (8, 2)
    np.testing.assert_array_equal(mixed[:, 0], [1, 1, 2, 2, 1, 1, 2, 2])
    assert stage_balanced_train_cache(bank, 8, cache, 2).filename == mixed.filename
