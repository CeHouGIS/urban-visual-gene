import numpy as np

from formal.analysis.analyze_heldout_scaling import classify, pair_matrices, select_anchors


def test_anchor_selection_balances_four_headings():
    panos = [{"city": "A", "source_pano": 3, "filtered_pano": 3}]
    anchors = select_anchors(panos, 128, seed=5)
    assert len(anchors) == 128
    assert {heading: sum(row["heading"] == heading for row in anchors) for heading in (0, 90, 180, 270)} == {0: 32, 90: 32, 180: 32, 270: 32}


def test_activation_metrics_include_full_support():
    left = np.array([[1, 0], [1, 0], [0, 2], [0, 0]], dtype=np.float16)
    right = np.array([[1, 0], [1, 0], [0, 0], [0, 2]], dtype=np.float16)
    decoder = np.eye(2, dtype=np.float32)
    metrics = pair_matrices(left, right, decoder, decoder)
    np.testing.assert_allclose(metrics["support_jaccard"][0, 0], 1)
    np.testing.assert_allclose(metrics["support_jaccard"][1, 1], 0)
    assert metrics["left_frequency"].tolist() == [2, 1]


def test_split_requires_complementary_child_support():
    left = np.ones((100, 1), dtype=np.float16)
    right = np.zeros((100, 3), dtype=np.float16)
    right[:50, 0] = 1
    right[50:, 1] = 1
    right[:50, 2] = 1  # a duplicate must not count as the second child
    metrics = pair_matrices(left, right, np.ones((1, 2), dtype=np.float32) / np.sqrt(2), np.ones((3, 2), dtype=np.float32) / np.sqrt(2))
    result = classify(left, right, metrics)
    assert result["split_candidates"][0]["children"] == [0, 1]
    np.testing.assert_allclose(result["split_candidates"][0]["parent_coverage"], 1)


def test_expansion_preserves_old_patches_and_excludes_training_panoramas():
    from formal.analysis.analyze_heldout_scaling import expanded_anchors
    from formal.analysis.discover_scene_templates import ROWS_PER_PANO, sample_panoramas
    manifest = {'cities': {'A': {'verified': True, 'filtered_rows': 20 * ROWS_PER_PANO}}}
    qc = {'cities': {'A': {'verified': True, 'bad_kept_positions': []}}}
    old_panos = sample_panoramas(manifest, qc, 2, 8, 4 * ROWS_PER_PANO)
    old = select_anchors(old_panos, 8, 8)
    panos, anchors = expanded_anchors(manifest, qc, old, 6, 8, 8, 4 * ROWS_PER_PANO)
    assert len(panos) == 6 and len(anchors) == 48
    assert len({r['source_pano'] for r in panos}) == 6
    assert all(r['filtered_pano'] >= 4 for r in panos)
    assert all(row in anchors for row in old)
    assert all(sum(r['heading'] == h for r in anchors) == 12 for h in (0, 90, 180, 270))
