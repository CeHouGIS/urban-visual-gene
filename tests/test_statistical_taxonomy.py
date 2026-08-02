import numpy as np

from formal.interpretation.build_statistical_taxonomy import (
    bh_fdr,
    canonical_labels,
    choose_cut,
    rank_fuse,
)


def test_bh_fdr_is_bounded_and_monotonic_by_pvalue():
    p = np.array([0.04, 0.001, 0.2, 0.01])
    q = bh_fdr(p)
    assert np.all((q >= 0) & (q <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_rank_fusion_preserves_shared_neighbours():
    first = np.eye(5, dtype=np.float32)
    second = np.eye(5, dtype=np.float32)
    first[0, 1] = first[1, 0] = 0.9
    second[0, 1] = second[1, 0] = 0.8
    fused = rank_fuse([first, second], neighbours=2)
    assert np.allclose(fused, fused.T)
    assert fused[0, 1] == fused.max()
    assert np.all(np.diag(fused) == 0)


def test_choose_cut_finds_separated_groups():
    rng = np.random.default_rng(2)
    embedding = np.vstack([
        rng.normal(-3, 0.1, (10, 3)),
        rng.normal(0, 0.1, (10, 3)),
        rng.normal(3, 0.1, (10, 3)),
    ])
    count, labels, scores = choose_cut(embedding, [2, 3, 4])
    assert count == 3
    assert len(set(labels)) == 3
    assert scores[3] > scores[2]


def test_canonical_labels_follow_first_gene_order():
    labels = np.array([9, 9, 3, 3, 7])
    assert canonical_labels(labels).tolist() == [0, 0, 1, 1, 2]
