import numpy as np

from scripts.multicity.build_semantic_parent_child_hierarchy import (
    DIMENSIONS,
    PARENT_CLASSES,
    assign_semantic_parents,
    bh_qvalues,
    choose_child_cut,
)


def test_bh_qvalues_are_bounded_and_monotone_by_rank():
    p = np.array([0.20, 0.001, 0.04, 0.01])
    q = bh_qvalues(p)
    assert q.shape == p.shape
    assert np.all((0 <= q) & (q <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all(q >= p)


def test_parent_assignment_is_complete_and_supports_secondary_membership():
    groups = list(PARENT_CLASSES.values())
    profiles = np.zeros((DIMENSIONS, 65), dtype=np.float64)
    prevalence = np.zeros(65, dtype=np.float64)
    for group in groups:
        prevalence[group] = 1 / (len(groups) * len(group))
    for dimension in range(DIMENSIONS):
        parent = dimension % len(groups)
        profiles[dimension, groups[parent][0]] = 1.0
    # D000 has strong evidence for two parents, which should retain an audit
    # secondary membership while keeping exactly one primary parent.
    profiles[0] = 0
    profiles[0, groups[0][0]] = 0.55
    profiles[0, groups[1][0]] = 0.45
    frame, primary, shares = assign_semantic_parents(profiles, prevalence)
    assert primary.shape == (DIMENSIONS,)
    assert shares.shape == (DIMENSIONS, len(groups))
    assert frame.loc[frame.membership.eq("primary")].dimension_id.nunique() == DIMENSIONS
    assert len(frame.loc[frame.dimension_id.eq(0)]) == 2


def test_child_cut_finds_separated_blocks():
    similarity = np.full((8, 8), 0.05)
    similarity[:4, :4] = 0.95
    similarity[4:, 4:] = 0.95
    np.fill_diagonal(similarity, 1)
    labels, audit = choose_child_cut(1 - similarity)
    assert len(np.unique(labels)) >= 2
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    assert audit.selected.sum() == 1
