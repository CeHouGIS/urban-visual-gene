import importlib

import numpy as np

from scripts.multicity.graph_visual_vocabulary.utils import canonical_labels


build_graphs = importlib.import_module(
    "scripts.multicity.graph_visual_vocabulary.02_build_graphs"
)
select_scales = importlib.import_module(
    "scripts.multicity.graph_visual_vocabulary.04_select_stable_scales"
)
cooccurrence = importlib.import_module(
    "scripts.multicity.graph_visual_vocabulary.09_rerun_cooccurrence"
)


def test_canonical_labels_follow_first_node():
    labels = canonical_labels(np.array([9, 9, 2, 2, 7]))
    assert labels.tolist() == [0, 0, 1, 1, 2]


def test_local_knn_graph_is_symmetric_without_self_loops():
    similarity = np.array(
        [[1, .9, .2, .1], [.9, 1, .3, .2], [.2, .3, 1, .8], [.1, .2, .8, 1]],
        dtype=float,
    )
    graph = build_graphs.local_knn_graph(similarity, 1, 1e-12)
    assert (graph != graph.T).nnz == 0
    assert np.allclose(graph.diagonal(), 0)
    assert graph.nnz // 2 == 2
    assert np.isclose(graph.data.mean(), 1)


def test_plateau_requires_stability_and_bounded_k():
    import pandas as pd

    frame = pd.DataFrame(
        {"gamma": [.1, .15, .2, .25, .3], "mean_seed_nmi": [.95] * 5,
         "n_communities": [4, 4, 5, 9, 9]}
    )
    adjacent = np.array([.96, .94, .70, .96])
    result = select_scales.candidate_plateaus(frame, adjacent, .9, 3, 2)
    assert len(result) == 1
    assert result[0]["gamma_min"] == .1
    assert result[0]["gamma_max"] == .2


def test_ppmi_preserves_existing_smoothed_observed_expected_definition():
    counts = np.array([[5, 4, 0], [3, 2, 0], [0, 4, 5]], dtype=np.uint16)
    marginal, observed, ppmi = cooccurrence.statistics(counts, 2, 1)
    expected = marginal[:, None] * marginal[None, :] / len(counts)
    reference = np.maximum(np.log((observed + .5) / (expected + .5)), 0)
    np.fill_diagonal(reference, 0)
    assert np.allclose(ppmi, reference)
