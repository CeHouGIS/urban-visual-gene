import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage

from formal.interpretation.compare_hierarchy_cuts import CANDIDATES, cut_labels


ROOT = Path(__file__).resolve().parents[1]


def test_cut_labels_are_canonical_and_have_requested_count():
    points = np.array([[0.0], [0.1], [1.0], [1.1], [3.0], [3.1]])
    labels = cut_labels(linkage(points, method="ward"), 3)
    assert labels.tolist() == [0, 0, 1, 1, 2, 2]


def test_published_hierarchy_comparison_is_complete_and_consistent():
    file = ROOT / "formal" / "site" / "w1024_statistical" / "hierarchy_comparison.json"
    comparison = json.loads(file.read_text())
    rows = comparison["rows"]
    assert [row["k"] for row in rows] == CANDIDATES
    assert comparison["recommendation"]["coarse_k"] == 32
    assert comparison["recommendation"]["fine_k"] == 64
    assert comparison["recommendation"]["fragmentation_starts_at"] == 112
    assert all(0 <= row["split_jaccard"] <= 1 for row in rows)
