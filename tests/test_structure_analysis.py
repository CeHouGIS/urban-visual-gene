import json
from collections import Counter
from pathlib import Path

from formal.interpretation.analyze_statistical_structure import normalized_entropy


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "formal" / "site" / "w1024_statistical" / "structure_analysis.json"


def test_normalized_entropy_handles_pure_and_balanced_branches():
    assert normalized_entropy(Counter({"stable": 8})) == 0.0
    assert abs(normalized_entropy(Counter({"stable": 4, "nuisance": 4})) - 1.0) < 1e-12


def test_published_structure_analysis_is_internally_consistent():
    data = json.loads(ANALYSIS.read_text())
    summary = data["summary"]

    assert summary["n_genes"] == 1024
    assert len(data["gene_evidence"]) == 1024
    assert len([row for row in data["branches"] if row["level"] == "coarse"]) == 32
    assert len([row for row in data["branches"] if row["level"] == "fine"]) == 64
    assert sum(summary["branch_status_by_level"]["fine"].values()) == 64

    equivalence = summary["equivalence"]
    strict_groups = [row for row in data["equivalence_groups"] if row["strict_collapse_candidate"]]
    assert equivalence["strict_groups"] == len(strict_groups)
    assert equivalence["strict_effective_dictionary_size"] == 932

    specialization = summary["specialization"]
    assert specialization["direct_edges"] == len(data["specialization_direct_edges"])
    assert specialization["direct_edges"] == 296
    assert specialization["longest_chain_edges"] == 3
