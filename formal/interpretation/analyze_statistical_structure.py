"""Audit W1024 branches, equivalence groups, and specialization structure.

This analysis is annotation-free. It summarizes independent evidence axes
instead of assigning natural-language semantics or collapsing them into one
opaque score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "formal" / "site" / "w1024_statistical"
DEFAULT_OUTPUT = DEFAULT_INPUT
STABLE_ROLES = {"stable_structured", "rare_stable_component"}
NUISANCE_ROLES = {"position_component", "quality_component", "unstable_component"}


def percentile(values: list[float]) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average") - 1
    return ranks / max(len(values) - 1, 1)


def normalized_entropy(counts: Counter[str]) -> float:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if len(values) <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(values)))


def choose_canonical(members: list[int], metrics: dict[int, dict]) -> int:
    """Choose a representative lexicographically, without a weighted score."""
    def key(gene: int) -> tuple:
        row = metrics[gene]
        nuisance = max(
            row["position_component_probability"],
            row["quality_component_probability"],
            row["unstable_probability"],
        )
        return (
            nuisance < 0.8,
            row["split_stability"],
            row["image_support"],
            row["nearest_fused_similarity"],
            -gene,
        )
    return max(members, key=key)


def equivalence_analysis(
    metrics: dict[int, dict], relations: list[dict], fused: np.ndarray,
) -> tuple[list[dict], dict[int, tuple[str, int]]]:
    graph = nx.Graph()
    graph.add_edges_from(
        (int(row["source"]), int(row["target"]))
        for row in relations if row["relation"] == "equivalent_to"
    )
    components = sorted(nx.connected_components(graph), key=lambda x: (-len(x), min(x)))
    groups = []
    membership: dict[int, tuple[str, int]] = {}
    for index, component in enumerate(components):
        members = sorted(int(gene) for gene in component)
        group_id = f"E{index:03d}"
        canonical = choose_canonical(members, metrics)
        for gene in members:
            membership[gene] = (group_id, canonical)
        pair_values = [
            float(fused[a, b]) for offset, a in enumerate(members) for b in members[offset + 1:]
        ]
        roles = Counter(metrics[gene]["statistical_role"] for gene in members)
        cross_fine = len({metrics[gene]["fine_cluster"] for gene in members}) > 1
        min_pair_fused = float(np.min(pair_values))
        groups.append({
            "group_id": group_id,
            "canonical_gene": canonical,
            "members": members,
            "n_members": len(members),
            "roles": dict(sorted(roles.items())),
            "coarse_clusters": sorted({metrics[gene]["coarse_cluster"] for gene in members}),
            "fine_clusters": sorted({metrics[gene]["fine_cluster"] for gene in members}),
            "cross_fine": cross_fine,
            "strict_collapse_candidate": not cross_fine and min_pair_fused >= 0.4,
            "stable_fraction": sum(metrics[gene]["statistical_role"] in STABLE_ROLES for gene in members) / len(members),
            "nuisance_fraction": sum(metrics[gene]["statistical_role"] in NUISANCE_ROLES for gene in members) / len(members),
            "mean_pair_fused": float(np.mean(pair_values)),
            "min_pair_fused": min_pair_fused,
        })
    return groups, membership


def gene_evidence(
    metrics: dict[int, dict], membership: dict[int, tuple[str, int]],
) -> dict[int, dict]:
    genes = sorted(metrics)
    stability_rank = percentile([metrics[gene]["split_stability"] for gene in genes])
    support_rank = percentile([math.log1p(metrics[gene]["image_support"]) for gene in genes])
    neighbour_rank = percentile([metrics[gene]["nearest_fused_similarity"] for gene in genes])
    adjacency_rank = percentile([metrics[gene]["self_adjacency_rate"] for gene in genes])
    moran_rank = percentile([metrics[gene]["spatial_morans_i"] for gene in genes])
    output = {}
    for offset, gene in enumerate(genes):
        row = metrics[gene]
        reliability = math.sqrt(stability_rank[offset] * support_rank[offset])
        structure = float(np.mean([
            neighbour_rank[offset], adjacency_rank[offset], moran_rank[offset],
        ]))
        nuisance = max(
            row["position_component_probability"],
            row["quality_component_probability"],
            row["unstable_probability"],
        )
        specificity = max(row["city_specific_probability"], row["rare_probability"])
        equivalent_group, canonical = membership.get(gene, (None, gene))
        if row["position_component_probability"] >= 0.8 or row["quality_component_probability"] >= 0.8:
            status = "nuisance_component"
        elif row["unstable_probability"] >= 0.8:
            status = "unstable_evidence"
        elif equivalent_group and gene != canonical:
            status = "redundant_variant"
        elif reliability >= 0.6 and specificity >= 0.8:
            status = "robust_specific"
        elif reliability >= 0.6 and structure >= 0.6:
            status = "robust_structured"
        elif reliability >= 0.45:
            status = "supported_mixed"
        else:
            status = "weak_evidence"
        output[gene] = {
            "gene_id": gene,
            "evidence_status": status,
            "reliability": reliability,
            "structure": structure,
            "nuisance_risk": nuisance,
            "specificity": specificity,
            "stability_percentile": float(stability_rank[offset]),
            "support_percentile": float(support_rank[offset]),
            "equivalent_group": equivalent_group,
            "canonical_gene": canonical,
        }
    return output


def branch_analysis(
    metrics: dict[int, dict], evidence: dict[int, dict], relations: list[dict],
    taxonomy: dict,
) -> list[dict]:
    nodes = []
    for coarse in taxonomy["children"]:
        nodes.append(("coarse", coarse, [gene for fine in coarse["children"] for gene in fine["genes"]]))
        nodes.extend(("fine", fine, fine["genes"]) for fine in coarse["children"])

    relation_sets: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in relations:
        relation_sets[row["relation"]].append((int(row["source"]), int(row["target"])))
    branches = []
    for level, node, raw_genes in nodes:
        genes = [int(gene) for gene in raw_genes]
        gene_set = set(genes)
        roles = Counter(metrics[gene]["statistical_role"] for gene in genes)
        evidence_counts = Counter(evidence[gene]["evidence_status"] for gene in genes)
        stable_fraction = sum(metrics[gene]["statistical_role"] in STABLE_ROLES for gene in genes) / len(genes)
        nuisance_fraction = sum(metrics[gene]["statistical_role"] in NUISANCE_ROLES for gene in genes) / len(genes)
        city_fraction = roles["city_specific_component"] / len(genes)
        dominant_role, dominant_count = roles.most_common(1)[0]
        if nuisance_fraction >= 0.5:
            status = "nuisance_dominant"
        elif stable_fraction >= 0.5 and nuisance_fraction < 0.3:
            status = "stable_dominant"
        elif city_fraction >= 0.5:
            status = "city_specific_dominant"
        else:
            status = "mixed"
        possible = max(len(genes) * (len(genes) - 1) / 2, 1)
        internal_counts = {
            relation: sum(a in gene_set and b in gene_set for a, b in edges)
            for relation, edges in relation_sets.items()
        }
        branches.append({
            "branch_id": node["id"],
            "level": level,
            "n_genes": len(genes),
            "genes": genes,
            "medoid_gene": int(node["medoid_gene"]),
            "branch_stability": float(node["stability"]),
            "branch_status": status,
            "dominant_role": dominant_role,
            "role_purity": dominant_count / len(genes),
            "role_entropy": normalized_entropy(roles),
            "role_counts": dict(sorted(roles.items())),
            "evidence_counts": dict(sorted(evidence_counts.items())),
            "stable_fraction": stable_fraction,
            "nuisance_fraction": nuisance_fraction,
            "city_specific_fraction": city_fraction,
            "weak_fraction": roles["weakly_structured"] / len(genes),
            "mean_reliability": float(np.mean([evidence[gene]["reliability"] for gene in genes])),
            "mean_structure": float(np.mean([evidence[gene]["structure"] for gene in genes])),
            "mean_nuisance_risk": float(np.mean([evidence[gene]["nuisance_risk"] for gene in genes])),
            "median_image_support": float(np.median([metrics[gene]["image_support"] for gene in genes])),
            "granularity_counts": dict(sorted(Counter(
                metrics[gene]["granularity_level"] for gene in genes
            ).items())),
            "internal_relations": internal_counts,
            "internal_relation_density": {
                relation: count / possible for relation, count in internal_counts.items()
            },
        })
    return branches


def specialization_analysis(
    metrics: dict[int, dict], relations: list[dict],
) -> tuple[dict, list[dict], list[dict]]:
    rows = [row for row in relations if row["relation"] == "specializes"]
    graph = nx.DiGraph()
    edge_lookup = {}
    for row in rows:
        child, parent = int(row["source"]), int(row["target"])
        graph.add_edge(child, parent)
        edge_lookup[(child, parent)] = row
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("specialization graph is not a DAG")
    reduced = nx.transitive_reduction(graph)
    direct_edges = []
    for child, parent in sorted(reduced.edges()):
        row = edge_lookup[(child, parent)]
        child_level = int(metrics[child]["granularity_level"][1:])
        parent_level = int(metrics[parent]["granularity_level"][1:])
        direct_edges.append({
            **row,
            "child_gene": child,
            "parent_gene": parent,
            "child_level": child_level,
            "parent_level": parent_level,
            "level_delta": child_level - parent_level,
            "child_image_support": metrics[child]["image_support"],
            "parent_image_support": metrics[parent]["image_support"],
            "support_ratio": metrics[child]["image_support"] / max(metrics[parent]["image_support"], 1),
        })
    weak_components = sorted(nx.weakly_connected_components(reduced), key=lambda x: (-len(x), min(x)))
    components = []
    for index, members_raw in enumerate(weak_components):
        members = sorted(int(gene) for gene in members_raw)
        subgraph = reduced.subgraph(members).copy()
        path = nx.dag_longest_path(subgraph)
        components.append({
            "component_id": f"S{index:03d}",
            "members": members,
            "n_members": len(members),
            "n_direct_edges": subgraph.number_of_edges(),
            "general_roots": sorted(gene for gene in members if subgraph.out_degree(gene) == 0),
            "specific_leaves": sorted(gene for gene in members if subgraph.in_degree(gene) == 0),
            "longest_path": [int(gene) for gene in path],
            "longest_path_edges": max(len(path) - 1, 0),
            "fine_clusters": sorted({metrics[gene]["fine_cluster"] for gene in members}),
        })
    delta_counts = Counter(
        "child_more_local" if row["level_delta"] < 0 else
        "same_scale" if row["level_delta"] == 0 else "child_more_global"
        for row in direct_edges
    )
    summary = {
        "nodes": graph.number_of_nodes(),
        "original_edges": graph.number_of_edges(),
        "direct_edges": reduced.number_of_edges(),
        "implied_edges_removed": graph.number_of_edges() - reduced.number_of_edges(),
        "components": len(components),
        "longest_chain_edges": nx.dag_longest_path_length(reduced),
        "longest_chain": [int(gene) for gene in nx.dag_longest_path(reduced)],
        "granularity_direction": dict(sorted(delta_counts.items())),
    }
    return summary, direct_edges, components


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def write_csv(rows: list[dict], file: Path, fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0])
    with file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(input_dir: Path, output_dir: Path) -> dict:
    metrics = {
        int(gene): row for gene, row in json.loads((input_dir / "gene_metrics.json").read_text()).items()
    }
    relations = json.loads((input_dir / "gene_relations.json").read_text())
    taxonomy = json.loads((input_dir / "taxonomy.json").read_text())
    with np.load(input_dir / "statistical_arrays.npz") as arrays:
        fused = arrays["fused_similarity"].astype(np.float32)

    equivalence_groups, membership = equivalence_analysis(metrics, relations, fused)
    evidence = gene_evidence(metrics, membership)
    branches = branch_analysis(metrics, evidence, relations, taxonomy)
    specialization_summary, direct_edges, specialization_components = specialization_analysis(metrics, relations)

    evidence_counts = Counter(row["evidence_status"] for row in evidence.values())
    branch_counts = Counter(row["branch_status"] for row in branches)
    branch_counts_by_level = {
        level: dict(sorted(Counter(
            row["branch_status"] for row in branches if row["level"] == level
        ).items()))
        for level in ("coarse", "fine")
    }
    strict_groups = [group for group in equivalence_groups if group["strict_collapse_candidate"]]
    summary = {
        "n_genes": len(metrics),
        "evidence_status_counts": dict(sorted(evidence_counts.items())),
        "branch_status_counts": dict(sorted(branch_counts.items())),
        "branch_status_by_level": branch_counts_by_level,
        "equivalence": {
            "groups": len(equivalence_groups),
            "genes_in_groups": sum(group["n_members"] for group in equivalence_groups),
            "groups_larger_than_two": sum(group["n_members"] > 2 for group in equivalence_groups),
            "cross_fine_groups": sum(group["cross_fine"] for group in equivalence_groups),
            "candidate_effective_dictionary_size": len(metrics) - sum(
                group["n_members"] - 1 for group in equivalence_groups
            ),
            "strict_groups": len(strict_groups),
            "strict_removable_variants": sum(group["n_members"] - 1 for group in strict_groups),
            "strict_effective_dictionary_size": len(metrics) - sum(
                group["n_members"] - 1 for group in strict_groups
            ),
        },
        "specialization": specialization_summary,
    }
    payload = round_floats({
        "method": {
            "labels": "none",
            "language_model": "none",
            "evidence_axes": ["reliability", "structure", "nuisance_risk", "specificity", "redundancy"],
            "canonical_selection": "lexicographic: non-nuisance, stability, support, fused neighbour",
            "strict_equivalence": "same fine branch and minimum all-pairs fused similarity >= 0.4",
        },
        "summary": summary,
        "gene_evidence": {str(gene): row for gene, row in evidence.items()},
        "branches": branches,
        "equivalence_groups": equivalence_groups,
        "specialization_direct_edges": direct_edges,
        "specialization_components": specialization_components,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure_analysis.json").write_text(json.dumps(payload, ensure_ascii=False))
    write_csv(list(payload["gene_evidence"].values()), output_dir / "gene_evidence.csv")
    write_csv(payload["branches"], output_dir / "branch_audit.csv", [
        "branch_id", "level", "n_genes", "medoid_gene", "branch_stability", "branch_status",
        "dominant_role", "role_purity", "role_entropy", "stable_fraction", "nuisance_fraction",
        "city_specific_fraction", "weak_fraction", "mean_reliability", "mean_structure",
        "mean_nuisance_risk", "median_image_support",
    ])
    write_csv(payload["equivalence_groups"], output_dir / "equivalence_groups.csv", [
        "group_id", "canonical_gene", "n_members", "cross_fine", "strict_collapse_candidate", "stable_fraction",
        "nuisance_fraction", "mean_pair_fused", "min_pair_fused",
    ])
    write_csv(payload["specialization_direct_edges"], output_dir / "specialization_direct_edges.csv")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
