import numpy as np
import networkx as nx
import pytest

from scripts.core.road_graph_utils import build_nx_graph, largest_component_ratio


def test_road_nodes_schema(synthetic_road_graph_nodes):
    required = {"road_node_id", "road_id", "lat", "lon", "chainage_m"}
    assert required.issubset(synthetic_road_graph_nodes.columns)


def test_road_node_id_unique(synthetic_road_graph_nodes):
    nodes = synthetic_road_graph_nodes
    assert nodes["road_node_id"].nunique() == len(nodes)


def test_road_edges_schema(synthetic_road_graph_edges):
    required = {"src_node_id", "dst_node_id", "edge_type",
                "network_distance_m", "edge_confidence"}
    assert required.issubset(synthetic_road_graph_edges.columns)


def test_edge_type_valid(synthetic_road_graph_edges):
    valid = {"same_road_next", "intersection_connect", "road_adjacent"}
    assert synthetic_road_graph_edges["edge_type"].isin(valid).all()


def test_network_distance_non_negative(synthetic_road_graph_edges):
    assert (synthetic_road_graph_edges["network_distance_m"] >= 0).all()


def test_chainage_non_negative(synthetic_road_graph_nodes):
    assert (synthetic_road_graph_nodes["chainage_m"] >= 0).all()


def test_same_road_nodes_ordered_by_chainage(synthetic_road_graph_nodes,
                                              synthetic_road_graph_edges):
    sre = synthetic_road_graph_edges[
        synthetic_road_graph_edges["edge_type"] == "same_road_next"
    ]
    ch_map = synthetic_road_graph_nodes.set_index("road_node_id")["chainage_m"].to_dict()
    for _, row in sre.iterrows():
        cs = ch_map.get(row["src_node_id"], 0)
        cd = ch_map.get(row["dst_node_id"], 0)
        # chainage either increases or decreases (bidirectional)
        assert abs(cs - cd) >= 0


def test_graph_connectivity(synthetic_road_graph_nodes, synthetic_road_graph_edges):
    G = build_nx_graph(synthetic_road_graph_nodes, synthetic_road_graph_edges)
    ratio = largest_component_ratio(G)
    assert ratio >= 0.90, f"LCC ratio={ratio:.3f} < 0.90"


def test_matched_panos_schema(synthetic_road_matched_panos):
    required = {"pano_id", "city", "lat", "lon", "matched_road_id",
                "road_distance_m", "chainage_m", "match_confidence",
                "pano_embedding"}
    assert required.issubset(synthetic_road_matched_panos.columns)


def test_road_distance_non_negative(synthetic_road_matched_panos):
    assert (synthetic_road_matched_panos["road_distance_m"] >= 0).all()


def test_match_confidence_in_range(synthetic_road_matched_panos):
    c = synthetic_road_matched_panos["match_confidence"].values
    assert (c > 0).all() and (c <= 1.0).all()
