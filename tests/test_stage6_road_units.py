import numpy as np
import pytest
import networkx as nx

from scripts.road_basis_model import RoadBasisAutoEncoder
from scripts.stage4_train_road_basis_model import train_basis_model
from scripts.stage5_infer_road_basis_activation import infer_activation
from scripts.stage6_extract_road_units import extract_road_units
from scripts.road_graph_utils import build_nx_graph


@pytest.fixture(scope="module")
def pipeline_outputs(synthetic_road_context_features, synthetic_road_graph_nodes,
                     synthetic_road_graph_edges, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("s6")
    model, _ = train_basis_model(
        synthetic_road_context_features, synthetic_road_graph_edges,
        K=8, hidden=32, epochs=20, output_dir=tmp,
    )
    act_df, _ = infer_activation(synthetic_road_context_features, model)
    b_gdf, u_gdf, stats, report = extract_road_units(
        act_df, synthetic_road_graph_nodes, synthetic_road_graph_edges,
        boundary_quantile=0.90,
        min_road_nodes=2,
        min_road_length_m=0.0,
        min_panos=0,
    )
    return b_gdf, u_gdf, stats, report, act_df


def test_boundaries_are_linestrings(pipeline_outputs):
    b_gdf = pipeline_outputs[0]
    if len(b_gdf) == 0:
        pytest.skip("no boundary edges")
    non_null = b_gdf[b_gdf.geometry.notna()]
    assert (non_null.geometry.geom_type == "LineString").all()


def test_boundary_properties(pipeline_outputs):
    b_gdf = pipeline_outputs[0]
    if len(b_gdf) == 0:
        pytest.skip("no boundary edges")
    required = {"boundary_id", "src_node_id", "dst_node_id",
                "activation_distance", "boundary_score"}
    assert required.issubset(b_gdf.columns)


def test_units_extracted(pipeline_outputs):
    _, u_gdf, stats, _, _ = pipeline_outputs
    assert len(stats) >= 1, "no road landscape units extracted"


def test_at_least_two_units(pipeline_outputs):
    _, _, stats, _, _ = pipeline_outputs
    assert len(stats) >= 2, f"only {len(stats)} unit(s), expected ≥2"


def test_unit_min_road_nodes(pipeline_outputs):
    _, _, stats, _, _ = pipeline_outputs
    assert (stats["n_road_nodes"] >= 2).all()


def test_unit_id_unique(pipeline_outputs):
    _, _, stats, _, _ = pipeline_outputs
    assert stats["unit_id"].nunique() == len(stats)


def test_within_variance_nonneg(pipeline_outputs):
    _, _, stats, _, _ = pipeline_outputs
    assert (stats["mean_within_activation_variance"] >= 0).all()


def test_dominant_basis_id_valid(pipeline_outputs):
    _, _, stats, _, _ = pipeline_outputs
    K = 8
    assert stats["dominant_basis_id"].between(0, K - 1).all()


def test_boundary_ratio_in_report(pipeline_outputs):
    _, _, _, report, _ = pipeline_outputs
    ratio = report["boundary_ratio"]
    expected = 0.10
    assert abs(ratio - expected) < 0.15, \
        f"boundary_ratio={ratio:.3f} far from expected ~{expected}"
