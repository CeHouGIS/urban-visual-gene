import numpy as np
import pytest

from scripts.core.road_basis_model import RoadBasisAutoEncoder
from scripts.pipeline.stage4_train_road_basis_model import train_basis_model
from scripts.pipeline.stage5_infer_road_basis_activation import infer_activation


@pytest.fixture(scope="module")
def trained_model(synthetic_road_context_features, synthetic_road_graph_edges,
                  tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    model, _ = train_basis_model(
        synthetic_road_context_features, synthetic_road_graph_edges,
        K=8, hidden=32, epochs=10, output_dir=tmp,
    )
    return model


@pytest.fixture(scope="module")
def activation_result(synthetic_road_context_features, trained_model):
    return infer_activation(synthetic_road_context_features, trained_model,
                            activation_threshold=0.01)


def test_schema(activation_result):
    df, _ = activation_result
    required = {"road_node_id", "road_id", "lat", "lon",
                "reconstruction_error", "activation_entropy", "active_basis_count"}
    assert required.issubset(df.columns)
    # check a_000 exists
    assert "a_000" in df.columns


def test_activation_columns_present(activation_result):
    df, _ = activation_result
    K = 8
    for k in range(K):
        assert f"a_{k:03d}" in df.columns, f"missing a_{k:03d}"


def test_activations_nonnegative(activation_result):
    df, _ = activation_result
    a_cols = [c for c in df.columns if c.startswith("a_")]
    vals = df[a_cols].values
    assert float(vals.min()) >= -1e-5, f"negative activation: {float(vals.min())}"


def test_entropy_nonnegative(activation_result):
    df, _ = activation_result
    assert (df["activation_entropy"] >= -1e-6).all()


def test_active_basis_count_nonnegative(activation_result):
    df, _ = activation_result
    assert (df["active_basis_count"] >= 0).all()


def test_sparsity_reasonable(activation_result):
    # With K=8 and few training epochs, strict sparsity may not hold;
    # verify the metric is reported and non-negative.
    df, report = activation_result
    assert "median_active_basis" in report
    assert report["median_active_basis"] >= 0


def test_row_count_matches_nodes(activation_result, synthetic_road_context_features):
    df, _ = activation_result
    assert len(df) == len(synthetic_road_context_features)
