"""End-to-end pipeline test on synthetic data. Must complete in < 60s."""
import pytest

from scripts.stage3_build_road_context_features import build_road_context_features
from scripts.stage4_train_road_basis_model import train_basis_model
from scripts.stage5_infer_road_basis_activation import infer_activation
from scripts.stage6_extract_road_units import extract_road_units


def test_e2e_pipeline(synthetic_road_matched_panos, synthetic_road_graph_nodes,
                      synthetic_road_graph_edges, tmp_path):
    # Stage 3
    ctx, r3 = build_road_context_features(
        synthetic_road_matched_panos,
        synthetic_road_graph_nodes,
        synthetic_road_graph_edges,
        search_radius_m=200.0,
        kernel_sigma_m=50.0,
    )
    assert len(ctx) == len(synthetic_road_graph_nodes)

    # Stage 4
    model, r4 = train_basis_model(
        ctx, synthetic_road_graph_edges,
        K=8, hidden=32, epochs=5,
        output_dir=tmp_path,
    )
    assert (tmp_path / "road_basis_model.pt").exists()

    # Stage 5
    act_df, r5 = infer_activation(ctx, model)
    assert len(act_df) == len(ctx)
    assert (act_df[[c for c in act_df.columns if c.startswith("a_")]].values.min()
            >= -1e-5)

    # Stage 6
    b_gdf, u_gdf, stats, r6 = extract_road_units(
        act_df, synthetic_road_graph_nodes, synthetic_road_graph_edges,
        boundary_quantile=0.90,
        min_road_nodes=2,
        min_road_length_m=0.0,
        min_panos=0,
    )

    assert len(stats) >= 2, f"E2E: only {len(stats)} unit(s)"

    # Save and verify files
    stats.to_csv(tmp_path / "unit_statistics.csv", index=False)
    assert (tmp_path / "unit_statistics.csv").exists()


def test_e2e_output_schemas(synthetic_road_matched_panos, synthetic_road_graph_nodes,
                             synthetic_road_graph_edges, tmp_path):
    ctx, _ = build_road_context_features(
        synthetic_road_matched_panos,
        synthetic_road_graph_nodes,
        synthetic_road_graph_edges,
        search_radius_m=200.0,
    )
    model, _ = train_basis_model(ctx, synthetic_road_graph_edges,
                                  K=8, hidden=32, epochs=3, output_dir=tmp_path)
    act_df, _ = infer_activation(ctx, model)

    # activation schema
    for col in ["road_node_id", "road_id", "lat", "lon",
                "reconstruction_error", "activation_entropy", "active_basis_count"]:
        assert col in act_df.columns, f"missing column: {col}"
    assert "a_000" in act_df.columns
