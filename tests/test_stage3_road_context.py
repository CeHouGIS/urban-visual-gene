import numpy as np
import pytest

from scripts.stage3_build_road_context_features import build_road_context_features


@pytest.fixture(scope="module")
def context_result(synthetic_road_matched_panos, synthetic_road_graph_nodes,
                   synthetic_road_graph_edges):
    ctx, report = build_road_context_features(
        synthetic_road_matched_panos,
        synthetic_road_graph_nodes,
        synthetic_road_graph_edges,
        search_radius_m=200.0,   # generous for synthetic data
        kernel_sigma_m=50.0,
        context_radius_m=100.0,
    )
    return ctx, report


def test_schema(context_result):
    ctx, _ = context_result
    required = {"road_node_id", "road_id", "lat", "lon",
                "road_context_embedding", "n_panos", "total_weight",
                "aggregation_method"}
    assert required.issubset(ctx.columns)


def test_all_nodes_have_embedding(context_result, synthetic_road_graph_nodes):
    ctx, _ = context_result
    assert len(ctx) == len(synthetic_road_graph_nodes)


def test_embedding_l2_normed(context_result):
    ctx, _ = context_result
    embs = np.stack(ctx["road_context_embedding"].values)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"max deviation: {abs(norms - 1).max():.6f}"


def test_intra_region_similarity_higher(context_result, synthetic_road_graph_nodes):
    ctx, _ = context_result
    # road_ids 'road_00','road_01' → region 0; 'road_02','road_03' → region 1
    embs = np.stack(ctx["road_context_embedding"].values)
    road_col = ctx["road_id"].values

    r0_mask = np.array([r in {"road_00", "road_01"} for r in road_col])
    r1_mask = np.array([r in {"road_02", "road_03"} for r in road_col])

    if r0_mask.sum() < 2 or r1_mask.sum() < 2:
        pytest.skip("Not enough region-0/1 nodes in synthetic data")

    r0 = embs[r0_mask]
    r1 = embs[r1_mask]
    intra = float((r0 @ r0.T).mean())
    inter = float((r0 @ r1.T).mean())
    assert intra > inter + 0.05, \
        f"intra={intra:.3f} should be > inter={inter:.3f}+0.05"


def test_n_panos_non_negative(context_result):
    ctx, _ = context_result
    assert (ctx["n_panos"] >= 0).all()


def test_coverage_ratio_in_report(context_result):
    _, report = context_result
    assert "coverage_ratio" in report
    assert report["coverage_ratio"] >= 0.0
