import numpy as np


def test_output_schema(synthetic_pano_features):
    df = synthetic_pano_features["panos"]
    required = {"pano_id", "city", "lat", "lon", "road_id",
                "pano_embedding", "feature_model"}
    assert required.issubset(df.columns)


def test_pano_id_unique(synthetic_pano_features):
    df = synthetic_pano_features["panos"]
    assert df["pano_id"].nunique() == len(df)


def test_embedding_dim_is_4d(synthetic_pano_features):
    D  = synthetic_pano_features["D"]          # 4 * D_SINGLE
    emb = synthetic_pano_features["panos"]["pano_embedding"].iloc[0]
    assert len(emb) == D


def test_l2_normalization(synthetic_pano_features):
    embs = np.stack(synthetic_pano_features["panos"]["pano_embedding"].values)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"max deviation: {abs(norms - 1).max():.6f}"


def test_lat_lon_range(synthetic_pano_features):
    df = synthetic_pano_features["panos"]
    assert df["lat"].between(-90, 90).all()
    assert df["lon"].between(-180, 180).all()


def test_intra_region_similarity_higher(synthetic_pano_features):
    embs   = np.stack(synthetic_pano_features["panos"]["pano_embedding"].values)
    labels = synthetic_pano_features["region_labels"]
    r0 = embs[labels == 0]
    r1 = embs[labels == 1]
    intra = float((r0 @ r0.T).mean())
    inter = float((r0 @ r1.T).mean())
    assert intra > inter + 0.10, \
        f"intra={intra:.3f} should be > inter={inter:.3f} + 0.10"
