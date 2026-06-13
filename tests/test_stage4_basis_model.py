import numpy as np
import pytest
import torch

from scripts.road_basis_model import RoadBasisAutoEncoder, cosine_recon_loss
from scripts.stage4_train_road_basis_model import train_basis_model


def test_model_instantiation():
    model = RoadBasisAutoEncoder(D=256, K=16, hidden=64)
    assert model is not None


def test_forward_shape():
    model = RoadBasisAutoEncoder(D=256, K=16, hidden=64)
    model.eval()
    z = torch.randn(8, 256)
    a, z_hat = model(z)
    assert a.shape    == (8, 16)
    assert z_hat.shape == (8, 256)


def test_activation_nonnegative():
    model = RoadBasisAutoEncoder(D=256, K=16, hidden=64)
    model.eval()
    z = torch.randn(32, 256)
    a, _ = model(z)
    assert float(a.min()) >= -1e-6, f"A has negative values: {float(a.min())}"


def test_basis_shape():
    model = RoadBasisAutoEncoder(D=256, K=16, hidden=64)
    assert model.basis.shape == (16, 256)


def test_basis_row_normalized():
    model = RoadBasisAutoEncoder(D=256, K=16, hidden=64)
    model._normalise_basis()
    norms = model.basis.norm(dim=1)
    assert torch.allclose(norms, torch.ones(16), atol=1e-3), \
        f"max norm dev: {float((norms-1).abs().max()):.4f}"


def test_training_smoke(synthetic_road_context_features, synthetic_road_graph_edges,
                        tmp_path):
    model, report = train_basis_model(
        synthetic_road_context_features,
        synthetic_road_graph_edges,
        K=8, hidden=32, epochs=3,
        lambda_sparse=1e-4, lambda_spatial=1e-4, lambda_div=1e-4,
        output_dir=tmp_path,
    )
    assert "final_train_loss" in report
    assert (tmp_path / "road_basis_model.pt").exists()
    assert (tmp_path / "road_landscape_basis.npy").exists()


def test_loss_decreases(synthetic_road_context_features, synthetic_road_graph_edges,
                        tmp_path):
    _, report = train_basis_model(
        synthetic_road_context_features,
        synthetic_road_graph_edges,
        K=8, hidden=32, epochs=30,
        output_dir=tmp_path,
    )
    assert report["final_train_loss"] < report["initial_train_loss"] * 0.99, (
        f"loss did not decrease: {report['initial_train_loss']:.4f} "
        f"→ {report['final_train_loss']:.4f}"
    )
