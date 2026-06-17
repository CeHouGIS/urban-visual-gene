"""Sparse Autoencoder for road landscape basis learning.

Z_road ≈ A X
  A ∈ R^{M×K}  activation (non-negative, sparse)
  X ∈ R^{K×D}  basis (row-normalised)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadBasisAutoEncoder(nn.Module):
    """Sparse autoencoder with linear decoder (basis = decoder weights).

    Parameters
    ----------
    D       : input / output dimension (road_context_embedding dim)
    K       : number of basis vectors
    hidden  : encoder hidden dim
    """

    def __init__(self, D: int, K: int, hidden: int = 512):
        super().__init__()
        self.D = D
        self.K = K

        # Encoder: D → hidden → K, final ReLU → A ≥ 0 with EXACT zeros.
        # (Softplus was used previously but never produces 0, so activations were
        # dense and near-identical between regions, collapsing the Stage-6
        # boundary threshold τ_c to 0. ReLU gives genuine sparsity.)
        self.encoder = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
            nn.ReLU(),   # ensures A ≥ 0, exact zeros
        )

        # Decoder: K → D  (no bias — linear combination of basis vectors)
        self.decoder = nn.Linear(K, D, bias=False)

        # Initialise basis rows as unit vectors
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight)
            self._normalise_basis()

    def _normalise_basis(self) -> None:
        """Row-normalise decoder weight matrix (X rows = unit vectors)."""
        with torch.no_grad():
            w = self.decoder.weight          # (D, K) — pytorch stores transposed
            # weight shape is (out, in) = (D, K), rows are output dims
            # basis vectors are columns of weight (each column = x_k ∈ R^D)
            norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.copy_(w / norms)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : (B, D)  road context embeddings

        Returns
        -------
        a     : (B, K)  activation (non-negative)
        z_hat : (B, D)  reconstruction
        """
        a     = self.encoder(z)        # (B, K), ≥ 0
        z_hat = self.decoder(a)        # (B, D)
        return a, z_hat

    @property
    def basis(self) -> torch.Tensor:
        """X : (K, D)  — each row is one basis vector."""
        return self.decoder.weight.T   # weight is (D, K), transpose → (K, D)


def cosine_recon_loss(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cosine_similarity) reconstruction loss."""
    return (1.0 - F.cosine_similarity(z, z_hat, dim=1)).mean()


def diversity_loss(model: RoadBasisAutoEncoder) -> torch.Tensor:
    """||X X^T - I||_F^2  — penalises redundant bases."""
    X = model.basis                         # (K, D)
    G = X @ X.T                            # (K, K)  gram matrix
    I = torch.eye(model.K, device=X.device)
    return ((G - I) ** 2).sum()


def spatial_smoothness_loss(
    a: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
) -> torch.Tensor:
    """Sum_{(i,j) in E} ||a_i - a_j||^2 — keeps A smooth along road graph."""
    if edge_src.numel() == 0:
        return torch.tensor(0.0, device=a.device)
    a_src = a[edge_src]
    a_dst = a[edge_dst]
    return ((a_src - a_dst) ** 2).sum(dim=1).mean()
