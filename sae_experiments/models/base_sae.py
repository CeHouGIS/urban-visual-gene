"""Small, dependency-free sparse autoencoders used by the DINOv3 pipeline.

The BatchTopK implementation applies the sparsity budget to the flattened
batch.  Consequently a row may receive fewer or more than ``k`` active
latents, while the batch average is at most ``k`` (and equals it when enough
positive preactivations are available).  This is different from selecting
``k`` latents independently for every row.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, decoder_unit_norm: bool = True):
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.k = 0
        self.b_pre = nn.Parameter(torch.zeros(self.input_dim))
        self.encoder = nn.Linear(self.input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=False)
        self.decoder_unit_norm = bool(decoder_unit_norm)
        nn.init.normal_(self.decoder.weight, std=1.0 / self.input_dim**0.5)
        with torch.no_grad():
            self.normalize_decoder_()
            self.encoder.weight.copy_(self.decoder.weight.t())
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        if self.decoder_unit_norm:
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            )

    # Historical callers use this shorter name.
    _norm = normalize_decoder_

    def _preactivate(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [batch, input_dim], got {tuple(x.shape)}")
        return self.encoder(x - self.b_pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return acts, self.decode(acts)


class BatchTopKSAE(_SparseAutoencoder):
    """Sparse autoencoder with a joint batch-level top-k operator."""

    def __init__(self, input_dim: int, latent_dim: int, k: int, decoder_unit_norm: bool = True):
        super().__init__(input_dim, latent_dim, decoder_unit_norm)
        if not 0 < k <= latent_dim:
            raise ValueError(f"k must be in [1, latent_dim], got {k}/{latent_dim}")
        self.k = int(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self._preactivate(x))
        flat = pre.reshape(-1)
        n_select = min(self.k * pre.shape[0], flat.numel())
        acts = torch.zeros_like(flat)
        if n_select:
            values, indices = torch.topk(flat, n_select, sorted=False)
            acts.scatter_(0, indices, values)
        return acts.reshape_as(pre)


class TopKSAE(_SparseAutoencoder):
    """Per-row Top-K SAE retained for explicit ablations and old checkpoints."""

    def __init__(self, input_dim: int, latent_dim: int, k: int, decoder_unit_norm: bool = True):
        super().__init__(input_dim, latent_dim, decoder_unit_norm)
        if not 0 < k <= latent_dim:
            raise ValueError(f"k must be in [1, latent_dim], got {k}/{latent_dim}")
        self.k = int(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self._preactivate(x))
        values, indices = torch.topk(pre, self.k, dim=-1, sorted=False)
        acts = torch.zeros_like(pre)
        return acts.scatter(-1, indices, values)


def build_sae(input_dim: int, config: dict[str, Any]) -> _SparseAutoencoder:
    """Construct an SAE from the repository YAML/CLI configuration format."""
    model_type = str(config.get("type", "batch_topk")).lower().replace("-", "_")
    latent = config.get("latent_dim", "auto")
    if latent in (None, "auto"):
        expansion = config.get("expansion_factor")
        if expansion is None:
            raise ValueError("latent_dim='auto' requires expansion_factor")
        latent = int(input_dim * int(expansion))
    latent = int(latent)
    k = int(config.get("k", config.get("topk", 8)))
    unit_norm = bool(config.get("decoder_unit_norm", True))
    if model_type in {"batch_topk", "batchtopk"}:
        return BatchTopKSAE(input_dim, latent, k, unit_norm)
    if model_type in {"topk", "per_row_topk"}:
        return TopKSAE(input_dim, latent, k, unit_norm)
    if model_type == "jumprelu":
        # JumpReLU is not part of the current paper method.  A regular Top-K
        # model is a deterministic fallback so old CLI paths remain usable.
        return TopKSAE(input_dim, latent, k, unit_norm)
    raise ValueError(f"unknown SAE model type: {model_type}")
