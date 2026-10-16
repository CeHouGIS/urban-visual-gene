"""Nonnegative convolutional dictionary with separable, multi-scale templates."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemplateBank(nn.Module):
    """B[m,d,y,x] = sum_r channels[m,r,d] * layout[m,r,y,x]."""

    def __init__(self, dimensions, count, size, rank):
        super().__init__()
        self.count, self.size, self.rank = count, size, rank
        self.channels = nn.Parameter(torch.rand(count, rank, dimensions))
        self.layout = nn.Parameter(torch.rand(count, rank, size, size))
        self.project_()

    @torch.no_grad()
    def project_(self):
        self.channels.clamp_(min=0)
        self.layout.clamp_(min=0)
        spatial_norm = self.layout.flatten(2).norm(dim=2).clamp_min(1e-10)
        self.layout.div_(spatial_norm[:, :, None, None])
        self.channels.mul_(spatial_norm[:, :, None])
        # Exact template norm, without materializing [M,D,S,S].
        gram_c = self.channels @ self.channels.transpose(1, 2)
        spatial = self.layout.flatten(2)
        gram_s = spatial @ spatial.transpose(1, 2)
        norm = (gram_c * gram_s).sum((1, 2)).clamp_min(1e-20).sqrt()
        self.channels.div_(norm[:, None, None])

    def forward(self, coefficients):
        batch, _, height, width = coefficients.shape
        repeated = coefficients[:, :, None].expand(-1, -1, self.rank, -1, -1)
        spread = F.conv_transpose2d(
            repeated.reshape(batch, self.count * self.rank, height, width),
            self.layout.reshape(self.count * self.rank, 1, self.size, self.size),
            padding=self.size // 2, groups=self.count * self.rank,
        )
        mixing = self.channels.permute(2, 0, 1).reshape(-1, self.count * self.rank, 1, 1)
        return F.conv2d(spread, mixing)

    def templates(self):
        return torch.einsum('mrd,mryx->mdyx', self.channels, self.layout)


class PrototypeAutoencoder(nn.Module):
    def __init__(self, dimensions=4096, scales=(3, 7, 15), per_scale=32,
                 rank=4, hidden=64):
        super().__init__()
        self.dimensions = dimensions
        self.scales = tuple(scales)
        self.per_scale = per_scale
        self.config = dict(dimensions=dimensions, scales=list(scales),
                           per_scale=per_scale, rank=rank, hidden=hidden)
        # The extra channel distinguishes a hidden patch from genuine zeros.
        self.encoder = nn.Sequential(nn.Conv2d(dimensions + 1, hidden, 1), nn.ReLU())
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, size, padding=size // 2, groups=hidden),
                          nn.ReLU(), nn.Conv2d(hidden, per_scale, 1))
            for size in scales
        ])
        self.banks = nn.ModuleList([
            TemplateBank(dimensions, per_scale, size, rank) for size in scales
        ])
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=.02)
            nn.init.constant_(head[-1].bias, .03)

    def forward(self, target, hidden_mask=None):
        if hidden_mask is None:
            hidden_mask = torch.zeros_like(target[:, :1], dtype=torch.bool)
        observed = target.masked_fill(hidden_mask, 0)
        encoded = self.encoder(torch.cat([observed, hidden_mask.to(target.dtype)], 1))
        maps = [F.relu(head(encoded)) for head in self.heads]
        reconstruction = sum(bank(c) for bank, c in zip(self.banks, maps))
        return reconstruction, torch.cat(maps, 1)

    @torch.no_grad()
    def project_(self):
        for bank in self.banks:
            bank.project_()

    def composition_penalty(self):
        return torch.stack([b.channels.abs().mean() for b in self.banks]).mean()


def balanced_loss(prediction, target, coefficients, hidden_mask=None,
                  zero_weight=32., sparsity=.01, masked_weight=1.):
    """Mean errors over positive/zero targets, plus an explicit response L1.

    Avoid masked_select: CUDA compaction of the mostly-zero tensor can allocate
    an int64 index buffer much larger than the original activations. Masking in
    place on a temporary keeps the same objective and its gradients, with only
    one dense float32 temporary per reduction.
    """
    difference = (prediction.float() - target.float()).square()
    positive = target > 0
    count = positive.sum().clamp_min(1)
    positive_error = difference.masked_fill(~positive, 0).sum() / count
    zero_count = positive.numel() - positive.sum()
    zero_error = difference.masked_fill(positive, 0).sum() / zero_count.clamp_min(1)
    hidden_error = difference.sum() * 0
    if hidden_mask is not None:
        hidden_positive = positive & hidden_mask
        hidden_error = difference.masked_fill(~hidden_positive, 0).sum() / hidden_positive.sum().clamp_min(1)
    response_l1 = coefficients.float().mean()
    loss = positive_error + zero_weight * zero_error + sparsity * response_l1 + masked_weight * hidden_error
    metrics = dict(positive_mse=float(positive_error.detach()), zero_mse=float(zero_error.detach()),
                   masked_positive_mse=float(hidden_error.detach()), response_l1=float(response_l1.detach()),
                   energy_nmse=float((difference.sum() / target.float().square().sum().clamp_min(1e-12)).detach()))
    return loss, metrics


def spatial_shuffle(x, generator):
    """One independent permutation per view, shared by all its dimensions."""
    batch, dimensions, height, width = x.shape
    order = torch.rand(batch, height * width, device=x.device, generator=generator).argsort(1)
    return x.flatten(2).gather(2, order[:, None].expand(-1, dimensions, -1)).reshape_as(x)


def feature_shuffle(x, generator):
    """Independently reassign each dimension's entire map across panoramas.

    Call with a single city and heading at a time. This exactly preserves each
    feature's empirical maps within that batch; it removes their scene pairing.
    """
    batch, dimensions = x.shape[:2]
    order = torch.rand(batch, dimensions, device=x.device, generator=generator).argsort(0)
    channels = torch.arange(dimensions, device=x.device)[None].expand(batch, -1)
    return x[order, channels]
