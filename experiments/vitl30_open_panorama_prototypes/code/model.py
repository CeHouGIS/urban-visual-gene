"""Whole-ring nonnegative dictionary and rotation-tied sparse NNLS codes."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def rotated_bank(templates):
    if templates.ndim != 4 or templates.shape[-1] % 4:
        raise ValueError('Expected [templates,channels,height,four-heading width]')
    return torch.stack([torch.roll(templates, r * (templates.shape[-1] // 4), -1)
                        for r in range(4)], 1).flatten(0, 1).flatten(1)

@torch.no_grad()
def sparse_code(x, bank, penalty=.015, active=8, steps=32):
    """Correlation-screened nonnegative L1 least squares; no learned encoder."""
    x = x.flatten(1)
    # Coefficients are inferred on the same complete four-view tensor.
    correlation = x @ bank.T
    n = min(active, bank.shape[0])
    scores, selected = correlation.topk(n, dim=1)
    gram = bank @ bank.T
    g = gram[selected[:, :, None], selected[:, None, :]]
    lipschitz = g.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-6)
    coefficient = torch.zeros_like(scores)
    y = coefficient.clone()
    t = 1.
    for _ in range(steps):
        gradient = torch.bmm(g, y.unsqueeze(-1)).squeeze(-1) - scores + penalty
        updated = (y - gradient / lipschitz).clamp_min(0)
        t_next = (1 + (1 + 4 * t * t) ** .5) / 2
        y = updated + (t - 1) / t_next * (updated - coefficient)
        coefficient, t = updated, t_next
    code = torch.zeros_like(correlation)
    code.scatter_(1, selected, coefficient)
    return code

class RingDictionary(nn.Module):
    def __init__(self, templates):
        super().__init__()
        self.templates = nn.Parameter(templates.detach().clone())
        self.project()

    @torch.no_grad()
    def project(self):
        self.templates.clamp_(min=0)
        self.templates.div_(self.templates.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None, None])

    def forward(self, x):
        bank = rotated_bank(self.templates)
        code = sparse_code(x, bank.detach())
        return (code @ bank).reshape_as(x), code

@torch.no_grad()
def best_rotation_similarity(templates):
    m = len(templates)
    bank = rotated_bank(templates)
    similarity = (templates.flatten(1) @ bank.T).reshape(m, m, 4)
    return similarity.max(-1)

def fit_loss(x, reconstruction):
    # The unit-norm panorama gives each location equal total reconstruction weight.
    return (x - reconstruction).square().flatten(1).sum(1).mean()
