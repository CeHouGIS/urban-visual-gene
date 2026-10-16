"""Discrete 32-slot element-layout dictionary with matched encoder ablations."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LandscapeDictionary(nn.Module):
    def __init__(self, families=15, slots=32, hidden=64, mode='relation'):
        super().__init__()
        if mode not in {'composition', 'context', 'relation'}:
            raise ValueError(mode)
        self.mode, self.families, self.slots = mode, families, slots
        self.family_embedding = nn.Embedding(families, 8)
        self.node = nn.Linear(19 + 8, hidden)
        self.neighbor = nn.Linear(hidden, hidden)
        self.context = nn.Linear(4 * families, hidden)
        self.readout = nn.Sequential(nn.Linear(families * hidden + hidden, hidden),
                                     nn.ReLU(), nn.Linear(hidden, slots))
        self.dictionary = nn.Parameter(torch.empty(slots, families, 16))
        nn.init.uniform_(self.dictionary, -.5, .5)

    def forward(self, x, panorama_context, temperature=.6):
        batch, families, cells = x.shape
        if families != self.families or cells != 16 or panorama_context.shape != (batch, 4, families):
            raise ValueError('Element layout or four-heading context shape differs')
        mass = x.mean(-1)
        grid = torch.tensor([(i // 4 / 3, i % 4 / 3) for i in range(16)],
                            dtype=x.dtype, device=x.device)
        centroid = (x @ grid) / x.sum(-1).clamp_min(1e-6).unsqueeze(-1)
        if self.mode == 'composition':
            visible = mass.unsqueeze(-1).expand_as(x)
            centroid = torch.full_like(centroid, .5)
        else:
            visible = x
        node = torch.cat([visible, mass.unsqueeze(-1), centroid,
                          self.family_embedding.weight.unsqueeze(0).expand(batch, -1, -1)], -1)
        node = F.relu(self.node(node))
        if self.mode == 'relation':
            delta = centroid[:, :, None] - centroid[:, None, :]
            active = (mass > .05).to(x.dtype)
            edge = torch.exp(-delta.square().sum(-1) / .18) * active[:, :, None] * active[:, None, :]
            edge = edge * (1 - torch.eye(families, device=x.device, dtype=x.dtype))
            edge = edge / edge.sum(-1, keepdim=True).clamp_min(1e-6)
            node = F.relu(node + self.neighbor(edge @ node))
        context = F.relu(self.context(panorama_context.flatten(1)))
        logits = self.readout(torch.cat([node.flatten(1), context], -1))
        probabilities = F.softmax(logits / temperature, -1)
        winner = F.one_hot(probabilities.argmax(-1), self.slots).to(x.dtype)
        straight_through = winner - probabilities.detach() + probabilities
        templates = F.softplus(self.dictionary)
        prediction = torch.einsum('bm,mfc->bfc', straight_through, templates)
        return prediction, probabilities, winner


def loss_terms(prediction, target, probabilities):
    weight = 1 + 2 * (target > .1).to(target.dtype)
    reconstruction = (weight * (prediction - target).square()).mean()
    mass_error = (prediction.mean(-1) - target.mean(-1)).square().mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
    use = probabilities.mean(0)
    balance = (use * (use.clamp_min(1e-8) * probabilities.shape[1]).log()).sum()
    objective = reconstruction + .3 * mass_error + .01 * entropy + .1 * balance
    return objective, {'reconstruction': reconstruction.detach(), 'mass_error': mass_error.detach(),
                       'entropy': entropy.detach(), 'usage_kl': balance.detach()}
