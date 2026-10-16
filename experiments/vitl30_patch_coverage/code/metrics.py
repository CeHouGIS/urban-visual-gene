"""Per-image patch quality, with fixed eligibility and exact hard pass counts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class CoveragePolicy:
    target_fraction: float = .8
    maximum_patch_nmse: float = .5
    minimum_target_energy: float = 1e-8
    temperature_start: float = .25
    temperature_end: float = .1
    coverage_weight: float = 5.
    robust_patch_weight: float = .1

    def __post_init__(self):
        if not 0 < self.target_fraction <= 1 or self.maximum_patch_nmse <= 0:
            raise ValueError('Invalid coverage or quality threshold')
        if self.minimum_target_energy <= 0 or min(self.temperature_start, self.temperature_end) <= 0:
            raise ValueError('Energy floor and temperatures must be positive')

    def record(self):
        return asdict(self)


def patch_statistics(prediction, target, policy=CoveragePolicy(), temperature=None):
    """Eligibility depends on TARGET only, never the prediction or its error.

    This measures SAE activation reconstruction, not validated semantic meaning.
    Zero-energy patches remain visible in counts and global reconstruction loss.
    An image without eligible patches always fails the image-level criterion.
    """
    if prediction.shape != target.shape or target.ndim != 4:
        raise ValueError('Expected matching [image, feature, height, width] tensors')
    error = (prediction.float() - target.float()).square().sum(1)
    energy = target.float().square().sum(1)
    valid = energy > policy.minimum_target_energy
    nmse = error / energy.clamp_min(policy.minimum_target_energy)
    covered = valid & torch.isfinite(nmse) & (nmse <= policy.maximum_patch_nmse)
    valid_count = valid.sum((1, 2))
    covered_count = covered.sum((1, 2))
    denominator = valid_count.clamp_min(1)
    fraction = covered_count.float() / denominator
    required_count = torch.ceil(valid_count.double() * policy.target_fraction).long()
    image_pass = (valid_count > 0) & (covered_count >= required_count)
    temp = policy.temperature_end if temperature is None else temperature
    smooth = torch.sigmoid((policy.maximum_patch_nmse - nmse) / temp)
    soft_fraction = (smooth * valid).sum((1, 2)) / denominator
    robust_patch_error = (torch.log1p(nmse) * valid).sum((1, 2)) / denominator
    image_nmse = error.sum((1, 2)) / energy.sum((1, 2)).clamp_min(policy.minimum_target_energy)
    return {'patch_nmse': nmse, 'valid_mask': valid, 'covered_mask': covered,
            'patch_sse': error, 'patch_energy': energy,
            'valid_count': valid_count, 'covered_count': covered_count,
            'required_count': required_count, 'coverage': fraction, 'image_pass': image_pass,
            'soft_coverage': soft_fraction, 'robust_patch_error': robust_patch_error,
            'image_nmse': image_nmse}


def coverage_objective(prediction, target, responses, composition_penalty,
                       progress, policy=CoveragePolicy()):
    """Penalize each image's shortfall; never take a dataset-mean shortfall.

    A temperature schedule makes the differentiable surrogate closer to hard
    coverage. Hard counts are always computed separately and never substituted.
    """
    temperature = policy.temperature_start + min(1., max(0., progress) / 2) * (
        policy.temperature_end - policy.temperature_start)
    stats = patch_statistics(prediction, target, policy, temperature)
    shortfall = torch.relu(policy.target_fraction - stats['soft_coverage'])
    eligible = stats['valid_count'] > 0
    coverage_penalty = (shortfall.square() * eligible).sum() / eligible.sum().clamp_min(1)
    # Global energy loss also penalizes spurious reconstruction on invalid patches.
    warmup = min(1., max(0., progress))
    loss = (stats['image_nmse'].mean() + policy.robust_patch_weight * stats['robust_patch_error'].mean()
            + policy.coverage_weight * coverage_penalty
            + warmup * (.01 * responses.float().mean() + .02 * composition_penalty))
    diagnostics = {'coverage_penalty': coverage_penalty, 'temperature': temperature,
                   'mean_patch_coverage': stats['coverage'].mean(),
                   'batch_images_meeting_80pct': stats['image_pass'].float().mean(),
                   'minimum_image_patch_coverage': stats['coverage'].min(),
                   'training_image_nmse': stats['image_nmse'].mean(),
                   'invalid_patch_fraction': (~stats['valid_mask']).float().mean()}
    return loss, diagnostics


def coverage_summary(table, policy=CoveragePolicy()):
    """Image pass fraction and patch average are deliberately separate metrics."""
    if not len(table):
        raise ValueError('No directional images scored')
    if table.duplicated(['row_id', 'heading']).any():
        raise ValueError('Duplicate directional images')
    if not table.heading.isin([0, 1, 2, 3]).all():
        raise ValueError('Heading must be an index in 0..3')
    if ((table.valid_patch_count < 0) | (table.covered_patch_count < 0) |
            (table.covered_patch_count > table.valid_patch_count) |
            (table.valid_patch_count > table.patch_count)).any():
        raise ValueError('Invalid patch counts')
    expected = table.valid_patch_count.apply(lambda n: math.ceil(int(n) * policy.target_fraction))
    passed = (table.valid_patch_count > 0) & (table.covered_patch_count >= expected)
    fraction = table.covered_patch_count / table.valid_patch_count.clip(lower=1)
    pano = table.assign(image_pass=passed).groupby('row_id').image_pass.agg(['all', 'size'])
    if not (pano['size'] == 4).all():
        raise ValueError('Evaluation must include all four directions of every panorama')
    return {'images': len(table), 'panoramas': len(pano), 'policy': policy.record(),
            'images_meeting_target': int(passed.sum()), 'images_below_target': int((~passed).sum()),
            'image_pass_fraction': float(passed.mean()),
            'all_images_meet_target': bool(passed.all()),
            'mean_image_patch_coverage': float(fraction.mean()),
            'pooled_patch_coverage': float(table.covered_patch_count.sum() / max(int(table.valid_patch_count.sum()), 1)),
            'coverage_quantiles': {str(q): float(fraction.quantile(q)) for q in [0, .05, .1, .25, .5, .75, .9, 1]},
            'panoramas_all_four_pass': int(pano['all'].sum()),
            'panorama_all_four_pass_fraction': float(pano['all'].mean()),
            'valid_patches': int(table.valid_patch_count.sum()),
            'invalid_patches': int(table.patch_count.sum() - table.valid_patch_count.sum()),
            'images_without_valid_patches': int((table.valid_patch_count == 0).sum()),
            'semantic_coverage': None, 'semantic_review': 'pending',
            'meaning': 'per-image patch reconstruction criterion, not semantic validation'}
