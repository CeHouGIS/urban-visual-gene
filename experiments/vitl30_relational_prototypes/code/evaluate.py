"""Paired, spatial-block-aware development audit for the three pilot arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.resources import configure_device
from .hierarchy import EXPERIMENT, read
from .model import LandscapeDictionary
from .regions import OUT, region_boxes
from .train import ARMS, load_arrays


@torch.inference_mode()
def score_arm(arm, x, context, ids, out, device='cuda:1', batch_size=256):
    folder = Path(out) / 'models' / arm
    config = read(folder / 'config.json')
    complete = read(folder / 'complete.json')
    checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=False)
    if (complete['identity'] != config['identity'] or
            checkpoint['identity'] != config['identity'] or complete['test_used']):
        raise ValueError('Unfinished or mismatched model')
    model = LandscapeDictionary(families=x.shape[1], slots=config['slots'], mode=arm).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    error = np.empty(len(ids), np.float32)
    assignment = np.empty(len(ids), np.uint8)
    for begin in range(0, len(ids), batch_size):
        chosen = ids[begin:begin + batch_size]
        target = torch.from_numpy(x[chosen]).to(device)
        pano = torch.from_numpy(context[chosen].copy()).to(device)
        pred, _, winner = model(target, pano)
        error[begin:begin + len(chosen)] = (pred - target).square().mean((1, 2)).cpu().numpy()
        assignment[begin:begin + len(chosen)] = winner.argmax(-1).cpu().numpy()
    return error, assignment, model


def block_difference(a, b, blocks, seed=42, draws=2000):
    """Resample spatial blocks rather than correlated regions or panoramas."""
    unique, inverse = np.unique(blocks, return_inverse=True)
    difference = np.asarray(a) - np.asarray(b)
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    estimates = sums[sampled].sum(1) / counts[sampled].sum(1)
    return {'mean_difference': float(difference.mean()),
            'block_bootstrap_95_ci': [float(v) for v in np.quantile(estimates, [.025, .975])],
            'spatial_blocks': len(unique), 'draws': draws}


def audit(out=OUT, device='cuda:1'):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Development inference requires CUDA; CPU fallback is disabled')
    configure_device(device)
    torch.set_num_threads(4)
    x, context, train_ids, dev_ids, meta = load_arrays(out)
    frame = pd.read_parquet(Path(out) / 'pilot_panoramas.parquet')
    pano = dev_ids // (4 * len(region_boxes()))
    if any(frame.iloc[pano].prototype_split != 'dev'):
        raise ValueError('Development index opened a training or test region')
    blocks = frame.iloc[pano].spatial_block.to_numpy()
    errors, assignments, models = {}, {}, {}
    for arm in ARMS:
        errors[arm], assignments[arm], models[arm] = score_arm(arm, x, context, dev_ids, out, device)
    families = meta['families']
    counts = np.bincount(assignments['relation'], minlength=32)
    slots = []
    for slot in range(32):
        chosen = np.flatnonzero(assignments['relation'] == slot)
        means = x[dev_ids[chosen]].mean((0, 2)) if len(chosen) else np.zeros(len(families))
        leaders = np.argsort(means)[::-1][:5]
        cities = frame.iloc[np.unique(pano[chosen])].city.nunique() if len(chosen) else 0
        slots.append({'slot': slot, 'development_regions': len(chosen), 'cities': int(cities),
                      'mean_element_strengths': [{'family': families[i], 'mean': float(means[i])}
                                                 for i in leaders]})
    report = {'schema_version': 1, 'pilot_identity': read(Path(out) / 'protocol.json')['identity'],
              'region_identity': meta['region_identity'], 'checkpoint_identity': read(EXPERIMENT / 'results/summary.json')['checkpoint_sha256'],
              'development_regions': len(dev_ids), 'development_panoramas': int(frame.prototype_split.eq('dev').sum()),
              'all_train_regions': meta['all_train_regions'], 'eligible_train_regions': len(train_ids),
              'all_dev_regions': meta['all_dev_regions'], 'eligible_dev_regions': len(dev_ids),
              'trained_arms': {'composition': 'large-region family strengths only, with four-heading context',
                               'context': 'complete four-by-four family layout, no graph messages',
                               'relation': 'complete layout plus element-centroid proximity graph messages'},
              'planned_local_template_refit': 'not_started; older frozen 3/7/15-patch template model is descriptive only',
              'arm_mse': {arm: float(values.mean()) for arm, values in errors.items()},
              'paired_differences': {
                  'relation_minus_context': block_difference(errors['relation'], errors['context'], blocks),
                  'context_minus_composition': block_difference(errors['context'], errors['composition'], blocks)},
              'slot_usage': slots, 'semantic_validation': 'pending', 'test_used': False,
              'interpretation': ('Development MSE measures family-layout reconstruction, not semantic landscape coverage. '
                                 'Qwen-derived element groups may be wrong; SAE and prior VLM evidence are not independent of all prototype-layer partitions.')}
    write_json(EXPERIMENT / 'results/relational_pilot_summary.json', report)
    write_json(Path(out) / 'development_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    result = audit(device=args.device)
    print(json.dumps({k: result[k] for k in ['arm_mse', 'paired_differences', 'development_regions']}, indent=2))
