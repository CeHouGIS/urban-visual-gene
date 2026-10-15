"""Exploratory angular adjacency test for five discovered feature pairs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from formal.analysis.plot_feature_activation_maps import load_checkpoint
from formal.analysis.test_feature_combinations import CHECKPOINT, FEATURES, OUT, ROWS_PER_PANO


def main():
    rows = json.loads((OUT / 'records.json').read_text())
    analysis = np.load(OUT / 'analysis_arrays.npz')
    validation = set(analysis['validation_city'].tolist())
    thresholds = analysis['thresholds']
    cities = sorted({r['city'] for r in rows})
    counts = np.concatenate([np.load(OUT / 'city_counts' / f'{city}.npy') for city in cities])
    pairs = list(csv.DictReader((OUT / 'pair_results.csv').open()))[:5]
    device = torch.device('cuda:0')
    model = load_checkpoint(CHECKPOINT, device).half()
    rng = np.random.default_rng(20261014)
    result = []
    for pair in pairs:
        a, b = int(pair['feature_a']), int(pair['feature_b'])
        eligible = [i for i, rec in enumerate(rows) if rec['city'] in validation
                    and counts[i, a] >= thresholds[a] and counts[i, b] >= thresholds[b]]
        # City-balanced evidence prevents a large city from dominating.
        city_groups = {}
        for i in eligible:
            city_groups.setdefault(rows[i]['city'], []).append(i)
        picked = []
        for city in sorted(city_groups):
            ids = rng.choice(city_groups[city], min(2, len(city_groups[city])), replace=False)
            picked.extend(int(i) for i in ids)
        if len(picked) > 120:
            picked = rng.choice(picked, 120, replace=False).tolist()
        true_scores, null_scores = [], []
        source = None
        prior_city = None
        for item in sorted(picked):
            rec = rows[item]
            if rec['city'] != prior_city:
                source = np.load(FEATURES / f"dict_{rec['city']}_p784.f16.npy", mmap_mode='r')
                prior_city = rec['city']
            start = rec['source_pano'] * ROWS_PER_PANO
            block = torch.from_numpy(np.asarray(source[start:start+ROWS_PER_PANO]).copy()).to(device)
            with torch.inference_mode():
                pre = torch.relu(model._preactivate(block)).flatten()
                values, ix = torch.topk(pre, model.k * ROWS_PER_PANO, sorted=False)
                active = ix[values > 0]
                patch = torch.div(active, model.latent_dim, rounding_mode='floor')
                feature = active.remainder(model.latent_dim)
                maps = []
                for f in (a, b):
                    mask = torch.zeros(ROWS_PER_PANO, device=device, dtype=torch.bool)
                    mask[patch[feature == f]] = True
                    maps.append(mask.cpu().numpy().reshape(4, 28, 28).transpose(1, 0, 2).reshape(28, 112))
            first = binary_dilation(maps[0], iterations=2)
            second = maps[1]
            true_scores.append(float(np.mean(first[second])))
            shifts = rng.integers(1, 112, 199)
            null_scores.append([float(np.mean(first[np.roll(second, int(shift), axis=1)])) for shift in shifts])
        true_scores = np.array(true_scores)
        null_scores = np.array(null_scores)
        observed = true_scores.mean()
        null = null_scores.mean(axis=0)
        p = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        result.append({'feature_a': a, 'feature_b': b, 'panoramas': len(picked),
                       'validation_cities': len({rows[i]['city'] for i in picked}),
                       'observed_adjacency': round(float(observed), 4),
                       'random_shift_adjacency': round(float(null.mean()), 4),
                       'spatial_lift': round(float(observed / max(null.mean(), 1e-9)), 3),
                       'spatial_p_exploratory': float(p)})
        print(result[-1], flush=True)
    with (OUT / 'spatial_exploratory.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, result[0].keys())
        writer.writeheader(); writer.writerows(result)


if __name__ == '__main__':
    main()
