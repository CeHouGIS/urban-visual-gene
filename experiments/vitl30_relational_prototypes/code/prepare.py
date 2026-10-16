"""Freeze a bounded, city-balanced train/dev pilot without opening activations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .hierarchy import CHECKPOINT, EXPERIMENT, ROOT, read, sha, write

BASE = ROOT / 'experiments/vitl30_prototype_coverage/outputs'


def select(frame, train_per_city=32, dev_per_city=8, seed=20261017):
    if train_per_city < 1 or dev_per_city < 1:
        raise ValueError('Pilot counts must be positive')
    allowed = frame[frame.prototype_split.isin(['train', 'dev'])]
    if allowed.groupby('spatial_block').prototype_split.nunique().max() > 1:
        raise ValueError('A spatial block crosses train/dev')
    parts = []
    for (city, split), group in allowed.groupby(['city', 'prototype_split'], sort=True):
        count = train_per_city if split == 'train' else dev_per_city
        ordered = sorted(group.index, key=lambda i: hashlib.sha256(
            f'{seed}/{city}/{frame.loc[i, "panoid"]}'.encode()).hexdigest())
        if len(ordered) < count:
            raise ValueError(f'Not enough {split} panoramas: {city}')
        part = group.loc[ordered[:count]].copy()
        part['row_id'] = part.index
        parts.append(part)
    result = pd.concat(parts).sort_values(['city', 'prototype_split', 'cache_pano_index'])
    if set(result.prototype_split) != {'train', 'dev'} or result.duplicated(['city', 'panoid']).any():
        raise ValueError('Invalid pilot split or duplicate panoramas')
    if (result.groupby('city').prototype_split.nunique() != 2).any():
        raise ValueError('Every city must have train and dev panoramas')
    return result


def prepare(out=EXPERIMENT / 'outputs', base=BASE, train_per_city=32, dev_per_city=8):
    out, base = Path(out), Path(base)
    source = read(base / 'protocol.json')
    hierarchy_path = EXPERIMENT / 'results/hierarchy.json'
    hierarchy = read(hierarchy_path)
    if source['checkpoint_sha256'] != CHECKPOINT or hierarchy['protocol']['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Different frozen SAE')
    frame = pd.read_parquet(base / 'panoramas.parquet')
    selected = select(frame, train_per_city, dev_per_city)
    protocol = {'version': 1, 'checkpoint_sha256': CHECKPOINT,
        'base_protocol_identity': source['identity'], 'manifest_sha256': sha(base / 'panoramas.parquet'),
        'training_scale_sha256': sha(base / 'positive_rms.npy'),
        'hierarchy_identity': hierarchy['identity'], 'taxonomy_sha256': hierarchy['protocol']['taxonomy_sha256'],
        'selection_code_sha256': sha(Path(__file__)), 'sampling_seed': 20261017,
        'sampling': {'train_per_city': train_per_city, 'dev_per_city': dev_per_city,
                     'all_four_headings': True, 'spatial_split': 'reuse frozen 1-km train/dev blocks'},
        'cities': int(selected.city.nunique()), 'panoramas': len(selected),
        'split_counts': {str(k): int(v) for k, v in selected.prototype_split.value_counts().items()},
        'images': len(selected) * 4, 'test_activations_opened': False,
        'prior_semantic_evidence_includes_test': True,
        'evaluation_scope': 'Exploratory prototype-layer comparison; SAE and previously generated semantic descriptions are not independent of prototype dev/test.',
        'regions': {'sizes': [14, 21, 28], 'coordinates': 'direction-local patches; no convolution across heading seams'},
        'arms': ['current_local_templates_frozen_reference', 'larger_context_reconstruction', 'element_relation_prototypes'],
        'comparability': 'The existing frozen template arm is descriptive only; fair architecture comparisons must refit all arms on the same pilot, budgets and seeds.',
        'prototype_slots_initial': 32,
        'planned_evaluation': ['composition_fidelity', 'relative_layout_fidelity', 'remove_constituent',
            'shuffle_layout_keep_composition', 'same_dominant_element_different_context', 'cross_city_retrieval', 'seed_matching'],
        'coverage': 'Keep original all-patch reconstruction coverage; independently audited composition coverage requires a separate criterion. Never count an entire matched bounding box as explained.',
        'stage': 'manifest_prepared', 'training_started': False,
        'semantic_review': 'pending; provisional grouping available, human validation is not complete'}
    protocol['identity'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != protocol:
        raise ValueError('Pilot identity changed; use a new output directory')
    write(out / 'protocol.json', protocol)
    selected.to_parquet(out / 'pilot_panoramas.parquet', index=False)
    write(EXPERIMENT / 'results/pilot_manifest_summary.json', {k: protocol[k] for k in [
        'identity', 'hierarchy_identity', 'cities', 'panoramas', 'images', 'split_counts',
        'sampling', 'regions', 'arms', 'prototype_slots_initial', 'stage', 'training_started',
        'semantic_review', 'test_activations_opened', 'prior_semantic_evidence_includes_test', 'comparability']})
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EXPERIMENT / 'outputs')
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), ensure_ascii=False, indent=2))
