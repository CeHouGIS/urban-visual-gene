"""Frozen three-way spatial split and training-only calibration of an existing CSR cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.vitl30_multiscale_prototypes.code.data import OUT as SOURCE, write_json
from sae_experiments.io_utils import digest

OUT = Path(__file__).resolve().parents[1] / 'outputs'


def spatial_split(frame, seed=20261016):
    """Assign whole 1-km blocks, with all three partitions represented in each city."""
    result = frame.copy()
    result['previous_prototype_split'] = result.prototype_split
    for city, group in result.groupby('city', sort=True):
        blocks = sorted(group.spatial_block.unique(), key=lambda b: hashlib.sha256(
            f'{seed}/{city}/{b}'.encode()).hexdigest())
        if len(blocks) < 7:
            raise ValueError(f'Too few blocks for a three-way split: {city}')
        n_train = int(len(blocks) * .70)
        n_dev = max(1, int(len(blocks) * .15))
        assignments = {b: 'train' if i < n_train else 'dev' if i < n_train + n_dev else 'test'
                       for i, b in enumerate(blocks)}
        result.loc[group.index, 'prototype_split'] = group.spatial_block.map(assignments)
    return result


def panel(frame, split='dev', per_city=1, seed=42):
    """A fixed city-balanced panel; final test is deliberately inaccessible here."""
    if split != 'dev':
        raise ValueError('This pilot may select development panels only; test remains sealed')
    selected = []
    for city, group in frame[frame.prototype_split == split].groupby('city', sort=True):
        ids = sorted(group.index, key=lambda i: hashlib.sha256(
            f'{seed}/{city}/{frame.loc[i, "panoid"]}'.encode()).hexdigest())
        selected.extend(ids[:per_city])
    return np.array(selected, dtype=np.int64)


def csr_training_moments(values, indices, ptr, train_panos, dimensions=4096, tokens=3136):
    """Accumulate selected panorama rows only, without densifying the patch tensor."""
    boundaries = np.asarray(ptr[::tokens], dtype=np.int64)
    if len(boundaries) != len(train_panos) + 1 or boundaries[-1] != len(values):
        raise ValueError('CSR panorama extents differ')
    keep = np.repeat(np.asarray(train_panos, dtype=bool), np.diff(boundaries))
    d = np.asarray(indices[keep], dtype=np.int64)
    v = np.asarray(values[keep], dtype=np.float64)
    return (np.bincount(d, minlength=dimensions),
            np.bincount(d, weights=v * v, minlength=dimensions))


def rms(count, square):
    scale = np.sqrt(square / np.maximum(count, 1)).astype(np.float32)
    scale[count == 0] = 1
    return np.maximum(scale, 1e-6)


def prepare(out=OUT, source=SOURCE):
    out, source = Path(out), Path(source)
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((source / 'corpus.json').read_text())
    frame = spatial_split(pd.read_parquet(source / 'panoramas.parquet'))
    protocol = {'version': 1, 'source': str(source.resolve()),
        'source_corpus_identity': corpus['identity'], 'checkpoint_sha256': corpus['checkpoint_sha256'],
        'source_manifest_sha256': digest(source / 'panoramas.parquet'),
        'split_seed': 20261016, 'split': '70/15/15 percent of 1-km blocks within each city',
        'split_counts': {str(k): int(v) for k, v in frame.prototype_split.value_counts().items()},
        'cities': len(frame.city.unique()), 'panoramas': len(frame),
        'city_counts': frame.groupby(['city', 'prototype_split']).size().unstack(fill_value=0).to_dict('index'),
        'test_policy': 'sealed during this pilot; no test activations used for initialization, scale, monitoring or selection',
        'scope': 'held out for the NEW prototype layer only; frozen SAE has seen some of these images',
        'target': 'at least 80% semantically covered panoramas; requires a fixed criterion and independent annotation',
        'panorama_rule': 'all four directional images must pass; also report directional and any-direction rates',
        'calibration': 'positive RMS from NEW training blocks only; clip at 8',
        'capacity': {'scales': [3, 7, 15], 'per_scale': 32, 'rank': 4},
        'pilot': 'exploratory old-template refit; two matched four-epoch fresh fits; full development scoring',
        'semantic_audit': 'pending human review; reconstruction reduction is not semantic coverage'}
    protocol['identity'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('Protocol differs; use a new output directory')
    write_json(out / 'protocol.json', protocol)
    if (out / 'panoramas.parquet').exists():
        pd.testing.assert_frame_equal(frame, pd.read_parquet(out / 'panoramas.parquet'))
    else:
        frame.to_parquet(out / 'panoramas.partial.parquet', index=False)
        (out / 'panoramas.partial.parquet').replace(out / 'panoramas.parquet')
    if (out / 'calibration.json').exists():
        saved = json.loads((out / 'calibration.json').read_text())
        if saved['identity'] != protocol['identity'] or digest(out / 'positive_rms.npy') != saved['scale_sha256']:
            raise ValueError('Calibration identity differs')
        return protocol
    count, square = np.zeros(4096, np.int64), np.zeros(4096, np.float64)
    completed = 0
    for city in corpus['cities']:
        city_frame = frame[frame.city == city['city']].sort_values('cache_pano_index')
        for start in range(0, city['panoramas'], corpus['chunk_panos']):
            chunk = source / 'activations' / city['city'].replace('/', '__') / f'{start:06d}'
            marker = json.loads((chunk / 'complete.json').read_text())
            if (marker['identity'] != corpus['identity'] or marker['start'] != start or
                    marker['stop'] != min(start + corpus['chunk_panos'], city['panoramas'])):
                raise ValueError(f'Invalid cache extent/identity: {chunk}')
            for name in ('values', 'indices', 'indptr'):
                if (chunk / f'{name}.npy').stat().st_size != marker['files'][name]['bytes']:
                    raise ValueError(f'Truncated cache: {chunk}/{name}')
            partial = out / 'calibration_chunks' / city['city'].replace('/', '__') / f'{start:06d}.npz'
            if partial.exists():
                with np.load(partial) as saved:
                    if str(saved['identity']) != protocol['identity']:
                        raise ValueError('Stale normalization partial')
                    c, s = saved['count'], saved['square']
            else:
                arrays = [np.load(chunk / f'{name}.npy', mmap_mode='r') for name in ('values', 'indices', 'indptr')]
                train = city_frame.prototype_split.iloc[start:marker['stop']].to_numpy() == 'train'
                c, s = csr_training_moments(*arrays, train)
                partial.parent.mkdir(parents=True, exist_ok=True)
                temporary = partial.with_suffix('.partial.npz')
                np.savez(temporary, identity=protocol['identity'], count=c, square=s)
                temporary.replace(partial)
            count += c
            square += s
            completed += marker['stop'] - start
        write_json(out / 'preparation_status.json', {'stage': 'training_only_calibration',
            'panoramas': completed, 'total': len(frame), 'city': city['city']})
        print('CALIBRATION', city['city'], completed, len(frame), flush=True)
    np.save(out / 'positive_rms.npy', rms(count, square))
    np.save(out / 'training_support.npy', count)
    write_json(out / 'calibration.json', {'identity': protocol['identity'],
        'scale_sha256': digest(out / 'positive_rms.npy'), 'unobserved_train_dimensions': int((count == 0).sum())})
    return protocol
