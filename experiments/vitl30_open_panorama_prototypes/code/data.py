"""Train/dev-only CSR preparation; heading joins are explicit and resumable."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / 'outputs'
SOURCE = ROOT / 'experiments/vitl30_multiscale_prototypes/outputs'
SPLIT = ROOT / 'experiments/vitl30_prototype_coverage/outputs'
GROUPS = ROOT / 'experiments/vitl30_panorama_w4096_k32/reports/semantic_umap/feature_semantic_umap_groups.csv'
TOKENS = 3136

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def status(out, state, **kwargs):
    write_json(Path(out) / 'status.json', {'status': state, 'pid': os.getpid(),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs})

def join_headings(array):
    """[...,4,H,W] -> [...,H,4W], never reshape across the y axis."""
    if array.shape[-3] != 4:
        raise ValueError('Exactly four headings required')
    return np.concatenate([array[..., h, :, :] for h in range(4)], axis=-1)

def group_lookup(path=GROUPS):
    frame = pd.read_csv(path).sort_values('feature')
    if frame.feature.tolist() != list(range(4096)):
        raise ValueError('Must retain all 4096 SAE dimensions, including excluded UMAP rows')
    groups = frame.decoder_group.to_numpy(np.int64)
    if set(groups) != set(range(64)):
        raise ValueError('Expected existing decoder groups 0..63')
    return groups

def aggregate_pano(values, indices, ptr, scale, groups):
    if len(ptr) != TOKENS + 1 or int(ptr[-1]) - int(ptr[0]) != len(values):
        raise ValueError('Panorama CSR extent differs')
    patch = np.repeat(np.arange(TOKENS, dtype=np.int64), np.diff(ptr).astype(np.int64))
    heading, local = patch // 784, patch % 784
    cell = (local // 28 // 2) * 56 + heading * 14 + (local % 28 // 2)
    dims = np.asarray(indices, dtype=np.int64)
    weights = np.minimum(np.asarray(values, np.float32) / scale[dims], 8)
    weights /= np.sqrt(np.bincount(groups, minlength=64))[groups[dims]]
    # Each pooled cell is the mean over its four source patch locations.
    return (np.bincount(groups[dims] * 784 + cell, weights=weights, minlength=64 * 784)
            .reshape(64, 14, 56).astype(np.float32) / 4)

def _chunk(job):
    chunk, rows, target, identity, corpus_identity = job
    target = Path(target)
    marker = target.with_suffix('.json')
    if target.exists() and marker.exists():
        meta = json.loads(marker.read_text())
        if meta['identity'] != identity or meta['rows'] != [r[1] for r in rows]:
            raise ValueError('Stale pooled input chunk')
        if target.stat().st_size != meta['bytes']:
            raise ValueError('Truncated pooled input chunk')
        return str(target), meta
    record = json.loads((chunk / 'complete.json').read_text())
    if record['identity'] != corpus_identity:
        raise ValueError('Sparse corpus changed')
    for name in ('values', 'indices', 'indptr'):
        if (chunk / f'{name}.npy').stat().st_size != record['files'][name]['bytes']:
            raise ValueError('Truncated source CSR')
    scale = np.load(SPLIT / 'positive_rms.npy')
    groups = group_lookup()
    v, ix, ptr = [np.load(chunk / f'{name}.npy', mmap_mode='r') for name in ('values', 'indices', 'indptr')]
    array = np.empty((len(rows), 64, 14, 56), np.float16)
    square, count = np.zeros(64, np.float64), np.zeros(64, np.int64)
    for j, (local, _, split) in enumerate(rows):
        if split not in ('train', 'dev'):
            raise ValueError('Test activations are sealed')
        p = ptr[local * TOKENS:(local + 1) * TOKENS + 1]
        a, b = int(p[0]), int(p[-1])
        x = aggregate_pano(v[a:b], ix[a:b], p, scale, groups)
        array[j] = x
        if split == 'train':
            square += np.square(x, dtype=np.float64).sum((1, 2))
            count += (x > 0).sum((1, 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.partial.npy')
    np.save(tmp, array, allow_pickle=False)
    tmp.replace(target)
    meta = {'identity': identity, 'rows': [r[1] for r in rows], 'square': square.tolist(),
            'count': count.tolist(), 'bytes': target.stat().st_size}
    write_json(marker, meta)
    return str(target), meta

def prepare(out=OUT, workers=4):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((SOURCE / 'corpus.json').read_text())
    split = json.loads((SPLIT / 'protocol.json').read_text())
    if corpus['identity'] != split['source_corpus_identity']:
        raise ValueError('Split and corpus identities differ')
    frame = pd.read_parquet(SPLIT / 'panoramas.parquet')
    frame = frame[frame.prototype_split.isin(['train', 'dev'])].reset_index(drop=True)
    protocol = {'version': 1, 'corpus_identity': corpus['identity'],
        'checkpoint_sha256': corpus['checkpoint_sha256'], 'split_identity': split['identity'],
        'split_manifest_sha256': digest(SPLIT / 'panoramas.parquet'),
        'groups_sha256': digest(GROUPS), 'rms_sha256': digest(SPLIT / 'positive_rms.npy'),
        'code_sha256': digest(__file__), 'counts': frame.prototype_split.value_counts().to_dict(),
        'cities': int(frame.city.nunique()), 'headings': [0, 90, 180, 270],
        'original_grid': [28, 112], 'model_grid': [14, 56], 'channels': 64,
        'aggregation': 'all4096; clip8 train-positive-RMS; sum/sqrt(group size); 2x2 mean per direction',
        'transform': 'log1p(raw / train-only group positive RMS), clip4, whole-panorama unit L2',
        'test_policy': 'No test activations or images; new prototype layer only, not full pipeline independence'}
    protocol['identity'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('Protocol changed; select a new output directory')
    write_json(out / 'protocol.json', protocol)
    if (out / 'inputs_complete.json').exists():
        saved = json.loads((out / 'inputs_complete.json').read_text())
        if saved['identity'] != protocol['identity'] or (out / 'inputs.npy').stat().st_size != saved['bytes']:
            raise ValueError('Invalid normalized inputs')
        return protocol
    frame.to_parquet(out / 'panoramas.parquet', index=False)
    jobs = []
    for city, city_rows in frame.groupby('city', sort=True):
        for start, selected in city_rows.groupby(city_rows.cache_pano_index // corpus['chunk_panos']):
            start = int(start) * corpus['chunk_panos']
            rows = [(int(row.cache_pano_index) - start, int(i), row.prototype_split) for i, row in selected.iterrows()]
            name = city.replace('/', '__')
            jobs.append((SOURCE / 'activations' / name / f'{start:06d}', rows,
                         out / 'chunks' / name / f'{start:06d}.npy', protocol['identity'], corpus['identity']))
    done, square, count, records = 0, np.zeros(64, np.float64), np.zeros(64, np.int64), []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, meta in pool.map(_chunk, jobs, chunksize=1):
            records.append((path, meta))
            square += meta['square']; count += meta['count']
            done += len(meta['rows'])
            status(out, 'preparing', stage='grouping_existing_activations', completed=done, total=len(frame),
                   elapsed_seconds=round(time.monotonic() - started, 1), test_sealed=True)
            if len(records) % 25 == 0:
                print('INPUT', done, '/', len(frame), flush=True)
    scale = np.sqrt(square / np.maximum(count, 1)).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    np.save(out / 'group_scale.npy', scale)
    np.save(out / 'dimension_groups.npy', group_lookup())
    normalized = np.lib.format.open_memmap(out / 'inputs.partial.npy', mode='w+', dtype=np.float16,
                                          shape=(len(frame), 64, 14, 56))
    for j, (path, meta) in enumerate(records):
        x = np.load(path).astype(np.float32)
        x = np.minimum(np.log1p(x / scale[None, :, None, None]), 4)
        norm = np.sqrt(np.square(x).sum((1, 2, 3), keepdims=True))
        normalized[meta['rows']] = x / np.maximum(norm, 1e-8)
        if j % 25 == 0:
            status(out, 'preparing', stage='packing_train_normalized_ring_inputs',
                   completed=j, total=len(records), test_sealed=True)
    normalized.flush(); del normalized
    (out / 'inputs.partial.npy').replace(out / 'inputs.npy')
    write_json(out / 'inputs_complete.json', {'identity': protocol['identity'],
        'bytes': (out / 'inputs.npy').stat().st_size, 'shape': [len(frame), 64, 14, 56],
        'group_scale_sha256': digest(out / 'group_scale.npy'), 'seconds': time.monotonic() - started})
    return protocol
