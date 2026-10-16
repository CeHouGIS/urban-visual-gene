"""Expand post-hoc interpretation using the existing complete sparse cache.

No DINO extraction, SAE inference/training, NAS directory scan, or evaluation
split changes. Top10 ranks are exact for the stored float16 activations;
ties use corpus order, then the earliest patch. The corpus includes images
used to train the SAE and must not be presented as independent validation.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from numba import njit
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'experiments/vitl30_multiscale_prototypes/outputs'
SOURCE = ROOT / 'experiments/vitl30_panorama_w4096_k32/outputs/analysis/interpretation_expanded_top10'
REPORT = ROOT / 'experiments/vitl30_panorama_w4096_k32/reports/vlm_dimensions_top10_expanded'
BASELINE = ROOT / 'experiments/vitl30_panorama_w4096_k32/reports/vlm_dimensions_top10_masked'
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'
WIDTH, PATCHES, TOKENS = 4096, 784, 3136


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def status(args, stage, **fields):
    value = {'stage': stage, 'pid': os.getpid(),
             'updated_utc': datetime.now(timezone.utc).isoformat(), **fields}
    write(args.report / 'expansion_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


@njit(nogil=True, cache=True)
def panorama_peaks(values, dimensions, indptr, panoramas, wanted, width=WIDTH, tokens=TOKENS):
    peaks = np.zeros((panoramas, width), np.float32)
    patches = np.zeros((panoramas, width), np.int16)
    for p in range(panoramas):
        for patch in range(tokens):
            row = p * tokens + patch
            for at in range(indptr[row], indptr[row + 1]):
                d, value = dimensions[at], values[at]
                if wanted[d] and value > peaks[p, d]:
                    peaks[p, d], patches[p, d] = value, patch
    return peaks, patches


def merge_top(values, ids, patches, peaks, positions, offset, top):
    """Stable merge: earlier panorama wins equal cached activation values."""
    selected = np.argsort(-peaks, axis=0, kind='stable')[:top]
    local_values = np.take_along_axis(peaks, selected, axis=0)
    local_patches = np.take_along_axis(positions, selected, axis=0)
    all_values = np.concatenate((values, local_values))
    all_ids = np.concatenate((ids, selected + offset))
    all_patches = np.concatenate((patches, local_patches))
    order = np.argsort(-all_values, axis=0, kind='stable')[:top]
    return tuple(np.take_along_axis(x, order, axis=0) for x in (all_values, all_ids, all_patches))


def missing_features(baseline):
    identity = read(baseline / 'identity.json')
    records = read(baseline / 'dimension_interpretations.json')
    if identity.get('checkpoint_sha256') != CHECKPOINT or records.get('identity') != identity.get('identity'):
        raise ValueError('Baseline interpretation identity differs')
    rows = records.get('dimensions', [])
    if [r['feature'] for r in rows] != list(range(WIDTH)):
        raise ValueError('Incomplete baseline dimensions')
    return [r['feature'] for r in rows if r['status'] != 'interpreted'], identity['identity']


def prepare(args):
    spec, complete = read(args.cache / 'corpus.json'), read(args.cache / 'cache_complete.json')
    if spec.get('checkpoint_sha256') != CHECKPOINT or complete.get('identity') != spec.get('identity'):
        raise ValueError('Incomplete cache or different SAE checkpoint')
    frame = pd.read_parquet(args.cache / 'panoramas.parquet')
    if (len(frame) != spec['panoramas'] or len(frame) != complete.get('panoramas')
            or frame.duplicated(['city', 'panoid']).any()):
        raise ValueError('Panorama manifest differs')
    expected_cities = [c['city'] for c in spec['cities'] for _ in range(c['panoramas'])]
    if frame.city.tolist() != expected_cities:
        raise ValueError('Manifest must follow corpus city order')
    features, baseline_identity = missing_features(args.baseline)
    manifest_hash = hashlib.sha256((args.cache / 'panoramas.parquet').read_bytes()).hexdigest()
    run = {'checkpoint_sha256': CHECKPOINT, 'width': WIDTH, 'k': 32, 'grid': 28,
           'cities': [c['city'] for c in spec['cities']], 'panoramas': len(frame),
           'images': len(frame) * 4, 'top': args.top, 'cache_identity': spec['identity'],
           'cache_manifest_sha256': manifest_hash, 'sample_kind': 'expanded_posthoc',
           'target_features': features, 'target_dimensions': len(features),
           'selection': 'Only dimensions without a previous VLM interpretation; previously interpreted dimensions are preserved',
           'baseline_report': str(args.baseline.resolve()), 'baseline_identity': baseline_identity,
           'sae_split_counts': frame.sae_split.value_counts().to_dict(),
           'scope': 'Expanded post-hoc interpretation corpus, including SAE training and validation images; not an independent evaluation set',
           'ranking': 'maximum stored positive activation per distinct panorama; best heading; float16 ties by manifest order then earliest patch',
           'activation_storage': 'existing float16 CSR values after float32 panorama-wise BatchTopK',
           'headings': [0, 90, 180, 270], 'source_cache': str(args.cache.resolve())}
    if (args.source / 'run.json').exists() and read(args.source / 'run.json') != run:
        raise ValueError('Expansion identity changed; choose a new source directory')
    write(args.source / 'run.json', run)
    return spec, frame, run


def rank(args, spec, frame):
    target = args.source / 'top.npz'
    if target.exists():
        return
    jobs, offset = [], 0
    for city in spec['cities']:
        slug = city['city'].replace('/', '__')
        rows = frame[frame.city == city['city']]
        if rows.cache_pano_index.tolist() != list(range(city['panoramas'])):
            raise ValueError('Cache indices differ from manifest')
        for start in range(0, city['panoramas'], spec['chunk_panos']):
            jobs.append((slug, start, offset + start))
        offset += city['panoramas']
    wanted = np.zeros(WIDTH, dtype=np.bool_)
    wanted[read(args.source / 'run.json')['target_features']] = True
    status(args, 'ranking', status='running', chunks=0, total_chunks=len(jobs),
           target_dimensions=int(wanted.sum()), total_panoramas=len(frame))
    # Compile once before entering the worker pool.
    panorama_peaks(np.array([1.], np.float32), np.array([0], np.uint16),
                   np.array([0, 1], np.uint32), 1, np.array([True]), 1, 1)

    def chunk(job):
        slug, start, offset = job
        cached = args.source / 'ranking' / slug / f'{start:06d}.npz'
        folder = args.cache / 'activations' / slug / f'{start:06d}'
        marker = read(folder / 'complete.json')
        if marker.get('identity') != spec['identity'] or marker.get('start') != start:
            raise ValueError(f'Invalid sparse chunk: {folder}')
        count = marker['stop'] - start
        if cached.exists():
            with np.load(cached) as saved:
                peaks, positions = saved['peaks'], saved['patches']
        else:
            arrays = [np.load(folder / f'{key}.npy', mmap_mode='r') for key in ('values', 'indices', 'indptr')]
            for key, array in zip(('values', 'indices', 'indptr'), arrays):
                if list(array.shape) != marker['files'][key]['shape']:
                    raise ValueError(f'Truncated sparse cache: {folder}/{key}')
            if len(arrays[2]) != count * TOKENS + 1 or arrays[2][-1] != len(arrays[0]):
                raise ValueError('Invalid CSR bounds')
            peaks, positions = panorama_peaks(arrays[0].astype(np.float32), arrays[1], arrays[2], count, wanted)
            cached.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, peaks=peaks, patches=positions)
            temporary.replace(cached)
        if peaks.shape != (count, WIDTH):
            raise ValueError('Invalid ranking shape')
        return offset, peaks, positions

    values = np.zeros((args.top, WIDTH), np.float32)
    ids = np.full((args.top, WIDTH), -1, np.int64)
    patches = np.zeros((args.top, WIDTH), np.int16)
    support = np.zeros(WIDTH, np.int64)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (offset, peaks, positions) in enumerate(pool.map(chunk, jobs), 1):
            values, ids, patches = merge_top(values, ids, patches, peaks, positions, offset, args.top)
            support += (peaks > 0).sum(axis=0)
            if i % 32 == 0 or i == len(jobs):
                status(args, 'ranking', status='running', chunks=i, total_chunks=len(jobs),
                       panoramas=offset + len(peaks), total_panoramas=len(frame),
                       observed_dimensions=int((support > 0).sum()), elapsed_seconds=round(time.monotonic() - started, 1))
    np.savez_compressed(target.with_suffix('.tmp.npz'), values=values, ids=ids, patches=patches, support=support)
    target.with_suffix('.tmp.npz').replace(target)
    summary = {'panoramas': len(frame), 'images': len(frame) * 4, 'cities': len(spec['cities']),
               'target_dimensions': int(wanted.sum()), 'preserved_dimensions': int((~wanted).sum()),
               'observed_dimensions': int((support[wanted] > 0).sum()),
               'unobserved_dimensions': int((support[wanted] == 0).sum()),
               'full_top10_dimensions': int((support[wanted] >= args.top).sum()),
               'top_instances': int(np.minimum(support[wanted], args.top).sum()),
               'previously_unobserved_now_observed': int((support[wanted] > 0).sum()),
               'sample_kind': 'expanded_posthoc', 'checkpoint_sha256': CHECKPOINT}
    write(args.source / 'ranking_summary.json', summary)
    print('RANKING_COMPLETE', json.dumps(summary), flush=True)


def crop_record(args, spec, frame, feature, rank, pid, position, activation):
    row = frame.iloc[int(pid)]
    start = int(row.cache_pano_index) // spec['chunk_panos'] * spec['chunk_panos']
    folder = args.cache / 'activations' / row.city.replace('/', '__') / f'{start:06d}'
    values, dimensions, ptr = [np.load(folder / f'{k}.npy', mmap_mode='r') for k in ('values', 'indices', 'indptr')]
    heading = int(position) // PATCHES
    first = ((int(row.cache_pano_index) - start) * 4 + heading) * PATCHES
    begin, end = int(ptr[first]), int(ptr[first + PATCHES])
    matches = np.flatnonzero(dimensions[begin:end] == feature) + begin
    patch_ids = np.searchsorted(ptr[first:first + PATCHES + 1], matches, side='right') - 1
    heat = np.zeros(PATCHES, np.float32)
    heat[patch_ids] = values[matches].astype(np.float32)
    if heat.max() != activation or int(heat.argmax()) != position % PATCHES:
        raise ValueError('Ranking and activation map differ')
    buffer = io.BytesIO()
    Image.fromarray(np.rint(heat.reshape(28, 28) / activation * 255).astype(np.uint8)).save(buffer, format='PNG')
    image_path = row[f'image_{heading * 90}']
    with Image.open(image_path) as image:
        image_size = list(image.size)
    return {'feature': int(feature), 'rank': int(rank), 'activation': float(activation),
            'pano_index': int(pid), 'cache_pano_index': int(row.cache_pano_index), 'city': row.city,
            'pano': str(row.panoid), 'heading': heading * 90, 'image': image_path,
            'native_view_size': image_size, 'peak_patch': int(position % PATCHES),
            'heatmap_png': base64.b64encode(buffer.getvalue()).decode(),
            'sae_split': row.sae_split, 'prototype_split': row.prototype_split}


def crops(args, spec, frame):
    target = args.source / 'crops.jsonl'
    if target.exists():
        return
    with np.load(args.source / 'top.npz') as saved:
        values, ids, patches = saved['values'], saved['ids'], saved['patches']
    def feature_records(feature):
        path = args.source / 'crop_shards' / f'f{feature:04d}.json'
        if path.exists():
            return read(path)
        rows = [crop_record(args, spec, frame, feature, r + 1, ids[r, feature], patches[r, feature], values[r, feature])
                for r in range(args.top) if values[r, feature] > 0]
        write(path, rows)
        return rows
    with ThreadPoolExecutor(max_workers=args.workers) as pool, target.with_suffix('.tmp').open('w') as out:
        for i, rows in enumerate(pool.map(feature_records, range(WIDTH)), 1):
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
            if i % 64 == 0 or i == WIDTH:
                status(args, 'crops', status='running', dimensions=i, total_dimensions=WIDTH)
    target.with_suffix('.tmp').replace(target)


def sheets(args):
    from sae_experiments.panorama_vitl30_dimension_vlm import contact_sheet, materialize_masked
    evidence = read(args.report / 'evidence.json')['dimensions']
    def build(row):
        if not row['count']:
            return
        target = args.report / row['sheets'][0]
        if not target.exists():
            materialized = materialize_masked(args.source, row['crops'], args.report, .5)
            contact_sheet(args.source, materialized, target, tile=224, columns=5)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(build, evidence), 1):
            if i % 64 == 0 or i == WIDTH:
                status(args, 'sheets', status='running', dimensions=i, total_dimensions=WIDTH)


def run_vlm(args):
    from sae_experiments.panorama_vitl30_dimension_vlm import MODEL
    args.model, args.max_new_tokens = MODEL, 512
    args.input_mode, args.mask_fraction = 'masked-top10', .5
    from sae_experiments.panorama_vitl30_dimension_vlm import prepare as prepare_vlm, report
    prepare_vlm(args)
    report(args)
    sheets(args)
    command = [sys.executable, '-u', '-m', 'sae_experiments.panorama_vitl30_dimension_vlm_runner',
               '--source', str(args.source), '--report', str(args.report),
               '--devices', args.devices, '--batch-size', '1', '--min-free-mib', str(args.min_free_mib)]
    status(args, 'vlm', status='waiting_for_gpu', gpu_only=True, devices=args.devices,
           detail='GPU inference only; no CPU model or offloading')
    with (args.report / 'runner.log').open('a', buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                time.sleep(15)
                runner = read(args.report / 'runner_status.json')
                summary = read(args.report / 'summary.json')
                status(args, 'vlm', gpu_only=True, devices=args.devices,
                       status='waiting_for_gpu' if runner.get('status', 'waiting_gpu') == 'waiting_gpu' else runner['status'],
                       interpreted=summary.get('status_counts', {}).get('interpreted', 0),
                       unobserved=summary.get('status_counts', {}).get('unobserved_in_sample', 0),
                       requested_dimensions=summary.get('requested_dimensions'))
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    summary = read(args.report / 'summary.json')
    if (process.returncode != 0 or summary.get('completed_or_explicitly_unobserved')
            != summary.get('requested_dimensions')):
        raise RuntimeError(f'VLM runner failed with exit {process.returncode}; see runner.log')
    status(args, 'complete', status='complete', **read(args.source / 'ranking_summary.json'))
    write(args.report / 'complete.json', {'checkpoint_sha256': CHECKPOINT,
           'source_scope': read(args.source / 'run.json'), 'summary': summary})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--report', type=Path, default=REPORT)
    p.add_argument('--baseline', type=Path, default=BASELINE)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--devices', default='1,3', help='CUDA devices only; no CPU inference or offloading')
    p.add_argument('--min-free-mib', type=int, default=7000)
    p.add_argument('--top', type=int, default=10, choices=[10])
    p.add_argument('--stage', choices=['rank', 'crops', 'vlm', 'all'], default='all')
    args = p.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    lock = (args.report / 'expansion.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def stop(signum, frame):
        raise KeyboardInterrupt('Expansion stopped')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        spec, frame, run = prepare(args)
        if args.stage in {'rank', 'all'}:
            rank(args, spec, frame)
        if args.stage in {'crops', 'all'}:
            crops(args, spec, frame)
        if args.stage in {'vlm', 'all'}:
            run_vlm(args)
    except KeyboardInterrupt as exc:
        status(args, 'stopped', status='stopped', error=str(exc))
        raise
    except BaseException as exc:
        status(args, 'failed', status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
