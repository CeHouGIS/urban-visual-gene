"""Dimension-level semantic UMAP from held-out ViT-L panorama activation crops.

Reuses the frozen SAE and verified cached features. One panorama contributes
at most one observation per dimension. No decoder vectors, geographical
coordinates or group labels are inputs to the semantic projection.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from formal.analysis.semantic_feature_umap import (
    TopImages, crop_region, digest, embed_crops, reduce_features, write_json,
    write_jsonl,
)
from sae_experiments.panorama_vitl30 import OUTPUT, DIM, PATCHES, WIDTH
from sae_experiments.panorama_vitl30_analysis import model_and_identity
from sae_experiments.panorama_vitl30_cache import cache_city, items

MODEL = '/root/.cache/urban_scene_models/siglip-base-patch16-224'
DATASET = '30-city panorama-aware ViT-L SAE W4096 K32: held-out image semantics'


def heldout_ids(city, total, seed=42, count=128):
    """Exactly the split made before training, never a new random sample."""
    city_seed = int.from_bytes(hashlib.sha256(f'{seed}/{city}'.encode()).digest()[:8], 'little')
    return np.sort(np.random.default_rng(city_seed).choice(np.arange(total), count, replace=False))


def prepare(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((args.output / 'experiment.json').read_text())
    model, checkpoint_hash = model_and_identity(args.output, 'cpu')
    del model
    identity = experiment['identity']
    rows, selected, excluded, identities = [], [], [], {}
    for item in items(args.output):
        city = item['city']
        frame, tokens, cache_hash = cache_city(args.output, city)
        if not frame.image_mapping_verified.all():
            excluded.append(city)
            continue
        ids = heldout_ids(city, len(tokens), identity['seed'], identity['val_panos'])
        identities[city] = cache_hash
        for pid in ids:
            pano_index = len(selected)
            views = frame.iloc[int(pid) * 4:int(pid) * 4 + 4]
            if views.heading.tolist() != [0, 90, 180, 270] or views.panoid.nunique() != 1:
                raise ValueError(f'Invalid panorama mapping: {city}/{pid}')
            for position, row in enumerate(views.itertuples()):
                if not Path(row.image_path).is_file():
                    raise FileNotFoundError(row.image_path)
                rows.append({'city': city, 'pano': str(row.panoid), 'heading': int(row.heading),
                             'image': str(row.image_path), 'feature_path': item['feature_path'],
                             'row_start': (int(pid) * 4 + position) * PATCHES,
                             'pano_index': pano_index, 'cache_pano_index': int(pid), 'grid': 28})
            selected.append({'city': city, 'cache_pano_index': int(pid), 'pano_index': pano_index,
                             'pano': str(views.panoid.iloc[0])})
    spec = {'dataset': DATASET, 'checkpoint': str(args.output / 'model_best.pt'),
            'checkpoint_sha256': checkpoint_hash, 'width': WIDTH, 'k': 32, 'grid': 28,
            'cities': list(identities), 'excluded_unmapped_cities': excluded,
            'images': len(rows), 'panoramas': len(selected), 'top': args.top,
            'crop_fraction': .5, 'margin': 1, 'min_side': 3, 'vlm_model': args.vlm_model,
            'encoder_weights_sha256': digest(Path(args.vlm_model) / 'model.safetensors'),
            'embedding_precision': args.embedding_precision,
            'activation': 'float32 joint BatchTopK per complete 3136-patch panorama',
            'ranking': 'maximum positive sparse activation per distinct panorama; best heading only; ties by manifest order',
            'sampling': 'exact pre-training held-out split, 128 panoramas per mapped city',
            'seed': identity['seed'], 'cache_identities': identities,
            'sample_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            'scope': 'Top100 within this explicit held-out sample, not a whole-NAS ranking'}
    old = out / 'run.json'
    if old.exists() and json.loads(old.read_text()) != spec:
        raise ValueError('Input/configuration changed; use a different output directory')
    write_json(old, spec)
    write_jsonl(out / 'images.jsonl', rows)
    write_jsonl(out / 'panoramas.jsonl', selected)
    return rows, selected, spec


@torch.inference_mode()
def rank(args, rows, selected):
    out = Path(args.outdir)
    if (out / 'top_images.jsonl').exists():
        return
    model, _ = model_and_identity(args.output, args.device)
    ranking = TopImages(WIDTH, args.top)
    per_city = defaultdict(list)
    for pano in selected:
        per_city[pano['city']].append(pano)
    for city, panos in per_city.items():
        cache = out / 'ranking' / (city.replace('/', '__') + '.npz')
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            with np.load(cache) as saved:
                peaks, patches = saved['peaks'], saved['patches']
                if saved['ids'].tolist() != [p['cache_pano_index'] for p in panos]:
                    raise ValueError('Stale ranking cache')
        else:
            _, tokens, _ = cache_city(args.output, city)
            peaks, patches = [], []
            begin = time.monotonic()
            with ThreadPoolExecutor(max_workers=1) as pool:
                def read(p):
                    return np.array(tokens[p['cache_pano_index']], np.float32).reshape(-1, DIM)
                future = pool.submit(read, panos[0])
                for i, pano in enumerate(panos):
                    block = future.result()
                    future = pool.submit(read, panos[i+1]) if i+1 < len(panos) else None
                    x = torch.from_numpy(block).to(args.device)
                    acts = model.encode(x)
                    v, ix = acts.max(0)
                    peaks.append(v.cpu().numpy())
                    patches.append(ix.cpu().numpy().astype(np.int16))
                    del x, acts
                    if (i+1) % 32 == 0:
                        print('RANK', city, i+1, '/', len(panos), round(time.monotonic()-begin, 1), 's', flush=True)
            peaks, patches = np.stack(peaks), np.stack(patches)
            temp = cache.with_suffix('.tmp.npz')
            np.savez_compressed(temp, peaks=peaks, patches=patches,
                                ids=[p['cache_pano_index'] for p in panos])
            temp.replace(cache)
        for pano, values in zip(panos, peaks):
            ranking.add(pano['pano_index'], values)
        print('RANK CITY COMPLETE', city, flush=True)
    # Recover the winning direction from saved per-panorama peak locations.
    selected_rows = list(ranking.rows())
    by_pano = defaultdict(list)
    for row in selected_rows:
        row['pano_index'] = row['image_id']
        by_pano[row['pano_index']].append(row)
    for city, panos in per_city.items():
        with np.load(out / 'ranking' / (city.replace('/', '__') + '.npz')) as saved:
            patches = saved['patches']
            for pano, peak in zip(panos, patches):
                for row in by_pano[pano['pano_index']]:
                    patch = int(peak[row['feature']])
                    row['image_id'] = 4 * pano['pano_index'] + patch // PATCHES
                    row['peak_patch'] = patch % PATCHES
    np.save(out / 'positive_panorama_support.npy', ranking.support)
    np.save(out / 'positive_image_support.npy', ranking.support)  # compatibility; unit recorded in run.json
    write_jsonl(out / 'top_images.jsonl', selected_rows)
    del model
    torch.cuda.empty_cache()


def save_crops(task):
    """One image per worker task; identical image/bbox pairs share clean pixels."""
    image_id, row, selected, heats, output = task
    out = Path(output)
    with Image.open(row['image']) as source:
        image = source.convert('RGB')  # same geometry as feature extraction
    result = []
    directory = out / 'crops' / f'{image_id // 1000:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    for item, heat in zip(selected, heats):
        if not np.isclose(heat.max(), item['activation'], rtol=1e-5, atol=1e-6):
            raise ValueError('Ranking/crop activation differs')
        bbox, info = crop_region(heat, image.size, .5, 1, 3)
        filename = f'i{image_id:05d}_' + '_'.join(map(str, bbox)) + '.jpg'
        target = directory / filename
        if not target.exists():
            image.crop(bbox).save(target, quality=92)
        buffer = io.BytesIO()
        Image.fromarray(np.rint(heat / heat.max() * 255).astype(np.uint8)).save(buffer, format='PNG')
        result.append({**item, 'city': row['city'], 'pano': row['pano'], 'heading': row['heading'],
                       'image': row['image'], 'crop': str(target.relative_to(out)),
                       'heatmap_png': base64.b64encode(buffer.getvalue()).decode(),
                       'bbox_native_view': bbox, 'native_view_size': list(image.size), **info})
    return result


@torch.inference_mode()
def crops(args, rows, selected):
    out = Path(args.outdir)
    if (out / 'crops.jsonl').exists():
        return
    by_pano = defaultdict(list)
    for line in (out / 'top_images.jsonl').open():
        row = json.loads(line)
        by_pano[row['pano_index']].append(row)
    model, _ = model_and_identity(args.output, args.device)
    all_crops, current_city, pending = [], None, []
    partial = out / 'crop_shards'
    partial.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.crop_workers) as pool:
        for number, (pid, values) in enumerate(sorted(by_pano.items()), 1):
            dest = partial / f'{pid:05d}.jsonl'
            if dest.exists():
                all_crops.extend(json.loads(x) for x in dest.read_text().splitlines())
                continue
            pano = selected[pid]
            if pano['city'] != current_city:
                _, tokens, _ = cache_city(args.output, pano['city'])
                current_city = pano['city']
            x = torch.from_numpy(np.array(tokens[pano['cache_pano_index']], np.float32).reshape(-1, DIM)).to(args.device)
            acts = model.encode(x)
            futures = []
            for heading in range(4):
                image_id = 4 * pid + heading
                view_items = [v for v in values if v['image_id'] == image_id]
                if not view_items:
                    continue
                ids = torch.tensor([v['feature'] for v in view_items], device=args.device)
                heat = acts[heading*PATCHES:(heading+1)*PATCHES, ids].T.contiguous().cpu().numpy().reshape(-1, 28, 28)
                futures.append(pool.submit(save_crops, (image_id, rows[image_id], view_items, heat, str(out))))
            del acts, x
            pending.append((dest, futures))
            if len(pending) >= args.crop_workers:
                path, jobs = pending.pop(0)
                saved = [r for future in jobs for r in future.result()]
                write_jsonl(path, saved)
                all_crops.extend(saved)
            if number % 64 == 0:
                print('CROP', number, '/', len(by_pano), 'panoramas;', len(all_crops), 'regions', flush=True)
        for path, jobs in pending:
            saved = [r for future in jobs for r in future.result()]
            write_jsonl(path, saved)
            all_crops.extend(saved)
    write_jsonl(out / 'crops.jsonl', sorted(all_crops, key=lambda r: (r['feature'], r['rank'])))
    write_json(out / 'crop_errors.json', [])  # image failures stop rather than silently dropping rank entries
    del model
    torch.cuda.empty_cache()


def embed(args):
    out = Path(args.outdir)
    if (out / 'embedding_run.json').exists():
        return
    rows = [json.loads(x) for x in (out / 'crops.jsonl').read_text().splitlines()]
    unique, ids, reverse = [], {}, []
    for row in rows:
        if row['crop'] not in ids:
            ids[row['crop']] = len(unique)
            unique.append({**row, 'crop': '../' + row['crop']})
        reverse.append(ids[row['crop']])
    unique_out = out / 'unique_embeddings'
    unique_out.mkdir(exist_ok=True)
    write_jsonl(unique_out / 'crops.jsonl', unique)
    unique_args = copy(args)
    unique_args.outdir = str(unique_out)
    print('ENCODER', len(unique), 'unique crops for', len(rows), 'dimension observations', flush=True)
    embed_crops(unique_args)
    encoded = np.load(unique_out / 'crop_embeddings.f32.npy', mmap_mode='r')
    expanded = np.lib.format.open_memmap(out / 'crop_embeddings.f32.npy', mode='w+',
                                        dtype=np.float32, shape=(len(rows), encoded.shape[1]))
    for start in range(0, len(rows), 8192):
        expanded[start:start+8192] = encoded[np.asarray(reverse[start:start+8192])]
    expanded.flush()
    np.save(out / 'embedding_completed.npy', np.ones(len(rows), dtype=bool))
    meta = json.loads((unique_out / 'embedding_run.json').read_text())
    write_json(out / 'embedding_run.json', {**meta, 'rows': len(rows), 'unique_crops': len(unique),
               'unique_crop_manifest_sha256': meta['crop_manifest_sha256'],
               'crop_manifest_sha256': digest(out / 'crops.jsonl'),
               'deduplication': 'Identical source image and native bbox encoded once; every per-dimension observation retains equal weight'})
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--outdir', default=str(OUTPUT / 'analysis/semantic_umap_top100'))
    parser.add_argument('--stage', choices=['all', 'prepare', 'rank', 'crop', 'embed', 'reduce'], default='all')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--vlm-model', default=MODEL)
    parser.add_argument('--top', type=int, default=100)
    parser.add_argument('--embedding-batch', type=int, default=16)
    parser.add_argument('--embedding-precision', default='bf16')
    parser.add_argument('--embedding-workers', type=int, default=4)
    parser.add_argument('--crop-workers', type=int, default=8)
    parser.add_argument('--min-examples', type=int, default=20)
    parser.add_argument('--neighbors', type=int, default=30)
    parser.add_argument('--min-dist', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.dataset = DATASET
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, selected, spec = prepare(args)
    print('PREPARED', len(spec['cities']), 'cities,', len(selected), 'panoramas;', len(rows), 'images', flush=True)
    for stage, function in [('rank', lambda: rank(args, rows, selected)),
                            ('crop', lambda: crops(args, rows, selected)),
                            ('embed', lambda: embed(args)),
                            ('reduce', lambda: reduce_features(args))]:
        if args.stage in ['all', stage]:
            begin = time.monotonic()
            print('START', stage, flush=True)
            function()
            print('DONE', stage, round(time.monotonic()-begin, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
