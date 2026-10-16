"""Bounded cache manifests, panorama-wise SAE inference, and sparse storage."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from sae_experiments.io_utils import digest
from sae_experiments.panorama_vitl30 import OUTPUT, DIM, WIDTH, PATCHES
from sae_experiments.panorama_vitl30_analysis import model_and_identity
from sae_experiments.panorama_vitl30_cache import cache_city, items
from sae_experiments.panorama_vitl30_semantic_umap import heldout_ids

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
TOKENS = 4 * PATCHES


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def block_split(city, lat, lon, seed=42, block_m=1000, validation_fraction=.2):
    lat, lon = np.asarray(lat), np.asarray(lon)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError(f'Missing geographical coordinates: {city}')
    cosine = max(math.cos(math.radians(float(np.median(lat)))), .01)
    east = np.floor(np.radians(lon) * 6371000 * cosine / block_m).astype(np.int64)
    north = np.floor(np.radians(lat) * 6371000 / block_m).astype(np.int64)
    blocks = [f'{city}/{x}/{y}' for x, y in zip(east, north)]
    validation = [int.from_bytes(hashlib.sha256(f'{seed}/{key}'.encode()).digest()[:8], 'little') / 2**64
                  < validation_fraction for key in blocks]
    return blocks, np.where(validation, 'validation', 'train')


def prepare(out=OUT, source=OUTPUT, seed=42, chunk_panos=64):
    out, source = Path(out), Path(source)
    experiment = json.loads((source / 'experiment.json').read_text())
    checkpoint_hash = digest(source / 'model_best.pt')
    frames, city_rows, excluded = [], [], []
    for city_index, item in enumerate(items(source)):
        city = item['city']
        frame, tokens, identity = cache_city(source, city)
        if not frame.image_mapping_verified.all():
            excluded.append(city)
            continue
        if frame.heading.tolist() != [0, 90, 180, 270] * len(tokens):
            raise ValueError(f'Heading order differs: {city}')
        if frame.panoid.astype(str).to_numpy().reshape(-1, 4).tolist() != [[str(p)] * 4 for p in frame.panoid.iloc[::4]]:
            raise ValueError(f'Panorama IDs differ across directions: {city}')
        pano = frame.iloc[::4].reset_index(drop=True)
        blocks, splits = block_split(city, pano.lat.to_numpy(), pano.lon.to_numpy(), seed)
        old_validation = set(heldout_ids(city, len(tokens), experiment['identity']['seed'], experiment['identity']['val_panos']))
        selected = pd.DataFrame({'city': city, 'city_index': city_index,
            'cache_pano_index': np.arange(len(tokens)), 'panoid': pano.panoid.astype(str),
            'lat': pano.lat.to_numpy(), 'lon': pano.lon.to_numpy(), 'spatial_block': blocks,
            'prototype_split': splits,
            'sae_split': ['validation' if i in old_validation else 'train' for i in range(len(tokens))]})
        for h in range(4):
            selected[f'image_{h * 90}'] = frame.image_path.iloc[h::4].to_numpy()
        frames.append(selected)
        city_rows.append({'city': city, 'city_index': city_index, 'panoramas': len(tokens),
            'train': int((splits == 'train').sum()), 'validation': int((splits == 'validation').sum()),
            'feature_path': item['feature_path'], 'cache_identity': identity,
            'manifest_sha256': digest(source / 'cache_manifests' / f'{city.replace("/", "__")}.parquet')})
    manifest = pd.concat(frames, ignore_index=True)
    if manifest.duplicated(['city', 'panoid']).any():
        raise ValueError('Duplicate panorama identifiers')
    identity = {'version': 1, 'source': str(source.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'seed': seed, 'width': WIDTH, 'grid': 28, 'headings': [0, 90, 180, 270], 'k': 32,
        'activation_selection': 'float32 exact BatchTopK independently over all 3136 patches of each panorama',
        'activation_storage': 'all positive activations, CSR patch rows, uint16 indices and float16 values',
        'chunk_panos': chunk_panos, 'prototype_split': 'hash of 1-km spatial blocks, 20% validation; all headings stay together',
        'sae_split': 'original training split preserved as metadata; frozen SAE saw some prototype-validation panoramas during SAE training',
        'cities': city_rows, 'excluded_unmapped_cities': excluded, 'panoramas': len(manifest),
        'directional_images': len(manifest) * 4,
        'train_panoramas': int((manifest.prototype_split == 'train').sum()),
        'validation_panoramas': int((manifest.prototype_split == 'validation').sum())}
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = out / 'corpus.json'
    if previous.exists() and json.loads(previous.read_text()) != identity:
        raise ValueError('Corpus identity changed; use another output directory')
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'panoramas.parquet').exists():
        temporary = out / 'panoramas.partial.parquet'
        manifest.to_parquet(temporary, index=False)
        temporary.replace(out / 'panoramas.parquet')
    else:
        pd.testing.assert_frame_equal(manifest, pd.read_parquet(out / 'panoramas.parquet'))
    write_json(previous, identity)
    return identity


@torch.inference_mode()
def sparse_encode(model, tokens):
    """Vectorized GEMM, but an independent TopK budget for EACH panorama."""
    batch = tokens.shape[0]
    pre = model._preactivate(tokens.reshape(-1, DIM)).relu().reshape(batch, -1)
    values, indices = pre.topk(model.k * TOKENS, dim=1, sorted=False)
    return values, indices


def csr_pack(values, indices, width=WIDTH, tokens=TOKENS):
    all_values, all_indices, ptrs = [], [], [np.zeros(1, np.uint32)]
    for val, ix in zip(values, indices):
        valid = val > 0
        val, ix = val[valid], ix[valid]
        order = np.argsort(ix)
        val, ix = val[order], ix[order]
        all_values.append(val.astype(np.float16))
        all_indices.append((ix % width).astype(np.uint16))
        counts = np.bincount(ix // width, minlength=tokens)
        ptrs.append((np.cumsum(counts, dtype=np.uint64) + int(ptrs[-1][-1])).astype(np.uint32))
    return np.concatenate(all_values), np.concatenate(all_indices), np.concatenate(ptrs)


def extract(out=OUT, device='cuda:3', shard=0, shards=1, batch_panos=4):
    out = Path(out)
    torch.set_num_threads(4)
    # Avoid TF32 changes in selection near the top-k boundary.
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = json.loads((out / 'corpus.json').read_text())
    frame = pd.read_parquet(out / 'panoramas.parquet')
    model, model_hash = model_and_identity(Path(spec['source']), device)
    if model_hash != spec['checkpoint_sha256']:
        raise ValueError('SAE checkpoint changed')
    started, completed = time.monotonic(), 0
    for index, city in enumerate(spec['cities']):
        if index % shards != shard:
            continue
        city_frame = frame[frame.city == city['city']].reset_index(drop=True)
        _, tokens, cache_identity = cache_city(Path(spec['source']), city['city'])
        if cache_identity != city['cache_identity']:
            raise ValueError('Source cache changed')
        folder = out / 'activations' / city['city'].replace('/', '__')
        folder.mkdir(parents=True, exist_ok=True)
        chunk_size = spec['chunk_panos']
        with ThreadPoolExecutor(max_workers=1) as pool:
            def read(start, stop):
                return np.array(tokens[start:stop], dtype=np.float32)
            future = None
            for start in range(0, len(tokens), chunk_size):
                stop = min(start + chunk_size, len(tokens))
                chunk = folder / f'{start:06d}'
                marker = chunk / 'complete.json'
                if marker.exists():
                    saved = json.loads(marker.read_text())
                    if saved['identity'] != spec['identity'] or saved['stop'] != stop:
                        raise ValueError('Stale sparse activation chunk')
                    continue
                chunk.mkdir(exist_ok=True)
                packed_values, packed_indices, packed_ptrs = [], [], [np.zeros(1, np.uint32)]
                sums = np.zeros((stop - start, WIDTH), np.float32)
                train_count, train_square = np.zeros(WIDTH, np.int64), np.zeros(WIDTH, np.float64)
                future = pool.submit(read, start, min(start + batch_panos, stop))
                for at in range(start, stop, batch_panos):
                    array = future.result()
                    end = min(at + batch_panos, stop)
                    future = pool.submit(read, end, min(end + batch_panos, stop)) if end < stop else None
                    x = torch.from_numpy(array).to(device)
                    values, indices = sparse_encode(model, x)
                    values, indices = values.cpu().numpy(), indices.cpu().numpy()
                    val, col, ptr = csr_pack(values, indices)
                    packed_values.append(val); packed_indices.append(col)
                    packed_ptrs.append(ptr[1:] + packed_ptrs[-1][-1])
                    for j, (v, ix) in enumerate(zip(values, indices)):
                        valid = v > 0
                        v, d = v[valid].astype(np.float64), ix[valid] % WIDTH
                        sums[at - start + j] = np.bincount(d, weights=v, minlength=WIDTH) / TOKENS
                        if city_frame.prototype_split.iloc[at + j] == 'train':
                            train_count += np.bincount(d, minlength=WIDTH)
                            train_square += np.bincount(d, weights=v * v, minlength=WIDTH)
                    del x, values, indices
                arrays = {'values': np.concatenate(packed_values), 'indices': np.concatenate(packed_indices),
                          'indptr': np.concatenate(packed_ptrs), 'composition': sums,
                          'train_count': train_count, 'train_square': train_square}
                for name, array in arrays.items():
                    temp = chunk / f'{name}.partial.npy'
                    np.save(temp, array, allow_pickle=False)
                    temp.replace(chunk / f'{name}.npy')
                write_json(marker, {'identity': spec['identity'], 'city': city['city'], 'start': start,
                    'stop': stop, 'nnz': int(len(arrays['values'])),
                    'files': {name: {'bytes': (chunk / f'{name}.npy').stat().st_size,
                                     'shape': list(value.shape)} for name, value in arrays.items()}})
                completed += stop - start
                progress = {'shard': shard, 'device': device, 'city': city['city'], 'city_completed': stop,
                    'city_total': len(tokens), 'new_panoramas': completed,
                    'elapsed_seconds': round(time.monotonic() - started, 2)}
                write_json(out / f'extraction_shard{shard}.json', progress)
                print('EXTRACT', json.dumps(progress), flush=True)
    write_json(out / f'extraction_shard{shard}.json', {'status': 'complete', 'new_panoramas': completed,
                                                        'elapsed_seconds': time.monotonic() - started})


def finish_cache(out=OUT):
    out = Path(out)
    spec = json.loads((out / 'corpus.json').read_text())
    count, squares = np.zeros(WIDTH, np.int64), np.zeros(WIDTH, np.float64)
    chunks, nnz = [], 0
    composition = np.lib.format.open_memmap(out / 'composition.partial.npy', mode='w+', dtype=np.float32,
                                           shape=(spec['panoramas'], WIDTH))
    offset = 0
    for city in spec['cities']:
        folder = out / 'activations' / city['city'].replace('/', '__')
        for start in range(0, city['panoramas'], spec['chunk_panos']):
            chunk = folder / f'{start:06d}'
            saved = json.loads((chunk / 'complete.json').read_text())
            if saved['identity'] != spec['identity']:
                raise ValueError('Sparse chunk identity differs')
            for name, info in saved['files'].items():
                if (chunk / f'{name}.npy').stat().st_size != info['bytes']:
                    raise ValueError(f'Sparse chunk truncated: {chunk}/{name}')
            count += np.load(chunk / 'train_count.npy')
            squares += np.load(chunk / 'train_square.npy')
            composition[offset + start:offset + saved['stop']] = np.load(chunk / 'composition.npy')
            chunks.append(str(chunk.relative_to(out)))
            nnz += saved['nnz']
        offset += city['panoramas']
    composition.flush(); del composition
    (out / 'composition.partial.npy').replace(out / 'composition.npy')
    scale = np.sqrt(squares / np.maximum(count, 1)).astype(np.float32)
    scale[count == 0] = 1
    scale = np.maximum(scale, 1e-6)
    np.save(out / 'positive_rms.npy', scale)
    np.save(out / 'training_support.npy', count)
    write_json(out / 'cache_complete.json', {'identity': spec['identity'], 'chunks': len(chunks),
        'panoramas': spec['panoramas'], 'nnz': nnz, 'scale': 'positive RMS, prototype TRAIN spatial blocks only',
        'scale_sha256': digest(out / 'positive_rms.npy'), 'unobserved_train_dimensions': int((count == 0).sum())})


class SparseViews:
    def __init__(self, out=OUT, device='cpu'):
        self.out, self.device = Path(out), device
        self.spec = json.loads((self.out / 'corpus.json').read_text())
        self.frame = pd.read_parquet(self.out / 'panoramas.parquet')
        self.scale = torch.from_numpy(np.load(self.out / 'positive_rms.npy')).to(device)
        self.cache = OrderedDict()

    def chunk(self, city, pano_index):
        start = int(pano_index) // self.spec['chunk_panos'] * self.spec['chunk_panos']
        key = (city, start)
        if key not in self.cache:
            path = self.out / 'activations' / city.replace('/', '__') / f'{start:06d}'
            self.cache[key] = tuple(np.load(path / f'{k}.npy', mmap_mode='r') for k in ('values', 'indices', 'indptr'))
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key], int(pano_index) - start

    def load(self, ids, headings):
        flats, vals = [], []
        for batch_index, (row_id, heading) in enumerate(zip(ids, headings)):
            row = self.frame.iloc[int(row_id)]
            (values, dimensions, ptr), local = self.chunk(row.city, row.cache_pano_index)
            begin = (local * 4 + int(heading)) * PATCHES
            offsets = ptr[begin:begin + PATCHES + 1].astype(np.int64)
            start, stop = int(offsets[0]), int(offsets[-1])
            patch = np.repeat(np.arange(PATCHES), np.diff(offsets))
            dimension = dimensions[start:stop].astype(np.int64)
            flats.append((batch_index * WIDTH + dimension) * PATCHES + patch)
            vals.append(values[start:stop].astype(np.float32))
        flat = torch.from_numpy(np.concatenate(flats)).to(self.device)
        values = torch.from_numpy(np.concatenate(vals)).to(self.device)
        dense = torch.zeros((len(ids), WIDTH, 28, 28), device=self.device)
        dense.view(-1).scatter_(0, flat, values)
        return (dense / self.scale[None, :, None, None]).clamp_(max=8.)
