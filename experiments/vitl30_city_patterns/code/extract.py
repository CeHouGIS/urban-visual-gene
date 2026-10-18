#!/usr/bin/env python3
"""Stream sampled four-view panoramas through the frozen main DINOv3-L and SAE.

No dense patch features are persisted. Each independently selected panorama
gets a 4096-dimension pooled SAE vector and four 64-group heading vectors.
Chunks are atomic and may be distributed across GPUs or resumed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from scipy import ndimage
import torch
from PIL import Image
from transformers import AutoModel

from sae_experiments.io_utils import digest
from sae_experiments.models.panorama_context import mix_panorama_context
from sae_experiments.panorama_vitl30 import (DIM, HEADINGS, MODEL_DIR, OUTPUT,
                                             PATCHES, WIDTH, preprocess)
from sae_experiments.panorama_vitl30_analysis import checked_groups, model_and_identity
from experiments.vitl30_multiscale_prototypes.code.data import sparse_encode
from formal.quality.quality_filter import RES as QC_RES, flag_row

ROOT = Path(__file__).resolve().parents[3]
METADATA = Path('/host/root/mnt/nas/huangyj/GoogleSV/metadata')
IMAGES = Path('/host/root/mnt/nas/huangyj/GoogleSV/images')
RUN_ROOT = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10')
DEFAULT_SAMPLE = RUN_ROOT / 'selection'
DEFAULT_OUT = RUN_ROOT / 'extraction'
EXPECTED_CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'
PANORAMA_PATCHES = 4 * PATCHES
GROUPS = 64


def cities():
    experiment = json.loads((OUTPUT / 'experiment.json').read_text())
    result = [name for name, _ in experiment['identity']['cities']]
    if len(result) != 30 or len(set(result)) != 30:
        raise ValueError('Main experiment city list changed')
    return result


def image_paths(city, panoid):
    pid = str(panoid)
    if len(pid) < 2 or '/' in pid or '\\' in pid:
        raise ValueError(f'Invalid panoid: {pid}')
    directory = IMAGES / city / pid[0] / pid[1]
    return tuple(directory / f'{pid}_{heading}.jpg' for heading in HEADINGS)


def eligible(row, flag, has_four_count):
    if row[flag] is not True:
        return False
    if has_four_count and (row['#images'] is None or row['#images'] < 4):
        return False
    return bool(row['panoid'])


def candidate_chunks(city, sample_root, rows_per_chunk):
    source = sample_root / city / 'selected.parquet'
    file = pq.ParquetFile(source)
    selected = ['panoid', 'lat', 'lon', 'grid_x', 'grid_y']
    if not set(selected).issubset(file.schema.names):
        raise ValueError(f'Incomplete selection: {source}')
    for index, batch in enumerate(file.iter_batches(batch_size=rows_per_chunk, columns=selected)):
        rows = batch.to_pylist()
        yield index, rows, len(rows)


def identity(city_list, sample_root, rows_per_chunk):
    manifest = json.loads((sample_root / 'manifest.json').read_text())
    if manifest['cell_meters'] != 1000 or manifest['max_panoramas_per_cell'] != 10:
        raise ValueError('Expected ten panoramas per 1-km cell')
    sample_records = {item['city']: item for item in manifest['cities']}
    sources = []
    for city in city_list:
        record = sample_records[city]
        path = sample_root / city / 'selected.parquet'
        stat = path.stat()
        if stat.st_size != record['selected_bytes'] or digest(path) != record['selected_sha256']:
            raise ValueError(f'Selection changed: {path}')
        sources.append({'city': city, 'selection_rows': pq.ParquetFile(path).metadata.num_rows,
                        'selection_sha256': record['selected_sha256']})
    model_path = MODEL_DIR / 'model.safetensors'
    model_stat = model_path.stat()
    payload = {'version': 3, 'cities': sources, 'rows_per_chunk': rows_per_chunk,
               'sample_root': str(sample_root.resolve()),
               'sample_seed': manifest['seed'], 'cell_meters': manifest['cell_meters'],
               'max_panoramas_per_cell': manifest['max_panoramas_per_cell'],
               'checkpoint_sha256': digest(OUTPUT / 'model_best.pt'),
               'dino_model_path': str(model_path), 'dino_weights_sha256': digest(model_path),
               'dino_weights_bytes': model_stat.st_size,
               'dino_weights_mtime_ns': model_stat.st_mtime_ns,
               'image_root': str(IMAGES), 'input_size': 448, 'grid': [28, 28],
               'context_weight': .25, 'sae_width': WIDTH, 'sae_k': 32,
               'quality_gate': 'existing heuristic thresholds: any black or two tunnel-like headings; blur retained',
               'selection': 'exact BatchTopK independently over 3136 patches per panorama',
               'representation': 'pooled positive SAE activation by 4096 dimensions and four heading/64 groups'}
    if payload['checkpoint_sha256'] != EXPECTED_CHECKPOINT:
        raise ValueError('Main SAE checkpoint changed')
    payload['identity'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)


def load_models(device):
    weights = MODEL_DIR / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    dino = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).eval().requires_grad_(False)
    if dino.config.hidden_size != DIM or dino.config.patch_size != 16:
        raise ValueError('Expected main DINOv3 ViT-L/16')
    dino = dino.to(device)
    if device.startswith('cuda'):
        dino = dino.to(torch.bfloat16)
    sae, sae_hash = model_and_identity(OUTPUT, device)
    if sae_hash != EXPECTED_CHECKPOINT:
        raise ValueError('Wrong SAE checkpoint')
    groups = checked_groups(OUTPUT, sae_hash)
    return dino, sae, torch.as_tensor(groups, dtype=torch.int64, device=device)


def quality_flags(rgb):
    array = np.asarray(rgb.resize((QC_RES, QC_RES)), dtype=np.float32)
    luminance = array @ np.array([.299, .587, .114], np.float32)
    bright = float(luminance.mean())
    dark = float((luminance < 20).mean())
    std = float(luminance.std())
    lapvar = float(ndimage.laplace(luminance).var())
    warm = float(array[..., 0].mean() - array[..., 2].mean())
    skytop = float(luminance[:QC_RES // 4].mean())
    threshold = np.percentile(luminance, 95)
    glow = float(luminance[luminance >= threshold].mean() - bright)
    black, _, tunnel = flag_row(bright, dark, std, lapvar, 0., warm, skytop, glow)
    return black, tunnel


def read_panorama(city, row, transform):
    paths = image_paths(city, row['panoid'])
    if not all(path.is_file() for path in paths):
        return 'missing', None
    try:
        images = []
        tunnels = 0
        for path in paths:
            with Image.open(path) as source:
                rgb = source.convert('RGB')
                black, tunnel = quality_flags(rgb)
                if black:
                    return 'black', None
                tunnels += int(tunnel)
                images.append(transform(rgb))
        if tunnels >= 2:
            return 'tunnel', None
        return 'ok', torch.stack(images)
    except (OSError, ValueError):
        return 'unreadable', None


@torch.inference_mode()
def encode_batch(dino, sae, groups, pixels, device):
    batch = len(pixels)
    image_batch = torch.cat(pixels).to(device)
    if device.startswith('cuda'):
        image_batch = image_batch.to(torch.bfloat16)
    prefix = 1 + int(getattr(dino.config, 'num_register_tokens', 0))
    hidden = dino(pixel_values=image_batch).last_hidden_state[:, prefix:].float()
    if hidden.shape[1:] != (PATCHES, DIM):
        raise ValueError(f'Unexpected DINO patch shape: {tuple(hidden.shape)}')
    hidden = torch.nn.functional.normalize(hidden, dim=-1)
    context = mix_panorama_context(hidden.reshape(batch, 4, PATCHES, DIM), grid_size=28)
    values, indices = sparse_encode(sae, context)
    dim = indices.remainder(WIDTH)
    composition = torch.zeros((batch, WIDTH), device=device, dtype=torch.float32)
    composition.scatter_add_(1, dim, values)
    composition.div_(PANORAMA_PATCHES)
    heading = torch.div(indices, WIDTH * PATCHES, rounding_mode='floor')
    heading_group = torch.zeros((batch, 4 * GROUPS), device=device, dtype=torch.float32)
    heading_group.scatter_add_(1, heading * GROUPS + groups[dim], values)
    heading_group.div_(PATCHES)
    return (composition.cpu().numpy().astype(np.float16),
            heading_group.reshape(batch, 4, GROUPS).cpu().numpy().astype(np.float16))


def save_chunk(folder, run_id, chunk_index, source_rows, candidates, output, skipped):
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'vectors.npz'
    temporary = folder / f'vectors.{os.getpid()}.partial.npz'
    np.savez(temporary,
             panoid=np.asarray([item[0] for item in output], dtype='U64'),
             lat=np.asarray([item[1] for item in output], np.float32),
             lon=np.asarray([item[2] for item in output], np.float32),
             grid_x=np.asarray([item[3] for item in output], np.int32),
             grid_y=np.asarray([item[4] for item in output], np.int32),
             composition=np.stack([item[5] for item in output]) if output else np.zeros((0, WIDTH), np.float16),
             heading_groups=np.stack([item[6] for item in output]) if output else np.zeros((0, 4, GROUPS), np.float16))
    temporary.replace(archive)
    write_json(folder / 'complete.json', {'identity': run_id, 'chunk': chunk_index,
        'selection_rows': source_rows, 'candidate_panoramas': candidates,
        'processed_panoramas': len(output), 'skipped': skipped,
        'archive_bytes': archive.stat().st_size})


def run(args):
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Full-city extraction requires a CUDA GPU')
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard index')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    selected = cities() if args.city is None else [args.city]
    if any(city not in cities() for city in selected):
        raise ValueError('City is not in the main 30-city experiment')
    specification = identity(selected, args.sample, args.rows_per_chunk)
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / 'run.json'
    if marker.exists() and json.loads(marker.read_text()) != specification:
        raise ValueError('Output directory belongs to another experiment')
    if not marker.exists():
        write_json(marker, specification)
    dino, sae, groups = load_models(args.device)
    transform = preprocess()
    started = time.monotonic()
    total = {'selection_rows': 0, 'candidates': 0, 'processed': 0,
             'missing': 0, 'black': 0, 'tunnel': 0, 'unreadable': 0, 'chunks': 0}
    task = 0
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        for city in selected:
            for chunk_index, rows, source_rows in candidate_chunks(city, args.sample, args.rows_per_chunk):
                assigned = task % args.shards == args.shard
                task += 1
                if not assigned:
                    continue
                folder = args.output / city.replace('/', '__') / f'{chunk_index:06d}'
                done = folder / 'complete.json'
                if done.exists():
                    saved = json.loads(done.read_text())
                    if saved['identity'] != specification['identity'] or saved['chunk'] != chunk_index:
                        raise ValueError(f'Stale result chunk: {folder}')
                    archive = folder / 'vectors.npz'
                    if not archive.exists() or archive.stat().st_size != saved['archive_bytes']:
                        raise ValueError(f'Incomplete result chunk: {folder}')
                    total['selection_rows'] += saved['selection_rows']
                    total['candidates'] += saved['candidate_panoramas']
                    total['processed'] += saved['processed_panoramas']
                    for reason, count in saved['skipped'].items():
                        total[reason] += count
                    total['chunks'] += 1
                    continue
                if args.max_chunks and total['chunks'] >= args.max_chunks:
                    break
                kept = []
                skipped = {'missing': 0, 'black': 0, 'tunnel': 0, 'unreadable': 0}
                for at in range(0, len(rows), args.batch_panos):
                    batch_rows = rows[at:at + args.batch_panos]
                    results = list(pool.map(lambda row: read_panorama(city, row, transform), batch_rows))
                    valid = [(row, pixels) for row, (status, pixels) in zip(batch_rows, results)
                             if status == 'ok']
                    for status, _ in results:
                        if status != 'ok':
                            skipped[status] += 1
                    if not valid:
                        continue
                    composition, heading_groups = encode_batch(dino, sae, groups,
                                            [pixels for _, pixels in valid], args.device)
                    for i, (row, _) in enumerate(valid):
                        kept.append((str(row['panoid']), row['lat'], row['lon'],
                                     row['grid_x'], row['grid_y'],
                                     composition[i], heading_groups[i]))
                save_chunk(folder, specification['identity'], chunk_index,
                           source_rows, len(rows), kept, skipped)
                total['selection_rows'] += source_rows
                total['candidates'] += len(rows)
                total['processed'] += len(kept)
                for reason, count in skipped.items():
                    total[reason] += count
                total['chunks'] += 1
                elapsed = time.monotonic() - started
                progress = {'status': 'running', 'shard': args.shard, 'shards': args.shards,
                            'city': city, 'chunk': chunk_index, **total,
                            'elapsed_seconds': round(elapsed, 2),
                            'new_panoramas_per_second': round(total['processed'] / max(elapsed, 1), 3)}
                write_json(args.output / f'progress_shard{args.shard}.json', progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            if args.max_chunks and total['chunks'] >= args.max_chunks:
                break
    write_json(args.output / f'progress_shard{args.shard}.json',
               {'status': 'complete', 'shard': args.shard, 'shards': args.shards,
                **total, 'elapsed_seconds': round(time.monotonic() - started, 2)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sample', type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument('--city', help='Restrict to one of the main 30 cities for a pilot')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--rows-per-chunk', type=int, default=1024)
    parser.add_argument('--batch-panos', type=int, default=2)
    parser.add_argument('--io-workers', type=int, default=4)
    parser.add_argument('--max-chunks', type=int, default=0)
    run(parser.parse_args())
