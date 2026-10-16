"""Current-checkpoint prototype exemplars, selected only from the dev partition.

Rank every dev view by its strongest template response. Retain city-specific
candidate pools, then require positive reconstruction contribution in float32.
No training, new feature extraction, or final-test access is performed.
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.evaluate import (
    SourceImageError, exemplar_evidence, native_evidence, template_figure,
)
from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder
from experiments.vitl30_prototype_coverage.code.trial import runtime
from sae_experiments.io_utils import digest
from .experiment import OUT, load_data


def response_maps(model, target):
    """The exact unmasked encoder path, omitting the unused dense decoder."""
    encoded = model.encoder(torch.cat([target, torch.zeros_like(target[:, :1])], 1))
    return torch.cat([F.relu(head(encoded)) for head in model.heads], 1)


def require_identity(record, identity):
    if record.get('identity') != identity:
        raise ValueError('Gallery checkpoint, partition, selection, or protocol changed')


@torch.inference_mode()
def run(device='cuda:1', batch_size=4, examples=8, per_city_pool=16):
    runtime(device)
    started = time.monotonic()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    checkpoint = OUT / 'model/best.pt'
    model_hash = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['experiment_config']['protocol_identity'] != protocol['identity']:
        raise ValueError('Checkpoint/protocol mismatch')
    model = PrototypeAutoencoder(**saved['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    data = load_data(OUT, device)
    frame = data.frame[data.frame.prototype_split == 'dev']
    count = len(model.scales) * model.per_scale
    folder = OUT / 'gallery'
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'version': 1, 'experiment_identity': protocol['identity'],
        'model_sha256': model_hash, 'sae_sha256': protocol['checkpoint_sha256'],
        'split': 'dev', 'panoramas': len(frame), 'images': len(frame) * 4,
        'examples_per_template': examples, 'per_city_candidate_pool': per_city_pool,
        'selection': 'peak response per directional view; unique panoramas; prefer <=2 per city; positive float32 single-response removal SSE change',
        'ranking_precision': 'bfloat16 autocast', 'evidence_precision': 'float32',
        'test_used': False}
    if (folder / 'protocol.json').exists():
        require_identity(json.loads((folder / 'protocol.json').read_text()), identity)
    write_json(folder / 'protocol.json', {'identity': identity})
    # Guard the fast ranking path against accidentally changing model semantics.
    first = int(frame.index[0])
    target = data.load([first], [0])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction, full_maps = model(target)
        fast_maps = response_maps(model, target)
    torch.testing.assert_close(full_maps, fast_maps, rtol=0, atol=0)
    del target, prediction, full_maps, fast_maps

    candidates = [[] for _ in range(count)]
    scanned = 0
    for city_index, (city, group) in enumerate(frame.groupby('city', sort=True)):
        city_path = folder / 'rankings' / (city.replace('/', '__') + '.json')
        if city_path.exists():
            city_result = json.loads(city_path.read_text())
            require_identity(city_result, identity)
        else:
            heaps = [[] for _ in range(count)]
            positive_views = np.zeros(count, np.int64)
            ids = group.index.to_numpy()
            for heading in range(4):
                for at in range(0, len(ids), batch_size):
                    selected = ids[at:at + batch_size]
                    target = data.load(selected, [heading] * len(selected))
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        maps = response_maps(model, target)
                    values, locations = maps.flatten(2).max(2)
                    values = values.float().cpu().numpy()
                    locations = locations.cpu().numpy()
                    positive_views += (values > 0).sum(0)
                    for b, row_id in enumerate(selected):
                        for p in np.flatnonzero(values[b] > 0):
                            location = int(locations[b, p])
                            row = (float(values[b, p]), int(row_id), heading, location // 28, location % 28)
                            if len(heaps[p]) < per_city_pool:
                                heapq.heappush(heaps[p], row)
                            elif row > heaps[p][0]:
                                heapq.heapreplace(heaps[p], row)
                    del target, maps, values, locations
            city_result = {'identity': identity, 'city': city, 'images': len(group) * 4,
                'positive_views': positive_views.tolist(),
                'candidates': [sorted(heap, reverse=True) for heap in heaps]}
            write_json(city_path, city_result)
        scanned += city_result['images']
        for p, rows in enumerate(city_result['candidates']):
            candidates[p].extend(rows)
        status = {'stage': 'ranking', 'cities': city_index + 1, 'total_cities': frame.city.nunique(),
            'images': scanned, 'total_images': len(frame) * 4,
            'elapsed_seconds': round(time.monotonic() - started, 1)}
        write_json(folder / 'status.json', status)
        print('RANK', json.dumps(status), flush=True)

    records = []
    for p in range(count):
        sub = folder / f'p{p:03d}'
        sub.mkdir(exist_ok=True)
        record_path = sub / 'prototype.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            require_identity(record, identity)
            records.append(record)
            continue
        bank, local = model.banks[p // model.per_scale], p % model.per_scale
        weights = (bank.channels[local] * bank.layout[local].sum((1, 2))[:, None]).sum(0).cpu().numpy()
        top = np.argsort(-weights)[:20]
        template_figure(bank, local, sub, weights)
        rows, rejected, failures, used, cities = [], [], [], set(), {}
        ranked = sorted(candidates[p], reverse=True)
        attempted = set()
        for city_cap in (2, examples):
            for candidate_index, (score, pano, heading, y, x) in enumerate(ranked):
                if pano in used or candidate_index in attempted:
                    continue
                source = data.frame.loc[pano]
                if source.prototype_split != 'dev':
                    raise ValueError('Non-dev exemplar')
                if cities.get(source.city, 0) >= city_cap:
                    continue
                attempted.add(candidate_index)
                target = data.load([pano], [heading])
                heat, evidence = exemplar_evidence(model, target, p, y, x)
                del target
                if evidence['coefficient_fp32'] <= 0 or evidence['removing_response_delta_sse'] <= 0:
                    rejected.append({'pano_row': pano, 'heading': heading * 90, **evidence})
                    continue
                stem = sub / f'r{len(rows) + 1:02d}'
                try:
                    native = native_evidence(source[f'image_{heading * 90}'], heat, evidence['box_patch'], stem)
                except SourceImageError as exc:
                    failures.append({'pano_row': pano, 'heading': heading * 90, 'error': str(exc)})
                    continue
                np.save(stem.with_suffix('.npy'), heat)
                rows.append({'rank': len(rows) + 1, 'pano_row': int(pano), 'panoid': str(source.panoid),
                    'city': str(source.city), 'heading': heading * 90, 'y': y, 'x': x,
                    'source_image': str(source[f'image_{heading * 90}']),
                    'ranking_response_bf16': score, **evidence, **native,
                    **{kind: stem.name + suffix for kind, suffix in {
                        'original': '_original.jpg', 'overlay': '_overlay.jpg',
                        'crop': '_crop.jpg', 'masked': '_masked.png'}.items()}})
                used.add(pano)
                cities[source.city] = cities.get(source.city, 0) + 1
                if len(rows) == examples:
                    break
            if len(rows) == examples:
                break
        status = 'candidate' if rows else ('no_positive_response' if not ranked else 'no_verified_helpful_example')
        record = {'prototype': p, 'scale': bank.size, 'identity': identity,
            'model_sha256': model_hash, 'status': status, 'instances': rows,
            'rejected_contribution': rejected, 'image_failures': failures,
            'candidates_checked': len(attempted),
            'top_dimensions': [{'dimension': int(d), 'weight': float(weights[d])} for d in top],
            'top20_weight_fraction': float(weights[top].sum() / max(float(weights.sum()), 1e-12)),
            'semantic_status': 'unreviewed; response/contribution evidence is not semantic validation'}
        write_json(record_path, record)
        records.append(record)
        status = {'stage': 'rendering', 'templates': p + 1, 'total_templates': count,
                  'current_template': p, 'examples': len(rows),
                  'elapsed_seconds': round(time.monotonic() - started, 1)}
        write_json(folder / 'status.json', status)
        print('GALLERY', json.dumps(status), flush=True)
    write_json(folder / 'prototypes.json', {'identity': identity, 'prototypes': records})
    status = {'stage': 'complete', 'identity': identity, 'templates': count,
              'templates_with_examples': sum(bool(p['instances']) for p in records),
              'examples': sum(len(p['instances']) for p in records),
              'elapsed_seconds': round(time.monotonic() - started, 1)}
    write_json(folder / 'complete.json', status)
    write_json(folder / 'status.json', status)
    print('COMPLETE', json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--examples', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.examples < 1:
        parser.error('batch size and example count must be positive')
    run(args.device, args.batch_size, args.examples)
