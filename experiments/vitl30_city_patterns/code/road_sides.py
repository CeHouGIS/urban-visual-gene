"""Independent road segmentation and frozen-SAE four-view evidence audit.

Geometric hypotheses are deliberately separate from verified roadside labels.
The source cohort lacks per-image camera calibration, so default final labels
remain undetermined, even when a stable road-axis hypothesis is available.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from .candidate_combinations import ELEMENTS, HIERARCHY, family_dimensions
from .extract import (DEFAULT_OUT, HEADINGS, PATCHES, WIDTH, image_paths,
                      load_models, preprocess, write_json)
from .followup import FOLLOWUP, sha256
from experiments.vitl30_multiscale_prototypes.code.data import sparse_encode
from sae_experiments.models.panorama_context import mix_panorama_context

SEGMENTER = Path('/root/.cache/urban-visual-gene/segformer-b0-cityscapes')
VERSION = 1
COLORS = ((255, 183, 0), (0, 220, 240))


def bearings(fov=90., size=28):
    # Pixel centres of a perspective view; horizontal yaw runs left to right.
    x = 2 * (np.arange(size) + .5) / size - 1
    local = np.arctan(x * np.tan(np.deg2rad(fov / 2)))
    return np.deg2rad(np.asarray(HEADINGS))[:, None] + local[None]


def axis_estimate(road, fov=90.):
    """Antipodal road evidence around the image horizon; not a fitted 3D road."""
    road = np.asarray(road, dtype=float)
    if road.shape != (4, 28, 28):
        raise ValueError('Expected independent road scores on four 28x28 grids')
    weights = np.where(road[:, 13:19] >= .6, road[:, 13:19], 0).sum(axis=1)
    mass = float(weights.sum())
    theta = bearings(fov)
    if mass < 12:
        return {'valid': False, 'reason': 'insufficient_visible_road', 'confidence': 0.}
    z = np.sum(weights * np.exp(2j * theta)) / mass
    concentration = float(abs(z))
    axis = float((np.angle(z) / 2) % np.pi)
    forward = float(weights[np.cos(theta - axis) >= 0].sum()) / mass
    valid = concentration >= .6 and .2 <= forward <= .8
    return {'valid': bool(valid), 'axis_degrees': float(np.rad2deg(axis)),
            'confidence': concentration, 'forward_mass_fraction': forward,
            'reason': 'provisional_straight_axis' if valid else 'ambiguous_or_non_antipodal_road'}


def side_assignment(activation, road, axis, fov=90.):
    values = np.asarray(activation, dtype=float).copy()
    if values.max() <= 0:
        return 0, 0.
    values[values < .35 * values.max()] = 0
    values[:, :5] = 0  # uncertain far-horizon/sky responses do not define sides
    values[road >= .5] = 0  # exclude attribution on the road surface itself
    theta = bearings(fov)[:, None, :]
    signed = np.broadcast_to(np.sin(theta - np.deg2rad(axis)), values.shape)
    left = values[signed > np.sin(np.deg2rad(15))].sum()
    right = values[signed < -np.sin(np.deg2rad(15))].sum()
    total = values.sum()
    if total <= 0:
        return 0, 0.
    confidence = float(max(left, right) / total)
    return (1 if left > right else -1) if confidence >= .85 else 0, confidence


def classify(maps, road, a, b):
    hypotheses, diagnostics = [], []
    for fov in (75., 90., 105.):
        axis = axis_estimate(road, fov)
        record = {'fov': fov, **axis}
        label = 'undetermined'
        if axis['valid']:
            sa, ca = side_assignment(maps[a], road, axis['axis_degrees'], fov)
            sb, cb = side_assignment(maps[b], road, axis['axis_degrees'], fov)
            record.update(side_a=sa, side_b=sb, confidence_a=ca, confidence_b=cb)
            if sa and sb:
                label = 'same_side' if sa == sb else 'opposite_side'
        hypotheses.append(label)
        diagnostics.append(record)
    stable = len(set(hypotheses)) == 1 and hypotheses[0] != 'undetermined'
    return {'side_label': 'undetermined',
            'tentative_side': hypotheses[0] if stable else 'undetermined',
            'reason': 'camera_calibration_unverified' if stable else 'ambiguous_road_or_element_masks',
            'road_axis_confidence': min(d['confidence'] for d in diagnostics),
            'calibration_verified': False, 'human_validated': False, 'diagnostics': diagnostics}


def choose_examples(frame, candidates, per_candidate=12):
    selected = []
    for candidate in candidates:
        a, b = candidate.split('+')
        pool = frame[frame[f'present_{a}'] & frame[f'present_{b}']].copy()
        pool['selection_hash'] = [hashlib.sha256(f'{candidate}/{c}/{p}'.encode()).hexdigest()
                                  for c, p in zip(pool.city, pool.panoid)]
        pool = pool.sort_values('selection_hash')
        used_cells, city_count, count = set(), Counter(), 0
        for row in pool.itertuples(index=False):
            cell = (row.city, row.grid_x, row.grid_y)
            if cell in used_cells or city_count[row.city] >= 3:
                continue
            selected.append({'candidate': candidate, 'city': row.city, 'panoid': row.panoid,
                             'grid_x': int(row.grid_x), 'grid_y': int(row.grid_y)})
            used_cells.add(cell)
            city_count[row.city] += 1
            count += 1
            if count >= per_candidate:
                break
    return selected


@torch.inference_mode()
def infer(images, dino, sae, segmenter, processor, lookup, device):
    pixels = torch.stack([preprocess()(im) for im in images]).to(device, dtype=torch.bfloat16)
    prefix = 1 + int(getattr(dino.config, 'num_register_tokens', 0))
    tokens = torch.nn.functional.normalize(dino(pixel_values=pixels).last_hidden_state[:, prefix:].float(), dim=-1)
    context = mix_panorama_context(tokens.reshape(1, 4, PATCHES, 1024), grid_size=28)
    values, indices = sparse_encode(sae, context)
    values, indices = values[0], indices[0]
    family = lookup[indices % WIDTH]
    valid = (family >= 0) & (values > 0)
    flat = torch.zeros(4 * PATCHES * len(ELEMENTS), device=device)
    flat.scatter_add_(0, (indices[valid] // WIDTH) * len(ELEMENTS) + family[valid], values[valid])
    array = flat.reshape(4, 28, 28, len(ELEMENTS)).cpu().numpy()
    # Cityscapes model receives its documented 1024x1024 preprocessing.
    road = []
    for im in images:
        inputs = processor(images=im, return_tensors='pt').to(device)
        logits = segmenter(**inputs).logits.float()
        prob = torch.nn.functional.interpolate(logits, size=(448, 448), mode='bilinear', align_corners=False).softmax(1)
        road.append(torch.nn.functional.adaptive_avg_pool2d(prob[:, 0:1], (28, 28))[0, 0].cpu().numpy())
    return {f: array[..., i] for i, f in enumerate(ELEMENTS)}, np.stack(road)


def render(images, maps, road, a, b, folder):
    folder.mkdir(parents=True, exist_ok=True)
    width, height = 336, 336
    original = Image.new('RGB', (width * 4, height + 24), '#141920')
    overlay = original.copy()
    for h, source in enumerate(images):
        rgb = np.asarray(source.resize((width, height)), dtype=float)
        shown = rgb * .4
        mask_road = np.asarray(Image.fromarray((road[h] * 255).astype('uint8')).resize((width, height), Image.Resampling.NEAREST)) >= 153
        shown[mask_road] = .55 * rgb[mask_road] + .45 * np.array([230, 100, 255])
        for family, color in zip((a, b), COLORS):
            field = maps[family][h]
            binary = field >= max(.35 * float(maps[family].max()), 1e-8)
            mask = np.asarray(Image.fromarray(binary).resize((width, height), Image.Resampling.NEAREST))
            shown[mask] = .5 * rgb[mask] + .5 * np.asarray(color)
        original.paste(Image.fromarray(rgb.astype('uint8')), (h * width, 24))
        overlay.paste(Image.fromarray(shown.clip(0, 255).astype('uint8')), (h * width, 24))
        for canvas in (original, overlay):
            ImageDraw.Draw(canvas).text((h * width + 8, 6), f'{HEADINGS[h]} degrees', fill='white')
    original.save(folder / 'original.webp', quality=88)
    overlay.save(folder / 'overlay.webp', quality=90)


def run(args):
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('GPU required; no CPU inference fallback')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.followup / 'stages/candidate_combinations'
    protocol = json.loads((source / 'protocol.json').read_text())
    if protocol['run_identity'] != args.run_identity:
        raise ValueError('Candidate extraction identity changed')
    candidates = pd.read_csv(source / 'candidates.csv')
    names = candidates.loc[candidates.status == 'co_presence_candidate', 'candidate'].tolist()
    frame = pd.read_parquet(source / 'panorama_elements.parquet')
    examples = choose_examples(frame, names, args.per_candidate)
    segmenter_hash = sha256(args.segmenter / 'pytorch_model.bin')
    experiment = {'version': VERSION, 'run_identity': args.run_identity,
                  'candidate_protocol_sha256': sha256(source / 'protocol.json'),
                  'hierarchy_sha256': sha256(HIERARCHY), 'segmenter_sha256': segmenter_hash,
                  'per_candidate': args.per_candidate}
    identity = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    processor = AutoImageProcessor.from_pretrained(args.segmenter, local_files_only=True)
    segmenter = SegformerForSemanticSegmentation.from_pretrained(args.segmenter, local_files_only=True).to(args.device).eval()
    if segmenter.config.id2label[0].lower() != 'road':
        raise ValueError('Unexpected segmentation class mapping')
    dino, sae, _ = load_models(args.device)
    dimensions = family_dimensions(json.loads(HIERARCHY.read_text()))
    lookup = torch.full((WIDTH,), -1, dtype=torch.long, device=args.device)
    for i, family in enumerate(ELEMENTS):
        lookup[torch.as_tensor(dimensions[family], device=args.device)] = i
    results = []
    for i, example in enumerate(examples):
        key = hashlib.sha256(f'{example["city"]}/{example["panoid"]}'.encode()).hexdigest()[:20]
        folder = args.output / 'evidence' / key
        cache = folder / 'activation.npz'
        paths = image_paths(example['city'], example['panoid'])
        images = []
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['identity']) != identity:
                    raise ValueError('Road audit cache identity changed')
                maps = {f: saved[f] for f in ELEMENTS}
                road = saved['independent_road']
        else:
            maps, road = infer(images, dino, sae, segmenter, processor, lookup, args.device)
            folder.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix('.partial.npz')
            np.savez_compressed(temp, identity=identity, independent_road=road, **maps)
            temp.replace(cache)
        a, b = example['candidate'].split('+')
        item = {**example, **classify(maps, road, a, b)}
        # Candidate-specific masks; a panorama can occur in several candidates.
        evidence = folder / example['candidate']
        render(images, maps, road, a, b, evidence)
        item.update(original=str(evidence / 'original.webp'), overlay=str(evidence / 'overlay.webp'))
        results.append(item)
        write_json(args.output / 'progress.json', {'status': 'running', 'processed': i + 1,
                    'total': len(examples), 'device': args.device, 'identity': identity})
        print(f'Road/SAE audit {i + 1}/{len(examples)}: {example["candidate"]} {item["tentative_side"]}', flush=True)
    write_json(args.output / 'examples.json', results)
    fields = ['candidate', 'city', 'panoid', 'grid_x', 'grid_y', 'road_axis_confidence',
              'side_label', 'tentative_side', 'reason', 'calibration_verified', 'human_validated', 'original', 'overlay']
    with (args.output / 'side_examples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    summary = {k: sum(r['side_label'] == k for r in results)
               for k in ('same_side', 'opposite_side', 'undetermined')}
    summary.update(examples=len(results), candidates=len(names),
                   tentative=dict(Counter(r['tentative_side'] for r in results)),
                   cities=len({r['city'] for r in results}),
                   cells=len({(r['city'], r['grid_x'], r['grid_y']) for r in results}),
                   human_validated=False, calibration_verified=False,
                   corpus_prevalence_estimable=False, status='audit_complete_review_required')
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'protocol.json', {**experiment, 'identity': identity,
        'segmenter': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024; road class 0',
        'road_mask_source': 'independent pretrained semantic segmentation, not SAE',
        'selection': 'hash order; maximum 3 panoramas/city/candidate; distinct cells',
        'family_mask': 'positive SAE family mass >=35% of panorama family maximum; attribution, not object segmentation',
        'camera_geometry': 'perspective atan projection; 75/90/105 degree sensitivity; nominal cardinal headings',
        'axis': 'second circular moment of near-horizon road probabilities, concentration>=0.6; antipodal mass>=0.2',
        'side': '>=85% non-road element mass on one signed half-plane; exclude +/-15 degrees around axis',
        'final_decision': 'undetermined pending acquisition calibration and human mask/road verification',
        'limitations': ['No metric object distance or object counts.',
                        'Intersections, curves, occlusions and segmentation domain shift can invalidate hypotheses.',
                        'Purposive audit sample is not a prevalence sample.',
                        'Qwen family names are provisional; geometry hypotheses are not verified prototypes.']})
    write_json(args.output / 'complete.json', {'schema_version': 1, 'stage': 'road_sides',
        'run_identity': args.run_identity, 'status': 'complete', 'evidence': summary})
    write_json(args.output / 'progress.json', {'status': 'complete', **summary})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    p.add_argument('--followup', type=Path, default=FOLLOWUP)
    p.add_argument('--output', type=Path, default=FOLLOWUP / 'stages/road_sides')
    p.add_argument('--run-identity', required=True)
    p.add_argument('--device', default='cuda:3')
    p.add_argument('--segmenter', type=Path, default=SEGMENTER)
    p.add_argument('--per-candidate', type=int, default=12)
    args = p.parse_args()
    if args.per_candidate < 1:
        p.error('per-candidate must be positive')
    run(args)
