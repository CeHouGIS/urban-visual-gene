"""Sample current landscape compositions for conservative roadside hypotheses.

This is an image audit, not corpus prevalence. A hypothesis requires a stable
road axis and one dominant non-road side for every participating response unit.
Without camera calibration and semantic mask review final labels stay unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from .composition_evidence import choose_examples, infer as infer_units, render
from .extract import HEADINGS, image_paths, load_models, write_json
from .followup import FOLLOWUP, sha256
from .road_sides import SEGMENTER, axis_estimate, side_assignment

VERSION = 1
NOT_LATERAL_FAMILIES = {'road', 'transport_structure'}
NOT_LATERAL_KEYS = {'pedestrian:人行横道', 'utility:电线'}


def lateral_units(pattern, units):
    return [i for i in pattern['units']
            if units[i]['family'] not in NOT_LATERAL_FAMILIES
            and units[i]['key'] not in NOT_LATERAL_KEYS]


def classify_pattern(maps, road, identifiers):
    hypotheses, details = [], []
    for fov in (75., 90., 105.):
        axis = axis_estimate(road, fov)
        sides, confidences = [], []
        if axis['valid']:
            for identifier in identifiers:
                side, confidence = side_assignment(maps[identifier], road,
                                                  axis['axis_degrees'], fov)
                sides.append(side)
                confidences.append(confidence)
        label = ('same_side' if len(set(sides)) == 1 else 'opposite_sides') if (
            len(sides) == len(identifiers) and all(sides)) else 'undetermined'
        hypotheses.append(label)
        details.append({'fov': fov, 'axis': axis, 'sides': sides,
                        'unit_confidences': confidences, 'hypothesis': label})
    stable = len(set(hypotheses)) == 1 and hypotheses[0] != 'undetermined'
    return {'tentative_side': hypotheses[0] if stable else 'undetermined',
            'final_side': 'undetermined', 'calibration_verified': False,
            'human_validated': False, 'diagnostics': details}


@torch.inference_mode()
def infer_road(images, processor, segmenter, device):
    result = []
    for image in images:
        batch = processor(images=image, return_tensors='pt').to(device)
        logits = segmenter(**batch).logits.float()
        probabilities = torch.nn.functional.interpolate(
            logits, size=(448, 448), mode='bilinear', align_corners=False).softmax(1)
        result.append(torch.nn.functional.adaptive_avg_pool2d(
            probabilities[:, :1], (28, 28))[0, 0].cpu().numpy())
    return np.stack(result)


def render_road(images, road, folder):
    """Show independent road segmentation on all four original headings."""
    size = 336
    canvas = Image.new('RGB', (4 * size, size + 24), '#141920')
    for heading, source in enumerate(images):
        rgb = np.asarray(source.resize((size, size)), dtype=np.float32)
        mask = Image.fromarray((road[heading] * 255).astype('uint8'))
        selected = np.asarray(mask.resize((size, size), Image.Resampling.NEAREST)) >= 153
        shown = rgb * .55
        shown[selected] = .55 * rgb[selected] + .45 * np.array([230, 100, 255])
        canvas.paste(Image.fromarray(shown.clip(0, 255).astype('uint8')),
                     (heading * size, 24))
        ImageDraw.Draw(canvas).text((heading * size + 8, 6),
                                    f'{HEADINGS[heading]} degrees', fill='white')
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / 'road.webp', quality=89)


def summarise(rows, all_patterns, units, requested):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['pattern_id']].append(row)
    table = []
    for pattern in all_patterns:
        lateral = lateral_units(pattern, units)
        selected = grouped[pattern['pattern_id']]
        counts = Counter(r['tentative_side'] for r in selected)
        table.append({'pattern_id': pattern['pattern_id'], 'label': pattern['label'],
            'eligible_for_side_audit': len(lateral) >= 2,
            'lateral_units': [units[i]['label'] for i in lateral],
            'sampled_panoramas': len(selected),
            'sampled_cities': len({r['city'] for r in selected}),
            'same_side_hypotheses': counts['same_side'],
            'opposite_sides_hypotheses': counts['opposite_sides'],
            'undetermined_hypotheses': counts['undetermined'],
            'final_same_side': 0, 'final_opposite_sides': 0,
            'final_undetermined': len(selected),
            'corpus_matching_panoramas': pattern['positive_panoramas']})
    counts = Counter(r['tentative_side'] for r in rows)
    total = len(rows)
    summary = {'version': VERSION, 'patterns_total': len(all_patterns),
        'patterns_side_eligible': sum(t['eligible_for_side_audit'] for t in table),
        'patterns_not_side_eligible': sum(not t['eligible_for_side_audit'] for t in table),
        'per_pattern_requested': requested, 'sampled_pattern_instances': total,
        'unique_panoramas': len({(r['city'], r['panoid']) for r in rows}),
        'sampled_cities': len({r['city'] for r in rows}),
        'tentative': {k: counts[k] for k in ('same_side', 'opposite_sides', 'undetermined')},
        'final': {'same_side': 0, 'opposite_sides': 0, 'undetermined': total},
        'human_validated': False, 'calibration_verified': False,
        'corpus_roadside_prevalence_estimable': False,
        'interpretation': 'Stratified retrieval audit; hypotheses cannot be extrapolated to 681,645 panoramas.'}
    return table, summary


def run(args):
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Roadside image audit requires GPU; no CPU inference fallback')
    torch.set_num_threads(4)
    source = args.followup / 'stages/composition_patterns'
    corpus = json.loads((args.followup / 'complete.json').read_text())
    marker = json.loads((source / 'complete.json').read_text())
    if marker['run_identity'] != corpus['identity']['extract_run_identity']:
        raise ValueError('Composition and corpus identities differ')
    units = json.loads((source / 'units.json').read_text())
    patterns = json.loads((source / 'patterns.json').read_text())
    eligible = [r for r in patterns if len(lateral_units(r, units)) >= 2]
    frame = pd.read_parquet(args.followup / 'stages/candidate_combinations/panorama_elements.parquet')
    assignment = np.load(source / 'assignment.npy', mmap_mode='r')
    if len(frame) != len(assignment) or len(frame) != corpus['processed']:
        raise ValueError('Panorama metadata and composition assignments differ')
    selected = choose_examples(frame, assignment, units, eligible, args.per_pattern)
    groups = defaultdict(list)
    for row in selected:
        groups[row['city'], row['panoid']].append(row)
    segmenter_hash = sha256(args.segmenter / 'pytorch_model.bin')
    protocol = {'version': VERSION, 'run_identity': marker['run_identity'],
        'composition_protocol_sha256': sha256(source / 'protocol.json'),
        'segmenter_sha256': segmenter_hash, 'per_pattern': args.per_pattern,
        'selection': 'deterministic hash; distinct cities per composition; eligibility before sampling',
        'side_eligibility': {'excluded_families': sorted(NOT_LATERAL_FAMILIES),
                             'excluded_keys': sorted(NOT_LATERAL_KEYS)},
        'road': 'independent Cityscapes SegFormer road probability at 28x28 per heading',
        'axis': 'near-horizon antipodal road evidence; concentration >=0.6 across 75/90/105 degree FOV',
        'unit_side': '>=85% non-road SAE response mass on one signed side; exclude axis +/-15 degrees',
        'combination': 'all lateral units same side vs at least one on each side; require agreement across all FOVs',
        'final_decision': 'undetermined pending heading/FOV calibration and independent semantic-mask review',
        'limitations': ['SAE maps are responses, not instance segmentation.',
                        'A unit can represent multiple objects on both sides.',
                        'The per-composition sample is not a corpus prevalence sample.',
                        'Curves and intersections can invalidate a straight road axis.']}
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'protocol.json', {**protocol, 'identity': identity})
    processor = AutoImageProcessor.from_pretrained(args.segmenter, local_files_only=True)
    segmenter = SegformerForSemanticSegmentation.from_pretrained(
        args.segmenter, local_files_only=True).to(args.device).eval()
    if segmenter.config.id2label[0].lower() != 'road':
        raise ValueError('Unexpected road class mapping')
    dino, sae, _ = load_models(args.device)
    results = []
    for index, ((city, panoid), requests) in enumerate(groups.items(), 1):
        ids = sorted({i for r in requests for i in lateral_units(r, units)})
        key = hashlib.sha256(f'{city}/{panoid}'.encode()).hexdigest()[:20]
        folder = args.output / 'evidence' / key
        cache = folder / 'activation.npz'
        with_images = []
        for path in image_paths(city, panoid):
            with Image.open(path) as image:
                with_images.append(image.convert('RGB'))
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['identity']) != identity or not np.array_equal(saved['unit_ids'], ids):
                    raise ValueError('Roadside cache identity or unit set changed')
                maps, road = saved['maps'], saved['road']
        else:
            maps = infer_units(with_images, dino, sae, ids, units, args.device)
            road = infer_road(with_images, processor, segmenter, args.device)
            folder.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix('.partial.npz')
            np.savez_compressed(temp, identity=identity, unit_ids=ids, maps=maps, road=road)
            temp.replace(cache)
        lookup = dict(zip(ids, maps))
        render_road(with_images, road, folder)
        for request in requests:
            relevant = lateral_units(request, units)
            outcome = classify_pattern(lookup, road, relevant)
            evidence = folder / request['pattern_id']
            render(with_images, np.stack([lookup[i] for i in relevant]), evidence)
            results.append({**request, 'lateral_units': relevant,
                'lateral_labels': [units[i]['label'] for i in relevant],
                **outcome, 'original': str(evidence / 'original.webp'),
                'overlay': str(evidence / 'overlay.webp'),
                'road_overlay': str(folder / 'road.webp')})
        write_json(args.output / 'examples.json', results)
        write_json(args.output / 'progress.json', {'status': 'running',
            'unique_panoramas_done': index, 'unique_panoramas_total': len(groups),
            'pattern_instances_done': len(results), 'pattern_instances_total': len(selected)})
        if index % 10 == 0 or index == len(groups):
            print(f'Roadside audit: {index}/{len(groups)} panoramas; {len(results)}/{len(selected)} pattern instances', flush=True)
    table, summary = summarise(results, patterns, units, args.per_pattern)
    summary.update(run_identity=marker['run_identity'], protocol_identity=identity)
    pd.DataFrame(table).to_csv(args.output / 'pattern_side_audit.csv', index=False)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'complete.json', {'status': 'complete',
        'run_identity': marker['run_identity'], 'identity': identity,
        'examples': len(results), 'patterns': len(patterns)})
    write_json(args.output / 'progress.json', {'status': 'complete', **summary})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup', type=Path, default=FOLLOWUP)
    parser.add_argument('--output', type=Path, default=FOLLOWUP / 'stages/composition_roadside_audit')
    parser.add_argument('--segmenter', type=Path, default=SEGMENTER)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--per-pattern', type=int, default=6)
    args = parser.parse_args()
    if args.per_pattern < 1:
        parser.error('per-pattern must be positive')
    run(args)
