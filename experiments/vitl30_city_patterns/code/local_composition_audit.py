"""Audit compact same-view landscape compositions from the full-corpus retrieval.

The 681,645 panorama vectors select matches; only sampled matches have saved
patch maps. This audit measures locality in those samples, not corpus prevalence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .followup import FOLLOWUP
from .extract import write_json

PATCHES = 28
VIEW_PIXELS = 336
COLORS = ((255, 184, 34), (34, 220, 226), (234, 110, 255))


def supplement_patterns(followup, out, device, order):
    """Recompute patch maps for full-corpus patterns omitted by the roadside audit."""
    import pandas as pd
    import torch

    from .composition_evidence import choose_examples, infer, render
    from .extract import image_paths, load_models

    if order not in (2, 3):
        raise ValueError('Only pair and triplet patterns were mined')
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Patch inference requires GPU')
    patterns_dir = followup / 'stages/composition_patterns'
    roadside_dir = followup / 'stages/composition_roadside_audit'
    marker = json.loads((patterns_dir / 'complete.json').read_text())
    roadside = json.loads((roadside_dir / 'complete.json').read_text())
    corpus = json.loads((followup / 'complete.json').read_text())
    if (marker['run_identity'] != roadside['run_identity']
            or marker['run_identity'] != corpus['identity']['extract_run_identity']):
        raise ValueError('Composition, roadside and corpus identities differ')
    patterns = json.loads((patterns_dir / 'patterns.json').read_text())
    units = json.loads((patterns_dir / 'units.json').read_text())
    audited = json.loads((roadside_dir / 'examples.json').read_text())
    already_complete = {row['pattern_id'] for row in audited
                        if len(row['units']) == len(row['lateral_units'])}
    missing = [p for p in patterns if len(p['units']) == order
               and p['pattern_id'] not in already_complete]
    identity = hashlib.sha256(json.dumps({
        'run_identity': marker['run_identity'],
        'composition_protocol': hashlib.sha256((patterns_dir / 'protocol.json').read_bytes()).hexdigest(),
        'patterns': [p['pattern_id'] for p in missing], 'per_pattern': 6,
        'selection': 'full-corpus matches, hash ranked, distinct cities',
    }, sort_keys=True).encode()).hexdigest()
    output = out / ('supplemental_pairs' if order == 2 else 'supplemental_triplets')
    manifest = output / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['identity'] != identity or saved['status'] != 'complete':
            raise ValueError('Supplement manifest differs from current experiment')
        return saved
    frame = pd.read_parquet(followup / 'stages/candidate_combinations/panorama_elements.parquet')
    assignment = np.load(patterns_dir / 'assignment.npy', mmap_mode='r')
    if len(frame) != len(assignment) or len(frame) != corpus['processed']:
        raise ValueError('Full-corpus assignment and metadata differ')
    selected = choose_examples(frame, assignment, units, missing, 6)
    if len(selected) != 6 * len(missing):
        raise ValueError('A composition has fewer than six distinct-city examples')
    requests = defaultdict(list)
    for row in selected:
        requests[row['city'], row['panoid']].append(row)
    dino, sae, _ = load_models(device)
    rows = []
    for index, ((city, panoid), group) in enumerate(requests.items(), 1):
        unit_ids = sorted({unit for row in group for unit in row['units']})
        key = hashlib.sha256(f'{city}/{panoid}'.encode()).hexdigest()[:20]
        folder = output / 'evidence' / key
        cache = folder / 'activation.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['identity']) != identity or not np.array_equal(saved['unit_ids'], unit_ids):
                    raise ValueError('Supplement patch cache identity or units differ')
                maps = saved['maps']
        else:
            images = []
            for path in image_paths(city, panoid):
                with Image.open(path) as source:
                    images.append(source.convert('RGB'))
            maps = infer(images, dino, sae, unit_ids, units, device)
            folder.mkdir(parents=True, exist_ok=True)
            partial = cache.with_suffix('.partial.npz')
            np.savez_compressed(partial, identity=identity, unit_ids=unit_ids, maps=maps)
            partial.replace(cache)
        lookup = dict(zip(unit_ids, maps))
        for row in group:
            image_file = folder / row['pattern_id'] / 'original.webp'
            if not image_file.exists():
                images = []
                for path in image_paths(city, panoid):
                    with Image.open(path) as source:
                        images.append(source.convert('RGB'))
                render(images, np.stack([lookup[uid] for uid in row['units']]), image_file.parent)
            rows.append({**row, 'lateral_units': row['units'], 'original': str(image_file)})
        if index % 25 == 0 or index == len(requests):
            print(f'Order-{order} patch audit: {index}/{len(requests)} panoramas', flush=True)
    result = {'status': 'complete', 'identity': identity,
              'run_identity': marker['run_identity'], 'patterns': len(missing),
              'panoramas': len(requests), 'examples': rows}
    write_json(manifest, result)
    return result


def components(field, panorama_peak, minimum_area=8):
    """Return substantial connected responses with the displayed 35% threshold."""
    if panorama_peak <= 0:
        return []
    mask = field >= .35 * panorama_peak
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    found = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        component = labels[slices] == index
        area = int(component.sum())
        if area < minimum_area:
            continue
        y, x = slices
        box = (x.start, y.start, x.stop, y.stop)
        found.append({'mask': labels == index, 'box': box, 'area': area,
                      'mass': float(field[labels == index].sum())})
    return sorted(found, key=lambda item: -item['mass'])[:5]


def gap(a, b):
    x0, y0, x1, y1 = a['box']
    u0, v0, u1, v1 = b['box']
    return max(0, u0 - x1, x0 - u1), max(0, v0 - y1, y0 - v1)


def compact_match(maps, max_gap=5, max_box_fraction=.5):
    """Find one compact, connected arrangement of all units in one heading."""
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 4 or maps.shape[1:] != (4, PATCHES, PATCHES) or not 2 <= len(maps) <= 3:
        raise ValueError('Expected 2-3 complete four-view patch maps')
    peaks = maps.max(axis=(1, 2, 3))
    best = None
    for heading in range(4):
        groups = [components(maps[i, heading], peaks[i]) for i in range(len(maps))]
        if any(not group for group in groups):
            continue
        for chosen in itertools.product(*groups):
            box = (min(c['box'][0] for c in chosen), min(c['box'][1] for c in chosen),
                   max(c['box'][2] for c in chosen), max(c['box'][3] for c in chosen))
            width, height = box[2] - box[0], box[3] - box[1]
            if width * height > PATCHES * PATCHES * max_box_fraction or width > 22 or height > 22:
                continue
            # All units must connect through at least one nearby pair, rather
            # than merely occupying unrelated corners of the same crop.
            reachable = {0}
            while True:
                added = {j for i in reachable for j, other in enumerate(chosen)
                         if j not in reachable and max(gap(chosen[i], other)) <= max_gap}
                if not added:
                    break
                reachable |= added
            if len(reachable) != len(chosen):
                continue
            score = min(c['mass'] / max(peaks[i], 1e-8) for i, c in enumerate(chosen))
            candidate = {'heading': heading, 'box': box, 'masks': [c['mask'] for c in chosen],
                         'component_areas': [c['area'] for c in chosen], 'score': float(score)}
            if best is None or score > best['score']:
                best = candidate
    return best


def draw_example(source, match, destination):
    """Keep all four views visible; highlight only the compact matched regions."""
    with Image.open(source) as original:
        canvas = original.convert('RGB')
    heading = match['heading']
    if canvas.size != (4 * VIEW_PIXELS, VIEW_PIXELS + 24):
        raise ValueError('Unexpected four-view evidence strip size')
    x_offset = heading * VIEW_PIXELS
    for index, mask in enumerate(match['masks']):
        enlarged = Image.fromarray((mask * 255).astype(np.uint8), 'L').resize(
            (VIEW_PIXELS, VIEW_PIXELS), Image.Resampling.NEAREST)
        region = canvas.crop((x_offset, 24, x_offset + VIEW_PIXELS, 24 + VIEW_PIXELS))
        tint = Image.new('RGB', region.size, COLORS[index])
        region.paste(Image.blend(region, tint, .55), (0, 0), enlarged)
        canvas.paste(region, (x_offset, 24))
    box = match['box']
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x_offset + box[0] * 12, 24 + box[1] * 12,
                    x_offset + box[2] * 12 - 1, 24 + box[3] * 12 - 1),
                   outline=(255, 255, 255), width=3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, 'WEBP', quality=90)


def run(followup=FOLLOWUP, out=None):
    followup = Path(followup)
    out = Path(out) if out else followup / 'stages/local_composition_audit'
    patterns_dir = followup / 'stages/composition_patterns'
    roadside_dir = followup / 'stages/composition_roadside_audit'
    corpus = json.loads((followup / 'complete.json').read_text())
    marker = json.loads((patterns_dir / 'complete.json').read_text())
    roadside = json.loads((roadside_dir / 'complete.json').read_text())
    summary = json.loads((patterns_dir / 'summary.json').read_text())
    if (marker['status'] != 'complete' or roadside['status'] != 'complete'
            or marker['run_identity'] != roadside['run_identity']
            or marker['run_identity'] != corpus['identity']['extract_run_identity']
            or summary['panoramas'] != corpus['processed']):
        raise ValueError('Full-corpus composition and patch audit identities differ')
    patterns = {r['pattern_id']: r for r in json.loads((patterns_dir / 'patterns.json').read_text())}
    examples = json.loads((roadside_dir / 'examples.json').read_text())
    if len(patterns) != summary['recurrent_patterns'] or len(examples) != roadside['examples']:
        raise ValueError('Incomplete composition or patch evidence catalogue')
    examples = [{**row, '_cache_root': str(roadside_dir),
                 '_cache_identity': roadside['identity']} for row in examples]
    supplements = {}
    for order, name in ((2, 'supplemental_pairs'), (3, 'supplemental_triplets')):
        supplement_file = out / name / 'manifest.json'
        if not supplement_file.exists():
            continue
        supplement = json.loads(supplement_file.read_text())
        if supplement['status'] != 'complete' or supplement['run_identity'] != marker['run_identity']:
            raise ValueError(f'{name} belongs to another corpus run')
        supplements[order] = supplement
        examples.extend({**row, '_cache_root': str(supplement_file.parent),
                         '_cache_identity': supplement['identity']}
                        for row in supplement['examples'])
    grouped = defaultdict(list)
    usable = 0
    for row in examples:
        pattern = patterns.get(row['pattern_id'])
        if pattern is None or row['units'] != pattern['units']:
            raise ValueError('Sample refers to a different composition')
        if len(row['units']) != len(row['lateral_units']):
            continue
        usable += 1
        key = hashlib.sha256(f"{row['city']}/{row['panoid']}".encode()).hexdigest()[:20]
        with np.load(Path(row['_cache_root']) / 'evidence' / key / 'activation.npz',
                     allow_pickle=False) as saved:
            unit_ids = list(saved['unit_ids'])
            if str(saved['identity']) != row['_cache_identity']:
                raise ValueError('Patch map belongs to another roadside run')
            maps = np.stack([saved['maps'][unit_ids.index(uid)] for uid in row['units']])
        match = compact_match(maps)
        if match is None:
            continue
        public = {'pattern_id': row['pattern_id'], 'city': row['city'], 'panoid': row['panoid'],
                  'grid_x': row['grid_x'], 'grid_y': row['grid_y'],
                  'heading': match['heading'] * 90, 'box_patches': match['box'],
                  'component_areas': match['component_areas'], 'score': match['score']}
        grouped[row['pattern_id']].append((public, match, row['original']))
    counts = defaultdict(int)
    for row in examples:
        if len(row['units']) == len(row['lateral_units']):
            counts[row['pattern_id']] += 1
    table = []
    for pid, sampled in counts.items():
        matches = grouped[pid]
        table.append({'pattern_id': pid, 'label': patterns[pid]['label'],
                      'order': len(patterns[pid]['units']),
                      'full_corpus_matching_panoramas': patterns[pid]['positive_panoramas'],
                      'sampled_panoramas': sampled, 'local_matches': len(matches),
                      'local_cities': len({r['city'] for r, _, _ in matches})})
    table.sort(key=lambda r: (-r['local_cities'], -r['local_matches'],
                              -r['full_corpus_matching_panoramas'], r['pattern_id']))
    recurring = [r for r in table if r['local_matches'] >= 2 and r['local_cities'] >= 2]
    # Put the scarce three-unit arrangements first so the gallery does not
    # consist solely of the most frequent, usually simpler, pairs.
    selected = [r for r in recurring if r['order'] == 3][:12]
    selected.extend(r for r in recurring if r['order'] == 2)
    selected = selected[:16]
    shown = []
    for row in selected:
        distinct = {}
        for item in sorted(grouped[row['pattern_id']], key=lambda entry: -entry[0]['score']):
            distinct.setdefault(item[0]['city'], item)
        display = []
        for index, (public, match, source) in enumerate(list(distinct.values())[:3]):
            relative = Path('gallery') / f"{row['pattern_id']}-{index + 1}.webp"
            draw_example(source, match, out / relative)
            display.append({**public, 'image': relative.as_posix()})
        shown.append({**row, 'examples': display})
    report = {'status': 'exploratory_sample_audit', 'source_run_identity': marker['run_identity'],
              'composition_protocol_sha256': hashlib.sha256((patterns_dir / 'protocol.json').read_bytes()).hexdigest(),
              'roadside_audit_identity': roadside['identity'], 'source_panoramas': summary['panoramas'],
              'source_patterns': len(patterns), 'eligible_patterns': len(counts),
              'selection': 'six hash-ranked full-corpus matches in distinct cities per composition',
              'supplement_identities': {str(order): item['identity']
                                        for order, item in supplements.items()},
              'supplemental_pair_patterns': supplements.get(2, {}).get('patterns', 0),
              'supplemental_pair_panoramas': supplements.get(2, {}).get('panoramas', 0),
              'supplemental_triplet_patterns': supplements.get(3, {}).get('patterns', 0),
              'supplemental_triplet_panoramas': supplements.get(3, {}).get('panoramas', 0),
              'sampled_instances': usable, 'local_matches': sum(len(x) for x in grouped.values()),
              'patterns_with_two_cities': sum(r['local_cities'] >= 2 for r in table),
              'rule': {'positive_patch_threshold': '35% of panorama unit maximum',
                       'minimum_component_area_patches': 8, 'maximum_pair_gap_patches': 5,
                       'maximum_union_box_fraction_of_one_view': .5,
                       'same_direction_required': True},
              'limitations': ['Full-corpus vectors are panorama means; locality is checked only in sampled patch maps.',
                              'Six selected instances per eligible pattern do not estimate corpus locality prevalence.',
                              'SAE masks and Qwen labels are provisional, not object instance segmentations.',
                              'A compact rectangle locates the response; it is not an object boundary.'],
              'patterns': table, 'gallery': shown}
    write_json(out / 'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup', type=Path, default=FOLLOWUP)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--augment-pairs', action='store_true')
    parser.add_argument('--augment-triplets', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.augment_pairs:
        supplement_patterns(args.followup, args.out or args.followup / 'stages/local_composition_audit',
                            args.device, 2)
    if args.augment_triplets:
        supplement_patterns(args.followup, args.out or args.followup / 'stages/local_composition_audit',
                            args.device, 3)
    result = run(args.followup, args.out)
    print(json.dumps({k: result[k] for k in ('source_panoramas', 'source_patterns',
        'eligible_patterns', 'sampled_instances', 'local_matches', 'patterns_with_two_cities')},
        ensure_ascii=False), flush=True)
