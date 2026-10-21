"""Audit cross-view arrangements in sampled full-corpus SAE composition matches.

Four headings form a circular yaw sequence. A seam match is a compact response
across adjacent views; a directional match records dominant headings without
claiming that the objects touch. Neither is a full-corpus prevalence estimate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .extract import write_json
from .followup import FOLLOWUP
from .local_composition_audit import COLORS, PATCHES, VIEW_PIXELS, components, gap


def _component_grid(maps):
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim != 4 or maps.shape[1:] != (4, PATCHES, PATCHES) or len(maps) not in (2, 3):
        raise ValueError('Expected 2-3 complete four-view patch maps')
    peaks = maps.max(axis=(1, 2, 3))
    groups = [[components(maps[i, heading], peaks[i]) for heading in range(4)]
              for i in range(len(maps))]
    return maps, peaks, groups


def seam_match(maps, max_gap=5, max_width=28, max_height=22):
    """Find compact components spanning one pair of adjacent yaw headings."""
    _, peaks, groups = _component_grid(maps)
    best = None
    for start in range(4):
        end = (start + 1) % 4
        choices = []
        for unit in range(len(groups)):
            left = [(start, item) for item in groups[unit][start]
                    if item['box'][2] >= PATCHES - max_gap]
            right = [(end, item) for item in groups[unit][end]
                     if item['box'][0] <= max_gap]
            choices.append(left + right)
        if any(not items for items in choices):
            continue
        for chosen in itertools.product(*choices):
            if {heading for heading, _ in chosen} != {start, end}:
                continue
            translated = []
            for heading, item in chosen:
                x0, y0, x1, y1 = item['box']
                offset = PATCHES if heading == end else 0
                translated.append({'box': (x0 + offset, y0, x1 + offset, y1)})
            box = (min(c['box'][0] for c in translated), min(c['box'][1] for c in translated),
                   max(c['box'][2] for c in translated), max(c['box'][3] for c in translated))
            if box[2] - box[0] > max_width or box[3] - box[1] > max_height:
                continue
            reachable = {0}
            while True:
                nearby = {j for i in reachable for j, candidate in enumerate(translated)
                          if j not in reachable and max(gap(translated[i], candidate)) <= max_gap}
                if not nearby:
                    break
                reachable |= nearby
            if len(reachable) != len(chosen):
                continue
            score = min(item['mass'] / max(peaks[i], 1e-8)
                        for i, (_, item) in enumerate(chosen))
            candidate = {'headings': [heading for heading, _ in chosen],
                         'masks': [item['mask'] for _, item in chosen],
                         'component_areas': [item['area'] for _, item in chosen],
                         'seam': (start, end), 'box_patches': box, 'score': float(score)}
            if best is None or score > best['score']:
                best = candidate
    return best


def directional_match(maps, minimum_share=.4, minimum_margin=.1):
    """Require a distinct, dominant response direction for every unit."""
    maps, peaks, groups = _component_grid(maps)
    totals = maps.sum(axis=(2, 3))
    selected = []
    for unit in range(len(maps)):
        ranked = np.argsort(-totals[unit], kind='stable')
        heading = int(ranked[0])
        mass = float(totals[unit, heading])
        overall = float(totals[unit].sum())
        if (peaks[unit] <= 0 or overall <= 0 or mass / overall < minimum_share
                or (mass - float(totals[unit, ranked[1]])) / overall < minimum_margin
                or not groups[unit][heading]):
            return None
        selected.append((heading, groups[unit][heading][0], mass / overall))
    headings = [heading for heading, _, _ in selected]
    if len(set(headings)) < 2:
        return None
    return {'headings': headings,
            'signature': [int((heading - headings[0]) % 4) for heading in headings],
            'masks': [item['mask'] for _, item, _ in selected],
            'component_areas': [item['area'] for _, item, _ in selected],
            'heading_shares': [float(share) for _, _, share in selected],
            'score': min(share for _, _, share in selected)}


def draw_example(source, match, destination):
    """Highlight each selected response in its own heading on the four-view strip."""
    with Image.open(source) as original:
        canvas = original.convert('RGB')
    if canvas.size != (4 * VIEW_PIXELS, VIEW_PIXELS + 24):
        raise ValueError('Unexpected four-view evidence strip size')
    for unit, (heading, mask) in enumerate(zip(match['headings'], match['masks'])):
        x0 = heading * VIEW_PIXELS
        region = canvas.crop((x0, 24, x0 + VIEW_PIXELS, 24 + VIEW_PIXELS))
        resized = Image.fromarray((mask * 255).astype(np.uint8), 'L').resize(
            (VIEW_PIXELS, VIEW_PIXELS), Image.Resampling.NEAREST)
        tint = Image.new('RGB', region.size, COLORS[unit])
        region.paste(Image.blend(region, tint, .58), (0, 0), resized)
        canvas.paste(region, (x0, 24))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, 'WEBP', quality=90)


def _evidence_rows(followup, local_dir, identity):
    roadside_dir = followup / 'stages/composition_roadside_audit'
    roadside = json.loads((roadside_dir / 'complete.json').read_text())
    if roadside['run_identity'] != identity or roadside['status'] != 'complete':
        raise ValueError('Roadside audit belongs to another corpus run')
    rows = [{**row, '_root': roadside_dir, '_identity': roadside['identity']}
            for row in json.loads((roadside_dir / 'examples.json').read_text())
            if len(row['units']) == len(row['lateral_units'])]
    if len(json.loads((roadside_dir / 'examples.json').read_text())) != roadside['examples']:
        raise ValueError('Incomplete roadside sample')
    for order, name in ((2, 'supplemental_pairs'), (3, 'supplemental_triplets')):
        path = local_dir / name / 'manifest.json'
        supplement = json.loads(path.read_text())
        if supplement['status'] != 'complete' or supplement['run_identity'] != identity:
            raise ValueError(f'{name} belongs to another corpus run')
        rows.extend({**row, '_root': path.parent, '_identity': supplement['identity']}
                    for row in supplement['examples'])
    return rows


def run(followup=FOLLOWUP, out=None):
    followup = Path(followup)
    local_dir = followup / 'stages/local_composition_audit'
    out = Path(out) if out else followup / 'stages/panorama_composition_audit'
    local = json.loads((local_dir / 'results.json').read_text())
    corpus = json.loads((followup / 'complete.json').read_text())
    identity = corpus['identity']['extract_run_identity']
    if (local['source_run_identity'] != identity or local['source_panoramas'] != corpus['processed']
            or local['eligible_patterns'] != 185 or local['sampled_instances'] != 1110):
        raise ValueError('Local audit does not cover all full-corpus compositions')
    patterns = {r['pattern_id']: r for r in local['patterns']}
    rows = _evidence_rows(followup, local_dir, identity)
    if len(rows) != local['sampled_instances']:
        raise ValueError('Cross-view sample does not match the local audit')
    by_pattern = defaultdict(list)
    seam_examples = defaultdict(list)
    directional_examples = defaultdict(list)
    for row in rows:
        pid = row['pattern_id']
        if pid not in patterns:
            raise ValueError('Sample refers to an unknown composition')
        key = hashlib.sha256(f"{row['city']}/{row['panoid']}".encode()).hexdigest()[:20]
        with np.load(row['_root'] / 'evidence' / key / 'activation.npz',
                     allow_pickle=False) as saved:
            if str(saved['identity']) != row['_identity']:
                raise ValueError('Patch map belongs to another inference run')
            unit_ids = list(saved['unit_ids'])
            maps = np.stack([saved['maps'][unit_ids.index(uid)] for uid in row['units']])
        seam = seam_match(maps)
        direction = directional_match(maps)
        by_pattern[pid].append({'city': row['city'], 'panoid': row['panoid'],
                                'seam': seam is not None,
                                'directional_signature': direction['signature'] if direction else None})
        basic = {'pattern_id': pid, 'city': row['city'], 'panoid': row['panoid']}
        if seam:
            seam_examples[pid].append((basic, seam, row['original']))
        if direction:
            directional_examples[pid].append((basic, direction, row['original']))
    table = []
    for pid, pattern in patterns.items():
        sample = by_pattern[pid]
        if len(sample) != 6:
            raise ValueError('Expected six distinct-city panoramas per composition')
        signatures = Counter(tuple(r['directional_signature']) for r in sample
                             if r['directional_signature'] is not None)
        top_signature, top_count = signatures.most_common(1)[0] if signatures else (None, 0)
        table.append({'pattern_id': pid, 'label': pattern['label'], 'order': pattern['order'],
                      'full_corpus_matching_panoramas': pattern['full_corpus_matching_panoramas'],
                      'sampled_panoramas': 6,
                      'same_view_local_matches': pattern['local_matches'],
                      'cross_seam_local_matches': sum(r['seam'] for r in sample),
                      'cross_seam_cities': len({r['city'] for r in sample if r['seam']}),
                      'directional_matches': sum(r['directional_signature'] is not None for r in sample),
                      'top_relative_heading_signature': list(top_signature) if top_signature else None,
                      'top_signature_matches': top_count})
    table.sort(key=lambda r: (-r['cross_seam_cities'], -r['directional_matches'],
                              -r['full_corpus_matching_panoramas'], r['pattern_id']))
    seam_gallery = []
    seam_selected = [r for r in table if r['cross_seam_cities'] >= 2]
    seam_selected.sort(key=lambda r: (-r['order'], -r['cross_seam_cities'],
                                      -r['directional_matches'], r['pattern_id']))
    for row in seam_selected[:16]:
        displays = []
        for index, (basic, match, source) in enumerate(
                sorted(seam_examples[row['pattern_id']], key=lambda item: -item[1]['score'])[:2], 1):
            relative = Path('gallery/seam') / f"{row['pattern_id']}-{index}.webp"
            draw_example(source, match, out / relative)
            displays.append({**basic, 'headings': [h * 90 for h in match['headings']],
                             'seam': [h * 90 for h in match['seam']],
                             'component_areas': match['component_areas'],
                             'image': relative.as_posix()})
        seam_gallery.append({**row, 'examples': displays})
    directional_gallery = []
    directional_selected = [r for r in table if r['top_signature_matches'] >= 2]
    directional_selected.sort(key=lambda r: (-r['order'], -r['top_signature_matches'],
                                             -r['directional_matches'], r['pattern_id']))
    for row in directional_selected[:16]:
        matches = [item for item in directional_examples[row['pattern_id']]
                   if item[1]['signature'] == row['top_relative_heading_signature']]
        displays = []
        for index, (basic, match, source) in enumerate(
                sorted(matches, key=lambda item: -item[1]['score'])[:2], 1):
            relative = Path('gallery/directional') / f"{row['pattern_id']}-{index}.webp"
            draw_example(source, match, out / relative)
            displays.append({**basic, 'headings': [h * 90 for h in match['headings']],
                             'relative_heading_signature': match['signature'],
                             'heading_shares': match['heading_shares'],
                             'image': relative.as_posix()})
        directional_gallery.append({**row, 'examples': displays})
    report = {'status': 'exploratory_sample_audit', 'source_run_identity': identity,
              'source_panoramas': corpus['processed'], 'source_patterns': len(patterns),
              'sampled_instances': len(rows),
              'cross_seam_local_matches': sum(r['cross_seam_local_matches'] for r in table),
              'patterns_cross_seam_two_cities': sum(r['cross_seam_cities'] >= 2 for r in table),
              'directional_matches': sum(r['directional_matches'] for r in table),
              'patterns_repeated_heading_signature': sum(r['top_signature_matches'] >= 2 for r in table),
              'rule': {'heading_order_degrees': [0, 90, 180, 270],
                       'ring_wrap': True, 'patch_threshold': '35% of unit panorama peak',
                       'minimum_component_area_patches': 8,
                       'seam_max_gap_patches': 5, 'seam_max_width_patches': 28,
                       'seam_max_height_patches': 22,
                       'directional_min_heading_mass_share': .4,
                       'directional_min_margin_over_second_heading': .1},
              'limitations': ['The four headings use nominal yaw order, not calibrated image overlap.',
                              'Seam adjacency is a patch-grid hypothesis, not verified physical continuity.',
                              'Directional layouts are cross-view co-responses, not proximity or object counts.',
                              'Six distinct-city matches per pattern cannot estimate full-corpus prevalence.',
                              'SAE activations and Qwen unit labels require visual review.'],
              'patterns': table, 'seam_gallery': seam_gallery,
              'directional_gallery': directional_gallery}
    write_json(out / 'results.json', report)
    return report


if __name__ == '__main__':
    result = run()
    print(json.dumps({key: result[key] for key in ('source_panoramas', 'source_patterns',
        'sampled_instances', 'cross_seam_local_matches', 'patterns_cross_seam_two_cities',
        'directional_matches', 'patterns_repeated_heading_signature')},
        ensure_ascii=False), flush=True)
