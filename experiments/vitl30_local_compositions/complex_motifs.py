#!/usr/bin/env python3
"""Audit recurring three-family layouts in the existing local-composition sample.

This is a descriptive search over provisional SAE regions, not object detection or
an inferential test. The prototype-layer test partition remains sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .run import (HEADINGS, OUT, SOURCE, digest, extract_nodes, relation, setup,
                  write_json)

COLORS = ((220, 59, 49), (22, 130, 205), (39, 163, 93))
COMMON = {'building', 'road', 'tree', 'low_vegetation'}


def triplet_instances(nodes):
    """Keep one strongest instance of each rotation-invariant layout per pano."""
    best = {}
    for heading in range(4):
        view = [(i, n) for i, n in enumerate(nodes) if n.heading == heading]
        for group in combinations(view, 3):
            ordered = sorted(group, key=lambda item: item[1].family)
            families = tuple(node.family for _, node in ordered)
            if len(set(families)) != 3:
                continue
            rels = tuple(relation(a[1], b[1]) for a, b in combinations(ordered, 2))
            if any(rel is None for rel in rels):
                continue
            key = (families, rels)
            score = min(n.strength / np.sqrt(n.area) for _, n in ordered)
            if key not in best or score > best[key][0]:
                best[key] = (score, tuple(i for i, _ in ordered))
    return best


def render(row, nodes, path):
    views = []
    for heading in HEADINGS:
        with Image.open(row['image_' + str(heading)]) as source:
            image = source.convert('RGB')
            image.thumbnail((350, 350))
        canvas = image.copy()
        for color, node in zip(COLORS, nodes):
            if node.heading != heading // 90:
                continue
            mask = Image.fromarray((node.unpack() * 255).astype(np.uint8), 'L')
            mask = mask.resize(canvas.size, Image.Resampling.NEAREST)
            overlay = Image.new('RGB', canvas.size, color)
            canvas.paste(Image.blend(image, overlay, .55), (0, 0), mask)
        views.append(canvas)
    width = max(view.width for view in views)
    height = max(view.height for view in views)
    output = Image.new('RGB', (4 * width, height + 32), (244, 245, 242))
    draw = ImageDraw.Draw(output)
    for heading, view in enumerate(views):
        output.paste(view, (heading * width, 32))
        draw.text((heading * width + 8, 9), str(HEADINGS[heading]) + ' deg',
                  fill=(25, 34, 42))
    output.save(path, 'WEBP', quality=83, method=4)


def extract_sample(rows, corpus, families, dim_family, scale, counts):
    all_nodes = []
    last_chunk = None
    arrays = None
    for i, row in rows.iterrows():
        start = int(row.cache_pano_index) // 64 * 64
        key = (row.city, start)
        if key != last_chunk:
            folder = SOURCE / 'activations' / row.city.replace('/', '__') / f'{start:06d}'
            marker = json.loads((folder / 'complete.json').read_text())
            if marker['identity'] != corpus['identity']:
                raise ValueError('Activation chunk identity mismatch')
            arrays = tuple(np.load(folder / f'{name}.npy', mmap_mode='r')
                           for name in ('values', 'indices', 'indptr'))
            last_chunk = key
        all_nodes.append(extract_nodes(arrays, int(row.cache_pano_index) - start,
                                       families, dim_family, scale, counts))
        if (i + 1) % 200 == 0:
            print(f'EXTRACT {i + 1}/{len(rows)}', flush=True)
    return all_nodes


def evaluate(rows, all_nodes, min_train=15, min_cities=6, min_dev=5):
    matches = defaultdict(list)
    for i, nodes in enumerate(all_nodes):
        for key, value in triplet_instances(nodes).items():
            matches[key].append((i, value))
    patterns = []
    for key, hits in matches.items():
        train = [(i, value) for i, value in hits if rows.iloc[i].prototype_split == 'train']
        dev = [(i, value) for i, value in hits if rows.iloc[i].prototype_split == 'dev']
        train_cities = {rows.iloc[i].city for i, _ in train}
        dev_cities = {rows.iloc[i].city for i, _ in dev}
        if len(train) < min_train or len(train_cities) < min_cities:
            continue
        patterns.append({'families': list(key[0]), 'relations': list(key[1]),
                         'train_panoramas': len(train), 'train_cities': len(train_cities),
                         'train_blocks': len({rows.iloc[i].spatial_block for i, _ in train}),
                         'dev_panoramas': len(dev), 'dev_cities': len(dev_cities),
                         'dev_blocks': len({rows.iloc[i].spatial_block for i, _ in dev}),
                         'specific_families': len(set(key[0]) - COMMON), '_dev': dev})
    patterns.sort(key=lambda p: (-p['specific_families'], -p['dev_cities'],
                                 -p['dev_panoramas'], -p['train_panoramas']))
    return patterns, [p for p in patterns if p['dev_panoramas'] >= min_dev
                      and p['dev_cities'] >= 3]


def select_gallery_patterns(repeated, limit):
    selected, seen = [], set()
    for item in repeated:
        families = tuple(item['families'])
        if families in seen:
            continue
        seen.add(families)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def main(args):
    corpus, protocol, hierarchy, frame, scale, families, dim_family, counts, _ = setup()
    sample = pd.read_csv(args.sample)
    rows = sample.merge(frame, on=['city', 'cache_pano_index', 'panoid',
                                   'spatial_block', 'prototype_split'], validate='one_to_one')
    if len(rows) != len(sample) or set(rows.prototype_split) != {'train', 'dev'}:
        raise ValueError('Sample changed or includes the sealed test partition')
    all_nodes = extract_sample(rows, corpus, families, dim_family, scale, counts)
    patterns, repeated = evaluate(rows, all_nodes, args.min_train,
                                  args.min_cities, args.min_dev)
    gallery = args.out / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    shown = []
    for item in select_gallery_patterns(repeated, args.max_gallery):
        per_city = {}
        for i, hit in item['_dev']:
            city = rows.iloc[i].city
            if city not in per_city or hit[0] > per_city[city][1][0]:
                per_city[city] = (i, hit)
        examples = []
        for number, (i, (_, indices)) in enumerate(
                sorted(per_city.values(), key=lambda pair: -pair[1][0])[:args.examples]):
            row = rows.iloc[i]
            selected = [all_nodes[i][index] for index in indices]
            filename = f'm{len(shown) + 1:02d}_{number + 1:02d}.webp'
            render(row, selected, gallery / filename)
            examples.append({'city': row.city, 'panoid': row.panoid,
                             'spatial_block': row.spatial_block, 'image': filename,
                             'heading': HEADINGS[selected[0].heading],
                             'areas_patches': [node.area for node in selected],
                             'top_dimensions': [list(node.top_dimensions[:3]) for node in selected]})
        shown.append({k: v for k, v in item.items() if k != '_dev'} | {'examples': examples})
    report = {'status': 'exploratory', 'checkpoint_sha256': corpus['checkpoint_sha256'],
              'corpus_identity': corpus['identity'], 'split_identity': protocol['identity'],
              'hierarchy_identity': hierarchy['identity'],
              'sample_sha256': digest(args.sample),
              'sample': {'panoramas': len(rows), 'train': int((rows.prototype_split == 'train').sum()),
                         'dev': int((rows.prototype_split == 'dev').sum()),
                         'cities': int(rows.city.nunique())},
              'method': {'view_scope': 'three distinct provisional families in one directional view',
                         'rotation_invariant': True, 'parent_family_instances': False,
                         'statistical_significance_tested': False,
                         'ranking': 'specificity, development-city count, development-pano count',
                         'minimum_train_panoramas': args.min_train,
                         'minimum_train_cities': args.min_cities,
                         'minimum_dev_panoramas': args.min_dev,
                         'minimum_dev_cities': 3},
              'counts': {'train_supported': len(patterns), 'repeated_in_dev': len(repeated),
                         'gallery_patterns': len(shown)},
              'candidate_summaries': [{k: v for k, v in item.items() if k != '_dev'}
                                      for item in repeated],
              'limitations': ['SAE family masks are not verified object instances.',
                              'Development examples select and rank motifs; no significance or out-of-sample claim.',
                              'One building plus two separate trees cannot be established from these masks.'],
              'patterns': shown}
    write_json(args.out / 'results.json', report)
    print('DONE', json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=OUT / 'sample.csv')
    parser.add_argument('--out', type=Path, default=OUT / 'complex_motifs')
    parser.add_argument('--min-train', type=int, default=15)
    parser.add_argument('--min-cities', type=int, default=6)
    parser.add_argument('--min-dev', type=int, default=5)
    parser.add_argument('--max-gallery', type=int, default=12)
    parser.add_argument('--examples', type=int, default=4)
    main(parser.parse_args())
