#!/usr/bin/env python3
"""Search recurring multi-element layouts on complete four-view panoramas.

The input is the frozen ViT-L/BatchTopK SAE sparse cache already used by the
local-composition pilot.  Each provisional family region is mapped from its
28x28 view to a circular horizontal panorama coordinate.  Motifs are then
searched over all four headings at once, with repeated families allowed (for
example ``building + tree + tree``).

This is an exploratory relationship audit.  A family mask is a visual feature
region, not an object segmentation mask; the output must therefore be reviewed
as candidate motifs rather than confirmed object-level prototypes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations, permutations, product
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .run import (HEADINGS, GRID, OUT, SOURCE, Node, digest, extract_nodes,
                  setup, write_json)

FOV_DEG = 90.0
ANGLE_BINS = (30.0, 75.0, 120.0, 160.0)
FAMILY_COLORS = {
    'building': (218, 61, 49), 'tree': (36, 158, 82),
    'road': (45, 116, 204), 'pedestrian': (238, 142, 30),
    'vehicle': (142, 71, 180), 'barrier': (190, 74, 135),
    'low_vegetation': (18, 157, 140), 'utility': (111, 83, 41),
    'sign': (240, 192, 48), 'water': (42, 168, 205),
    'person': (170, 80, 54), 'transport_structure': (99, 103, 107),
    'furniture': (126, 92, 160), 'terrain': (109, 137, 61),
    'construction': (160, 98, 49),
}
FOCUS_SIGNATURES = [
    ('building', 'tree', 'tree'),
    ('building', 'pedestrian', 'tree'),
    ('barrier', 'road', 'tree'),
    ('building', 'road', 'utility'),
    ('low_vegetation', 'pedestrian', 'tree'),
]


def panorama_angle(node: Node) -> float:
    """Return a circular bearing in degrees for a local node centroid."""
    local = (node.x / max(GRID - 1, 1) - 0.5) * FOV_DEG
    return (HEADINGS[node.heading] + local) % 360.0


def circular_distance(a: float, b: float) -> float:
    delta = abs((a - b) % 360.0)
    return min(delta, 360.0 - delta)


def signed_circular_delta(a: float, b: float) -> float:
    """Shortest signed angle from a to b, in [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0


def angle_bin(distance: float) -> str:
    for index, edge in enumerate(ANGLE_BINS):
        if distance < edge:
            return ('near', 'adjacent', 'wide', 'very_wide')[index]
    return 'opposite'


def vertical_relation(a: Node, b: Node) -> str:
    delta = (b.y - a.y) / GRID
    return 'up' if delta < -0.20 else 'down' if delta > 0.20 else 'level'


def overlaps_same_view(a: Node, b: Node) -> bool:
    if a.heading != b.heading:
        return False
    ma, mb = a.unpack(), b.unpack()
    return int((ma & mb).sum()) / max(1, min(a.area, b.area)) > 0.30


def likely_duplicate_observation(a: Node, b: Node) -> bool:
    """Remove same-family edge duplicates seen at adjacent headings."""
    if a.family != b.family:
        return False
    if overlaps_same_view(a, b):
        return True
    distance = circular_distance(panorama_angle(a), panorama_angle(b))
    if distance < 10.0 and abs(a.y - b.y) / GRID < 0.18:
        return True
    return False


def pair_descriptor(a: Node, b: Node) -> tuple[str, str]:
    distance = circular_distance(panorama_angle(a), panorama_angle(b))
    return angle_bin(distance), vertical_relation(a, b)


def motif_key(nodes: list[Node], indices: tuple[int, int, int]) -> tuple:
    """Canonical rotation-invariant key, also symmetric for repeated families."""
    chosen = [nodes[i] for i in indices]
    # Sorting by family first keeps semantic roles readable.  Enumerating the
    # small set of permutations avoids assigning an arbitrary identity to the
    # two trees in a building+tree+tree motif.
    # Families are already sorted in the key.  Only permutations within a
    # repeated-family group can change the descriptor, so avoid six-way
    # permutation work for the common all-distinct case.
    groups = []
    for family in sorted({n.family for n in chosen}):
        groups.append([i for i, n in enumerate(chosen) if n.family == family])
    candidates = []
    for group_order in product(*[list(permutations(group)) for group in groups]):
        order = tuple(i for group in group_order for i in group)
        ordered = [chosen[i] for i in order]
        families = tuple(n.family for n in ordered)
        if families != tuple(sorted(families)):
            continue
        descriptors = tuple(pair_descriptor(ordered[i], ordered[j])
                            for i, j in ((0, 1), (0, 2), (1, 2)))
        candidates.append((families, descriptors))
    if not candidates:
        raise AssertionError('No canonical permutation')
    return min(candidates)


def family_signature(nodes: list[Node], indices: tuple[int, int, int]) -> tuple[str, ...]:
    return tuple(sorted(nodes[i].family for i in indices))


def present_motif_keys(nodes: list[Node], allowed_signatures=None) -> set[tuple]:
    """Return keys present in a pano, skipping family signatures not requested."""
    result = set()
    for indices in combinations(range(len(nodes)), 3):
        if allowed_signatures is not None and family_signature(nodes, indices) not in allowed_signatures:
            continue
        chosen = [nodes[i] for i in indices]
        if any(likely_duplicate_observation(a, b)
               for a, b in combinations(chosen, 2)):
            continue
        result.add(motif_key(nodes, indices))
    return result


def motif_score(nodes: list[Node], indices: tuple[int, int, int]) -> float:
    return min(nodes[i].strength / np.sqrt(max(1, nodes[i].area)) for i in indices)


def triplet_instances_panorama(nodes: list[Node]) -> dict[tuple, tuple[float, tuple[int, ...]]]:
    """Find one strongest instance of every three-node panorama motif."""
    best: dict[tuple, tuple[float, tuple[int, ...]]] = {}
    for indices in combinations(range(len(nodes)), 3):
        chosen = [nodes[i] for i in indices]
        # The three regions must be distinct.  Repeated families are allowed,
        # but a feature duplicated at a view boundary is not counted twice.
        if any(likely_duplicate_observation(a, b)
               for a, b in combinations(chosen, 2)):
            continue
        key = motif_key(nodes, indices)
        score = motif_score(nodes, indices)
        if key not in best or score > best[key][0]:
            best[key] = (score, tuple(indices))
    return best


def pair_support(nodes: list[Node], families: tuple[str, ...]) -> bool:
    """Whether a pano has enough distinct family regions for a candidate."""
    counts = Counter(n.family for n in nodes)
    return all(counts[f] >= need for f, need in Counter(families).items())


def candidate_family_specificity(families: tuple[str, ...]) -> int:
    common = {'building', 'road', 'tree', 'low_vegetation'}
    return len(set(families) - common) + (1 if len(set(families)) < 3 else 0)


def focus_rank(families: tuple[str, ...]) -> int:
    try:
        return FOCUS_SIGNATURES.index(tuple(families))
    except ValueError:
        return len(FOCUS_SIGNATURES) + candidate_family_specificity(families)


def extract_all(rows, corpus, families, dim_family, scale, counts):
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


def city_block_counts(rows, matches, split):
    cities = set()
    blocks = set()
    for index, _ in matches:
        row = rows.iloc[index]
        if row.prototype_split == split:
            cities.add(row.city)
            blocks.add(row.spatial_block)
    return len(cities), len(blocks)


def _match_counts(rows, all_nodes, candidate_keys, split=None):
    counts = Counter()
    for index, nodes in enumerate(all_nodes):
        if split is not None and rows.iloc[index].prototype_split != split:
            continue
        counts.update(set(triplet_instances_panorama(nodes)).intersection(candidate_keys))
    return counts


def null_counts(rows, all_nodes, candidates, repeats=199, seed=6138):
    """Generate two frozen dev nulls: family co-occurrence and layout."""
    dev_indices = [i for i, split in enumerate(rows.prototype_split)
                   if split == 'dev']
    rng = random.Random(seed)
    allowed_signatures = {tuple(key[0]) for key in candidates}
    results = {'cooccurrence': {key: [] for key in candidates},
               'layout': {key: [] for key in candidates}}
    # Preserve city and heading-level marginal family rates while breaking the
    # association between a family and its original panorama.
    city_heading = defaultdict(list)
    for idx in dev_indices:
        for node_id, node in enumerate(all_nodes[idx]):
            city_heading[(rows.iloc[idx].city, node.heading)].append((idx, node_id))
    for repeat in range(repeats):
        labels = {}
        for sites in city_heading.values():
            original = [all_nodes[i][j].family for i, j in sites]
            rng.shuffle(original)
            labels.update(zip(sites, original))
        replicate_cooccurrence = Counter()
        for idx in dev_indices:
            shuffled = [replace(n, family=labels[(idx, j)])
                        for j, n in enumerate(all_nodes[idx])]
            present = present_motif_keys(shuffled, allowed_signatures) & candidates
            replicate_cooccurrence.update(present)
        for key in candidates:
            results['cooccurrence'][key].append(replicate_cooccurrence[key])
        # For layout, retain each pano's family multiset and shuffle geometry.
        replicate_layout = Counter()
        for idx in dev_indices:
            nodes = all_nodes[idx]
            geometry = [(n.heading, n.y, n.x, n.area, n.mask)
                        for n in nodes]
            rng.shuffle(geometry)
            shuffled = [replace(n, heading=h, y=y, x=x, area=area, mask=mask)
                        for n, (h, y, x, area, mask) in zip(nodes, geometry)]
            present = present_motif_keys(shuffled, allowed_signatures) & candidates
            replicate_layout.update(present)
        for key in candidates:
            results['layout'][key].append(replicate_layout[key])
        if (repeat + 1) % 25 == 0:
            print(f'NULL {repeat + 1}/{repeats}', flush=True)
    return results


def bh(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [1.0] * len(values)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted


def _open_view(row, heading, size=360):
    with Image.open(row[f'image_{heading}']) as source:
        image = source.convert('RGB')
        image.thumbnail((size, size))
        return image.copy()


def render_panorama(row, nodes, path):
    """Render the four views as a labelled 360-degree ring strip."""
    views = []
    for heading_index, heading in enumerate(HEADINGS):
        view = _open_view(row, heading)
        base = view.copy()
        for node in nodes:
            if node.heading != heading_index:
                continue
            color = FAMILY_COLORS.get(node.family, (233, 58, 116))
            mask = Image.fromarray((node.unpack() * 255).astype(np.uint8), 'L')
            mask = mask.resize(base.size, Image.Resampling.NEAREST)
            overlay = Image.new('RGB', base.size, color)
            base.paste(Image.blend(base, overlay, .58), (0, 0), mask)
        views.append(base)
    width = max(v.width for v in views)
    height = max(v.height for v in views)
    header = 42
    output = Image.new('RGB', (width * 4, height + header), (244, 245, 242))
    draw = ImageDraw.Draw(output)
    for i, view in enumerate(views):
        x = i * width
        output.paste(view, (x, header))
        draw.text((x + 8, 8), f'{HEADINGS[i]}°  ({"→" if i < 3 else "→ 0°"})',
                  fill=(25, 34, 42))
    draw.text((8, height + header - 18),
              'left-to-right: 0° → 90° → 180° → 270° → 0°  |  colored regions are candidate SAE families',
              fill=(25, 34, 42))
    output.save(path, 'WEBP', quality=84, method=4)


def p_value(observed, null):
    return (1 + sum(value >= observed for value in null)) / (1 + len(null))


def build_report(rows, all_nodes, args, corpus, protocol, hierarchy, scale_info,
                 output, labels):
    observed = defaultdict(list)
    for index, nodes in enumerate(all_nodes):
        for key, value in triplet_instances_panorama(nodes).items():
            observed[key].append((index, value))
    candidates = []
    for key, matches in observed.items():
        train = [(i, value) for i, value in matches
                 if rows.iloc[i].prototype_split == 'train']
        train_cities, train_blocks = city_block_counts(rows, train, 'train')
        if len(train) >= args.min_train and train_cities >= args.min_cities:
            candidates.append((key, matches, train_cities, train_blocks))
    candidates.sort(key=lambda x: (-candidate_family_specificity(x[0][0]),
                                   -len(x[1])))
    candidate_keys = {key for key, *_ in candidates}
    print(f'PANORAMA CANDIDATES {len(candidates)}', flush=True)
    if not candidates:
        raise ValueError('No recurring panorama-level triple candidates')
    null = null_counts(rows, all_nodes, candidate_keys,
                       repeats=args.permutations, seed=args.seed + 2)
    patterns = []
    for key, matches, train_cities, train_blocks in candidates:
        train_matches = [(i, v) for i, v in matches
                         if rows.iloc[i].prototype_split == 'train']
        dev_matches = [(i, v) for i, v in matches
                       if rows.iloc[i].prototype_split == 'dev']
        dev_cities, dev_blocks = city_block_counts(rows, dev_matches, 'dev')
        p = {mode: p_value(len(dev_matches), null[mode][key])
             for mode in ('cooccurrence', 'layout')}
        fams = key[0]
        # Pair support is the strictest pairwise prerequisite among the three
        # family pairs.  This makes the extra support of the triple explicit.
        pair_keys = [(fams[i], fams[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        pair_train = []
        pair_dev = []
        for pair in pair_keys:
            pair_train.append(sum(pair_support(all_nodes[i], pair)
                                  for i, _ in train_matches))
            pair_dev.append(sum(pair_support(all_nodes[i], pair)
                                for i, _ in dev_matches))
        patterns.append({'families': list(fams), 'layout': [list(x) for x in key[1]],
                         'train_panoramas': len(train_matches),
                         'train_cities': train_cities, 'train_blocks': train_blocks,
                         'dev_panoramas': len(dev_matches), 'dev_cities': dev_cities,
                         'dev_blocks': dev_blocks, 'pair_train_support': pair_train,
                         'pair_dev_support': pair_dev,
                         'p': p, 'null_means': {mode: float(np.mean(null[mode][key]))
                                              for mode in ('cooccurrence', 'layout')},
                         '_matches': dev_matches, '_key': key})
    for mode in ('cooccurrence', 'layout'):
        corrected = bh([p['p'][mode] for p in patterns])
        for item, value in zip(patterns, corrected):
            item.setdefault('q', {})[mode] = value
    patterns.sort(key=lambda p: (focus_rank(tuple(p['families'])),
                                 -(p['q']['cooccurrence'] < .05 and
                                   p['q']['layout'] < .05),
                                 -p['dev_cities'], -p['dev_panoramas']))

    gallery = output / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    public = []
    selected_items = []
    selected_keys = set()
    # Reserve one gallery slot for each requested relation family, so the
    # results page cannot be dominated by trivial road-road combinations.
    for signature in FOCUS_SIGNATURES:
        for item in patterns:
            if tuple(item['families']) != signature:
                continue
            if item['dev_cities'] >= args.min_dev_cities and item['dev_panoramas'] >= args.min_dev:
                selected_items.append(item)
                selected_keys.add(tuple(item['families']) + tuple(map(tuple, item['layout'])))
                break
    for item in patterns:
        key = tuple(item['families']) + tuple(map(tuple, item['layout']))
        if len(selected_items) >= args.max_gallery or key in selected_keys:
            continue
        if item['dev_cities'] < args.min_dev_cities or item['dev_panoramas'] < args.min_dev:
            continue
        selected_items.append(item)
    for item in selected_items[:args.max_gallery]:
        by_city = {}
        for index, (score, node_ids) in item['_matches']:
            city = rows.iloc[index].city
            if city not in by_city or score > by_city[city][1][0]:
                by_city[city] = (index, (score, node_ids))
        selected = sorted(by_city.values(), key=lambda x: -x[1][0])[:args.examples]
        examples = []
        for number, (index, (score, node_ids)) in enumerate(selected):
            row = rows.iloc[index]
            nodes = [all_nodes[index][node_id] for node_id in node_ids]
            filename = f'm{len(public) + 1:02d}_{number + 1:02d}.webp'
            render_panorama(row, nodes, gallery / filename)
            examples.append({'city': row.city, 'panoid': row.panoid,
                             'spatial_block': row.spatial_block, 'image': filename,
                             'score': score,
                             'nodes': [{'family': n.family, 'heading': HEADINGS[n.heading],
                                        'panorama_angle_deg': round(panorama_angle(n), 2),
                                        'area_patches': n.area,
                                        'top_dimensions': list(n.top_dimensions),
                                        'top_labels': [labels[d] for d in n.top_dimensions[:3]]}
                                       for n in nodes]})
        public.append({k: v for k, v in item.items()
                       if k not in {'_matches', '_key'}} | {'examples': examples})

    result = {
        'status': 'exploratory',
        'scope': 'complete four-heading panorama; sealed test partition not accessed',
        'checkpoint_sha256': corpus['checkpoint_sha256'],
        'corpus_identity': corpus['identity'], 'split_identity': protocol['identity'],
        'hierarchy_identity': hierarchy['identity'],
        'sampling': {'sample_sha256': digest(args.sample), 'panoramas': len(rows),
                     'train_panoramas': int((rows.prototype_split == 'train').sum()),
                     'dev_panoramas': int((rows.prototype_split == 'dev').sum()),
                     'cities': int(rows.city.nunique()), 'nodes': sum(map(len, all_nodes)),
                     'directional_images': len(rows) * 4},
        'method': {'coordinate_system': 'circular horizontal bearing', 'fov_deg': FOV_DEG,
                   'view_order': list(HEADINGS), 'patch_grid': [GRID, GRID],
                   'motif_size': 3, 'repeated_families_allowed': True,
                   'same_family_boundary_duplicates_removed': True,
                   'canonical_layout': 'family-sorted permutation-minimum pair angle bins and vertical relations',
                   'minimum_train_panoramas': args.min_train,
                   'minimum_train_cities': args.min_cities,
                   'minimum_dev_panoramas': args.min_dev,
                   'minimum_dev_cities': args.min_dev_cities,
                   'permutations': args.permutations,
                   'multiple_testing': 'BH-FDR across frozen train candidates on dev'},
        'counts': {'all_observed_keys': len(observed), 'train_candidates': len(candidates),
                   'gallery_patterns': len(public),
                   'both_nulls_q_below_005': sum(p['q']['cooccurrence'] < .05 and
                                               p['q']['layout'] < .05 for p in patterns)},
        'limitations': [
            'SAE family regions are not object segmentation masks; repeated regions are only provisional instances.',
            'FOV=90 degrees is an explicit coordinate assumption for the four perspective images.',
            'The ring strip is a display of four views, not a geometrically rectified equirectangular panorama.',
            'Permutation p-values and BH q-values are exploratory support, not independent semantic validation.',
            'A candidate is called a motif only after cross-city support and manual inspection; no candidate is an established prototype yet.'
        ],
        'patterns': public,
        'focus_signatures': [list(x) for x in FOCUS_SIGNATURES],
        'all_candidate_summaries': [{k: v for k, v in p.items()
                                     if k not in {'_matches', '_key'}} for p in patterns],
    }
    write_json(output / 'results.json', result)
    return result


def main(args):
    corpus, protocol, hierarchy, frame, scale, families, dim_family, counts, labels = setup()
    sample = pd.read_csv(args.sample)
    rows = sample.merge(frame, on=['city', 'cache_pano_index', 'panoid',
                                   'spatial_block', 'prototype_split'],
                        validate='one_to_one')
    if len(rows) != len(sample) or set(rows.prototype_split) != {'train', 'dev'}:
        raise ValueError('Sample changed or includes sealed test partition')
    all_nodes = extract_all(rows, corpus, families, dim_family, scale, counts)
    result = build_report(rows, all_nodes, args, corpus, protocol, hierarchy,
                          {'positive_rms': str(scale)}, Path(args.out), labels)
    rows[['city', 'cache_pano_index', 'panoid', 'spatial_block', 'prototype_split']].to_csv(
        Path(args.out) / 'sample.csv', index=False)
    print('DONE', json.dumps(result['counts'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=OUT / 'sample.csv')
    parser.add_argument('--out', type=Path, default=OUT / 'panorama_motifs')
    parser.add_argument('--min-train', type=int, default=15)
    parser.add_argument('--min-cities', type=int, default=6)
    parser.add_argument('--min-dev', type=int, default=5)
    parser.add_argument('--min-dev-cities', type=int, default=3)
    parser.add_argument('--permutations', type=int, default=199)
    parser.add_argument('--max-gallery', type=int, default=16)
    parser.add_argument('--examples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261017)
    main(parser.parse_args())
