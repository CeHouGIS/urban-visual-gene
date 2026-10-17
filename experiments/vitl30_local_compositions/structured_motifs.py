#!/usr/bin/env python3
"""Audit panorama motifs using the curated Qwen label structure.

The geometry and sample come from panorama_motifs. Labels are provisional:
component labels describe evidence for a region, not detected object instances.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .panorama_motifs import (bh, circular_distance, extract_all,
                              likely_duplicate_observation, motif_key,
                              panorama_angle, p_value, render_panorama)
from .run import OUT, digest, setup, write_json

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / 'docs/site/label_structure_rules.json'
COMMON = {'building', 'road', 'tree', 'low_vegetation'}
FOCUS = [
    ('building', 'tree', 'tree'),
    ('barrier', 'road', 'utility'),
    ('road', 'utility', 'vehicle'),
    ('building', 'road', 'sign'),
    ('road', 'transport_structure', 'tree'),
    ('barrier', 'building', 'tree'),
]


@dataclass(frozen=True)
class TaggedNode:
    region: object
    observed_label: str
    entity_label: str


def label_index(hierarchy, rules):
    parents = {'is_a': {}, 'part_of': {}}
    linked = set()
    for group in rules['hierarchy']:
        family, parent = group['family'], group['parent']
        for child in group['children']:
            key = (family, child)
            if key in parents[group['relation']]:
                raise ValueError(f'Ambiguous label parent: {key}')
            parents[group['relation']][key] = parent
            linked.update(((family, parent), key))
    for group in rules['parallel']:
        linked.update((group['family'], label) for label in group['labels'])

    # Resolve with separate chains: the first part-of edge changes the entity.
    def resolve(family, label):
        current, seen = label, set()
        while (family, current) not in seen:
            seen.add((family, current))
            part = parents['part_of'].get((family, current))
            if part:
                current = part
                while (family, current) in parents['part_of']:
                    current = parents['part_of'][(family, current)]
                return current
            current = parents['is_a'].get((family, current))
            if current is None:
                break
        return label

    mapping = {}
    for row in hierarchy['dimensions']:
        key = (row['family'], row['label_zh'])
        if row['provisional_graph_candidate'] and key in linked:
            mapping[row['feature']] = (row['label_zh'], resolve(*key))
    return mapping


def tag_nodes(nodes, mapping):
    tagged = []
    for node in nodes:
        if not node.top_dimensions:
            continue
        labels = next((mapping[dim] for dim in node.top_dimensions if dim in mapping), None)
        if labels is None:
            continue
        tagged.append(TaggedNode(node, *labels))
    return tagged


def separate_objects(a: TaggedNode, b: TaggedNode):
    first, second = a.region, b.region
    if first.family != second.family:
        return True
    # Two masks from one building, vehicle, or road are not two objects.
    if first.family != 'tree':
        return False
    if likely_duplicate_observation(first, second):
        return False
    # A large angular gap gives a reviewable two-tree candidate, not proof of
    # two physical trees; contiguous canopy fragments remain excluded.
    return circular_distance(panorama_angle(first), panorama_angle(second)) >= 30


def instances(nodes, allowed=None):
    best = {}
    for indices in combinations(range(len(nodes)), 3):
        selected = [nodes[i] for i in indices]
        regions = [n.region for n in selected]
        if any(not separate_objects(a, b) for a, b in combinations(selected, 2)):
            continue
        signature = tuple(sorted(n.family for n in regions))
        if allowed is not None and signature not in allowed:
            continue
        key = motif_key(regions, (0, 1, 2))
        score = min(n.strength / np.sqrt(n.area) for n in regions)
        if key not in best or score > best[key][0]:
            best[key] = (score, indices)
    return best


def null_counts(rows, nodes, candidates, repeats, seed):
    dev = [i for i, split in enumerate(rows.prototype_split) if split == 'dev']
    sites = defaultdict(list)
    for i in dev:
        for j, node in enumerate(nodes[i]):
            sites[(rows.iloc[i].city, node.region.heading)].append((i, j))
    rng = random.Random(seed)
    signatures = {key[0] for key in candidates}
    result = {mode: {key: [] for key in candidates}
              for mode in ('cooccurrence', 'layout')}
    for repeat in range(repeats):
        shuffled_tags = {}
        for positions in sites.values():
            tags = [(nodes[i][j].region.family, nodes[i][j].observed_label,
                     nodes[i][j].entity_label) for i, j in positions]
            rng.shuffle(tags)
            shuffled_tags.update(zip(positions, tags))
        observed = Counter()
        for i in dev:
            changed = []
            for j, node in enumerate(nodes[i]):
                family, label, entity = shuffled_tags[(i, j)]
                changed.append(TaggedNode(replace(node.region, family=family), label, entity))
            observed.update(set(instances(changed, signatures)) & candidates)
        for key in candidates:
            result['cooccurrence'][key].append(observed[key])

        observed = Counter()
        for i in dev:
            geometry = [(n.region.heading, n.region.y, n.region.x,
                         n.region.area, n.region.mask) for n in nodes[i]]
            rng.shuffle(geometry)
            changed = [replace(node, region=replace(node.region, heading=h,
                       y=y, x=x, area=area, mask=mask))
                       for node, (h, y, x, area, mask) in zip(nodes[i], geometry)]
            observed.update(set(instances(changed, signatures)) & candidates)
        for key in candidates:
            result['layout'][key].append(observed[key])
        if (repeat + 1) % 25 == 0:
            print(f'NULL {repeat + 1}/{repeats}', flush=True)
    return result


def role_labels(nodes, indices):
    selected = sorted((nodes[i] for i in indices),
                      key=lambda n: (n.region.family, n.entity_label, n.observed_label))
    return tuple((n.region.family, n.entity_label) for n in selected)


def main(args):
    corpus, protocol, hierarchy, frame, scale, families, dim_family, counts, _ = setup()
    rules = json.loads(RULES.read_text())
    mapping = label_index(hierarchy, rules)
    sample = pd.read_csv(args.sample)
    rows = sample.merge(frame, on=['city', 'cache_pano_index', 'panoid',
                                   'spatial_block', 'prototype_split'], validate='one_to_one')
    if len(rows) != len(sample) or set(rows.prototype_split) != {'train', 'dev'}:
        raise ValueError('Sample changed or includes sealed test partition')
    raw = extract_all(rows, corpus, families, dim_family, scale, counts)
    nodes = [tag_nodes(group, mapping) for group in raw]
    observed = defaultdict(list)
    for i, group in enumerate(nodes):
        for key, hit in instances(group).items():
            observed[key].append((i, hit))
    candidates = {}
    for key, matches in observed.items():
        train = [i for i, _ in matches if rows.iloc[i].prototype_split == 'train']
        if len(train) >= args.min_train and len(set(rows.iloc[i].city for i in train)) >= args.min_cities:
            candidates[key] = matches
    print(f'CANDIDATES {len(candidates)}', flush=True)
    if not candidates:
        raise ValueError('No structurally eligible recurring motifs')
    null = null_counts(rows, nodes, set(candidates), args.permutations, args.seed)
    summaries = []
    for key, matches in candidates.items():
        train = [(i, hit) for i, hit in matches if rows.iloc[i].prototype_split == 'train']
        dev = [(i, hit) for i, hit in matches if rows.iloc[i].prototype_split == 'dev']
        support = {split: {'panoramas': len(hits),
                           'cities': len(set(rows.iloc[i].city for i, _ in hits)),
                           'blocks': len(set(rows.iloc[i].spatial_block for i, _ in hits))}
                   for split, hits in (('train', train), ('dev', dev))}
        label_train = Counter(role_labels(nodes[i], hit[1]) for i, hit in train)
        label_dev = Counter(role_labels(nodes[i], hit[1]) for i, hit in dev)
        fine = []
        for labels, count in label_train.most_common():
            if count >= args.min_fine_train and label_dev[labels] >= args.min_fine_dev:
                fine.append({'roles': [list(x) for x in labels], 'train_panoramas': count,
                             'dev_panoramas': label_dev[labels]})
        p = {mode: p_value(len(dev), null[mode][key]) for mode in null}
        summaries.append({'families': list(key[0]), 'layout': [list(x) for x in key[1]],
                          'support': support, 'p': p,
                          'null_means': {mode: float(np.mean(null[mode][key])) for mode in null},
                          'fine_compositions': fine[:12], '_dev': dev, '_key': key})
    for mode in null:
        for item, q in zip(summaries, bh([x['p'][mode] for x in summaries])):
            item.setdefault('q', {})[mode] = q
    summaries.sort(key=lambda x: (-len(set(x['families']) - COMMON),
                                  -(x['q']['cooccurrence'] < .05 and x['q']['layout'] < .05),
                                  -x['support']['dev']['cities'],
                                  -x['support']['dev']['panoramas']))
    gallery = args.out / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    shown, signatures = [], set()
    priority = [next((item for item in summaries if tuple(item['families']) == signature
                      and item['support']['dev']['panoramas'] >= args.min_dev), None)
                for signature in FOCUS]
    for item in [x for x in priority if x is not None] + summaries:
        signature = tuple(item['families'])
        if signature in signatures or item['support']['dev']['panoramas'] < args.min_dev:
            continue
        signatures.add(signature)
        by_city = {}
        for i, hit in item['_dev']:
            city = rows.iloc[i].city
            if city not in by_city or hit[0] > by_city[city][1][0]:
                by_city[city] = (i, hit)
        examples = []
        for number, (i, (score, ids)) in enumerate(sorted(by_city.values(),
                                       key=lambda pair: -pair[1][0])[:args.examples], 1):
            chosen = [nodes[i][j] for j in ids]
            filename = f's{len(shown)+1:02d}_{number:02d}.webp'
            render_panorama(rows.iloc[i], [n.region for n in chosen], gallery / filename)
            examples.append({'city': rows.iloc[i].city, 'panoid': rows.iloc[i].panoid,
                             'image': filename, 'score': score,
                             'nodes': [{'family': n.region.family, 'observed_label': n.observed_label,
                                        'entity_label': n.entity_label,
                                        'top_dimensions': list(n.region.top_dimensions),
                                        'heading': n.region.heading * 90} for n in chosen]})
        shown.append({k: v for k, v in item.items() if not k.startswith('_')} | {'examples': examples})
        if len(shown) >= args.max_gallery:
            break
    report = {'status': 'exploratory', 'checkpoint_sha256': corpus['checkpoint_sha256'],
              'corpus_identity': corpus['identity'], 'split_identity': protocol['identity'],
              'hierarchy_identity': hierarchy['identity'], 'rules_sha256': digest(RULES),
              'sample_sha256': digest(args.sample),
              'sampling': {'panoramas': len(rows), 'cities': int(rows.city.nunique()),
                           'train': int((rows.prototype_split == 'train').sum()),
                           'dev': int((rows.prototype_split == 'dev').sum()),
                           'family_regions': sum(map(len, raw)),
                           'structured_regions': sum(map(len, nodes))},
              'method': {'unit': 'complete four-heading panorama', 'motif_size': 3,
                         'label_selection': 'highest-contributing dimension per family region, explicit linked labels only',
                         'part_of': 'component maps to containing entity; no component counted independently',
                         'same_family': 'only distant tree regions may repeat',
                         'min_tree_angle_deg': 30, 'permutations': args.permutations,
                         'fdr': 'BH over train-selected layout keys, evaluated on dev',
                         'minimum_train': args.min_train, 'minimum_cities': args.min_cities},
              'counts': {'observed_keys': len(observed), 'train_candidates': len(candidates),
                         'both_nulls_q_below_005': sum(x['q']['cooccurrence'] < .05 and
                             x['q']['layout'] < .05 for x in summaries),
                         'gallery_patterns': len(shown)},
              'limitations': ['Qwen labels and family masks are provisional, not object segmentation.',
                              'Two separated tree regions may still be one physical tree.',
                              'Fine compositions are descriptive; their p/q values were not separately tested.',
                              'VLM evidence may overlap later test images; sealed test activations were not accessed.'],
              'patterns': shown,
              'all_candidate_summaries': [{k: v for k, v in x.items() if not k.startswith('_')}
                                          for x in summaries]}
    write_json(args.out / 'results.json', report)
    print('DONE', json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=OUT / 'sample.csv')
    parser.add_argument('--out', type=Path, default=OUT / 'structured_motifs')
    parser.add_argument('--min-train', type=int, default=15)
    parser.add_argument('--min-cities', type=int, default=6)
    parser.add_argument('--min-dev', type=int, default=5)
    parser.add_argument('--min-fine-train', type=int, default=3)
    parser.add_argument('--min-fine-dev', type=int, default=1)
    parser.add_argument('--permutations', type=int, default=99)
    parser.add_argument('--max-gallery', type=int, default=16)
    parser.add_argument('--examples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261017)
    main(parser.parse_args())
