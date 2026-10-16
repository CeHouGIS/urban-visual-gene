#!/usr/bin/env python3
"""Exploratory open mining of recurring, masked, multi-element street-view layouts.

This is a discovery pilot, not a trained object segmenter or a semantic validation.
The sealed prototype-layer test partition is never read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, gaussian_filter, label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'experiments/vitl30_multiscale_prototypes/outputs'
SPLITS = ROOT / 'experiments/vitl30_prototype_coverage/outputs'
HIERARCHY = ROOT / 'experiments/vitl30_relational_prototypes/results/hierarchy.json'
OUT = Path(__file__).resolve().parent / 'outputs'
GRID = 28
HEADINGS = (0, 90, 180, 270)
COLORS = ((234, 65, 54), (33, 142, 213))


def family_map(values, indices, indptr, local, heading, dim_to_family, scale, family_counts):
    """Aggregate positive SAE activations by candidate element family on a 28x28 view."""
    first = (local * 4 + heading) * (GRID * GRID)
    offsets = indptr[first:first + GRID * GRID + 1]
    start, stop = int(offsets[0]), int(offsets[-1])
    dimensions = np.asarray(indices[start:stop], dtype=np.int32)
    family = dim_to_family[dimensions]
    valid = family >= 0
    if not np.any(valid):
        return np.zeros((len(family_counts), GRID, GRID), np.float32)
    patch = np.repeat(np.arange(GRID * GRID, dtype=np.int32), np.diff(offsets))[valid]
    dimensions = dimensions[valid]
    family = family[valid]
    raw = np.asarray(values[start:stop], dtype=np.float32)[valid]
    weighted = np.minimum(raw / scale[dimensions], 8.) / np.sqrt(family_counts[family])
    grouped = np.bincount(family * GRID * GRID + patch, weights=weighted,
                          minlength=len(family_counts) * GRID * GRID)
    return grouped.reshape(len(family_counts), GRID, GRID).astype(np.float32)


@dataclass(frozen=True)
class Node:
    family: str
    heading: int
    y: float
    x: float
    area: int
    strength: float
    mask: str
    top_dimensions: tuple[int, ...]

    def unpack(self):
        return np.unpackbits(np.frombuffer(bytes.fromhex(self.mask), dtype=np.uint8),
                             count=GRID * GRID).reshape(GRID, GRID).astype(bool)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def setup():
    corpus = json.loads((SOURCE / 'corpus.json').read_text())
    cache = json.loads((SOURCE / 'cache_complete.json').read_text())
    protocol = json.loads((SPLITS / 'protocol.json').read_text())
    hierarchy = json.loads(HIERARCHY.read_text())
    if (cache['identity'] != corpus['identity'] or protocol['source_corpus_identity'] != corpus['identity']
            or protocol['checkpoint_sha256'] != corpus['checkpoint_sha256']
            or len(hierarchy['dimensions']) != 4096):
        raise ValueError('Incompatible sparse SAE cache, split or interpretation')
    frame = pd.read_parquet(SPLITS / 'panoramas.parquet')
    if frame.prototype_split.value_counts().to_dict() != protocol['split_counts']:
        raise ValueError('Prototype partitions changed')
    scale = np.load(SPLITS / 'positive_rms.npy')
    families = list(hierarchy['summary']['graph_candidate_family_counts'])
    dimension_family = np.full(4096, -1, np.int16)
    lookup = {key: i for i, key in enumerate(families)}
    labels = {}
    for row in hierarchy['dimensions']:
        if row['provisional_graph_candidate']:
            dimension_family[row['feature']] = lookup[row['family']]
        labels[row['feature']] = row['label_zh']
    counts = np.bincount(dimension_family[dimension_family >= 0], minlength=len(families))
    return corpus, protocol, hierarchy, frame, scale, families, dimension_family, counts, labels


def sample(frame, chunks_per_city=4, train_cap=64, dev_cap=20, seed=20261017):
    rng = random.Random(seed)
    chosen = []
    for city, group in frame.groupby('city', sort=True):
        starts = sorted(set(int(i) // 64 * 64 for i in group.cache_pano_index))
        selected_starts = set(rng.sample(starts, min(chunks_per_city, len(starts))))
        subset = group[group.cache_pano_index.map(lambda i: int(i) // 64 * 64 in selected_starts)]
        for split, cap in (('train', train_cap), ('dev', dev_cap)):
            candidates = subset[subset.prototype_split == split]
            indices = rng.sample(list(candidates.index), min(cap, len(candidates)))
            chosen.extend(indices)
    result = frame.loc[chosen].sort_values(['city', 'cache_pano_index']).reset_index(drop=True)
    if set(result.prototype_split) != {'train', 'dev'} or result.city.nunique() != 29:
        raise ValueError('Incomplete city or partition sample')
    return result


def node_dimensions(values, indices, ptr, local, heading, mask, family_id, dimension_family):
    first = (local * 4 + heading) * GRID * GRID
    offsets = ptr[first:first + GRID * GRID + 1]
    start, stop = int(offsets[0]), int(offsets[-1])
    patch = np.repeat(np.arange(GRID * GRID), np.diff(offsets))
    dim = np.asarray(indices[start:stop], dtype=np.int32)
    keep = mask.reshape(-1)[patch] & (dimension_family[dim] == family_id)
    if not np.any(keep):
        return ()
    totals = np.bincount(dim[keep], weights=np.asarray(values[start:stop], dtype=np.float32)[keep],
                         minlength=4096)
    return tuple(int(d) for d in np.argsort(totals)[-8:][::-1] if totals[d] > 0)


def extract_nodes(arrays, local, families, dimension_family, scale, counts):
    candidates = []
    for heading in range(4):
        maps = family_map(*arrays, local, heading, dimension_family, scale, counts)
        for family_id, raw in enumerate(maps):
            smooth = gaussian_filter(raw, .8)
            peak = float(smooth.max())
            if peak < .04:
                continue
            positive = smooth[smooth > peak * .02]
            threshold = max(peak * .33, float(np.quantile(positive, .68)))
            binary = binary_closing(smooth >= threshold, iterations=1)
            regions, total = label(binary)
            family_nodes = []
            for region_id in range(1, total + 1):
                mask = regions == region_id
                area = int(mask.sum())
                if area < 10 or area > 520:
                    continue
                weight = smooth * mask
                strength = float(weight.sum())
                yy, xx = np.indices(mask.shape)
                y = float((weight * yy).sum() / strength)
                x = float((weight * xx).sum() / strength)
                family_nodes.append(Node(families[family_id], heading, y, x, area, strength,
                       np.packbits(mask.reshape(-1)).tobytes().hex(),
                       node_dimensions(*arrays, local, heading, mask, family_id, dimension_family)))
            family_nodes.sort(key=lambda n: n.strength, reverse=True)
            candidates.extend(family_nodes[:2])
    # Avoid letting a large common surface crowd out smaller distinct elements.
    candidates.sort(key=lambda n: n.strength / np.sqrt(n.area), reverse=True)
    selected, counts_by_family = [], Counter()
    for node in candidates:
        if counts_by_family[node.family] < 3:
            selected.append(node)
            counts_by_family[node.family] += 1
        if len(selected) == 12:
            break
    return selected


def relation(a, b):
    if a.family == b.family:
        return None
    if a.heading == b.heading:
        ma, mb = a.unpack(), b.unpack()
        intersection = int((ma & mb).sum())
        if intersection / min(a.area, b.area) > .30:
            return None
        dy, dx = (b.y - a.y) / GRID, (b.x - a.x) / GRID
        vertical = 'D' if dy > .20 else 'U' if dy < -.20 else 'M'
        horizontal = 'R' if dx > .20 else 'L' if dx < -.20 else 'M'
        if vertical == horizontal == 'M':
            return None
        return 'same_' + vertical + horizontal
    angle = (b.heading - a.heading) % 4
    dy = (b.y - a.y) / GRID
    vertical = 'D' if dy > .25 else 'U' if dy < -.25 else 'M'
    return 'cross_' + str(angle * 90) + '_' + vertical


def pair_instances(nodes):
    best = {}
    for ai, a in enumerate(nodes):
        for bi in range(ai + 1, len(nodes)):
            b = nodes[bi]
            if a.family == b.family:
                continue
            if a.family > b.family:
                left, right, indices = b, a, (bi, ai)
            else:
                left, right, indices = a, b, (ai, bi)
            rel = relation(left, right)
            if rel is None:
                continue
            key = (left.family, right.family, rel)
            strength = min(left.strength / np.sqrt(left.area), right.strength / np.sqrt(right.area))
            if key not in best or strength > best[key][0]:
                best[key] = (strength, indices)
    return best


def shuffled_nodes(nodes, rng, mode):
    if mode == 'layout':
        positions = [(n.heading, n.y, n.x, n.area, n.mask) for n in nodes]
        rng.shuffle(positions)
        return [replace(n, heading=h, y=y, x=x, area=area, mask=mask)
                for n, (h, y, x, area, mask) in zip(nodes, positions)]
    raise ValueError(mode)


def city_frequency(rows, matches, split):
    by_city, blocks = defaultdict(set), set()
    for idx in matches:
        row = rows.iloc[idx]
        if row.prototype_split != split:
            continue
        by_city[row.city].add(idx)
        blocks.add(row.spatial_block)
    return len([x for x in by_city.values() if x]), len(blocks)


def null_counts(rows, all_nodes, candidates, repeats=399, seed=6138):
    """Frozen dev candidates; shuffle family labels within city and view, or geometry within pano."""
    dev_indices = [i for i, split in enumerate(rows.prototype_split) if split == 'dev']
    city_heading = defaultdict(list)
    for idx in dev_indices:
        for node_id, node in enumerate(all_nodes[idx]):
            city_heading[(rows.iloc[idx].city, node.heading)].append((idx, node_id))
    rng = random.Random(seed)
    results = {'cooccurrence': {key: [] for key in candidates},
               'layout': {key: [] for key in candidates}}
    for repeat in range(repeats):
        labels = {}
        for sites in city_heading.values():
            original = [all_nodes[i][j].family for i, j in sites]
            rng.shuffle(original)
            labels.update(zip(sites, original))
        counts = Counter()
        for idx in dev_indices:
            nodes = [replace(n, family=labels[(idx, j)]) for j, n in enumerate(all_nodes[idx])]
            counts.update(set(pair_instances(nodes)) & candidates)
        for key in candidates:
            results['cooccurrence'][key].append(counts[key])
        counts = Counter()
        for idx in dev_indices:
            counts.update(set(pair_instances(shuffled_nodes(all_nodes[idx], rng, 'layout'))) & candidates)
        for key in candidates:
            results['layout'][key].append(counts[key])
        if (repeat + 1) % 50 == 0:
            print('NULL', repeat + 1, '/', repeats, flush=True)
    return results


def bh(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [1.] * len(values)
    running = 1.
    for rank in range(len(values), 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted


def gallery_image(row, pair, path):
    images = []
    for heading in HEADINGS:
        with Image.open(row['image_' + str(heading)]) as source:
            view = source.convert('RGB')
            view.thumbnail((350, 350))
            width, height = view.size
            canvas = view.copy()
            tint = Image.new('RGB', view.size)
            tint.paste((20, 25, 32), (0, 0, width, height))
            canvas = Image.blend(canvas, tint, .15)
            for color, node in zip(COLORS, pair):
                if node.heading != heading // 90:
                    continue
                mask = Image.fromarray((node.unpack() * 255).astype(np.uint8), 'L')
                mask = mask.resize(view.size, Image.Resampling.NEAREST)
                overlay = Image.new('RGB', view.size, color)
                canvas.paste(Image.blend(view, overlay, .52), (0, 0), mask)
            images.append(canvas)
    width = max(i.width for i in images)
    height = max(i.height for i in images)
    output = Image.new('RGB', (width * 4, height + 34), (244, 245, 242))
    draw = ImageDraw.Draw(output)
    for i, image in enumerate(images):
        output.paste(image, (i * width, 34))
        draw.text((i * width + 8, 9), str(HEADINGS[i]) + '°', fill=(25, 34, 42))
    output.save(path, 'WEBP', quality=82, method=4)


def main(args):
    corpus, protocol, hierarchy, frame, scale, families, dim_family, counts, labels = setup()
    rows = sample(frame, args.chunks_per_city, args.train_cap, args.dev_cap, args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    same_chunk, arrays, all_nodes = None, None, []
    for idx, row in rows.iterrows():
        chunk_start = int(row.cache_pano_index) // 64 * 64
        key = (row.city, chunk_start)
        if key != same_chunk:
            folder = SOURCE / 'activations' / row.city.replace('/', '__') / ('%06d' % chunk_start)
            marker = json.loads((folder / 'complete.json').read_text())
            if marker['identity'] != corpus['identity']:
                raise ValueError('Activation chunk identity mismatch')
            arrays = tuple(np.load(folder / (name + '.npy'), mmap_mode='r')
                           for name in ('values', 'indices', 'indptr'))
            same_chunk = key
        all_nodes.append(extract_nodes(arrays, int(row.cache_pano_index) - chunk_start,
                                       families, dim_family, scale, counts))
        if (idx + 1) % 200 == 0:
            print('EXTRACT', idx + 1, '/', len(rows), flush=True)
    observed = defaultdict(list)
    for idx, nodes in enumerate(all_nodes):
        for key, pair in pair_instances(nodes).items():
            observed[key].append((idx, pair))
    train = [key for key, match in observed.items() if
             sum(rows.iloc[i].prototype_split == 'train' for i, _ in match) >= args.min_train and
             city_frequency(rows, [i for i, _ in match], 'train')[0] >= args.min_train_cities]
    train.sort(key=lambda key: -len(observed[key]))
    print('CANDIDATES', len(train), 'ALL_KEYS', len(observed), flush=True)
    if not train:
        raise ValueError('No recurring cross-city pair candidates; inspect region quality')
    null = null_counts(rows, all_nodes, set(train), args.permutations, args.seed + 2)
    patterns = []
    for key in train:
        matches = observed[key]
        train_matches = [x for x in matches if rows.iloc[x[0]].prototype_split == 'train']
        dev_matches = [x for x in matches if rows.iloc[x[0]].prototype_split == 'dev']
        pvalues, expected = {}, {}
        for mode in ('cooccurrence', 'layout'):
            controls = null[mode][key]
            pvalues[mode] = (1 + sum(n >= len(dev_matches) for n in controls)) / (1 + len(controls))
            expected[mode] = float(np.mean(controls))
        train_cities, train_blocks = city_frequency(rows, [i for i, _ in train_matches], 'train')
        dev_cities, dev_blocks = city_frequency(rows, [i for i, _ in dev_matches], 'dev')
        patterns.append({'families': list(key[:2]), 'relation': key[2],
              'train_panoramas': len(train_matches), 'train_cities': train_cities,
              'train_blocks': train_blocks, 'dev_panoramas': len(dev_matches),
              'dev_cities': dev_cities, 'dev_blocks': dev_blocks, 'p': pvalues,
              'null_means': expected, '_matches': dev_matches})
    for mode in ('cooccurrence', 'layout'):
        corrected = bh([item['p'][mode] for item in patterns])
        for item, q in zip(patterns, corrected):
            item.setdefault('q', {})[mode] = q
    # A display filter, not a claim of validated semantics or full-data support.
    patterns.sort(key=lambda p: (p['q']['cooccurrence'] + p['q']['layout'],
                                  -p['dev_cities'], -p['dev_panoramas']))
    gallery = output / 'gallery'
    gallery.mkdir(exist_ok=True)
    public = []
    for item in patterns:
        if len(public) >= args.max_gallery:
            break
        if item['dev_cities'] < 4 or item['dev_panoramas'] < 8:
            continue
        by_city = {}
        for idx, pair in item['_matches']:
            city = rows.iloc[idx].city
            if city not in by_city or pair[0] > by_city[city][1][0]:
                by_city[city] = (idx, pair)
        selections = sorted(by_city.values(), key=lambda x: -x[1][0])[:6]
        examples = []
        for number, (idx, (score, node_ids)) in enumerate(selections):
            row = rows.iloc[idx]
            nodes = [all_nodes[idx][n] for n in node_ids]
            filename = 'p%02d_%02d.webp' % (len(public) + 1, number + 1)
            gallery_image(row, nodes, gallery / filename)
            examples.append({'city': row.city, 'panoid': row.panoid,
                'spatial_block': row.spatial_block, 'image': filename, 'score': score,
                'nodes': [{'family': n.family, 'heading': HEADINGS[n.heading],
                           'area_patches': n.area, 'top_dimensions': list(n.top_dimensions),
                           'top_labels': [labels[d] for d in n.top_dimensions[:3]]} for n in nodes]})
        public.append({k: v for k, v in item.items() if k != '_matches'} | {'examples': examples})
    result = {'status': 'complete', 'scope': 'exploratory balanced city sample; sealed test not accessed',
              'checkpoint_sha256': corpus['checkpoint_sha256'], 'corpus_identity': corpus['identity'],
              'split_identity': protocol['identity'], 'hierarchy_identity': hierarchy['identity'],
              'hierarchy_sha256': digest(HIERARCHY),
              'sampling': {'seed': args.seed, 'chunks_per_city': args.chunks_per_city,
                           'train_cap_per_city': args.train_cap, 'dev_cap_per_city': args.dev_cap,
                           'train_panoramas': int((rows.prototype_split == 'train').sum()),
                           'dev_panoramas': int((rows.prototype_split == 'dev').sum()),
                           'cities': int(rows.city.nunique()), 'nodes': sum(map(len, all_nodes)),
                           'sampled_images': len(rows) * 4},
              'search': {'minimum_train_panoramas': args.min_train,
                         'minimum_train_cities': args.min_train_cities,
                         'frozen_candidates': len(train), 'permutations': args.permutations,
                         'multiple_testing': 'BH-FDR across every frozen train candidate on dev',
                         'display_rule': 'at least 8 dev panoramas and 4 dev cities; top 6 city-distinct examples'},
              'findings': {'displayed': len(public),
                           'q_below_005_both_nulls': sum(p['q']['cooccurrence'] < .05 and
                               p['q']['layout'] < .05 for p in patterns),
                           'all_candidates': len(patterns)},
              'limitations': ['Region masks arise from smoothed VLM-assisted SAE family activations, not independent object segmentation or pixel truth.',
                              'Family labels are provisional; shared family parts are collapsed only approximately.',
                              'The pilot does not train a continuous 4096-dimensional soft matcher, search triples, or demonstrate 80% semantic patch coverage.',
                              'Frozen SAE and VLM interpretation may have seen prototype-layer development images; inference is exploratory.'],
              'patterns': public}
    write_json(output / 'results.json', result)
    rows[['city', 'cache_pano_index', 'panoid', 'spatial_block', 'prototype_split']].to_csv(
        output / 'sample.csv', index=False)
    print('DONE', json.dumps({'sampling': result['sampling'], 'findings': result['findings']},
                             ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--chunks-per-city', type=int, default=4)
    parser.add_argument('--train-cap', type=int, default=64)
    parser.add_argument('--dev-cap', type=int, default=20)
    parser.add_argument('--min-train', type=int, default=30)
    parser.add_argument('--min-train-cities', type=int, default=8)
    parser.add_argument('--permutations', type=int, default=399)
    parser.add_argument('--max-gallery', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261017)
    main(parser.parse_args())
