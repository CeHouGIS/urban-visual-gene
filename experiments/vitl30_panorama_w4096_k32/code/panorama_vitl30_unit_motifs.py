"""Audit every SAE dimension, then discover and test panorama compositions.

This CPU analysis reuses frozen Top100 crop embeddings and per-panorama SAE
peak locations. Embedding agreement is a screening measure, never a semantic
label. City-disjoint discovery and confirmation avoid testing mined hypotheses
on the observations used to select them. Peak relations are image evidence,
not recovered 3-D geometry or segmentation masks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from formal.analysis.semantic_feature_umap import digest, write_json
from sae_experiments.panorama_vitl30 import OUTPUT, WIDTH
from sae_experiments.panorama_vitl30_landscape_prototypes import load_peaks

SOURCE = OUTPUT / 'analysis/semantic_umap_top100'
REPORT = Path('/root/code/urban-visual-gene/reports/vitl30_unit_motifs')
STATES = {
    'unobserved_in_sample': '样本内未观察到正激活',
    'insufficient_examples': '实例不足，暂不能判断',
    'coherent_visual_candidate': '视觉一致性候选，语义待核对',
    'possible_multiple_patterns': '可能存在多种视觉模式',
    'inconclusive_visual_evidence': '视觉证据尚不明确',
}


def bh_adjust(p):
    p = np.asarray(p, float)
    if not len(p):
        return p
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def unit_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pair_agreement(x):
    """Average off-diagonal cosine for unit vectors, including batched inputs."""
    n = x.shape[-2]
    if n < 2:
        return np.zeros(x.shape[:-2])
    total = x.sum(axis=-2)
    return (np.sum(total * total, axis=-1) - np.sum(x * x, axis=(-2, -1))) / (n * (n - 1))


def cross_city_agreement(x, cities):
    """Average cosine over pairs from different cities; NaN if no such pair."""
    total = x.sum(axis=-2)
    numerator = np.sum(total * total, axis=-1)
    denominator = len(cities) ** 2
    for city in np.unique(cities):
        selected = np.flatnonzero(cities == city)
        part = x[..., selected, :].sum(axis=-2)
        numerator = numerator - np.sum(part * part, axis=-1)
        denominator -= len(selected) ** 2
    return numerator / denominator if denominator else np.full(x.shape[:-2], np.nan)


def spherical_split(x, start):
    """Small two-mode screen, not a semantic classifier."""
    first = int(start % len(x))
    second = int(np.argmin(x @ x[first]))
    centers = x[[first, second]].copy()
    previous = None
    labels = np.zeros(len(x), int)
    for _ in range(25):
        labels = (x @ centers.T).argmax(axis=1)
        if previous is not None and np.array_equal(previous, labels):
            break
        if len(np.unique(labels)) < 2:
            return labels, 1.
        centers = unit_rows(np.stack([x[labels == k].mean(0) for k in range(2)]))
        previous = labels.copy()
    return labels, float(centers[0] @ centers[1])


def read_crops(source):
    """Keep the array/manifest row identity; avoid retaining encoded heatmaps."""
    data = []
    for line in (source / 'crops.jsonl').open():
        row = json.loads(line)
        row.pop('heatmap_png', None)
        data.append(row)
    embeddings = np.load(source / 'crop_embeddings.f32.npy', mmap_mode='r')
    meta = json.loads((source / 'embedding_run.json').read_text())
    if len(data) != len(embeddings) or meta['crop_manifest_sha256'] != digest(source / 'crops.jsonl'):
        raise ValueError('Embedding/manifest identity mismatch')
    if not np.load(source / 'embedding_completed.npy').all():
        raise ValueError('Incomplete embedding cache')
    return data, embeddings


def audit_dimensions(source, crops, embeddings, seed, repeats, workers):
    features = np.array([r['feature'] for r in crops])
    pano_ids = np.array([r['pano_index'] for r in crops])
    city_names = sorted({r['city'] for r in crops})
    city_index = {name: i for i, name in enumerate(city_names)}
    cities = np.array([city_index[r['city']] for r in crops])
    _, crop_ids = np.unique([r['crop'] for r in crops], return_inverse=True)
    support = np.load(source / 'positive_panorama_support.npy')
    by_feature = [np.flatnonzero(features == f) for f in range(WIDTH)]
    # An equal-chance panorama, then an equal-chance eligible crop in that pano.
    # This prevents frequently retrieved panoramas dominating the null pool.
    pano_order = np.argsort(pano_ids, kind='stable')
    unique_panos, starts, counts = np.unique(pano_ids[pano_order], return_index=True, return_counts=True)
    start_lookup = dict(zip(unique_panos.tolist(), zip(starts.tolist(), counts.tolist())))
    city_panos = {c: np.unique(pano_ids[cities == c]) for c in range(len(city_names))}

    def one(feature):
        ids = by_feature[feature]
        n = len(ids)
        result = {'feature': feature, 'positive_panoramas': int(support[feature]),
                  'top_crop_count': n, 'top_crop_cities': int(len(np.unique(cities[ids]))),
                  'semantic_verdict': 'not_independently_verified', 'semantic_label': '',
                  'coherence': None, 'cross_city_coherence': None, 'null_coherence': None,
                  'coherence_excess': None, 'coherence_p': 1., 'coherence_q': 1.,
                  'rank_continuation_excess': None, 'two_mode_center_cosine': None,
                  'two_mode_partition_ari': None, 'two_mode_minor_fraction': None}
        if n < 20:
            result['status'] = 'unobserved_in_sample' if n == 0 else 'insufficient_examples'
            return result
        x = unit_rows(np.asarray(embeddings[ids], np.float32))
        target_cities = cities[ids]
        rng = np.random.default_rng(seed + feature * 1009)
        reference = np.empty((repeats, n), np.int64)
        forbidden_crops = set(crop_ids[ids].tolist())
        for city in np.unique(target_cities):
            columns = np.flatnonzero(target_cities == city)
            pool = city_panos[int(city)]
            chosen_panos = pool[np.argsort(rng.random((repeats, len(pool))), axis=1)[:, :len(columns)]]
            flat_panos = chosen_panos.ravel()
            start = np.array([start_lookup[int(p)][0] for p in flat_panos])
            count = np.array([start_lookup[int(p)][1] for p in flat_panos])
            pick = pano_order[start + (rng.random(len(start)) * count).astype(int)]
            bad = (features[pick] == feature) | np.isin(crop_ids[pick], list(forbidden_crops))
            attempts = 0
            while bad.any() and attempts < 30:
                pick[bad] = pano_order[start[bad] + (rng.random(int(bad.sum())) * count[bad]).astype(int)]
                bad = (features[pick] == feature) | np.isin(crop_ids[pick], list(forbidden_crops))
                attempts += 1
            for position in np.flatnonzero(bad):
                candidates = pano_order[start[position]:start[position] + count[position]]
                candidates = candidates[(features[candidates] != feature) &
                                        ~np.isin(crop_ids[candidates], list(forbidden_crops))]
                if not len(candidates):
                    raise ValueError('No independent crop reference in a matched panorama')
                pick[position] = rng.choice(candidates)
            reference[:, columns] = pick.reshape(repeats, len(columns))
        controls = unit_rows(np.asarray(embeddings[reference], np.float32))
        use_cross = len(np.unique(target_cities)) >= 2
        real = float(cross_city_agreement(x, target_cities) if use_cross else pair_agreement(x))
        null = cross_city_agreement(controls, target_cities) if use_cross else pair_agreement(controls)
        top = min(20, n // 2)
        centroid = unit_rows(x[:top].mean(0))
        actual_continuation = float(x[top:].mean(0) @ centroid)
        control_continuation = float(np.mean(controls[:, top:, :] @ centroid))
        labels, separation = spherical_split(x, 0)
        other_labels, _ = spherical_split(x, n // 2)
        minor = float(np.bincount(labels, minlength=2).min() / n)
        result.update(coherence=float(pair_agreement(x)),
                      cross_city_coherence=real if use_cross else None,
                      null_coherence=float(np.mean(null)), coherence_excess=real - float(np.mean(null)),
                      coherence_p=float((1 + np.count_nonzero(null >= real)) / (repeats + 1)),
                      rank_continuation_excess=actual_continuation - control_continuation,
                      two_mode_center_cosine=separation,
                      two_mode_partition_ari=float(adjusted_rand_score(labels, other_labels)),
                      two_mode_minor_fraction=minor)
        return result

    results = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(one, range(WIDTH)), 1):
            results.append(result)
            if i % 256 == 0:
                print('AUDIT', i, '/', WIDTH, flush=True)
    eligible = [r for r in results if r['top_crop_count'] >= 20]
    for row, q in zip(eligible, bh_adjust([r['coherence_p'] for r in eligible])):
        row['coherence_q'] = float(q)
        multiple = (row['two_mode_minor_fraction'] >= .2 and row['two_mode_center_cosine'] < .8 and
                    row['two_mode_partition_ari'] >= .7)
        coherent = (q <= .05 and row['coherence_excess'] >= .03 and
                    row['rank_continuation_excess'] >= .03)
        row['status'] = ('possible_multiple_patterns' if multiple else
                         'coherent_visual_candidate' if coherent else 'inconclusive_visual_evidence')
    return results


def spatial_units(rows, experiment, cell_meters, seed):
    """Read only verified saved manifests, choose one panorama per city/grid cell."""
    coordinates = np.full((len(rows), 2), np.nan)
    for city in sorted({r['city'] for r in rows}):
        frame = pd.read_parquet(experiment / 'cache_manifests' / f'{city.replace("/", "__")}.parquet')
        for row in (r for r in rows if r['city'] == city):
            source = frame.iloc[row['cache_pano_index'] * 4]
            if str(source.panoid) != row['pano'] or not source.image_mapping_verified:
                raise ValueError('Coordinate/image manifest identity mismatch')
            coordinates[row['pano_index']] = [source.lat, source.lon]
    groups = defaultdict(list)
    for row, (lat, lon) in zip(rows, coordinates):
        if not np.isfinite([lat, lon]).all():
            continue
        # A local equirectangular grid per city, adequate for 1-km thinning.
        center_lat = np.nanmedian(coordinates[[i for i, r in enumerate(rows) if r['city'] == row['city']], 0])
        x = lon * 111_320 * np.cos(np.deg2rad(center_lat))
        y = lat * 111_320
        key = (row['city'], int(np.floor(x / cell_meters)), int(np.floor(y / cell_meters)))
        groups[key].append(row['pano_index'])
    keep = []
    cells = {}
    for key, values in groups.items():
        selected = min(values, key=lambda i: hashlib.sha256(f'{seed}/{rows[i]["pano"]}'.encode()).digest())
        keep.append(selected)
        cells[selected] = key[1:]
    return np.array(sorted(keep), int), coordinates, cells


def discovery_split(rows, keep, seed):
    city_names = sorted({rows[i]['city'] for i in keep})
    rng = np.random.default_rng(seed)
    rng.shuffle(city_names)
    n = max(1, min(len(city_names) - 1, round(len(city_names) * .65)))
    train_cities = set(city_names[:n])
    discovery = np.array([i for i in keep if rows[i]['city'] in train_cities], int)
    confirmation = np.array([i for i in keep if rows[i]['city'] not in train_cities], int)
    return discovery, confirmation


def fit_presence(peaks, discovery, quantile=.8):
    train = peaks[discovery]
    thresholds = np.full(peaks.shape[1], np.inf, np.float32)
    for j in range(peaks.shape[1]):
        positive = train[:, j][train[:, j] > 0]
        if len(positive) >= 20:
            thresholds[j] = np.quantile(positive, quantile)
    return (peaks > 0) & (peaks >= thresholds), thresholds


def expected_count(presence, row_ids, cities, feature_ids):
    result = 0.
    for city in np.unique(cities[row_ids]):
        ids = row_ids[cities[row_ids] == city]
        marginals = presence[np.ix_(ids, feature_ids)].mean(axis=0)
        result += len(ids) * float(np.prod(marginals))
    return result


def mine_candidates(presence, discovery, cities, max_pairs, max_triples):
    """Bounded Apriori pair/triple mining, selected entirely on discovery cities."""
    x = presence[discovery].astype(np.float32)
    joint = x.T @ x
    expected = np.zeros_like(joint)
    for city in np.unique(cities[discovery]):
        ids = discovery[cities[discovery] == city]
        counts = presence[ids].sum(0, dtype=np.float32)
        expected += np.outer(counts, counts) / len(ids)
    lift = joint / np.maximum(expected, 1e-9)
    score = (joint - expected) / np.sqrt(np.maximum(expected, 1))
    a, b = np.where(np.triu((joint >= 20) & (lift >= 1.5), k=1))
    order = np.argsort(-score[a, b], kind='stable')
    pairs = []
    for position in order:
        first, second = int(a[position]), int(b[position])
        ids = discovery[presence[discovery, first] & presence[discovery, second]]
        if len(np.unique(cities[ids])) >= 3:
            pairs.append((first, second))
            if len(pairs) >= max_pairs:
                break
    graph = defaultdict(set)
    for first, second in pairs:
        graph[first].add(second)
        graph[second].add(first)
    triples = []
    for first in sorted(graph):
        for second in sorted(j for j in graph[first] if j > first):
            for third in sorted(j for j in graph[first] & graph[second] if j > second):
                fs = (first, second, third)
                ids = discovery[presence[np.ix_(discovery, fs)].all(1)]
                if len(ids) < 12 or len(np.unique(cities[ids])) < 3:
                    continue
                expected_n = expected_count(presence, discovery, cities, fs)
                triples.append(((len(ids) - expected_n) / np.sqrt(max(expected_n, 1)), fs))
    triples.sort(key=lambda item: (-item[0], item[1]))
    return pairs + [fs for _, fs in triples[:max_triples]]


def conditional_independence_null(presence, ids, cities, features, repeats, rng):
    """Fixed city-wise feature counts; simulate random intersections exactly."""
    totals = np.zeros(repeats, int)
    for city in np.unique(cities[ids]):
        selected = ids[cities[ids] == city]
        n = len(selected)
        counts = presence[np.ix_(selected, features)].sum(0).astype(int)
        overlap = np.full(repeats, counts[0], int)
        for count in counts[1:]:
            overlap = rng.hypergeometric(overlap, n - overlap, int(count))
        totals += overlap
    return totals


def relation_codes(first, second):
    """Rotation-invariant heading/within-view peak relations, no metric 3-D claim."""
    heading_first, heading_second = first // 784, second // 784
    vertical = np.where((second % 784) // 28 - (first % 784) // 28 < -3, 0,
                        np.where((second % 784) // 28 - (first % 784) // 28 > 3, 2, 1))
    horizontal = np.where(second % 28 - first % 28 < -3, 0,
                          np.where(second % 28 - first % 28 > 3, 2, 1))
    delta = (heading_second - heading_first) % 4
    return np.where(delta == 0, vertical * 3 + horizontal, 8 + delta).astype(int)


def relation_name(code):
    if code >= 9:
        return {9: 'second peak in next heading (+90°)', 10: 'peaks in opposite headings',
                11: 'second peak in previous heading (−90°)'}[code]
    return ('same heading: second peak ' + ['above', 'at similar height', 'below'][code // 3] +
            ', ' + ['left', 'at similar horizontal position', 'right'][code % 3])


def spatial_test(patches, presence, discovery, confirmation, cities, features, repeats, rng):
    first = features[0]
    train_ids = discovery[presence[np.ix_(discovery, features)].all(1)]
    test_ids = confirmation[presence[np.ix_(confirmation, features)].all(1)]
    train_rel = np.stack([relation_codes(patches[train_ids, first], patches[train_ids, j])
                          for j in features[1:]], axis=1)
    patterns = Counter(map(tuple, train_rel.tolist()))
    template, count = sorted(patterns.items(), key=lambda item: (-item[1], item[0]))[0]
    result = {'spatial_template_codes': list(template),
              'spatial_template': [relation_name(code) for code in template],
              'discovery_template_fraction': count / len(train_ids),
              'confirmation_template_fraction': None, 'spatial_null_fraction': None,
              'spatial_excess': None, 'spatial_p': 1., 'spatial_q': 1.,
              'spatial_status': 'insufficient_confirmation_support'}
    if len(test_ids) < 10:
        return result
    test_rel = np.stack([relation_codes(patches[test_ids, first], patches[test_ids, j])
                        for j in features[1:]], axis=1)
    actual = int(np.all(test_rel == template, axis=1).sum())
    # Shuffle the entire (heading, x, y) peak location of each non-anchor feature
    # among co-present panoramas in the same city. Sky/ground positional biases
    # and within-city marginals remain intact.
    matches = np.ones((repeats, len(test_ids)), bool)
    for f, code in zip(features[1:], template):
        shuffled = np.broadcast_to(patches[test_ids, f], (repeats, len(test_ids))).copy()
        for city in np.unique(cities[test_ids]):
            positions = np.flatnonzero(cities[test_ids] == city)
            orders = rng.random((repeats, len(positions))).argsort(1)
            shuffled[:, positions] = patches[test_ids[positions[orders]], f]
        matches &= relation_codes(patches[test_ids, first][None, :], shuffled) == code
    counts = matches.sum(1)
    result.update(confirmation_template_fraction=actual / len(test_ids),
                  spatial_null_fraction=float(counts.mean() / len(test_ids)),
                  spatial_excess=float((actual - counts.mean()) / len(test_ids)),
                  spatial_p=float((1 + np.count_nonzero(counts >= actual)) / (repeats + 1)))
    return result


def test_motifs(candidates, presence, patches, discovery, confirmation, cities, repeats, seed):
    results = []
    for number, features in enumerate(candidates):
        train_ids = discovery[presence[np.ix_(discovery, features)].all(1)]
        test_ids = confirmation[presence[np.ix_(confirmation, features)].all(1)]
        expected = expected_count(presence, confirmation, cities, features)
        null = conditional_independence_null(presence, confirmation, cities, features, repeats,
                                             np.random.default_rng(seed + number * 1009))
        row = {'motif': number, 'features': list(features), 'size': len(features),
               'discovery_support': len(train_ids), 'discovery_cities': len(np.unique(cities[train_ids])),
               'confirmation_support': len(test_ids), 'confirmation_cities': len(np.unique(cities[test_ids])),
               'city_conditioned_expected': expected,
               'confirmation_lift': len(test_ids) / expected if expected else 0.,
               'cooccurrence_p': float((1 + np.count_nonzero(null >= len(test_ids))) / (repeats + 1))}
        row.update(spatial_test(patches, presence, discovery, confirmation, cities, features, repeats,
                                np.random.default_rng(seed + 9_000_001 + number)))
        results.append(row)
        if (number + 1) % 50 == 0:
            print('MOTIFS TESTED', number + 1, '/', len(candidates), flush=True)
    co_q = bh_adjust([r['cooccurrence_p'] for r in results])
    # Include untestable hypotheses as p=1, rather than reducing the family after
    # looking at which candidates have convenient confirmation support.
    space_q = bh_adjust([r['spatial_p'] for r in results])
    for row, cq, sq in zip(results, co_q, space_q):
        row['cooccurrence_q'], row['spatial_q'] = float(cq), float(sq)
        supported = (cq <= .05 and row['confirmation_lift'] >= 1.3 and row['confirmation_support'] >= 10)
        row['cooccurrence_status'] = ('cross_city_composition_candidate' if supported and row['confirmation_cities'] >= 3
                                      else 'regional_composition_candidate' if supported
                                      else 'not_confirmed')
        if row['confirmation_template_fraction'] is not None:
            row['spatial_status'] = ('repeatable_peak_relation_candidate' if supported and sq <= .05 and
                                     row['spatial_excess'] >= .1 and row['discovery_template_fraction'] >= .35
                                     else 'no_additional_spatial_evidence')
    return results


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def read_patches(source, rows, run):
    patches = np.zeros((len(rows), WIDTH), np.int16)
    for city in run['cities']:
        selected = [r for r in rows if r['city'] == city]
        with np.load(source / 'ranking' / f'{city.replace("/", "__")}.npz') as saved:
            patches[[r['pano_index'] for r in selected]] = saved['patches']
    if patches.min() < 0 or patches.max() >= 3136:
        raise ValueError('Invalid saved peak patch')
    return patches


def exemplar_ids(row, presence, peaks, thresholds, discovery, confirmation, pano_rows, limit=6):
    features = row['features']
    ids = np.concatenate([confirmation, discovery])
    ids = ids[presence[np.ix_(ids, features)].all(1)]
    strength = np.min(peaks[np.ix_(ids, features)] / thresholds[features], axis=1)
    confirm = set(confirmation.tolist())
    order = sorted(zip(ids.tolist(), strength.tolist()), key=lambda item: (item[0] not in confirm, -item[1], item[0]))
    chosen, used_cities = [], set()
    for distinct in (True, False):
        for i, _ in order:
            if i in chosen or (distinct and pano_rows[i]['city'] in used_cities):
                continue
            chosen.append(i)
            used_cities.add(pano_rows[i]['city'])
            if len(chosen) == limit:
                return chosen
    return chosen


def montage(source, target, selected, rows, patches, features, confirmation):
    colors = ['#ffdd00', '#00e5ff', '#ff55dd']
    canvas = Image.new('RGB', (960, len(selected) * 265), '#fff')
    draw = ImageDraw.Draw(canvas)
    confirm = set(confirmation.tolist())
    for line, pid in enumerate(selected):
        for heading in range(4):
            with Image.open(source / 'sources' / f'{pid * 4 + heading}.jpg') as original:
                canvas.paste(original.convert('RGB').resize((240, 240)), (heading * 240, line * 265))
        for f, color in zip(features, colors):
            patch = int(patches[pid, f])
            heading, inner = divmod(patch, 784)
            y, x = divmod(inner, 28)
            px, py = heading * 240 + (x + .5) * 240 / 28, line * 265 + (y + .5) * 240 / 28
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline='black', width=5)
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=color, width=3)
            tx = min(heading * 240 + 187, max(heading * 240 + 2, px + 9))
            ty = min(line * 265 + 225, max(line * 265 + 2, py - 8))
            draw.rectangle((tx, ty, tx + 52, ty + 13), fill='black')
            draw.text((tx + 2, ty), f'f{f}', fill=color)
        split = 'CONFIRMATION' if pid in confirm else 'DISCOVERY'
        draw.text((5, line * 265 + 244), f'{rows[pid]["city"]} | {rows[pid]["pano"]} | {split}', fill='#243744')
    canvas.save(target, quality=90)


def render(source, report, audit, motifs, crops, pano_rows, presence, patches, peaks, thresholds, discovery, confirmation):
    report.mkdir(exist_ok=True, parents=True)
    examples = defaultdict(list)
    grouped = defaultdict(list)
    for crop in crops:
        grouped[crop['feature']].append(crop)
    for f, values in grouped.items():
        wanted = sorted(set(np.linspace(0, len(values) - 1, min(12, len(values)), dtype=int).tolist()))
        examples[f] = [{'rank': values[i]['rank'], 'city': values[i]['city'],
                        'crop': str((source / values[i]['crop']).resolve()),
                        'evidence': str((source / 'features' / f'{f}.html').resolve())} for i in wanted]
    cards = []
    figures = report / 'panoramas'
    figures.mkdir(exist_ok=True)
    ranked = sorted(motifs, key=lambda r: (r['cooccurrence_status'] == 'not_confirmed',
                                         r['spatial_status'] != 'repeatable_peak_relation_candidate',
                                         r['cooccurrence_q'], -r['confirmation_lift']))
    supported = [r for r in ranked if r['cooccurrence_status'] != 'not_confirmed']
    illustrated = supported[:30] if supported else ranked[:12]
    for row in illustrated:
        ids = exemplar_ids(row, presence, peaks, thresholds, discovery, confirmation, pano_rows)
        row['example_pano_indices'] = ids
        target = figures / f'm{row["motif"]}.jpg'
        montage(source, target, ids, pano_rows, patches, row['features'], confirmation)
        links = ' + '.join(f'<a href="#f{f}">f{f}</a> ({html.escape(STATES[audit[f]["status"]])})' for f in row['features'])
        cards.append(f'<article id="m{row["motif"]}"><h3>组合 {row["motif"]}: {links}</h3>'
                     f'<p>留出城市：{row["confirmation_support"]} 个全景 / {row["confirmation_cities"]} 城；'
                     f'lift={row["confirmation_lift"]:.2f}，共现 q={row["cooccurrence_q"]:.4f}。'
                     f'空间：{html.escape(row["spatial_status"])}。语义名称尚未独立核验。</p>'
                     '<p>圆圈仅标记各维度最强响应 patch；不表示物体轮廓或完整激活范围。颜色依次为黄、青、粉。</p>'
                     f'<img loading="lazy" src="panoramas/{target.name}" alt="四方向真实全景与特征峰值位置"></article>')
    payload = [{'feature': r['feature'], 'status': r['status'], 'support': r['positive_panoramas'],
                'coherence_excess': r['coherence_excess'], 'q': r['coherence_q'],
                'examples': examples[r['feature']]} for r in audit]
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAE 维度与景观组合证据</title><style>
body{margin:0;background:#f4f6f8;color:#243344;font:15px system-ui,sans-serif}header,main{max-width:1120px;margin:auto;padding:22px}h1{font-size:27px}p{line-height:1.7}a{color:#176a95}section,article{padding:20px;background:white;border-radius:12px;margin:18px 0}article img{width:100%;height:auto}label,button,select,input{font:inherit}input,button,select{padding:8px;border:1px solid #abb8c5;border-radius:5px;background:white}button{cursor:pointer}.controls{display:flex;gap:8px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}.grid img{width:100%;aspect-ratio:1;object-fit:contain;background:#e8edf1}.muted{color:#607184;font-size:13px}#listing{max-height:280px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;border-bottom:1px solid #eee;padding:7px}tr[data-feature]{cursor:pointer}tr[data-feature]:hover{background:#e7f2fa}#inspector{scroll-margin-top:12px}code{overflow-wrap:anywhere}@media(max-width:600px){header,main{padding:12px}section,article{padding:12px}h1{font-size:22px}}</style>
<header><h1>SAE 维度 → 特征组合 → 空间证据</h1><p>全量保留 4,096 个维度。视觉 embedding 一致性只能筛选候选，不能证明每个维度都有明确语义，也不能把一致纹理自动命名为景观单元。</p><p><a href="REPORT.md">实验报告</a> · <a href="dimension_audit.csv">全部维度</a> · <a href="motifs.csv">全部组合</a> · <a href="EXPERIMENT_DESIGN.md">判据与限制</a></p></header><main>
<section><h2>逐维审查</h2><div class="controls"><input id="feature" type="number" min="0" max="4095" placeholder="维度编号"><button id="open">查看维度</button><select id="status"><option value="">全部状态</option></select></div><p id="count" class="muted"></p><div id="listing"></div></section><section id="inspector"><h2 id="dimension-title">选择一个维度查看实例</h2><p id="dimension-text"></p><div id="examples" class="grid"></div></section><section><h2>组合与四方向实例</h2><p>组合从发现城市挖掘，在另一组城市确认。图例展示确认结果优先的最多 30 个组合；完整结果在 CSV 中。这里的关系来自激活峰值位置，尚不是经过物体识别确认的空间模板。</p></section>
'''
    page += ''.join(cards)
    page += '<script id="data" type="application/json">' + json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c') + '</script>'
    page += '<script>const rows=JSON.parse(document.getElementById("data").textContent),states=' + json.dumps(STATES, ensure_ascii=False) + ''';
const select=document.querySelector('#status');Object.entries(states).forEach(([k,v])=>{const o=document.createElement('option');o.value=k;o.textContent=v;select.append(o)});
function listing(){const subset=rows.filter(r=>!select.value||r.status===select.value);document.querySelector('#count').textContent=`${subset.length} / 4096 个维度；所有语义名称均待独立核验`;const table=document.createElement('table');table.innerHTML='<thead><tr><th>维度</th><th>正激活全景数</th><th>证据状态</th></tr></thead>';const body=document.createElement('tbody');subset.forEach(r=>{const tr=document.createElement('tr');tr.dataset.feature=r.feature;[String(r.feature),String(r.support),states[r.status]].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});tr.onclick=()=>{location.hash='f'+r.feature};body.append(tr)});table.append(body);document.querySelector('#listing').replaceChildren(table)}
function inspect(){const m=location.hash.match(/^#f(\\d+)$/);if(!m)return;const r=rows[Number(m[1])];if(!r)return;document.querySelector('#feature').value=r.feature;document.querySelector('#dimension-title').textContent=`f${r.feature}：${states[r.status]}`;document.querySelector('#dimension-text').textContent=`样本内 ${r.support} 个全景正激活。跨城市视觉一致性差值：${r.coherence_excess===null?'无足够证据':r.coherence_excess.toFixed(3)}；q=${r.q.toFixed(4)}。下方横跨激活排名展示，点击进入原图和完整激活证据。`;const box=document.querySelector('#examples');box.replaceChildren();r.examples.forEach(e=>{const a=document.createElement('a');a.href=e.evidence;a.target='_blank';a.rel='noopener';const img=document.createElement('img');img.src=e.crop;img.loading='lazy';img.alt=`f${r.feature} rank ${e.rank}`;const text=document.createElement('div');text.className='muted';text.textContent=`Top ${e.rank} · ${e.city}`;a.append(img,text);box.append(a)});document.querySelector('#inspector').scrollIntoView({block:'start'})}
select.onchange=listing;document.querySelector('#open').onclick=()=>{const n=Number(document.querySelector('#feature').value);if(Number.isInteger(n)&&n>=0&&n<4096)location.hash='f'+n};window.addEventListener('hashchange',inspect);listing();inspect();</script></main></html>'''
    (report / 'index.html').write_text(page)
    return [r['motif'] for r in illustrated]


def write_design(report):
    (report / 'EXPERIMENT_DESIGN.md').write_text('''# 维度、组合与空间结构的分层实验

## 研究问题

1. 每个 SAE 维度是否有足够证据支持一致的视觉含义？
2. 哪些维度组合跨点位、跨城市重复出现？
3. 这些组合是否还包含超出各维度固有位置偏好的重复空间关系？

## 对“明确语义”的判据

不预设一维等于一个物体。可能对应物体／物体部件、表面纹理、颜色／光照、多个视觉模式、成像异常，或当前样本证据不足。每个维度都保留，缺少激活不能解释为整个训练集的死亡维度。

第一轮使用已有 SigLIP 裁剪 embedding。至少 20 个不同全景实例才做一致性检验；跨城市两两余弦相似度与城市组成匹配的其他维度裁剪比较，同一裁剪不进入参照，每次参照内每个城市的全景不重复。单城市维度退回同城比较并保留地域范围。所有可检验维度一起做 BH 校正。候选要求 q≤0.05、相似度差≥0.03，以及 Top20 描述对后续排名实例的相似度高于参照至少 0.03。

两次初始化的球面二分若均得到相似分组（ARI≥0.7）、小组占比≥0.2、中心余弦<0.8，标为“可能存在多种视觉模式”。这些阈值是本轮操作判据，均不是校准后的语义准确率。Embedding 一致的维度仍需分辨物体、纹理和异常；不一致也可能源于编码器、裁剪或视角问题，不能直接判为无意义。

每维独立核验应由不同于发现阶段的实例支持：用 Top20 提出名称，再用后续排名、不同城市及非最高激活实例核对；允许 mixed / texture / artifact / uncertain。当前所有维度的 semantic_verdict 保持未独立验证，不继承 64 个 decoder group 的名称。VLM 自述 confidence 不作为准确率。最终语义纯度需要独立人工标注子样本，报告一致率和不确定性。

## 组合发现与确认

读取已有地理元数据，在每城 1-km 网格中按固定哈希保留一个全景，避免同一位置附近的密集重复采样。按城市划分约 65% 发现集与 35% 确认集。强激活阈值只在发现集上估计，为每维正激活的第 80 百分位，且需要至少 20 个发现集正激活实例。

Apriori 挖掘二项／三项组合。二项组合要求发现集支持≥20、至少 3 城、城市条件 lift≥1.5，按发现集超额共现得分选最多 300 项；三项从上述二项候选的闭合三角形生成，支持≥12、至少 3 城，选最多 100 项。上限是计算预算，不是预设景观类型数量。确认集模拟保持每城每维边际计数的随机交集，对全部选定假设一起做 BH 校正。确认支持≥10、lift≥1.3、q≤0.05 为组合候选；至少 3 个确认城市标跨城市候选，其余保留区域候选。

三项检验针对完全独立基线，不能据此声称存在超出二阶关联的三阶相互作用。维度视觉审查不用于挑选确认集的组合假设；未命名、混合和可能冗余的特征均明确标记。两个近义维度共同出现可能只是编码冗余，不能直接视为两个不同景观元素。

## 空间结构

复用逐全景逐维最强 patch 的已存位置。关系使用同一方向图中的相对上下／左右（3 个 patch 容差），以及四个 heading 的相邻／相对关系；整体旋转四个 heading 时编码不变。发现集确定最常见关系，确认集检验该关系。参照是在同城、同一组合已出现的全景间，打乱非锚点维度的完整峰值位置；因此天空偏上、道路偏下等固有偏好仍被保留。全部候选一起做空间检验 BH 校正，空间差值≥0.1、q≤0.05 且发现集模板占比≥0.35 才标为重复峰值关系候选。

峰值位置不等于物体范围、相邻实体或真实 3-D 几何。成为景观空间模板还需要干净原图、激活区域及多实例关系核验。四方向图的未知投影和视场角不被当作已知角度几何。

## 范围与限制

本轮复用 29 个已验证图像映射城市的 3,712 个验证全景。它们未用于 SAE 训练，但曾用于其他探索分析；本轮城市确认集是本分析内部的留出集，不等于从未使用的外部测试集。Top100 是这个有限候选集中的排名，不代表 NAS 全量。1-km 稀疏化减少近邻重复，不能消除所有空间自相关；随机交集与空间置换的 p/q 均应视作探索性证据，外部确认需空间块重采样和额外城市／点位。这里只对当前冻结 SAE 做分析，不证明换训练种子后维度仍相同。
''')


def overview(report, audit, motifs):
    counts = Counter(r['status'] for r in audit)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    labels = ['Unobserved', 'Too few examples', 'Coherent candidate', 'Possible multiple modes', 'Inconclusive']
    axes[0].barh(labels, [counts[k] for k in STATES], color=['#98a5b4', '#bac4cf', '#168b83', '#d58935', '#6b7ca3'])
    axes[0].set_title('All 4,096 SAE dimensions')
    eligible = [r for r in audit if r['coherence_excess'] is not None]
    axes[1].hist([r['coherence_excess'] for r in eligible], bins=35, color='#168b83', alpha=.85)
    axes[1].axvline(.03, color='#be5e35', linestyle='--')
    axes[1].set_title('Crop embedding agreement above control')
    axes[1].set_xlabel('Cosine agreement difference')
    axes[1].set_ylabel('Dimensions')
    supported = [r for r in motifs if r['cooccurrence_status'] != 'not_confirmed']
    spatial = [r for r in supported if r['spatial_status'] == 'repeatable_peak_relation_candidate']
    axes[2].bar(['Tested\ncombinations', 'Composition\ncandidates', 'Peak relation\ncandidates'],
                [len(motifs), len(supported), len(spatial)], color=['#6b7ca3', '#168b83', '#d58935'])
    axes[2].set_title('Confirmation on held-out cities')
    fig.suptitle('Evidence screening: visual agreement is not verified semantic purity', fontsize=13)
    fig.tight_layout()
    fig.savefig(report / 'Fig_Dimension_and_Motif_Evidence.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--experiment', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--stage', choices=['all', 'audit', 'motifs', 'render'], default='all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audit-repeats', type=int, default=99)
    parser.add_argument('--motif-repeats', type=int, default=1999)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-pairs', type=int, default=300)
    parser.add_argument('--max-triples', type=int, default=100)
    parser.add_argument('--cell-meters', type=int, default=1000)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    write_design(args.report)
    begin = time.monotonic()
    rows, peaks, run = load_peaks(args.source)
    patches = read_patches(args.source, rows, run)
    crops, embeddings = read_crops(args.source)
    identity = {'version': 1, 'checkpoint_sha256': run['checkpoint_sha256'],
                'crop_manifest_sha256': digest(args.source / 'crops.jsonl'),
                'panorama_manifest_sha256': digest(args.source / 'panoramas.jsonl'),
                'seed': args.seed, 'audit_repeats': args.audit_repeats,
                'motif_repeats': args.motif_repeats, 'cell_meters': args.cell_meters,
                'max_pairs': args.max_pairs, 'max_triples': args.max_triples,
                'code_sha256': digest(Path(__file__))}
    old = args.report / 'identity.json'
    if old.exists() and json.loads(old.read_text()) != identity:
        raise ValueError('Analysis identity changed; use a new report directory')
    write_json(old, identity)
    audit_path = args.report / 'dimension_audit.json'
    if args.stage in ['all', 'audit'] and not audit_path.exists():
        audit = audit_dimensions(args.source, crops, embeddings, args.seed, args.audit_repeats, args.workers)
        write_json(audit_path, audit)
        write_csv(args.report / 'dimension_audit.csv', audit)
    if args.stage == 'audit':
        return
    audit = json.loads(audit_path.read_text())
    keep, coordinates, cells = spatial_units(rows, args.experiment, args.cell_meters, args.seed)
    discovery, confirmation = discovery_split(rows, keep, args.seed)
    presence, thresholds = fit_presence(peaks, discovery)
    cities = np.array([r['city'] for r in rows])
    print('SPATIAL UNITS', len(keep), 'DISCOVERY', len(discovery), 'CONFIRMATION', len(confirmation), flush=True)
    motif_path = args.report / 'motifs.json'
    if args.stage in ['all', 'motifs'] and not motif_path.exists():
        with threadpool_limits(limits=args.workers):
            candidates = mine_candidates(presence, discovery, cities, args.max_pairs, args.max_triples)
        print('CANDIDATE COMBINATIONS', len(candidates), flush=True)
        motifs = test_motifs(candidates, presence, patches, discovery, confirmation, cities,
                             args.motif_repeats, args.seed)
        # Similar crop centroids can flag redundant dimensions; they cannot
        # establish semantic equivalence. Keep them visible in the result.
        means = np.zeros((WIDTH, embeddings.shape[1]), np.float32)
        counts = np.zeros(WIDTH, int)
        discovery_cities = set(cities[discovery])
        for start in range(0, len(crops), 4096):
            ix = np.array([i for i in range(start, min(start + 4096, len(crops)))
                           if crops[i]['city'] in discovery_cities], int)
            if len(ix):
                fs = np.array([crops[i]['feature'] for i in ix])
                np.add.at(means, fs, embeddings[ix])
                np.add.at(counts, fs, 1)
        means = unit_rows(means)
        for row in motifs:
            fs = row['features']
            similarities = [float(means[a] @ means[b]) for pos, a in enumerate(fs) for b in fs[pos + 1:]
                            if counts[a] and counts[b]]
            row['max_discovery_crop_centroid_cosine'] = max(similarities, default=None)
            row['possible_redundant_dimensions'] = any(v >= .9 for v in similarities)
            row['dimension_evidence_states'] = [audit[f]['status'] for f in fs]
            row['semantic_verdict'] = 'composition_not_yet_semantically_verified'
        write_json(motif_path, motifs)
        write_csv(args.report / 'motifs.csv', motifs)
        split_rows = [{'pano_index': i, 'city': rows[i]['city'], 'pano': rows[i]['pano'],
                       'split': 'discovery' if i in set(discovery.tolist()) else 'confirmation',
                       'lat': float(coordinates[i, 0]), 'lon': float(coordinates[i, 1]),
                       'grid_x': cells[i][0], 'grid_y': cells[i][1]} for i in keep]
        write_csv(args.report / 'analysis_sample.csv', split_rows)
        np.save(args.report / 'presence_thresholds.npy', thresholds)
    motifs = json.loads(motif_path.read_text())
    illustrated = render(args.source, args.report, audit, motifs, crops, rows, presence, patches,
                         peaks, thresholds, discovery, confirmation)
    overview(args.report, audit, motifs)
    states = Counter(r['status'] for r in audit)
    co_states = Counter(r['cooccurrence_status'] for r in motifs)
    sp_states = Counter(r['spatial_status'] for r in motifs)
    summary = {**identity, 'source_panoramas': len(rows), 'source_cities': len(run['cities']),
               'dimensions': WIDTH, 'dimension_status_counts': dict(states),
               'independently_verified_semantic_dimensions': 0,
               'crop_observations': len(crops), 'spatially_thinned_panoramas': len(keep),
               'discovery_panoramas': len(discovery), 'confirmation_panoramas': len(confirmation),
               'discovery_cities': sorted(set(cities[discovery])),
               'confirmation_cities': sorted(set(cities[confirmation])),
               'motifs_tested': len(motifs), 'cooccurrence_status_counts': dict(co_states),
               'spatial_status_counts': dict(sp_states), 'illustrated_motifs': illustrated,
               'scope': 'Exploratory frozen-SAE evidence; no verified semantic purity or 3-D scene templates',
               'elapsed_this_stage_seconds': time.monotonic() - begin}
    write_json(args.report / 'summary.json', summary)
    report = '# SAE 维度是否构成明确语义单元？\n\n'
    report += f'本轮使用 {len(rows):,} 个验证全景、{len(run["cities"])} 城、{len(crops):,} 份已缓存激活区域 embedding。没有重新提取 DINO 特征、训练 SAE 或占用 GPU。\n\n'
    report += '## 逐维证据\n\n所有 4,096 维都保留。自动筛查结果：\n\n'
    report += '\n'.join(f'- {STATES[k]}：{states[k]:,} 维。' for k in STATES)
    report += '\n\n**视觉一致性不等于明确语义。** 本轮尚未对任何维度完成独立语义标注验证；状态不代表物体准确率或语义纯度。纹理、光照和异常也可能一致。未激活仅针对本样本，不能判为整个训练集的死亡维度。\n\n'
    report += '## 组合与空间证据\n\n'
    report += f'每城 1-km 网格保留一个点后，共 {len(keep):,} 个全景。发现集为 {len(discovery):,} 点／{len(set(cities[discovery]))} 城；确认集为 {len(confirmation):,} 点／{len(set(cities[confirmation]))} 城，两者城市不重叠。\n\n'
    report += f'在发现集选出 {len(motifs)} 个二项／三项组合，确认集有 {sum(v for k, v in co_states.items() if k != "not_confirmed")} 个达到组合候选判据；其中 {sp_states["repeatable_peak_relation_candidate"]} 个还达到重复峰值关系候选判据。完整结果包括未确认的候选。\n\n'
    report += '这些是统计与激活位置证据，尚未完成景观语义核验。可能冗余的维度已单独标记；三项共现不等于三阶相互作用。1-km 稀疏化未消除全部空间相关性，p/q 为探索性证据。\n\n'
    report += '[交互实例](index.html) · [全部维度](dimension_audit.csv) · [全部组合](motifs.csv) · [分析样本](analysis_sample.csv) · [完整设计和限制](EXPERIMENT_DESIGN.md)\n\n'
    report += '![Evidence overview](Fig_Dimension_and_Motif_Evidence.png)\n'
    (args.report / 'REPORT.md').write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
