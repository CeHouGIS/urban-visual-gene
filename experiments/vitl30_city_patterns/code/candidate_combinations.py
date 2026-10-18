"""Local co-presence screen on frozen SAE means; no API or roadside inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .extract import DEFAULT_OUT, write_json
from .followup import FOLLOWUP, sha256

VERSION = 2
HIERARCHY = (Path(__file__).resolve().parents[3] /
             'experiments/vitl30_relational_prototypes/results/hierarchy.json')
ELEMENTS = ('building', 'tree', 'low_vegetation', 'road', 'pedestrian',
            'barrier', 'vehicle', 'person', 'water', 'terrain',
            'transport_structure', 'utility', 'sign', 'furniture', 'construction')


def stable_city(city):
    return int(hashlib.sha256(city.encode()).hexdigest()[:8], 16)


def bh(values):
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    best = 1.
    for rank in range(len(values) - 1, -1, -1):
        index = order[rank]
        best = min(best, values[index] * len(values) / (rank + 1))
        result[index] = best
    return result


def family_dimensions(hierarchy):
    families = {r['id'] for r in hierarchy['families'] if r.get('counts_as_element')}
    dimensions = {family: [] for family in ELEMENTS}
    seen = set()
    for row in hierarchy['dimensions']:
        if (row.get('family') in families and row.get('family') in dimensions
                and row.get('counts_as_element') and row.get('provisional_graph_candidate')):
            feature = int(row['feature'])
            if feature in seen or not 0 <= feature < 4096:
                raise ValueError('Duplicate or invalid hierarchy feature')
            seen.add(feature)
            dimensions[row['family']].append(feature)
    return {k: np.asarray(v, dtype=np.int32) for k, v in dimensions.items()}


def family_shares(composition, dimensions):
    """All 4096 dimensions contribute to the denominator, even unnamed ones."""
    composition = np.asarray(composition, dtype=np.float32)
    if (composition.ndim != 2 or composition.shape[1] != 4096
            or not np.isfinite(composition).all() or (composition < 0).any()):
        raise ValueError('Invalid nonnegative 4096-dimensional SAE means')
    total = composition.sum(axis=1, dtype=np.float64)
    scores = np.stack([composition[:, dimensions[f]].sum(axis=1, dtype=np.float64)
                       for f in ELEMENTS], axis=1)
    return np.divide(scores, total[:, None], out=np.zeros_like(scores),
                     where=total[:, None] > 0).astype(np.float32)


def load_city(city, args, dimensions, identity):
    cache = args.output / 'cache' / f"{city.replace('/', '__')}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved['identity']) != identity:
                raise ValueError(f'Cache identity mismatch: {cache}')
            data = {k: saved[k] for k in ('panoid', 'grid_x', 'grid_y', 'shares')}
    else:
        collected = {k: [] for k in ('panoid', 'grid_x', 'grid_y', 'shares')}
        archives = sorted((args.extraction / city.replace('/', '__')).glob('*/vectors.npz'))
        if not archives:
            raise ValueError(f'No completed archives for {city}')
        for archive in archives:
            marker = json.loads(archive.with_name('complete.json').read_text())
            if marker['identity'] != args.run_identity or marker['archive_bytes'] != archive.stat().st_size:
                raise ValueError(f'Archive identity mismatch: {archive}')
            with np.load(archive, allow_pickle=False) as saved:
                if len(saved['panoid']) != marker['processed_panoramas']:
                    raise ValueError(f'Archive count mismatch: {archive}')
                for k in ('panoid', 'grid_x', 'grid_y'):
                    collected[k].append(saved[k])
                collected['shares'].append(family_shares(saved['composition'], dimensions))
        data = {k: np.concatenate(v) for k, v in collected.items()}
        if len(np.unique(data['panoid'])) != len(data['panoid']):
            raise ValueError(f'Duplicate panorama in {city}')
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.partial.npz')
        np.savez_compressed(temporary, identity=identity, **data)
        temporary.replace(cache)
    data['present'] = data['shares'] >= args.threshold
    coords = np.column_stack((data['grid_x'], data['grid_y']))
    for name, scale in [('cell_index', 1), ('block_index', 5)]:
        _, data[name] = np.unique(coords // scale, axis=0, return_inverse=True)
    return data


def support(events, cities, pair):
    counts = dict(panoramas=0, cells=0, blocks=0, cities=0)
    for city in cities:
        data = events[city]
        mask = data['present'][:, pair[0]] & data['present'][:, pair[1]]
        counts['panoramas'] += int(mask.sum())
        counts['cells'] += len(np.unique(data['cell_index'][mask]))
        counts['blocks'] += len(np.unique(data['block_index'][mask]))
        counts['cities'] += int(mask.any())
    return counts


def strata(events, cities, pair, spatial):
    totals, first, second = [], [], []
    for city in cities:
        data = events[city]
        index = data['block_index'] if spatial else np.zeros(len(data['panoid']), dtype=int)
        totals.extend(np.bincount(index).astype(np.int64))
        first.extend(np.bincount(index, weights=data['present'][:, pair[0]]).astype(np.int64))
        second.extend(np.bincount(index, weights=data['present'][:, pair[1]]).astype(np.int64))
    return tuple(np.asarray(v, dtype=np.int64) for v in (totals, first, second))


def permutation_test(totals, first, second, observed, permutations, rng):
    """Hypergeometric overlap draws equal shuffling one binary column per stratum."""
    active = (totals > 0) & (first > 0) & (second > 0)
    n, a, b = totals[active], first[active], second[active]
    expected = float(np.sum(a * b / n)) if len(n) else 0.
    exceedances = 0
    for start in range(0, permutations, 32):
        size = min(32, permutations - start)
        draws = rng.hypergeometric(a, n - a, b, size=(size, len(n))).sum(axis=1)
        exceedances += int(np.count_nonzero(draws >= observed))
    return (exceedances + 1) / (permutations + 1), expected


def screen(events, permutations=999, seed=20261018, min_pano=30, min_cells=8,
           min_cities=3, min_lift=1.05):
    discovery = sorted(c for c in events if stable_city(c) % 3 != 0)
    confirmation = sorted(set(events) - set(discovery))
    if not discovery or not confirmation:
        raise ValueError('Need both discovery and confirmation cities')
    results = []
    for pair in itertools.combinations(range(len(ELEMENTS)), 2):
        discovered = support(events, discovery, pair)
        confirmed = support(events, confirmation, pair)
        if (discovered['panoramas'] < min_pano or discovered['cells'] < min_cells
                or discovered['cities'] < min_cities):
            continue
        row = {'candidate': '+'.join(ELEMENTS[i] for i in pair),
               'element_a': ELEMENTS[pair[0]], 'element_b': ELEMENTS[pair[1]]}
        for prefix, values in [('discovery', discovered), ('confirmation', confirmed)]:
            row.update({f'{prefix}_{k}': v for k, v in values.items()})
        for name, spatial in [('city', False), ('spatial', True)]:
            rng = np.random.default_rng(seed + pair[0] * 1009 + pair[1] * 11 + spatial)
            pvalue, expected = permutation_test(*strata(events, confirmation, pair, spatial),
                                                confirmed['panoramas'], permutations, rng)
            row[f'p_{name}'] = pvalue
            row[f'null_mean_{name}'] = expected
            row[f'lift_{name}'] = confirmed['panoramas'] / expected if expected > 0 else 0.
        row['p_both'] = max(row['p_city'], row['p_spatial'])
        row['layout_evidence'] = 'unavailable_from_pooled_vectors'
        results.append(row)
        print(f'Tested {row["candidate"]}: p(city)={row["p_city"]:.4g}, p(spatial)={row["p_spatial"]:.4g}', flush=True)
    for row, qvalue in zip(results, bh([r['p_both'] for r in results])):
        row['q_both'] = float(qvalue)
        row['status'] = ('co_presence_candidate' if qvalue <= .05
                         and min(row['lift_city'], row['lift_spatial']) >= min_lift
                         and row['confirmation_panoramas'] >= min_pano
                         and row['confirmation_cells'] >= min_cells
                         and row['confirmation_cities'] >= min_cities else 'not_confirmed')
    return discovery, confirmation, results


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    completed = json.loads((args.followup / 'complete.json').read_text())
    extraction = json.loads((args.extraction / 'run.json').read_text())
    if (extraction['identity'] != args.run_identity
            or completed['identity']['extract_run_identity'] != args.run_identity
            or completed['cities'] != 30 or len(extraction['cities']) != 30):
        raise ValueError('Expected the verified complete 30-city extraction')
    dimensions = family_dimensions(json.loads(args.hierarchy.read_text()))
    identity = hashlib.sha256(json.dumps({'version': VERSION, 'run': args.run_identity,
                              'hierarchy': sha256(args.hierarchy)}, sort_keys=True).encode()).hexdigest()
    events = {}
    for entry in extraction['cities']:
        city = entry['city']
        events[city] = load_city(city, args, dimensions, identity)
        message = f'Local element cache: {len(events)}/30 cities; {city}; {len(events[city]["panoid"]):,} panoramas'
        print(message, flush=True)
        write_json(args.output / 'progress.json', {'status': 'loading', 'detail': message,
                                                  'cities_done': len(events)})
    total = sum(len(data['panoid']) for data in events.values())
    if total != completed['processed']:
        raise ValueError('Panorama count differs from verified screening')
    print(f'Testing {total:,} panoramas with {args.permutations} null draws per test', flush=True)
    write_json(args.output / 'progress.json', {'status': 'testing', 'panoramas': total})
    discovery, confirmation, results = screen(events, args.permutations)
    fields = ['candidate', 'element_a', 'element_b'] + [
        f'{prefix}_{key}' for prefix in ('discovery', 'confirmation')
        for key in ('panoramas', 'cells', 'blocks', 'cities')] + [
        'p_city', 'null_mean_city', 'lift_city', 'p_spatial', 'null_mean_spatial',
        'lift_spatial', 'p_both', 'q_both', 'layout_evidence', 'status']
    with (args.output / 'candidates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    partial = args.output / 'panorama_elements.partial.parquet'
    writer = None
    try:
        for city, data in events.items():
            columns = {'city': [city] * len(data['panoid']), 'panoid': data['panoid'],
                       'grid_x': data['grid_x'], 'grid_y': data['grid_y']}
            for i, family in enumerate(ELEMENTS):
                columns[f'share_{family}'] = data['shares'][:, i]
                columns[f'present_{family}'] = data['present'][:, i]
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    partial.replace(args.output / 'panorama_elements.parquet')
    retained = sum(r['status'] == 'co_presence_candidate' for r in results)
    write_json(args.output / 'protocol.json', {
        'schema_version': VERSION, 'run_identity': args.run_identity,
        'execution': 'direct local Python; no API', 'hierarchy': str(args.hierarchy),
        'hierarchy_sha256': sha256(args.hierarchy), 'cache_identity': identity,
        'family_dimensions': {k: v.tolist() for k, v in dimensions.items()},
        'labels_human_validated': False, 'relative_mass_threshold': args.threshold,
        'score_denominator': 'all 4096 SAE activation dimensions',
        'unit': 'complete four-direction panorama', 'cell_meters': 1000,
        'split_rule': 'SHA256(city) first eight hex modulo 3; nonzero=discovery',
        'discovery_cities': discovery, 'confirmation_cities': confirmation,
        'support_minima_each_split': {'panoramas': 30, '1km_cells': 8, 'cities': 3},
        'minimum_lift_both_nulls': 1.05,
        'null_models': ['within-city binary-column permutation',
                        'within city and 5x5 km block binary-column permutation'],
        'null_sampler': 'exact hypergeometric overlaps equivalent to binary shuffles',
        'permutations': args.permutations, 'seed': 20261018,
        'multiple_testing': 'BH FDR 0.05 on max(p_city,p_spatial) for discovery-qualified pairs',
        'limitations': ['Qwen family labels remain provisional.',
                        'Parts within each family are merged; object counts unavailable.',
                        'Within-block spatial dependence is not eliminated.',
                        'Pooled means do not establish layout, road axes or roadside identity.',
                        'Co-presence candidates are not validated landscape prototypes.']})
    summary = {'schema_version': VERSION, 'run_identity': args.run_identity,
               'cities': 30, 'panoramas': total, 'tested_pairs': len(results),
               'candidate_count': retained, 'layout_verified': False,
               'interpretation': 'Provisional co-presence; patch and roadside verification pending.'}
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'complete.json', {'schema_version': 1,
               'stage': 'candidate_combinations', 'run_identity': args.run_identity,
               'status': 'complete', 'evidence': summary})
    write_json(args.output / 'progress.json', {'status': 'complete', **summary})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--followup', type=Path, default=FOLLOWUP)
    parser.add_argument('--output', type=Path, default=FOLLOWUP / 'stages/candidate_combinations')
    parser.add_argument('--hierarchy', type=Path, default=HIERARCHY)
    parser.add_argument('--run-identity', required=True)
    parser.add_argument('--threshold', type=float, default=.02)
    parser.add_argument('--permutations', type=int, default=999)
    options = parser.parse_args()
    if not 0 < options.threshold <= 1 or options.permutations < 99:
        parser.error('threshold must be in (0,1]; permutations must be >=99')
    run(options)
