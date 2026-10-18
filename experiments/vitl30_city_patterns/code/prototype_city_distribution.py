"""Full-corpus, cell-balanced distributions of provisional co-presence candidates.

Roadside hypotheses from a purposive image audit are never extrapolated to
the corpus. Their missing prevalence is stored as null, not zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .extract import DEFAULT_OUT, write_json
from .followup import FOLLOWUP, sha256


def cell_statistics(frame, a, b):
    data = frame[['city', 'grid_x', 'grid_y']].copy()
    data['positive'] = (frame[f'present_{a}'] & frame[f'present_{b}']).astype(int)
    cells = data.groupby(['city', 'grid_x', 'grid_y'], sort=True).positive.agg(['sum', 'size']).reset_index()
    cells = cells.rename(columns={'sum': 'positive_panoramas', 'size': 'sampled_panoramas'})
    cells['prevalence'] = cells.positive_panoramas / cells.sampled_panoramas
    return cells


def spatial_interval(cells, seed=20261018, draws=999):
    """Resample 5-km blocks; ratio gives equal weight to observed 1-km cells."""
    blocks = cells.assign(block_x=cells.grid_x // 5, block_y=cells.grid_y // 5).groupby(
        ['block_x', 'block_y']).prevalence.agg(['sum', 'size'])
    if len(blocks) < 2:
        return float('nan'), float('nan'), len(blocks)
    rng = np.random.default_rng(seed)
    sums, sizes = blocks['sum'].to_numpy(), blocks['size'].to_numpy()
    rates = []
    for _ in range(draws):
        idx = rng.integers(0, len(blocks), size=len(blocks))
        rates.append(sums[idx].sum() / sizes[idx].sum())
    lo, hi = np.quantile(rates, [.025, .975])
    return float(lo), float(hi), len(blocks)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.followup / 'stages/candidate_combinations'
    candidate_protocol = json.loads((source / 'protocol.json').read_text())
    if candidate_protocol['run_identity'] != args.run_identity:
        raise ValueError('Extraction identity mismatch')
    candidates = pd.read_csv(source / 'candidates.csv')
    candidates = candidates[candidates.status == 'co_presence_candidate']
    frame = pd.read_parquet(source / 'panorama_elements.parquet')
    expected = json.loads((args.followup / 'complete.json').read_text())
    if len(frame) != expected['processed'] or frame.city.nunique() != 30:
        raise ValueError('Distribution must cover all usable panoramas and all cities')
    if frame.duplicated(['city', 'panoid']).any():
        raise ValueError('Duplicate panorama')
    cities, cell_tables = [], []
    for row in candidates.itertuples(index=False):
        cells = cell_statistics(frame, row.element_a, row.element_b)
        cells['prototype_id'] = row.candidate
        cells['detection_kind'] = 'provisional_co_presence'
        # Missing geometric coverage stays missing in machine-readable outputs.
        cells['same_side_prevalence'] = np.nan
        cells['opposite_side_prevalence'] = np.nan
        cell_tables.append(cells)
        for city, group in cells.groupby('city', sort=True):
            seed = int(hashlib.sha256(f'{city}/{row.candidate}'.encode()).hexdigest()[:8], 16)
            lo, hi, blocks = spatial_interval(group, seed, args.bootstrap)
            cities.append({'city': city, 'prototype_id': row.candidate,
                'detection_kind': 'provisional_co_presence',
                'cities_split': 'confirmation' if city in candidate_protocol['confirmation_cities'] else 'discovery',
                'panoramas': int(group.sampled_panoramas.sum()), 'cells': len(group), 'blocks_5km': blocks,
                'positive_panoramas': int(group.positive_panoramas.sum()),
                'positive_cells': int((group.positive_panoramas > 0).sum()),
                'cell_weighted_prevalence': float(group.prevalence.mean()),
                'bootstrap_low': lo, 'bootstrap_high': hi,
                'same_side_prevalence': np.nan, 'opposite_side_prevalence': np.nan,
                'geometry_status': 'not_measured_corpus_wide', 'human_validated': False})
        print(f'Full-corpus distribution: {row.candidate}; {len(cells):,} cells', flush=True)
    city_table = pd.DataFrame(cities)
    cell_table = pd.concat(cell_tables, ignore_index=True)
    city_table.to_csv(args.output / 'city_prototype_distribution.csv', index=False)
    cell_table.to_parquet(args.output / 'cell_prototype_distribution.parquet', index=False)
    summary = {'cities': 30, 'panoramas': len(frame), 'cells': expected['valid_cells'],
               'candidate_count': len(candidates), 'validated_prototypes': 0,
               'distribution_rows': len(city_table), 'cell_rows': len(cell_table),
               'detection_kind': 'provisional_co_presence', 'roadside_prevalence_available': False,
               'status': 'descriptive_distribution_complete_validation_pending'}
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'protocol.json', {
        'schema_version': 1, 'run_identity': args.run_identity,
        'candidate_protocol_sha256': sha256(source / 'protocol.json'),
        'candidates_sha256': sha256(source / 'candidates.csv'),
        'detector': 'both provisional family mass shares >=0.02 of all 4096 activation mass',
        'detector_frozen_before_distribution': True, 'unit': 'complete four-direction panorama',
        'city_weighting': 'mean of within-1km-cell positive fractions; each observed cell equal weight',
        'uncertainty': '95% percentile bootstrap over city-specific 5x5km blocks, ratio of resampled cell-rate sums to cell counts',
        'bootstrap_draws': args.bootstrap,
        'null_geometry': 'same/opposite prevalence missing; unmeasured does not mean absent',
        'limitations': ['Describes selected candidate co-presence, not validated landscape prototypes.',
            'Inference is conditional on observed cells and the sampled, quality-screened corpus.',
            'Confidence intervals do not account for semantic mislabelling or candidate selection.',
            'Road audit is purposive and cannot support roadside prevalence.',
            'Acquisition density and city boundaries differ; rates are not city land-area estimates.']})
    write_json(args.output / 'complete.json', {'schema_version': 1,
        'stage': 'prototype_city_distribution', 'run_identity': args.run_identity,
        'status': 'complete', 'evidence': summary})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    p.add_argument('--followup', type=Path, default=FOLLOWUP)
    p.add_argument('--output', type=Path, default=FOLLOWUP / 'stages/prototype_city_distribution')
    p.add_argument('--run-identity', required=True)
    p.add_argument('--bootstrap', type=int, default=999)
    args = p.parse_args()
    if args.bootstrap < 99:
        p.error('At least 99 bootstrap draws required')
    run(args)
