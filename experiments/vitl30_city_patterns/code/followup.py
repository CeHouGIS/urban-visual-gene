#!/usr/bin/env python3
"""Wait for all extraction shards, then produce auditable city-cell summaries."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .extract import DEFAULT_OUT, DEFAULT_SAMPLE, GROUPS, WIDTH, write_json

VERSION = 1
FOLLOWUP = DEFAULT_OUT.parent / 'followup'

NEXT_TASKS = [
    {'id': 'candidate_combinations', 'label': 'Repeated element combinations',
     'depends_on': ['city_cell_screening'],
     'criterion': 'Screen recurring element combinations across panoramas, cells and cities using reviewed labels.'},
    {'id': 'road_sides', 'label': 'Same-side and opposite-side relations',
     'depends_on': ['candidate_combinations'],
     'criterion': 'Recompute patch masks, verify the road axis, and separate same-side, opposite-side and undetermined examples.'},
    {'id': 'prototype_city_distribution', 'label': 'Prototype city distributions',
     'depends_on': ['road_sides'],
     'criterion': 'Apply validated prototype detectors to the sampled corpus and publish cell-weighted city prevalence.'},
    {'id': 'chapter5_figures', 'label': 'Chapter 5 figure refresh',
     'depends_on': ['prototype_city_distribution'],
     'criterion': 'Regenerate every affected Chapter 5 figure, verify captions and compile the thesis.'},
]


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def ready(extraction, selection, expected_cities=30):
    """Only final complete statuses unlock the expensive validation pass."""
    run_file = extraction / 'run.json'
    manifest_file = selection / 'manifest.json'
    if not run_file.is_file() or not manifest_file.is_file():
        return False, 'selection or extraction run identity missing'
    run = json.loads(run_file.read_text())
    manifest = json.loads(manifest_file.read_text())
    if run.get('version') != 3 or len(run.get('cities', [])) != expected_cities:
        raise ValueError('Not the expected 30-city extraction')
    if run.get('cell_meters') != 1000 or run.get('max_panoramas_per_cell') != 10:
        raise ValueError('Not the 1-km, ten-panorama selection')
    selected = sum(item['selected_panoramas'] for item in manifest['cities'])
    if selected != sum(item['selection_rows'] for item in run['cities']):
        raise ValueError('Extraction city counts differ from the selection manifest')
    reports = [extraction / f'progress_shard{shard}.json' for shard in range(4)]
    if not all(path.is_file() for path in reports):
        return False, 'waiting for four shard progress files'
    states = [json.loads(path.read_text()) for path in reports]
    if any(item.get('shards') != 4 or item.get('shard') != shard
           for shard, item in enumerate(states)):
        raise ValueError('Shard progress identity mismatch')
    if not all(item.get('status') == 'complete' for item in states):
        done = sum(item.get('candidates', 0) for item in states)
        return False, f'extracting {done:,} / {selected:,} candidate panoramas'
    if sum(item['candidates'] for item in states) != selected:
        raise ValueError('Completed shards did not cover all selected panoramas')
    return True, 'four shards complete'


def validate_chunks(extraction, selection):
    run = json.loads((extraction / 'run.json').read_text())
    manifest = json.loads((selection / 'manifest.json').read_text())
    records = {record['city']: record for record in manifest['cities']}
    if set(records) != {item['city'] for item in run['cities']}:
        raise ValueError('City sets do not match')
    chunk_rows = int(run['rows_per_chunk'])
    if chunk_rows < 1:
        raise ValueError('Invalid chunk size')
    index = []
    candidates = processed = 0
    rejection = {'missing': 0, 'black': 0, 'tunnel': 0, 'unreadable': 0}
    for city_record in run['cities']:
        city = city_record['city']
        sample = records[city]
        if city_record['selection_rows'] != sample['selected_panoramas']:
            raise ValueError(f'Selection count changed: {city}')
        selected_file = selection / city / 'selected.parquet'
        selected_hash = sha256(selected_file)
        if selected_hash != sample['selected_sha256'] or selected_hash != city_record['selection_sha256']:
            raise ValueError(f'Selection file changed: {city}')
        count = sample['selected_panoramas']
        for chunk in range(math.ceil(count / chunk_rows)):
            folder = extraction / city.replace('/', '__') / f'{chunk:06d}'
            marker_file = folder / 'complete.json'
            archive = folder / 'vectors.npz'
            if not marker_file.is_file() or not archive.is_file():
                raise ValueError(f'Missing completed chunk: {folder}')
            marker = json.loads(marker_file.read_text())
            expected_rows = min(chunk_rows, count - chunk * chunk_rows)
            if (marker.get('identity') != run['identity'] or marker.get('chunk') != chunk
                    or marker.get('selection_rows') != expected_rows
                    or marker.get('candidate_panoramas') != expected_rows
                    or marker.get('archive_bytes') != archive.stat().st_size):
                raise ValueError(f'Invalid completed chunk: {folder}')
            skipped = marker.get('skipped', {})
            if set(skipped) != set(rejection) or any(not isinstance(v, int) or v < 0 for v in skipped.values()):
                raise ValueError(f'Invalid skip accounting: {folder}')
            if marker['processed_panoramas'] + sum(skipped.values()) != expected_rows:
                raise ValueError(f'Inconsistent panorama accounting: {folder}')
            index.append((city, archive, marker['processed_panoramas']))
            candidates += expected_rows
            processed += marker['processed_panoramas']
            for reason, value in skipped.items():
                rejection[reason] += value
    if candidates != sum(item['selected_panoramas'] for item in manifest['cities']):
        raise ValueError('Chunk count does not cover the selection')
    return run, manifest, index, {'candidates': candidates, 'processed': processed, 'skipped': rejection}


def group_shares(heading_groups):
    values = np.asarray(heading_groups, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, GROUPS):
        raise ValueError('Expected four headings and 64 groups')
    mass = values.sum(axis=1, dtype=np.float64)
    denominator = mass.sum(axis=1)
    shares = np.divide(mass, denominator[:, None], out=np.zeros_like(mass),
                       where=denominator[:, None] > 0)
    return shares, denominator > 0


def queue_snapshot(extraction, selection, phase, detail):
    manifest = json.loads((selection / 'manifest.json').read_text())
    run = json.loads((extraction / 'run.json').read_text())
    total = sum(item['selected_panoramas'] for item in manifest['cities'])
    reports = [json.loads(path.read_text()) for path in sorted(extraction.glob('progress_shard*.json'))]
    processed = sum(item.get('candidates', 0) for item in reports)
    extraction_state = 'complete' if phase in {'analyzing', 'complete'} else 'running'
    tasks = [
        {'id': 'extract', 'label': 'Four-GPU SAE extraction', 'status': extraction_state,
         'depends_on': [], 'execution': 'automatic'},
        {'id': 'validate_chunks', 'label': 'Verify all selected chunks and panoid/grid identity',
         'status': 'complete' if phase == 'complete' else 'running' if phase == 'analyzing' else 'waiting',
         'depends_on': ['extract'], 'execution': 'automatic'},
        {'id': 'city_cell_screening', 'label': 'Balanced city and 1-km cell summary',
         'status': 'complete' if phase == 'complete' else 'running' if phase == 'analyzing' else 'waiting',
         'depends_on': ['validate_chunks'], 'execution': 'automatic'},
    ]
    for item in NEXT_TASKS:
        tasks.append({**item, 'status': 'pending' if phase == 'complete' else 'blocked',
                      'execution': 'requires new analysis and review'})
    if phase == 'error':
        tasks[1]['status'] = 'error'
    return {'version': 1, 'updated_utc': datetime.now(timezone.utc).isoformat(),
            'run_identity': run['identity'], 'phase': phase, 'detail': detail,
            'selected_panoramas': total, 'extraction_candidates_done': processed,
            'tasks': tasks}


def write_queue(extraction, selection, output, phase, detail):
    write_json(output / 'task_queue.json', queue_snapshot(extraction, selection, phase, detail))


def scan_city(city, archives, expected_processed, selection_file):
    selected = pq.read_table(selection_file, columns=['panoid', 'grid_x', 'grid_y']).to_pydict()
    selected_cells = {panoid: (int(x), int(y)) for panoid, x, y in zip(
        selected['panoid'], selected['grid_x'], selected['grid_y'])}
    if len(selected_cells) != len(selected['panoid']):
        raise ValueError(f'Duplicate selected panoid: {city}')
    cells = {}
    total = 0
    zero_mass = 0
    for archive, expected_rows in archives:
        with np.load(archive, allow_pickle=False) as data:
            panoids = data['panoid']
            coords = np.column_stack((data['grid_x'], data['grid_y']))
            vectors = data['composition']
            heading = data['heading_groups']
            if (len(panoids) != expected_rows or coords.shape != (expected_rows, 2)
                    or vectors.shape != (expected_rows, WIDTH)
                    or heading.shape != (expected_rows, 4, GROUPS)
                    or len(set(panoids.tolist())) != expected_rows):
                raise ValueError(f'Invalid archive arrays: {archive}')
            if not np.isfinite(vectors).all() or not np.isfinite(heading).all():
                raise ValueError(f'Non-finite SAE output: {archive}')
            if any(selected_cells.get(str(panoid)) != (int(x), int(y))
                   for panoid, (x, y) in zip(panoids, coords)):
                raise ValueError(f'Panoid/grid identity mismatch: {archive}')
            shares, active = group_shares(heading)
            zero_mass += int((~active).sum())
            for (grid_x, grid_y), share, good in zip(coords, shares, active):
                key = (int(grid_x), int(grid_y))
                if key not in cells:
                    cells[key] = [np.zeros(GROUPS, dtype=np.float64), 0, 0]
                record = cells[key]
                record[1] += 1
                if good:
                    record[0] += share
                    record[2] += 1
            total += expected_rows
    if total != expected_processed:
        raise ValueError(f'City count mismatch: {city}')
    return cells, zero_mass


def render_preview(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matrix = np.asarray([[row[f'group_{i:02d}'] for i in range(GROUPS)]
                         for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
    image = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_yticks(range(len(rows)), [row['city'].split('/')[-1] for row in rows], fontsize=8)
    ax.set_xticks(range(0, GROUPS, 4))
    ax.set_xlabel('SAE decoder group')
    ax.set_title('30-city screening: mean group activation share across 1-km cells')
    fig.colorbar(image, ax=ax, label='Mean within-panorama activation mass share', shrink=.75)
    fig.savefig(output / 'city_group_mass_preview.png', dpi=170)
    plt.close(fig)


def analyze(extraction, selection, output, expected_cities=30):
    is_ready, reason = ready(extraction, selection, expected_cities)
    if not is_ready:
        raise RuntimeError(reason)
    run, manifest, index, totals = validate_chunks(extraction, selection)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'version': VERSION, 'extract_run_identity': run['identity'],
                'selection_sha256': sha256(selection / 'manifest.json'),
                'unit': 'equal weight to each observed 1-km cell within a city',
                'quantity': 'positive 64-group activation mass share, not dominant-patch prevalence'}
    complete_file = output / 'complete.json'
    if complete_file.exists():
        saved = json.loads(complete_file.read_text())
        if saved['identity'] != identity:
            raise ValueError('Existing follow-up output has another run identity')
        return saved
    groups = {}
    for city, archive, count in index:
        groups.setdefault(city, []).append((archive, count))
    records = {item['city']: item for item in manifest['cities']}
    cell_file = output / 'cells.partial.parquet'
    writer = None
    city_rows = []
    try:
        for city in (item['city'] for item in run['cities']):
            cells, zero_mass = scan_city(city, groups[city], sum(n for _, n in groups[city]),
                                        selection / city / 'selected.parquet')
            epsg = records[city]['epsg']
            keys = sorted(cells)
            valid = [key for key in keys if cells[key][2] > 0]
            if not valid:
                raise ValueError(f'No positive group activation cells: {city}')
            means = np.stack([cells[key][0] / cells[key][2] for key in valid])
            city_mean = means.mean(axis=0)
            city_row = {'city': city, 'utm_epsg': epsg,
                        'selected_panoramas': records[city]['selected_panoramas'],
                        'processed_panoramas': sum(cells[key][1] for key in keys),
                        'occupied_cells': len(keys), 'valid_cells': len(valid),
                        'zero_mass_panoramas': zero_mass}
            city_row.update({f'group_{i:02d}': float(city_mean[i]) for i in range(GROUPS)})
            city_rows.append(city_row)
            columns = {'city': pa.array([city] * len(keys)),
                       'utm_epsg': pa.array([epsg] * len(keys), type=pa.int32()),
                       'grid_x': pa.array([key[0] for key in keys], type=pa.int32()),
                       'grid_y': pa.array([key[1] for key in keys], type=pa.int32()),
                       'panoramas': pa.array([cells[key][1] for key in keys], type=pa.int32()),
                       'positive_mass_panoramas': pa.array([cells[key][2] for key in keys], type=pa.int32())}
            for i in range(GROUPS):
                columns[f'group_{i:02d}'] = pa.array([
                    float(cells[key][0][i] / cells[key][2]) if cells[key][2] else None
                    for key in keys], type=pa.float32())
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(cell_file, table.schema, compression='zstd')
            writer.write_table(table)
            print(json.dumps({'city': city, 'processed': city_row['processed_panoramas'],
                              'cells': len(keys)}, ensure_ascii=False), flush=True)
    finally:
        if writer is not None:
            writer.close()
    if sum(row['processed_panoramas'] for row in city_rows) != totals['processed']:
        raise ValueError('City totals disagree with extraction markers')
    cell_file.replace(output / 'cells.parquet')
    fields = list(city_rows[0])
    city_partial = output / 'cities.partial.csv'
    with city_partial.open('w', newline='') as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(city_rows)
    city_partial.replace(output / 'cities.csv')
    render_preview(city_rows, output)
    tasks = [{**task, 'status': 'pending'} for task in NEXT_TASKS]
    write_json(output / 'next_tasks.json', {'identity': identity, 'tasks': tasks})
    result = {'identity': identity, 'cities': len(city_rows),
              'occupied_cells': sum(row['occupied_cells'] for row in city_rows),
              'valid_cells': sum(row['valid_cells'] for row in city_rows),
              'zero_mass_panoramas': sum(row['zero_mass_panoramas'] for row in city_rows),
              **totals, 'products': ['cells.parquet', 'cities.csv', 'city_group_mass_preview.png',
                                    'next_tasks.json'],
              'interpretation': 'descriptive group-mass screening; no roadside or semantic prototype claim'}
    write_json(complete_file, result)
    return result


def watch(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            is_ready, reason = ready(args.extraction, args.selection)
            if is_ready:
                write_json(output / 'status.json', {'status': 'analyzing', 'detail': reason})
                write_queue(args.extraction, args.selection, output, 'analyzing', reason)
                result = analyze(args.extraction, args.selection, output)
                write_json(output / 'status.json', {'status': 'complete',
                                                    'detail': 'City-cell screening complete; spatial prototype and figures pending',
                                                    'result': result})
                write_queue(args.extraction, args.selection, output, 'complete',
                            'City-cell screening complete; spatial prototype and figures pending')
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return
            write_json(output / 'status.json', {'status': 'waiting', 'detail': reason})
            write_queue(args.extraction, args.selection, output, 'waiting', reason)
            print(reason, flush=True)
        except Exception as exc:
            write_json(output / 'status.json', {'status': 'error', 'detail': str(exc)})
            write_queue(args.extraction, args.selection, output, 'error', str(exc))
            raise
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument('--output', type=Path, default=FOLLOWUP)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('--interval must be at least 10 seconds')
    watch(args)
