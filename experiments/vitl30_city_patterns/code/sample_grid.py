#!/usr/bin/env python3
"""Select at most ten four-view panorama candidates per city and 1-km UTM cell."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

from .extract import DEFAULT_SAMPLE, METADATA, cities, eligible, write_json

CELL_METERS = 1000
PER_CELL = 10
SEED = 42


def utm_epsg(lat, lon):
    if not (-80 <= lat <= 84 and -180 <= lon <= 180):
        raise ValueError(f'Coordinate outside UTM bounds: {lat}, {lon}')
    zone = min(60, max(1, int(math.floor((lon + 180) / 6)) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def cell_xy(transformer, lat, lon):
    east, north = transformer.transform(lon, lat)
    return int(math.floor(east / CELL_METERS)), int(math.floor(north / CELL_METERS))


def score_panoid(panoid, seed=SEED):
    payload = f'{seed}:{panoid}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'big')


def offer(heaps, counts, cell, row, seed=SEED, limit=PER_CELL):
    counts[cell] += 1
    panoid = str(row['panoid'])
    score = score_panoid(panoid, seed)
    item = (-score, panoid, float(row['lat']), float(row['lon']))
    heap = heaps.setdefault(cell, [])
    if any(entry[1] == panoid for entry in heap):
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def read_batches(path, batch_size=65536):
    file = pq.ParquetFile(path)
    columns = file.schema.names
    flag = next((name for name in ('exists', 'download', 'marker') if name in columns), None)
    if flag is None:
        raise ValueError(f'Missing image availability flag: {path}')
    has_four_count = '#images' in columns
    selected = ['panoid', 'lat', 'lon', flag]
    if has_four_count:
        selected.append('#images')
    for batch in file.iter_batches(batch_size=batch_size, columns=selected):
        yield batch.to_pylist(), flag, has_four_count


def city_projection(path):
    for rows, flag, has_four_count in read_batches(path, batch_size=16384):
        coords = [(row['lat'], row['lon']) for row in rows
                  if eligible(row, flag, has_four_count)
                  and row['lat'] is not None and row['lon'] is not None
                  and math.isfinite(row['lat']) and math.isfinite(row['lon'])]
        if coords:
            lat, lon = np.median(np.asarray(coords, dtype=np.float64), axis=0)
            return utm_epsg(float(lat), float(lon))
    raise ValueError(f'No eligible coordinates: {path}')


def sample_city(city, output, seed=SEED, limit=PER_CELL):
    path = METADATA / city / 'panoids.parquet'
    epsg = city_projection(path)
    transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    heaps = {}
    counts = Counter()
    source_rows = eligible_rows = invalid_coords = 0
    for rows, flag, has_four_count in read_batches(path):
        source_rows += len(rows)
        valid = [row for row in rows if eligible(row, flag, has_four_count)]
        eligible_rows += len(valid)
        valid_coords = [row for row in valid if row['lat'] is not None and row['lon'] is not None
                        and math.isfinite(row['lat']) and math.isfinite(row['lon'])
                        and -90 <= row['lat'] <= 90 and -180 <= row['lon'] <= 180]
        invalid_coords += len(valid) - len(valid_coords)
        if not valid_coords:
            continue
        east, north = transformer.transform([row['lon'] for row in valid_coords],
                                            [row['lat'] for row in valid_coords])
        for row, x, y in zip(valid_coords, east, north):
            if math.isfinite(x) and math.isfinite(y):
                cell = (int(math.floor(x / CELL_METERS)), int(math.floor(y / CELL_METERS)))
                offer(heaps, counts, cell, row, seed, limit)
            else:
                invalid_coords += 1
    selected = []
    for (grid_x, grid_y), heap in sorted(heaps.items()):
        for neg_score, panoid, lat, lon in sorted(heap, reverse=True):
            selected.append({'panoid': panoid, 'lat': lat, 'lon': lon,
                             'grid_x': grid_x, 'grid_y': grid_y})
    if len({row['panoid'] for row in selected}) != len(selected):
        raise ValueError(f'Duplicate panoid selected in {city}')
    destination = output / city / 'selected.parquet'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.partial')
    schema = pa.schema([('panoid', pa.string()), ('lat', pa.float64()),
                        ('lon', pa.float64()), ('grid_x', pa.int32()), ('grid_y', pa.int32())])
    pq.write_table(pa.Table.from_pylist(selected, schema=schema), temporary, compression='zstd')
    temporary.replace(destination)
    source_stat = path.stat()
    return {'city': city, 'epsg': epsg, 'source_rows': source_rows,
            'eligible_rows': eligible_rows, 'invalid_coordinates': invalid_coords,
            'grid_cells': len(counts), 'underfilled_cells': sum(n < limit for n in counts.values()),
            'selected_panoramas': len(selected), 'selected_bytes': destination.stat().st_size,
            'selected_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'source_bytes': source_stat.st_size, 'source_mtime_ns': source_stat.st_mtime_ns}


def run(args):
    if args.limit < 1:
        raise ValueError('limit must be positive')
    selection = cities() if args.city is None else [args.city]
    if any(city not in cities() for city in selection):
        raise ValueError('City is not in the main 30-city experiment')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    records = []
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing['seed'] != args.seed or existing['max_panoramas_per_cell'] != args.limit:
            raise ValueError('Selection directory has different parameters')
        records = existing['cities']
    completed = {record['city']: record for record in records}
    for city in selection:
        if city in completed:
            record = completed[city]
            selected_file = args.output / city / 'selected.parquet'
            if not selected_file.is_file() or hashlib.sha256(selected_file.read_bytes()).hexdigest() != record['selected_sha256']:
                raise ValueError(f'Corrupt prior selection: {city}')
            print(json.dumps({'skipped_complete': city, 'selected_panoramas': record['selected_panoramas']}), flush=True)
            continue
        record = sample_city(city, args.output, args.seed, args.limit)
        records.append(record)
        completed[city] = record
        payload = {'version': 1, 'cell_meters': CELL_METERS,
                   'max_panoramas_per_cell': args.limit, 'seed': args.seed,
                   'selection_rule': 'lowest stable BLAKE2b-64 hash per city/UTM grid cell',
                   'cities': records}
        write_json(manifest, payload)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(json.dumps({'cities_complete': len(records),
                      'selected_panoramas': sum(item['selected_panoramas'] for item in records),
                      'grid_cells': sum(item['grid_cells'] for item in records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument('--city', help='Select one city for a pilot')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--limit', type=int, default=PER_CELL)
    run(parser.parse_args())
