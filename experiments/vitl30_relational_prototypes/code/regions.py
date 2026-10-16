"""Build compact, panorama-aligned element grids from the frozen SAE CSR cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.vitl30_multiscale_prototypes.code.data import OUT as SOURCE, write_json
from .hierarchy import EXPERIMENT, read, sha

OUT = EXPERIMENT / 'outputs'
GRID = 28
HEADINGS = (0, 90, 180, 270)


def region_boxes():
    boxes = []
    for size, starts in [(14, (0, 7, 14)), (21, (0, 7)), (28, (0,))]:
        for y in starts:
            for x in starts:
                boxes.append((size, y, x, y + size, x + size))
    return boxes


def pooled_cells(maps, boxes=None):
    """Area means for 4x4 cells of each selected region."""
    boxes = boxes or region_boxes()
    integral = np.pad(maps.astype(np.float32), ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
    y0, x0, y1, x1, area = [], [], [], [], []
    for size, top, left, _, _ in boxes:
        by = np.linspace(0, size, 5, dtype=np.int64) + top
        bx = np.linspace(0, size, 5, dtype=np.int64) + left
        for i in range(4):
            for j in range(4):
                y0.append(by[i]); y1.append(by[i + 1])
                x0.append(bx[j]); x1.append(bx[j + 1])
                area.append((by[i + 1] - by[i]) * (bx[j + 1] - bx[j]))
    values = (integral[:, y1, x1] - integral[:, y0, x1]
              - integral[:, y1, x0] + integral[:, y0, x0]) / np.asarray(area)[None]
    return values.reshape(maps.shape[0], len(boxes), 16).transpose(1, 0, 2).clip(min=0)


def family_map(values, indices, indptr, local, heading, dim_to_family, scale, family_counts):
    start_row = (local * 4 + heading) * (GRID * GRID)
    offsets = indptr[start_row:start_row + GRID * GRID + 1]
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


def build(out=OUT, source=SOURCE):
    out, source = Path(out), Path(source)
    protocol = read(out / 'protocol.json')
    hierarchy = read(EXPERIMENT / 'results/hierarchy.json')
    corpus = read(source / 'corpus.json')
    cache = read(source / 'cache_complete.json')
    manifest = pd.read_parquet(out / 'pilot_panoramas.parquet')
    if (protocol['checkpoint_sha256'] != corpus['checkpoint_sha256'] or
            protocol['hierarchy_identity'] != hierarchy['identity'] or
            protocol['base_protocol_identity'] != read(EXPERIMENT.parent / 'vitl30_prototype_coverage/outputs/protocol.json')['identity'] or
            cache['identity'] != corpus['identity'] or len(manifest) != protocol['panoramas'] or
            any(s not in {'train', 'dev'} for s in manifest.prototype_split)):
        raise ValueError('Pilot, hierarchy, SAE cache, or split identity differs')
    calibration = EXPERIMENT.parent / 'vitl30_prototype_coverage/outputs/positive_rms.npy'
    if sha(calibration) != protocol['training_scale_sha256']:
        raise ValueError('Training-block calibration changed')
    families = list(hierarchy['summary']['graph_candidate_family_counts'])
    lookup = {name: i for i, name in enumerate(families)}
    dim_to_family = np.full(4096, -1, np.int16)
    for row in hierarchy['dimensions']:
        if row['provisional_graph_candidate']:
            dim_to_family[row['feature']] = lookup[row['family']]
    counts = np.bincount(dim_to_family[dim_to_family >= 0], minlength=len(families))
    scale = np.load(calibration)
    boxes = region_boxes()
    identity = {'version': 1, 'pilot_identity': protocol['identity'],
                'hierarchy_identity': hierarchy['identity'], 'corpus_identity': corpus['identity'],
                'cache_identity': cache['identity'], 'scale_sha256': protocol['training_scale_sha256'],
                'code_sha256': sha(Path(__file__)), 'families': families,
                'headings': HEADINGS, 'boxes': boxes,
                'normalization': 'clip each positive SAE value / training RMS at 8; sum per family / sqrt(candidate dimensions)',
                'features': 'four-by-four cell area means, each region kept within one direction; no test activation access'}
    identity['identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = out / 'regions'
    folder.mkdir(parents=True, exist_ok=True)
    previous = read(folder / 'identity.json') if (folder / 'identity.json').exists() else None
    if previous and previous != identity:
        raise ValueError('Region identity changed; use another output directory')
    write_json(folder / 'identity.json', identity)
    target = folder / 'pooled.npy'
    shape = (len(manifest), 4, len(boxes), len(families), 16)
    data = np.lib.format.open_memmap(target, mode='r+' if target.exists() else 'w+',
                                     dtype=np.float16, shape=shape)
    progress = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
    at = int(progress.get('completed_panoramas', 0))
    if at > len(manifest):
        raise ValueError('Invalid region progress')
    if (folder / 'complete.json').exists():
        return read(folder / 'complete.json')
    current_key, arrays = None, None
    for index in range(at, len(manifest)):
        row = manifest.iloc[index]
        chunk_start = int(row.cache_pano_index) // corpus['chunk_panos'] * corpus['chunk_panos']
        key = (row.city, chunk_start)
        if key != current_key:
            chunk = source / 'activations' / row.city.replace('/', '__') / f'{chunk_start:06d}'
            marker = read(chunk / 'complete.json')
            if marker['identity'] != corpus['identity']:
                raise ValueError(f'Sparse chunk identity differs: {chunk}')
            arrays = tuple(np.load(chunk / f'{name}.npy', mmap_mode='r') for name in ('values', 'indices', 'indptr'))
            current_key = key
        local = int(row.cache_pano_index) - chunk_start
        for heading in range(4):
            grid = family_map(*arrays, local, heading, dim_to_family, scale, counts)
            data[index, heading] = pooled_cells(grid, boxes).astype(np.float16)
        if (index + 1) % 32 == 0 or index + 1 == len(manifest):
            data.flush()
            write_json(folder / 'status.json', {'status': 'building', 'completed_panoramas': index + 1,
                       'total_panoramas': len(manifest), 'city': row.city})
            print('REGIONS', index + 1, '/', len(manifest), row.city, flush=True)
    result = {'identity': identity['identity'], 'shape': list(shape), 'dtype': 'float16',
              'panoramas': len(manifest), 'images': len(manifest) * 4,
              'regions': len(manifest) * 4 * len(boxes), 'test_used': False,
              'sha256': sha(target)}
    write_json(folder / 'complete.json', result)
    write_json(folder / 'status.json', {'status': 'complete', **result})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build(args.out), ensure_ascii=False, indent=2))
