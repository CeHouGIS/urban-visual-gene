"""Recompute the full 22-class city figures from saved panorama SAE means."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .extract import DEFAULT_OUT, write_json
from .followup import FOLLOWUP, sha256
from .semantic_city_figures import city_atlas, city_figures, load, pipeline

ROOT = Path(__file__).resolve().parents[3]
HIERARCHY = ROOT / 'experiments/vitl30_city_patterns/results/current_semantic_hierarchy.json'
OUTPUT = ROOT / 'experiments/vitl30_city_patterns/results/figures_22'
STAGE = FOLLOWUP / 'stages/semantic_22_classes'
THESIS = Path('/root/code/HKUST-Thesis/img_gene/vitl30_w4096')

# This is the complete, mutually exclusive dimensional taxonomy, including
# non-landscape and uncertain responses. It is not the combination-mining set.
CLASSES = (
    'unresolved', 'building', 'ground_unspecified', 'tree', 'road', 'sky',
    'vegetation_unspecified', 'vehicle', 'material_unspecified',
    'low_vegetation', 'scene_context', 'sign', 'barrier',
    'transport_structure', 'utility', 'person', 'terrain', 'pedestrian',
    'construction', 'furniture', 'water', 'spatial_encoding',
)
LABELS = {
    'unresolved': 'Unresolved / low evidence',
    'building': 'Buildings and components',
    'ground_unspecified': 'Ground, unspecified',
    'tree': 'Trees and components',
    'road': 'Road and carriageway',
    'sky': 'Sky and clouds',
    'vegetation_unspecified': 'Vegetation, unspecified',
    'vehicle': 'Vehicles and components',
    'material_unspecified': 'Material, object unclear',
    'low_vegetation': 'Low vegetation',
    'scene_context': 'Scene context',
    'sign': 'Signs and traffic control',
    'barrier': 'Barriers and enclosures',
    'transport_structure': 'Transport structures',
    'utility': 'Utility infrastructure',
    'person': 'People and body parts',
    'terrain': 'Natural terrain',
    'pedestrian': 'Pedestrian space',
    'construction': 'Construction equipment',
    'furniture': 'Street furniture',
    'water': 'Water',
    'spatial_encoding': 'Spatial encoding candidates',
}
STEMS = (
    'Fig_ViTL30_Pipeline_Current',
    'Fig_ViTL30_Semantic_City_Composition',
    'Fig_ViTL30_Semantic_City_Excess',
    'Fig_ViTL30_Semantic_City_Similarity',
    'Fig_ViTL30_Semantic_Cooccurrence',
    'Fig_ViTL30_Semantic_Spatial_Structure',
    'Fig_ViTL30_Semantic_Cell_UMAP',
)


def class_indices(hierarchy):
    rows = hierarchy['dimensions']
    if len(rows) != 4096 or set(LABELS) != set(CLASSES):
        raise ValueError('The complete taxonomy must define 4096 dimensions and 22 classes')
    by_id = {int(row['feature']): row['family'] for row in rows}
    if set(by_id) != set(range(4096)) or set(by_id.values()) != set(CLASSES):
        raise ValueError('Dimensions must partition exactly into the 22 classes')
    return [np.asarray([i for i in range(4096) if by_id[i] == key], dtype=np.int32)
            for key in CLASSES]


def score_chunk(composition, indices):
    response = np.asarray(composition, dtype=np.float32)
    if response.ndim != 2 or response.shape[1] != 4096 or not np.isfinite(response).all():
        raise ValueError('Invalid saved SAE composition')
    if (response < 0).any():
        raise ValueError('Negative SAE response')
    totals = response.sum(axis=1, dtype=np.float64)
    if (totals <= 0).any():
        raise ValueError('Panorama has no positive SAE response')
    masses = np.stack([response[:, cols].sum(axis=1, dtype=np.float64)
                       for cols in indices], axis=1)
    if not np.allclose(masses.sum(axis=1), totals, rtol=1e-6, atol=1e-6):
        raise ValueError('22 classes do not conserve total positive SAE mass')
    return (masses / totals[:, None]).astype(np.float32)


def compute_scores(args, frame, indices, identity):
    cache = args.stage / 'panorama_class_shares.npy'
    marker = args.stage / 'scores_identity.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved != {'identity': identity, 'rows': len(frame), 'classes': list(CLASSES)}:
            raise ValueError('Existing 22-class score cache has a different identity')
        scores = np.load(cache, mmap_mode='r')
        if scores.shape != (len(frame), len(CLASSES)):
            raise ValueError('Wrong saved 22-class score shape')
        return scores

    values = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32,
                                       shape=(len(frame), len(CLASSES)))
    run = json.loads((args.extraction / 'run.json').read_text())
    cursor = 0
    for city_record in run['cities']:
        city = city_record['city']
        city_count = 0
        for path in sorted((args.extraction / city.replace('/', '__')).glob('*/vectors.npz')):
            marker_path = path.with_name('complete.json')
            verified = json.loads(marker_path.read_text())
            if verified['identity'] != args.run_identity or verified['archive_bytes'] != path.stat().st_size:
                raise ValueError(f'Archive identity mismatch: {path}')
            with np.load(path, allow_pickle=False) as data:
                count = len(data['panoid'])
                rows = frame.iloc[cursor:cursor + count]
                if (len(rows) != count or not (rows.city == city).all()
                        or not np.array_equal(rows.panoid.to_numpy(), data['panoid'])
                        or not np.array_equal(rows.grid_x.to_numpy(), data['grid_x'])
                        or not np.array_equal(rows.grid_y.to_numpy(), data['grid_y'])):
                    raise ValueError(f'Panorama ordering mismatch: {path}')
                values[cursor:cursor + count] = score_chunk(data['composition'], indices)
                cursor += count
                city_count += count
        if city_count != int((frame.city == city).sum()):
            raise ValueError(f'Incomplete city: {city}')
        values.flush()
        write_json(args.stage / 'progress.json',
                   {'status': 'scoring', 'city': city, 'panoramas_done': cursor,
                    'total': len(frame)})
        print(f'22-class scores: {city}; {cursor:,}/{len(frame):,}', flush=True)
    if cursor != len(frame):
        raise ValueError('Incomplete 30-city corpus')
    write_json(marker, {'identity': identity, 'rows': len(frame), 'classes': list(CLASSES)})
    return np.load(cache, mmap_mode='r')


def cell_composition(frame, scores):
    keys = pd.MultiIndex.from_frame(frame[['city', 'grid_x', 'grid_y']])
    cell_id, unique = pd.factorize(keys, sort=True)
    counts = np.bincount(cell_id)
    cells = pd.DataFrame(list(unique), columns=['city', 'grid_x', 'grid_y'])
    cells['sampled_panoramas'] = counts
    for j, label in enumerate(CLASSES):
        cells[label] = np.bincount(cell_id, weights=scores[:, j],
                                   minlength=len(cells)) / counts
    if not np.allclose(cells[list(CLASSES)].sum(axis=1), 1, atol=1e-6):
        raise ValueError('Cell shares must sum to one across all 22 classes')
    return cells


def render(args, cells, pattern_stage):
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    palette = list(plt.get_cmap('tab20').colors) + ['#484848', '#c19d55']
    pipeline(args.output)
    city_figures(cells, list(CLASSES), args.output, labels=LABELS,
                 colors=palette, residuals=[])
    summary, units, patterns, _, _, _ = load(pattern_stage)
    if summary['panoramas'] != int(cells.sampled_panoramas.sum()) or summary['cells'] != len(cells):
        raise ValueError('Landscape-combination corpus differs from 22-class corpus')
    city_atlas(pattern_stage, units, patterns, cells, list(CLASSES), args.output,
               labels=LABELS, colors=palette, residuals=[])
    if args.install_thesis:
        THESIS.mkdir(parents=True, exist_ok=True)
        for stem in STEMS:
            shutil.copy2(args.output / f'{stem}.pdf', THESIS / f'{stem}.pdf')
        for source in sorted((args.output / 'cities').glob('*.pdf')):
            filename = source.name.replace('Brasília', 'Brasilia')
            target = THESIS / 'cities' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return summary


def output_manifest(args, identity, hierarchy, frame, cells):
    outputs = {str(path.relative_to(args.output)): sha256(path)
               for path in sorted(args.output.rglob('*'))
               if path.is_file() and path.name != 'semantic_22_figure_manifest.json'}
    value = {
        'status': 'complete', 'run_identity': args.run_identity,
        'score_identity': identity,
        'hierarchy_sha256': sha256(HIERARCHY),
        'panoramas': len(frame), 'cities': int(frame.city.nunique()),
        'cells': len(cells), 'class_count': len(CLASSES),
        'dimension_count': len(hierarchy['dimensions']),
        'dimensions_per_class': dict(Counter(r['family'] for r in hierarchy['dimensions'])),
        'classes': list(CLASSES),
        'aggregation': 'mean panoramas per observed 1-km cell; equal cells within city',
        'denominator': 'positive response over all 4096 SAE dimensions',
        'combination_scope': 'existing landscape-eligible semantic-unit search; excludes non-landscape and insufficient-evidence classes',
        'outputs_sha256': outputs,
    }
    write_json(args.output / 'semantic_22_figure_manifest.json', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--pattern-stage', type=Path,
                        default=FOLLOWUP / 'stages/composition_patterns_screened')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--run-identity', required=True)
    parser.add_argument('--install-thesis', action='store_true')
    args = parser.parse_args()
    args.stage.mkdir(parents=True, exist_ok=True)
    source = args.stage.parent / 'candidate_combinations/panorama_elements.parquet'
    frame = pd.read_parquet(source, columns=['city', 'panoid', 'grid_x', 'grid_y'])
    corpus = json.loads((args.stage.parent.parent / 'complete.json').read_text())
    if (len(frame) != corpus['processed'] or frame.city.nunique() != 30
            or corpus['identity']['extract_run_identity'] != args.run_identity):
        raise ValueError('The 30-city source corpus is incomplete or has a different identity')
    hierarchy = json.loads(HIERARCHY.read_text())
    indices = class_indices(hierarchy)
    identity = hashlib.sha256(json.dumps({'run': args.run_identity,
        'hierarchy_sha256': sha256(HIERARCHY), 'classes': list(CLASSES)},
        sort_keys=True).encode()).hexdigest()
    scores = compute_scores(args, frame, indices, identity)
    cells = cell_composition(frame, scores)
    cells.to_parquet(args.stage / 'cell_class_shares.parquet', index=False)
    write_json(args.stage / 'progress.json',
               {'status': 'rendering', 'panoramas_done': len(frame), 'cells': len(cells)})
    summary = render(args, cells, args.pattern_stage)
    output_manifest(args, identity, hierarchy, frame, cells)
    write_json(args.stage / 'complete.json', {
        'status': 'complete', 'identity': identity, 'panoramas': len(frame),
        'cities': 30, 'cells': len(cells), 'classes': len(CLASSES),
        'dimension_counts': dict(Counter(r['family'] for r in hierarchy['dimensions'])),
        'non_landscape_classes': ['unresolved', 'ground_unspecified', 'sky',
            'vegetation_unspecified', 'vehicle', 'material_unspecified',
            'scene_context', 'person', 'spatial_encoding'],
        'landscape_combinations': summary['recurrent_patterns'],
        'thesis_figures_installed': args.install_thesis,
        'interpretation': 'Descriptive SAE response shares across all 22 provisional classes; '
            'non-landscape and uncertain classes are retained in city figures but excluded '
            'from landscape-combination mining.'})
    write_json(args.stage / 'progress.json', {'status': 'complete', 'panoramas_done': len(frame)})
    print(f'Completed 22-class figures for {len(frame):,} panoramas in {len(cells):,} cells', flush=True)


if __name__ == '__main__':
    main()
