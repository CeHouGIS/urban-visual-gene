"""Render city case studies of broadly shared and city-enriched SAE responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
import torch

from .composition_evidence import COLORS, infer, render
from .composition_patterns import pattern_mask
from .extract import image_paths, load_models, write_json
from .followup import FOLLOWUP
from .render_city_distinctive_examples import choose_indices


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/city_internal_compositions'
STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
FRAME = FOLLOWUP / 'stages/candidate_combinations/panorama_elements.parquet'
CASES = {
    'rio': {
        'city': 'Brazil/RiodeJaneiro', 'display': 'Rio de Janeiro', 'short': 'Rio',
        'patterns': (
            ('d8393482e056', 'Broadly shared', 'Road + trees', '#4e817d', 1),
            ('ec57f37a1aab', 'Regional candidate',
             'Wall + tile-roof building + sidewalk + wires', '#ae7652', 1),
            ('627c338c01db', 'Rio-enriched candidate',
             'Wall + tile-roof building + hillside', '#866a94', 1),
        ),
    },
    'dhaka': {
        'city': 'Bangladesh/Dhaka', 'display': 'Dhaka', 'short': 'Dhaka',
        'patterns': (
            ('d8393482e056', 'Broadly shared', 'Road + trees', '#4e817d', 1),
            ('6c4c7543befd', 'Two-city shared candidate',
             'Fence + older structures + trees', '#ae7652', 1),
            ('cd8d335c382b', 'Dhaka-enriched candidate',
             'Older structures + signs/posters + trees', '#866a94', 3),
        ),
    },
}


def prepare(stage: Path, frame_path: Path, results: Path, case: dict):
    rows = pd.read_csv(results / 'city_patterns.csv')
    support = pd.read_csv(results / 'pattern_city_support.csv').set_index('pattern_id')
    units = json.loads((stage / 'units.json').read_text())
    calibration = json.loads((stage / 'calibration.json').read_text())
    frame = pd.read_parquet(frame_path, columns=['city', 'panoid', 'grid_x', 'grid_y'])
    assignment = np.load(stage / 'assignment.npy', mmap_mode='r')
    scores = np.load(stage / 'scores.npy', mmap_mode='r')
    if len(frame) != len(assignment) or len(frame) != len(scores):
        raise ValueError('Panorama metadata and response arrays are misaligned')
    thresholds = np.asarray(calibration['unit_thresholds'], dtype=float)
    cities = sorted(frame.city.unique())
    if len(cities) != 30:
        raise ValueError(f'Expected 30 cities, found {len(cities)}')
    city = case['city']
    if city not in cities:
        raise ValueError(f'{city} is absent')
    selected = []
    for pattern_id, scope, english_label, color, rank in case['patterns']:
        matches = rows[(rows.city == city) & (rows.pattern_id == pattern_id)]
        if len(matches) != 1 or not bool(matches.iloc[0].replicated):
            raise ValueError(f'{city} pattern is absent or unconfirmed: {pattern_id}')
        record = matches.iloc[0]
        ids = json.loads(record.unit_ids)
        candidates = choose_indices(frame, assignment, scores, thresholds, units,
                                    city, ids, rank)
        position, strength = candidates[rank - 1]
        positive = pattern_mask(assignment, units, ids) & (frame.city.to_numpy() == city)
        actual_cells = frame.loc[positive, ['grid_x', 'grid_y']].drop_duplicates()
        if positive.sum() != record.panoramas or len(actual_cells) != record.cells:
            raise ValueError(f'Saved {city} counts changed: {pattern_id}')
        if int(support.loc[pattern_id, 'cities']) != int(record.cities_replicated):
            raise ValueError(f'Cross-city support changed: {pattern_id}')
        selected.append({'id': pattern_id, 'scope': scope, 'label': english_label,
                         'color': color, 'record': record, 'unit_ids': ids,
                         'support': support.loc[pattern_id], 'position': position,
                         'strength': strength, 'rank': rank, 'cells': actual_cells})
    if not 15 < int(selected[0]['support'].cities) <= 30:
        raise ValueError('Broadly shared example no longer spans most cities')
    if not (int(selected[1]['support'].cities) == 2 and
            int(selected[2]['support'].cities) == 1):
        raise ValueError('Regional/enriched examples changed scope')
    return frame, units, cities, selected


def draw_support(ax, item, cities, case):
    support_cities = set(item['support'].city_names.split(' | '))
    color = item['color']
    ax.set_xlim(-.35, 10.35)
    ax.set_ylim(-1.85, 4.5)
    for index, city in enumerate(cities):
        x, y = index % 10, 1.5 - index // 10
        present = city in support_cities
        ax.scatter(x, y, marker='s', s=102, facecolor=color if present else '#e4e8e7',
                   edgecolor='#1e292b' if city == case['city'] else 'none',
                   linewidth=1.4 if city == case['city'] else 0, zorder=3)
    ax.text(0, 2.8, f"{int(item['support'].cities)}/30 cities", fontsize=14,
            weight='bold', color='#263232', va='center')
    ax.text(0, 2.35, f"{case['short']} rate: {item['record'].city_rate:.2%}",
            fontsize=7.7, color='#52615f', va='center')
    ax.text(0, -1.05, f"Outlined square: {case['short']}",
            fontsize=7.7, color='#52615f')
    if int(item['support'].cities) == 2:
        pair = ', '.join(city.split('/')[-1].replace('Bras\u00edlia', 'Brasilia')
                         for city in item['support'].city_names.split(' | '))
        ax.text(0, -1.45, f'Confirmed: {pair}', fontsize=7.7, color='#52615f')
    elif int(item['support'].cities) == 1:
        ax.text(0, -1.45, 'Below-cutoff responses elsewhere', fontsize=7.3,
                color='#52615f')
    ax.axis('off')


def draw_map(ax, item, all_cells, limits, row):
    color = item['color']
    base_size = min(5, max(1.1, 5000 / len(all_cells)))
    ax.scatter(all_cells.grid_x, all_cells.grid_y, s=base_size, color='#d6dddc',
               marker='s', linewidths=0, rasterized=True)
    ax.scatter(item['cells'].grid_x, item['cells'].grid_y, s=base_size * 2.6, color=color,
               marker='s', linewidths=0, rasterized=True)
    ax.scatter(item['sample'].grid_x, item['sample'].grid_y, s=80,
               marker='o', facecolor='none', edgecolor='#1e292b', linewidth=1.2,
               zorder=5)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.01, .97, f"{int(item['record'].cells):,} of {len(all_cells):,} observed cells",
            transform=ax.transAxes, va='top', fontsize=8, color='#263232',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    if row == 2:
        xmin = limits[0][0] + 2
        ymin = limits[1][0] + 2
        ax.plot([xmin, xmin + 10], [ymin, ymin], lw=1.4, color='#263232')
        ax.text(xmin + 5, ymin + .8, '10 km', ha='center', va='bottom', fontsize=7.5,
                color='#263232', path_effects=[patheffects.withStroke(
                    linewidth=3, foreground='white')])


def draw_evidence(ax, item):
    with Image.open(item['original']) as opened:
        original = np.asarray(opened.convert('RGB'))
    with Image.open(item['overlay']) as opened:
        overlay = np.asarray(opened.convert('RGB'))
    # The two strips preserve the native four-heading aspect ratio.
    ax.imshow(original, extent=(0, 4, 1.05, 2.05), aspect='auto')
    ax.imshow(overlay, extent=(0, 4, 0, 1), aspect='auto')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index in range(1, 4):
        ax.axvline(index, color='white', lw=.5, alpha=.8)
    label_box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': 2}
    ax.text(3.96, 1.89, 'Original', fontsize=8, color='#263232', va='top',
            ha='right', bbox=label_box)
    ax.text(3.96, .84, 'SAE response', fontsize=8, color='#263232', va='top',
            ha='right', bbox=label_box)
    x = 0.0
    for label, unit_color in zip(item['unit_labels'], COLORS):
        english = {
            '道路': 'road', '树木': 'trees', '围墙': 'wall',
            '瓦顶建筑': 'tile-roof building', '人行道': 'sidewalk',
            '电线': 'wires', '山地': 'hillside',
            '围栏': 'fence', '老屋': 'older structures', '广告牌': 'signs/posters',
        }.get(label, label)
        ax.text(x, -.095, '\u25a0', color=unit_color, transform=ax.transAxes,
                fontsize=9, va='center')
        ax.text(x + .027, -.095, english, color='#465452', transform=ax.transAxes,
                fontsize=7.6, va='center')
        x += .045 + .015 * len(english)


def render_figure(frame, cities, selected, output: Path, case: dict):
    city_rows = frame.loc[frame.city == case['city']]
    all_cells = city_rows[['grid_x', 'grid_y']].drop_duplicates()
    limits = ((int(all_cells.grid_x.min()) - 2, int(all_cells.grid_x.max()) + 2),
              (int(all_cells.grid_y.min()) - 2, int(all_cells.grid_y.max()) + 2))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.linewidth': .6})
    fig = plt.figure(figsize=(16.5, 17), facecolor='white')
    gs = GridSpec(3, 3, figure=fig, width_ratios=[2.3, 3.4, 7.5],
                  left=.045, right=.985, top=.865, bottom=.10,
                  hspace=.33, wspace=.20)
    for index, item in enumerate(selected):
        left = fig.add_subplot(gs[index, 0])
        middle = fig.add_subplot(gs[index, 1])
        right = fig.add_subplot(gs[index, 2])
        draw_support(left, item, cities, case)
        draw_map(middle, item, all_cells, limits, index)
        draw_evidence(right, item)
        bounds = left.get_position()
        fig.text(bounds.x0, bounds.y1 + .008,
                 f"{chr(97 + index)}  {item['scope']}", fontsize=12.5,
                 fontweight='bold', color=item['color'], ha='left')
        fig.text(bounds.x0, bounds.y1 - .010, item['label'], fontsize=9.3,
                 color='#263232', ha='left')
    fig.text(.045, .963, 'Shared and city-enriched landscape combinations',
             fontsize=20, fontweight='bold', color='#24312f', ha='left')
    fig.text(.045, .938, f"{case['display']} case study  |  30-city screened SAE sample",
             fontsize=10.5, color='#52615f', ha='left')
    fig.text(.045, .900, 'CROSS-CITY SUPPORT', fontsize=8.6, weight='bold', color='#42514e')
    fig.text(.245, .900, f"WITHIN {case['short'].upper()}",
             fontsize=8.6, weight='bold', color='#42514e')
    fig.text(.505, .900, 'FOUR-VIEW EXAMPLE', fontsize=8.6, weight='bold', color='#42514e')
    fig.text(.045, .049,
             'Filled squares: cities passing the within-city discovery and 5-km-block holdout criteria. '
             'Maps: 1-km cells with at least one matching sampled panorama; rings mark illustrated locations.',
             fontsize=8.2, color='#51605d', ha='left')
    fig.text(.045, .032,
             'Combinations are panorama-level SAE co-responses with provisional semantic labels; '
             'overlays show relative activation, not object segmentation or verified spatial relations.',
             fontsize=8.2, color='#51605d', ha='left')
    figure_name = f"Fig_{case['short']}_Three_Scopes"
    fig.savefig(output / f'{figure_name}.png', dpi=220, facecolor='white')
    fig.savefig(output / f'{figure_name}.pdf', facecolor='white')
    plt.close(fig)


def run(stage: Path, frame_path: Path, results: Path, output: Path,
        device: str, case_key: str = 'rio'):
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for four-view attribution')
    case = CASES[case_key]
    city = case['city']
    frame, units, cities, selected = prepare(stage, frame_path, results, case)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    dino, sae, _ = load_models(device)
    manifest = {'city': city, 'corpus_panoramas': len(frame),
                'sampled_city_panoramas': int((frame.city == city).sum()),
                'scope_rule': 'Most-city example has >15 confirmed cities; region/city labels are descriptive candidates.',
                'selection': 'Strength-ranked panorama-level matches with all four image files; chosen rank recorded per pattern.',
                'map': 'Observed 1-km cells; a cell is colored if at least one sampled panorama matches.',
                'evidence': 'Frozen DINOv3-L and BatchTopK SAE; response overlay is not segmentation.',
                'patterns': []}
    for item in selected:
        sample = frame.iloc[item['position']]
        images = []
        for path in image_paths(city, sample.panoid):
            with Image.open(path) as opened:
                images.append(opened.convert('RGB'))
        maps = infer(images, dino, sae, item['unit_ids'], units, device)
        if maps.shape != (len(item['unit_ids']), 4, 28, 28) or not np.isfinite(maps).all():
            raise ValueError(f'Invalid activation map for {item["id"]}')
        folder = output / item['id']
        render(images, maps, folder)
        item['original'] = folder / 'original.webp'
        item['overlay'] = folder / 'overlay.webp'
        item['sample'] = sample
        item['unit_labels'] = [units[uid]['label'] for uid in item['unit_ids']]
        manifest['patterns'].append({
            'pattern_id': item['id'], 'scope': item['scope'], 'label': item['label'],
            'unit_ids': item['unit_ids'], 'unit_labels': item['unit_labels'],
            'confirming_cities': item['support'].city_names.split(' | '),
            'city_support': int(item['support'].cities),
            'city_matching_panoramas': int(item['record'].panoramas),
            'city_matching_cells': int(item['record'].cells),
            'city_cell_balanced_rate': float(item['record'].city_rate),
            'panoid': str(sample.panoid), 'grid_x': int(sample.grid_x),
            'grid_y': int(sample.grid_y),
            'strength_rank': item['rank'],
            'minimum_unit_score_ratio': item['strength'],
            'original': str(item['original'].resolve()),
            'overlay': str(item['overlay'].resolve())})
        print(f'Rendered {item["id"]}: {sample.panoid}', flush=True)
    render_figure(frame, cities, selected, output, case)
    write_json(output / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--frame', type=Path, default=FRAME)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--case', choices=tuple(CASES), default='rio')
    args = parser.parse_args()
    output = args.output or RESULTS / f'{args.case}_three_scopes'
    run(args.stage, args.frame, args.results, output, args.device, args.case)
