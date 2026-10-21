"""Render six-city grid bubbles from the screened panorama composition corpus."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from .composition_patterns import pattern_mask
from .followup import FOLLOWUP
from .semantic_city_figures import FIGURES, PARENTS, load
from sae_experiments.panorama_vitl30_figures import name


STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
CITY_ORDER = [
    'Argentina/BuenosAires',
    'Australia/Melbourne',
    'Bangladesh/Dhaka',
    'Portugal/Lisbon',
    'SouthAfrica/CapeTown',
    'US/Chicago',
]
TOP_TYPES = 12
AREA_PER_MATCH = 4.0
TYPE_COLORS = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9',
    '#8C564B', '#9467BD', '#B07C43', '#17A398', '#C44E52', '#6C8EBF',
]
OTHER_COLOR = '#AEB5B7'
EMPTY_COLOR = '#FFFFFF'
FAMILY_ORDER = {
    'building': 'Buildings', 'tree': 'Trees', 'low_vegetation': 'Low vegetation',
    'road': 'Roads', 'pedestrian': 'Pedestrian space', 'barrier': 'Barriers',
    'water': 'Water', 'terrain': 'Terrain', 'transport_structure': 'Transport structures',
    'utility': 'Utilities', 'sign': 'Signs', 'furniture': 'Street furniture',
    'construction': 'Fixed construction',
}


def signature_label(signature):
    return ' + '.join(FAMILY_ORDER.get(family, PARENTS.get(family, family))
                      for family in signature)


def build_cell_metrics(stage):
    summary, units, patterns, frame, _scores, assignment = load(stage)
    keys = pd.MultiIndex.from_frame(frame[['city', 'grid_x', 'grid_y']])
    cell_id, unique = pd.factorize(keys, sort=True)
    n_cells = len(unique)
    signatures = defaultdict(lambda: np.zeros(len(frame), dtype=bool))
    any_pattern = np.zeros(len(frame), dtype=bool)

    for pattern in patterns:
        matched = pattern_mask(assignment, units, pattern['units'])
        signature = tuple(sorted({units[i]['family'] for i in pattern['units']}))
        signatures[signature] |= matched
        any_pattern |= matched

    # Rank a cell's type by the number of sampled panoramas with each family signature.
    signature_counts = {}
    for signature, matched in signatures.items():
        signature_counts[signature] = np.bincount(
            cell_id, weights=matched.astype(np.uint8), minlength=n_cells)

    signature_order = sorted(signature_counts,
                             key=lambda signature: (-int(signature_counts[signature].sum()), signature))
    matrix = np.column_stack([signature_counts[signature] for signature in signature_order])
    max_count = matrix.max(axis=1)
    winner = matrix.argmax(axis=1)
    winner_counts = np.bincount(winner[max_count > 0], minlength=len(signature_order))
    top_indices = sorted(range(len(signature_order)),
                         key=lambda i: (-winner_counts[i], signature_order[i]))[:TOP_TYPES]
    top_signatures = [signature_order[i] for i in top_indices]

    cells = pd.DataFrame(list(unique), columns=['city', 'grid_x', 'grid_y'])
    cells['sampled_panoramas'] = np.bincount(cell_id, minlength=n_cells)
    cells['panoramas_with_composition'] = np.bincount(
        cell_id, weights=any_pattern.astype(np.uint8), minlength=n_cells)
    cells['composition_match_rate'] = (
        cells['panoramas_with_composition'] / cells['sampled_panoramas'])
    labels = []
    for cell, count in zip(winner, max_count):
        labels.append(signature_label(signature_order[cell]) if count > 0 else 'No detected composition')
    cells['dominant_signature'] = labels
    cells['distinct_family_signatures'] = np.sum(matrix > 0, axis=1)
    cells['dominant_signature_panoramas'] = max_count.astype(np.int16)

    selected = cells[cells.city.isin(CITY_ORDER)].copy()
    if selected.city.nunique() != len(CITY_ORDER):
        raise ValueError('One or more selected cities are absent from the screened corpus')
    selected['composition_type'] = selected['dominant_signature']
    has_composition = selected['panoramas_with_composition'] > 0
    outside_top = ~selected['composition_type'].isin(
        [signature_label(s) for s in top_signatures])
    selected.loc[has_composition & outside_top, 'composition_type'] = 'Other signatures'
    return summary, selected, top_signatures


def render(stage=STAGE, output=FIGURES):
    summary, cells, top_signatures = build_cell_metrics(stage)
    labels = [signature_label(signature) for signature in top_signatures]
    colors = {label: TYPE_COLORS[i] for i, label in enumerate(labels)}
    colors['Other signatures'] = OTHER_COLOR
    colors['No detected composition'] = EMPTY_COLOR

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / 'Fig_ViTL30_SixCity_Grid_Bubbles.csv'
    pdf_path = output / 'Fig_ViTL30_SixCity_Grid_Bubbles.pdf'
    png_path = output / 'Fig_ViTL30_SixCity_Grid_Bubbles.png'
    cells.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(16.8, 12.4), facecolor='white')
    grid = fig.add_gridspec(2, 3, left=.055, right=.985, top=.885, bottom=.30,
                            wspace=.19, hspace=.26)
    fig.text(.5, .965, 'Landscape composition detections across six cities',
             ha='center', va='top', fontsize=17, weight='bold', color='#202729')
    fig.text(.5, .928,
             'One circle per observed 1-km cell; cities selected across six world regions',
             ha='center', va='top', fontsize=10.2, color='#4f5d60')

    for panel, city in enumerate(CITY_ORDER):
        ax = fig.add_subplot(grid[panel // 3, panel % 3])
        group = cells[cells.city == city].copy()
        gx = group.grid_x.to_numpy(dtype=float)
        gy = group.grid_y.to_numpy(dtype=float)
        x = gx - gx.min() + .5
        y = gy - gy.min() + .5
        for label in labels + ['Other signatures']:
            mask = group.composition_type.to_numpy() == label
            if not mask.any():
                continue
            values = group.loc[mask, 'panoramas_with_composition'].to_numpy()
            sizes = np.maximum(values, 0) * AREA_PER_MATCH
            nonzero = values > 0
            if nonzero.any():
                ax.scatter(x[mask][nonzero], y[mask][nonzero], s=sizes[nonzero],
                           c=colors[label], edgecolors='white', linewidths=.22,
                           alpha=.86, rasterized=True, zorder=2)
        zero = group.panoramas_with_composition.to_numpy() == 0
        if zero.any():
            ax.scatter(x[zero], y[zero], s=3, facecolors='white',
                       edgecolors='#70797b', linewidths=.55, rasterized=True, zorder=3)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-.3, max(x.max() + .8, 1))
        ax.set_ylim(-.3, max(y.max() + .8, 1))
        x_step = max(5, int(np.ceil(x.max() / 6 / 5) * 5))
        y_step = max(5, int(np.ceil(y.max() / 6 / 5) * 5))
        ax.xaxis.set_major_locator(MultipleLocator(x_step))
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.tick_params(labelsize=7, length=2.5, colors='#566164')
        ax.grid(color='#e7ebec', linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel('Local easting (km)', fontsize=8, color='#3f4a4d')
        ax.set_ylabel('Local northing (km)', fontsize=8, color='#3f4a4d')
        ax.set_title(f'{name(city)}   |   {len(group):,} cells', loc='left',
                     fontsize=11, weight='bold', pad=7, color='#202729')
        for spine in ax.spines.values():
            spine.set_color('#aeb8bb')
            spine.set_linewidth(.65)

    type_handles = [Line2D([0], [0], marker='o', linestyle='None', markersize=7,
                           markerfacecolor=colors[label], markeredgecolor='white',
                           label=label) for label in labels]
    type_handles += [Line2D([0], [0], marker='o', linestyle='None', markersize=7,
                            markerfacecolor=OTHER_COLOR, markeredgecolor='white',
                            label='Other signatures'),
                     Line2D([0], [0], marker='o', linestyle='None', markersize=6,
                            markerfacecolor='white', markeredgecolor='#70797b',
                            label='No detected composition')]
    fig.legend(handles=type_handles, loc='lower center', bbox_to_anchor=(.52, .102),
               ncol=4, frameon=False, fontsize=8.3, handletextpad=.45,
               columnspacing=1.35, labelspacing=.65)

    size_counts = [0, 2, 5, 8, 10]
    size_handles = []
    for count in size_counts:
        size_handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                   markersize=2.5 if count == 0 else np.sqrt(AREA_PER_MATCH * count),
                                   markerfacecolor='white' if count == 0 else '#566164',
                                   markeredgecolor='#70797b' if count == 0 else 'white',
                                   label=str(count)))
    fig.legend(handles=size_handles, title='Sample panoramas with at least one recurrent composition',
               loc='lower center', bbox_to_anchor=(.5, .212), ncol=len(size_handles),
               frameon=False, fontsize=8.3, title_fontsize=8.5,
               handletextpad=.55, columnspacing=1.5)
    fig.text(.5, .025,
             'Bubble area encodes the number of matching sampled panoramas (up to 10 per cell), not physical object counts. '
             'Color is the most frequent family signature among recurrent SAE combinations; labels remain exploratory and unvalidated.',
             ha='center', va='bottom', fontsize=8.1, color='#4f5d60', wrap=True)

    fig.savefig(pdf_path, facecolor='white', bbox_inches='tight', pad_inches=.12)
    fig.savefig(png_path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)

    counts = cells.groupby('city').agg(
        cells=('city', 'size'),
        sampled_panoramas=('sampled_panoramas', 'sum'),
        panoramas_with_composition=('panoramas_with_composition', 'sum'),
        median_match_rate=('composition_match_rate', 'median'),
    ).reset_index()
    manifest = {
        'figure': png_path.name,
        'run_identity': summary['run_identity'],
        'cities': CITY_ORDER,
        'selection': 'One city per world region; at least 850 observed 1-km cells per selected city.',
        'grid_count_metric': 'Number of sampled panorama instances in the cell matching at least one of 184 recurrent semantic combinations; 0-10 under the sampling design.',
        'color_metric': 'Most frequent high-level family signature among recurrent semantic combinations matched by sampled panoramas in the cell.',
        'color_classes': labels + ['Other signatures', 'No detected composition'],
        'top_signature_definitions': {signature_label(s): list(s) for s in top_signatures},
        'cell_counts': counts.to_dict(orient='records'),
        'limitations': [
            'Combination labels and family signatures are exploratory and not human-validated.',
            'Bubble sizes count sampled panorama instances, not physical objects or all landscapes in a cell.',
            'Each cell contains up to 10 sampled panoramas; counts are conditional on this sample.',
            'Local axes are relative grid indices in kilometres, not common coordinates across cities.',
        ],
        'files': [pdf_path.name, png_path.name, csv_path.name],
    }
    (output / 'Fig_ViTL30_SixCity_Grid_Bubbles.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pdf': str(pdf_path), 'png': str(png_path), 'csv': str(csv_path),
                      'cells': int(len(cells)), 'city_summary': counts.to_dict(orient='records')},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--output', type=Path, default=FIGURES)
    args = parser.parse_args()
    render(args.stage, args.output)


if __name__ == '__main__':
    main()
