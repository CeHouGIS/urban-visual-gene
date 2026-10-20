"""Render semantic parent-family classes for recurrent landscape combinations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .followup import FOLLOWUP
from sae_experiments.panorama_vitl30_figures import name

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'experiments/vitl30_city_patterns/results/figures/six_city_2p5km_category_maps'
DEFAULT_STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
PARENTS = {
    'building': 'Buildings', 'tree': 'Trees', 'low_vegetation': 'Low vegetation',
    'road': 'Roads', 'pedestrian': 'Pedestrian space', 'barrier': 'Barriers',
    'water': 'Water', 'terrain': 'Terrain',
    'transport_structure': 'Transport structures', 'utility': 'Utilities',
    'sign': 'Signs', 'furniture': 'Street furniture',
    'construction': 'Fixed construction',
}


def load_classes(stage: Path):
    summary = json.loads((stage / 'summary.json').read_text())
    units = json.loads((stage / 'units.json').read_text())
    patterns = json.loads((stage / 'patterns.json').read_text())
    rates = pd.read_csv(stage / 'city_distribution.csv')
    if len(patterns) != summary['recurrent_patterns'] or rates.pattern_id.nunique() != len(patterns):
        raise ValueError('Pattern data does not match the screened summary')
    if rates.city.nunique() != summary['cities'] or len(rates) != len(patterns) * summary['cities']:
        raise ValueError('City distribution is incomplete')

    grouped: dict[tuple[str, ...], list[dict]] = {}
    for pattern in patterns:
        families = tuple(sorted({units[index]['family'] for index in pattern['units']}))
        if len(families) != pattern['order']:
            raise ValueError(f'Pattern family/order mismatch: {pattern["pattern_id"]}')
        grouped.setdefault(families, []).append(pattern)

    city_order = sorted(rates.city.unique())
    city_values: dict[tuple[str, ...], np.ndarray] = {}
    class_rows = []
    for families, members in grouped.items():
        ids = {pattern['pattern_id'] for pattern in members}
        selected = rates[rates.pattern_id.isin(ids)]
        matrix = selected.pivot(index='city', columns='pattern_id', values='cell_weighted_rate')
        matrix = matrix.loc[city_order, sorted(ids)]
        values = matrix.sum(axis=1).to_numpy(dtype=float)
        if (values > 1 + 1e-9).any():
            raise ValueError(f'Within-class pattern matches overlap for {families}')
        city_values[families] = values
        class_rows.append({
            'families': families,
            'label': ' + '.join(PARENTS[family] for family in families),
            'order': len(families),
            'fine_pattern_count': len(members),
            'mean_city_cell_rate': float(values.mean()),
            'positive_panoramas': int(sum(pattern['positive_panoramas'] for pattern in members)),
        })

    class_rows.sort(key=lambda row: (-row['mean_city_cell_rate'], row['label']))
    return summary, class_rows, city_values, city_order


def draw_inventory(summary, classes, output: Path, pdf: PdfPages):
    pair = sorted((row for row in classes if row['order'] == 2),
                  key=lambda row: (-row['mean_city_cell_rate'], row['label']))
    triple = sorted((row for row in classes if row['order'] == 3),
                    key=lambda row: (-row['mean_city_cell_rate'], row['label']))
    fig = plt.figure(figsize=(16.8, 14.4), facecolor='white')
    fig.text(.5, .975, 'Landscape combination classes', ha='center', va='top',
             fontsize=19, weight='bold', color='#202729')
    fig.text(.5, .946,
             f"{summary['recurrent_patterns']} recurrent fine-grained patterns grouped into "
             f"{len(classes)} unordered parent-family signatures",
             ha='center', va='top', fontsize=10, color='#4f5d60')
    grid = fig.add_gridspec(1, 2, left=.26, right=.98, top=.89, bottom=.095,
                            wspace=.55, width_ratios=[1, 1])
    colors = {2: '#3d7d78', 3: '#bd7357'}
    for column, (order, rows) in enumerate(((2, pair), (3, triple))):
        ax = fig.add_subplot(grid[0, column])
        y = np.arange(len(rows))
        counts = np.asarray([row['fine_pattern_count'] for row in rows])
        rates = np.asarray([row['mean_city_cell_rate'] * 100 for row in rows])
        labels = [row['label'] for row in rows]
        ax.barh(y, counts, color=colors[order], height=.68, alpha=.88)
        ax.set_yticks(y, labels, fontsize=7.1)
        ax.invert_yaxis()
        ax.set_xlim(0, max(counts) * 1.35)
        ax.set_xlabel('Distinct recurrent fine-grained patterns', fontsize=8.5)
        ax.set_title(f'{order}-family classes  |  {len(rows)} signatures',
                     fontsize=11, weight='bold', loc='left', pad=9)
        ax.xaxis.grid(True, color='#e4e8e8', linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=7.5, length=2, colors='#566164')
        ax.tick_params(axis='y', length=0, pad=5, colors='#263033')
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#aeb8bb')
        for index, (count, rate) in enumerate(zip(counts, rates)):
            ax.text(max(count - .48, .15), index, str(count), va='center', ha='right',
                    fontsize=7, weight='bold', color='white')
        rate_ax = ax.twiny()
        rate_ax.set_xlim(0, max(float(rates.max()) * 1.16, .1))
        rate_ax.scatter(rates, y + .2, color='#263033', s=12, marker='|', linewidths=1.4,
                        zorder=4)
        rate_ax.set_xlabel('Mean city-level cell-balanced panorama rate (%)', fontsize=8.2)
        rate_ax.tick_params(axis='x', labelsize=7, length=2, colors='#566164')
        rate_ax.spines['top'].set_color('#aeb8bb')
        for spine in ('right', 'left', 'bottom'):
            rate_ax.spines[spine].set_visible(False)
        rate_ax.set_ylim(ax.get_ylim())

    fig.text(.26, .045,
             'Class = the unordered set of semantic parent families in a combination. Bar length counts distinct '
             'subtype-level patterns; marker shows their mean city-level rate, weighting observed 1-km cells equally '
             'within each city and cities equally. The subtype patterns within one class are mutually exclusive '
             'under the one-unit-per-family assignment.',
             ha='left', va='bottom', fontsize=8, color='#4f5d60', wrap=True)
    fig.savefig(output, dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.14)
    pdf.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=.14)
    plt.close(fig)


def draw_city_heatmap(summary, classes, city_values, cities, output: Path, pdf: PdfPages):
    selected = classes[:20]
    labels = [f"{row['label']}  (n={row['fine_pattern_count']})" for row in selected]
    matrix = np.stack([city_values[row['families']] for row in selected], axis=0) * 100
    positive = matrix[matrix > 0]
    vmax = max(float(np.quantile(positive, .97)) if len(positive) else 1., .1)
    fig, ax = plt.subplots(figsize=(16.5, 10.8), facecolor='white')
    fig.subplots_adjust(left=.24, right=.95, top=.88, bottom=.23)
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis',
                   vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8.2)
    ax.set_xticks(np.arange(len(cities)), [name(city) for city in cities],
                  rotation=90, fontsize=7.3)
    ax.set_title('City distribution of the 20 most common landscape combination classes',
                 fontsize=15, weight='bold', loc='left', pad=14, color='#202729')
    ax.tick_params(axis='both', length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(im, ax=ax, shrink=.78, pad=.015)
    colorbar.set_label('Cell-balanced sampled panoramas matching class (%)', fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)
    fig.text(.24, .045,
             f"Showing 20 of {len(classes)} parent-family signatures. Class rates sum the mutually exclusive "
             'fine-grained patterns with the same parent-family signature. These are SAE-response retrieval rates, '
             'not physical object counts, patch coverage, or human-validated landscape prototypes.',
             ha='left', va='bottom', fontsize=8, color='#4f5d60', wrap=True)
    fig.savefig(output, dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.14)
    pdf.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=.14)
    plt.close(fig)


def render(stage=DEFAULT_STAGE, output=OUTPUT, append_to_main=True):
    summary, classes, city_values, cities = load_classes(stage)
    output.mkdir(parents=True, exist_ok=True)
    inventory_png = output / 'Fig_LandscapeCombinationClassInventory.png'
    city_png = output / 'Fig_LandscapeCombinationClassCityDistribution.png'
    combo_pdf = output / 'LandscapeCombinationClasses.pdf'
    with PdfPages(combo_pdf) as pdf:
        draw_inventory(summary, classes, inventory_png, pdf)
        draw_city_heatmap(summary, classes, city_values, cities, city_png, pdf)

    manifest_path = output / 'SixCity_30x30_CategoryAtlas.json'
    manifest = json.loads(manifest_path.read_text())
    class_pages = {
        'pdf': combo_pdf.name,
        'figures': [inventory_png.name, city_png.name],
        'class_count': len(classes),
        'recurrent_fine_patterns': summary['recurrent_patterns'],
        'class_signatures_by_order': {str(order): sum(row['order'] == order for row in classes)
                                      for order in (2, 3)},
        'fine_patterns_by_order': summary['orders'],
        'city_distribution_classes_shown': min(20, len(classes)),
        'class_definition': 'Unordered combination of semantic parent families; each class groups mutually exclusive subtype-level recurrent SAE-response patterns under the one-unit-per-family assignment.',
        'rate_definition': 'Mean across cities of within-city observed 1-km-cell-balanced sampled-panorama rates; response retrieval, not physical object count or validated landscape prevalence.',
    }
    manifest['combination_classes'] = class_pages
    manifest['files'] = sorted(set(manifest.get('files', [])) |
                               {combo_pdf.name, inventory_png.name, city_png.name})

    if append_to_main:
        main_pdf = output / 'LandscapeCategoryAtlas.pdf'
        merged = output / '.LandscapeCategoryAtlas.merged.pdf'
        subprocess.run(['pdfunite', str(main_pdf), str(combo_pdf), str(merged)], check=True)
        merged.replace(main_pdf)
        info = subprocess.run(['pdfinfo', str(main_pdf)], check=True, capture_output=True, text=True).stdout
        manifest['combined_atlas_page_count'] = int(next(
            line.split(':', 1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'classes': len(classes), 'patterns': summary['recurrent_patterns'],
                      'orders': class_pages['fine_patterns_by_order'], 'class_signatures': class_pages['class_signatures_by_order'],
                      'pdf': str(combo_pdf),
                      'combined_atlas': str(output / 'LandscapeCategoryAtlas.pdf') if append_to_main else None},
                     ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--no-append', action='store_true')
    args = parser.parse_args()
    render(args.stage, args.output, append_to_main=not args.no_append)
