"""Map class proportions on city-normalized 30 by 30 grids for six cities."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd

from .followup import FOLLOWUP
from .render_landscape_combination_classes import load_classes
from sae_experiments.panorama_vitl30_figures import name

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'experiments/vitl30_city_patterns/results/figures/six_city_2p5km_category_maps'
STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
CITY_ORDER = [
    'Argentina/BuenosAires', 'Australia/Melbourne', 'Bangladesh/Dhaka',
    'Portugal/Lisbon', 'SouthAfrica/CapeTown', 'US/Chicago',
]
MAX_MARKER_AREA = 40.0
TOP_CLASSES = 20
GRID_SIDE = 30


def load_class_cells(stage: Path, classes):
    patterns = json.loads((stage / 'patterns.json').read_text())
    units = json.loads((stage / 'units.json').read_text())
    observed = pd.read_csv(stage / 'observed_cells.csv')
    encoded = json.loads((stage / 'cell_patterns.json').read_text())
    if len(observed) == 0 or len(encoded) != len(patterns):
        raise ValueError('Incomplete panorama-to-grid pattern counts')
    expected_ids = {row['pattern_id'] for row in patterns}
    if {row['pattern_id'] for row in encoded} != expected_ids:
        raise ValueError('Cell-count pattern identities differ')

    pattern_families = {
        row['pattern_id']: tuple(sorted({units[index]['family'] for index in row['units']}))
        for row in patterns
    }
    class_patterns = {
        row['families']: [pattern_id for pattern_id, signature in pattern_families.items()
                          if signature == row['families']]
        for row in classes
    }
    wanted = {pattern_id for ids in class_patterns.values() for pattern_id in ids}
    counts_by_pattern = {}
    for row in encoded:
        if row['pattern_id'] not in wanted:
            continue
        values = np.frombuffer(base64.b64decode(row['positive_counts_u8']), dtype=np.uint8)
        if len(values) != len(observed):
            raise ValueError(f"Cell-count length mismatch for {row['pattern_id']}")
        counts_by_pattern[row['pattern_id']] = values.astype(np.int32)

    output_cells = []
    for class_index, class_row in enumerate(classes, start=1):
        signature = class_row['families']
        matched = np.zeros(len(observed), dtype=np.int32)
        for pattern_id in class_patterns[signature]:
            matched += counts_by_pattern[pattern_id]
        if (matched > observed.sampled_panoramas.to_numpy()).any():
            raise ValueError(f"Combination class double-counts panoramas: {class_row['label']}")

        selected = observed.assign(matched_panoramas=matched)
        selected = selected[selected.city.isin(CITY_ORDER)].copy()
        for city in CITY_ORDER:
            city_mask = selected.city == city
            city_cells = selected.loc[city_mask]
            for coordinate, target in [('grid_x', 'cell_x'), ('grid_y', 'cell_y')]:
                lower = float(city_cells[coordinate].min())
                extent = float(city_cells[coordinate].max() - lower + 1)
                position = (city_cells[coordinate].to_numpy(dtype=float) + .5 - lower)
                selected.loc[city_mask, target] = np.clip(
                    np.floor(position / extent * GRID_SIDE), 0, GRID_SIDE - 1).astype(np.int32)
        grouped = selected.groupby(['city', 'cell_x', 'cell_y'], sort=True, as_index=False).agg(
            sampled_panoramas=('sampled_panoramas', 'sum'),
            matched_panoramas=('matched_panoramas', 'sum'),
            source_1km_cells=('sampled_panoramas', 'size'))
        grouped['class_id'] = f'C{class_index:02d}'
        grouped['class_label'] = class_row['label']
        grouped['fine_pattern_count'] = class_row['fine_pattern_count']
        grouped['matched_share'] = (grouped.matched_panoramas /
                                    grouped.sampled_panoramas.replace(0, np.nan))
        if (grouped.matched_share > 1 + 1e-9).any():
            raise ValueError(f'Class proportion exceeds one: {class_row["label"]}')
        output_cells.append(grouped)

    cells = pd.concat(output_cells, ignore_index=True)
    return cells


def city_figure(class_row, class_id, color, cells, output_path, pdf):
    positive = cells.loc[cells.matched_share > 0, 'matched_share'].to_numpy(dtype=float)
    if not len(positive):
        return None

    fig = plt.figure(figsize=(16.2, 10.6), facecolor='white')
    grid = fig.add_gridspec(2, 3, left=.06, right=.985, top=.85, bottom=.19,
                            wspace=.20, hspace=.32)
    fig.text(.5, .965,
             f"{class_id}  |  {class_row['label']}", ha='center', va='top',
             fontsize=17, weight='bold', color='#202729')
    fig.text(.5, .923,
             f"{class_row['fine_pattern_count']} subtype-level patterns  |  30 x 30 normalized city grid  |  "
             'circle area = matching-panorama proportion (shared 0-100% scale)',
             ha='center', va='top', fontsize=9, color='#4f5d60')

    city_statistics = {}
    for panel, city in enumerate(CITY_ORDER):
        ax = fig.add_subplot(grid[panel // 3, panel % 3])
        group = cells[cells.city == city].copy()
        if group.empty:
            raise ValueError(f'Missing city grid cells: {city}')
        x = group.cell_x.to_numpy(dtype=float) + .5
        y = group.cell_y.to_numpy(dtype=float) + .5
        values = group.matched_share.to_numpy(dtype=float)
        ax.scatter(x, y, s=5, c='#e1e6e7', linewidths=0, rasterized=True, zorder=1)
        active = values > 0
        if active.any():
            sizes = MAX_MARKER_AREA * values[active]
            ax.scatter(x[active], y[active], s=sizes, c=[color], edgecolors='white',
                       linewidths=.3, alpha=.94, rasterized=True, zorder=2)

        ax.set_xlim(0, GRID_SIDE)
        ax.set_ylim(0, GRID_SIDE)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which='minor', color='#eef1f1', linewidth=.25, zorder=0)
        ax.grid(which='major', color='#e0e5e5', linewidth=.45, zorder=0)
        ax.tick_params(labelsize=6.8, length=2, colors='#566164')
        ax.set_xlabel('Normalized east-west position', fontsize=7.3, color='#3f4a4d')
        ax.set_ylabel('Normalized north-south position', fontsize=7.3, color='#3f4a4d')
        ax.set_title(name(city),
                     loc='left', fontsize=9.4, weight='bold', pad=5, color='#202729')
        for spine in ax.spines.values():
            spine.set_color('#aeb8bb')
            spine.set_linewidth(.6)
        city_statistics[city] = {
            'observed_grid_bins': int(len(group)),
            'bins_with_class': int(active.sum()),
            'mean_class_share_across_observed_bins': float(values[values > 0].mean()) if active.any() else 0.,
        }

    reference_shares = [.25, .5, .75, 1.]
    handles = [Line2D(
        [0], [0], marker='o', linestyle='None',
        markersize=np.sqrt(MAX_MARKER_AREA * value),
        markerfacecolor=color, markeredgecolor='white', label=f'{value:.0%}')
        for value in reference_shares]
    fig.legend(handles=handles, title='Matching panorama share per grid cell',
               loc='lower center', bbox_to_anchor=(.5, .115), ncol=len(handles),
               frameon=False, fontsize=8, title_fontsize=8.2,
               handletextpad=.5, columnspacing=1.2)
    fig.text(.5, .025,
             'Faint circles mark observed grid bins with no matching panorama. Each city is independently mapped '
             'to 30 x 30 equal-width bins; bin widths vary by city and are not fixed kilometers. Colored-circle '
             'area is proportional to matching sampled panoramas divided by all sampled panoramas in that bin. '
             'This is an SAE-response share, not physical object prevalence or validated landscape prevalence.',
             ha='center', va='bottom', fontsize=7.6, color='#4f5d60', wrap=True)

    fig.savefig(output_path, dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.12)
    pdf.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return city_statistics


def render(stage=STAGE, output=OUTPUT, append_to_main=False):
    summary, classes, _city_values, _cities = load_classes(stage)
    selected = classes[:TOP_CLASSES]
    cells = load_class_cells(stage, selected)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / 'LandscapeCombinationClassCityBubbles.pdf'
    cell_path = output / 'LandscapeCombinationClassCityCells.csv'
    cells.to_csv(cell_path, index=False)

    palette = []
    for red, green, blue in plt.get_cmap('tab20').colors:
        luminance = .2126 * red + .7152 * green + .0722 * blue
        factor = .65 if luminance > .55 else 1.
        palette.append((red * factor, green * factor, blue * factor))
    results = []
    with PdfPages(pdf_path) as pdf:
        for index, class_row in enumerate(selected):
            class_id = f'C{index + 1:02d}'
            color = to_hex(palette[index])
            image_path = output / f'Fig_LandscapeCombination_{class_id}_SixCity_Bubbles.png'
            subset = cells[cells.class_id == class_id]
            stats = city_figure(class_row, class_id, color, subset, image_path, pdf)
            if stats is None:
                continue
            results.append({
                'class_id': class_id,
                'class_label': class_row['label'],
                'families': list(class_row['families']),
                'fine_pattern_count': class_row['fine_pattern_count'],
                'mean_city_cell_rate': class_row['mean_city_cell_rate'],
                'color': color,
                'png': image_path.name,
                'city_statistics': stats,
            })

    manifest_path = output / 'SixCity_30x30_CategoryAtlas.json'
    manifest = json.loads(manifest_path.read_text())
    bubble_metadata = {
        'pdf': pdf_path.name,
        'cells_csv': cell_path.name,
        'cities': CITY_ORDER,
        'grid_shape_per_city': [GRID_SIDE, GRID_SIDE],
        'grid_definition': 'Each city extent is independently divided into 30 equal-width bins on each axis from the observed 1-km source-cell centroids; physical bin widths differ by city and no fixed-km grid is used.',
        'class_count': len(results),
        'selection': 'Twenty semantic parent-family signatures ranked by mean city-level cell-balanced panorama rate.',
        'circle_area': 'Proportional to the within-bin share of sampled panoramas matching the class; a fixed 0-100% area scale is shared across all six cities and all class pages.',
        'circle_color': 'One stable unique darkened tab20 color per combination class; class labels and matching size legends are repeated on each page.',
        'proportion_definition': 'Matching sampled panorama count divided by all sampled panoramas in the normalized bin.',
        'counts_are': 'SAE-response shares, not physical object prevalence or validated landscape prevalence.',
        'classes': results,
    }
    manifest['combination_city_bubbles'] = bubble_metadata
    manifest['files'] = sorted(set(manifest.get('files', [])) |
                               {pdf_path.name, cell_path.name} |
                               {item['png'] for item in results})

    if append_to_main:
        main_pdf = output / 'LandscapeCategoryAtlas.pdf'
        current_pages = int(manifest.get('combined_atlas_page_count', 0))
        expected_pages = int(manifest.get('category_count', 0)) + len(
            manifest.get('combination_classes', {}).get('figures', []))
        if current_pages != expected_pages:
            raise ValueError(
                f'Expected the {expected_pages}-page category and class atlas before appending; '
                f'got {current_pages}')
        merged = output / '.LandscapeCategoryAtlas.merged.pdf'
        subprocess.run(['pdfunite', str(main_pdf), str(pdf_path), str(merged)], check=True)
        merged.replace(main_pdf)
        info = subprocess.run(['pdfinfo', str(main_pdf)], check=True, capture_output=True, text=True).stdout
        manifest['combined_atlas_page_count'] = int(next(
            line.split(':', 1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))

    (output / 'LandscapeCombinationClassCityBubbles.json').write_text(
        json.dumps(bubble_metadata, ensure_ascii=False, indent=2) + '\n')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'class_count': len(results), 'cities': len(CITY_ORDER),
                      'grid_shape_per_city': [GRID_SIDE, GRID_SIDE], 'pdf': str(pdf_path),
                      'combined_atlas_pages': manifest.get('combined_atlas_page_count') if append_to_main else None},
                     ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--append-to-main', action='store_true')
    args = parser.parse_args()
    render(args.stage, args.output, append_to_main=args.append_to_main)
