"""Draw six-city 2.5-km proportional pie maps from screened SAE compositions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Wedge
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from .composition_patterns import pattern_mask
from .followup import FOLLOWUP
from .semantic_city_figures import FIGURES, PARENTS, load
from sae_experiments.panorama_vitl30_figures import name


STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
OUTPUT = FIGURES / 'six_city_2p5km_pie_bubbles'
CITY_ORDER = [
    'Argentina/BuenosAires', 'Australia/Melbourne', 'Bangladesh/Dhaka',
    'Portugal/Lisbon', 'SouthAfrica/CapeTown', 'US/Chicago',
]
MAX_DISPLAY_FAMILIES = len(PARENTS)
MAX_RADIUS_POINTS = 2.55
FAMILY_COLORS = {
    'building': '#327f9d', 'tree': '#438f45', 'low_vegetation': '#91aa35',
    'road': '#bd5c3e', 'pedestrian': '#ca8b53', 'barrier': '#8687b5',
    'water': '#4eacb2', 'terrain': '#a37767', 'transport_structure': '#577bb0',
    'utility': '#d1a344', 'sign': '#a86688', 'furniture': '#808385',
    'construction': '#b96857', 'other': '#b7bec0',
}


def aggregate(stage):
    summary, units, patterns, frame, _scores, assignment = load(stage)
    families = list(PARENTS)
    family_presence = np.zeros((len(frame), len(families)), dtype=bool)
    family_index = {family: i for i, family in enumerate(families)}
    for index, pattern in enumerate(patterns):
        matched = pattern_mask(assignment, units, pattern['units'])
        for family in {units[i]['family'] for i in pattern['units']}:
            if family in family_index:
                family_presence[:, family_index[family]] |= matched
        if index and index % 50 == 0:
            print(f'Pattern masks: {index:,}/{len(patterns):,}', flush=True)

    selected_mask = frame.city.isin(CITY_ORDER).to_numpy()
    selected = frame.loc[selected_mask, ['city', 'grid_x', 'grid_y']].reset_index(drop=True)
    family_presence = family_presence[selected_mask]

    # Assign each source 1-km grid by its UTM-bin centroid to a fixed 2.5-km grid.
    selected['grid25_x'] = np.floor((selected.grid_x.to_numpy() + .5) / 2.5).astype(np.int32)
    selected['grid25_y'] = np.floor((selected.grid_y.to_numpy() + .5) / 2.5).astype(np.int32)
    keys = pd.MultiIndex.from_frame(selected[['city', 'grid25_x', 'grid25_y']])
    bin_id, unique = pd.factorize(keys, sort=True)
    n_bins = len(unique)

    counts = np.column_stack([
        np.bincount(bin_id, weights=family_presence[:, i], minlength=n_bins)
        for i in range(len(families))
    ]).astype(np.int32)
    cells = pd.DataFrame(list(unique), columns=['city', 'grid25_x', 'grid25_y'])
    cells['sampled_panoramas'] = np.bincount(bin_id, minlength=n_bins)
    unique_source_cells = np.unique(np.column_stack([
        bin_id, selected.grid_x.to_numpy(), selected.grid_y.to_numpy()
    ]), axis=0)
    cells['source_1km_cells'] = np.bincount(unique_source_cells[:, 0], minlength=n_bins)
    cells['grid_center_x_km'] = cells.grid25_x * 2.5 + 1.25
    cells['grid_center_y_km'] = cells.grid25_y * 2.5 + 1.25

    for i, family in enumerate(families):
        cells[f'count_{family}'] = counts[:, i]
    cells['landscape_total'] = counts.sum(axis=1)
    for city in CITY_ORDER:
        city_mask = cells.city.to_numpy() == city
        city_total = counts[city_mask].sum(axis=0)
        cells.loc[city_mask, 'share_of_city_total'] = (
            cells.loc[city_mask, 'landscape_total'] /
            max(int(city_total.sum()), 1))
        for i, family in enumerate(families):
            total = int(city_total[i])
            count = cells.loc[city_mask, f'count_{family}'].to_numpy()
            cells.loc[city_mask, f'within_grid_share_{family}'] = np.divide(
                count, cells.loc[city_mask, 'landscape_total'].to_numpy(),
                out=np.zeros(len(count), dtype=float),
                where=cells.loc[city_mask, 'landscape_total'].to_numpy() > 0)
            cells.loc[city_mask, f'citywide_share_{family}'] = count / max(total, 1)

    city_counts = cells.groupby('city')[[f'count_{family}' for family in families]].sum()
    active_totals = city_counts.sum(axis=0)
    active_families = [f for f in families if active_totals[f'count_{f}'] > 0]
    active_families.sort(key=lambda f: (-int(active_totals[f'count_{f}']), f))
    main_families = active_families[:MAX_DISPLAY_FAMILIES]
    other_families = active_families[MAX_DISPLAY_FAMILIES:]

    display_types = list(main_families) + (['other'] if other_families else [])
    display_counts = []
    for family in main_families:
        display_counts.append(cells[f'count_{family}'].to_numpy(dtype=np.int32))
    if other_families:
        display_counts.append(cells[[f'count_{f}' for f in other_families]]
                              .sum(axis=1).to_numpy(dtype=np.int32))
    cells['landscape_total'] = np.column_stack(display_counts).sum(axis=1)
    return summary, cells, families, active_families, main_families, other_families, display_types


def type_citywide_shares(cells, city, families, display_types, other_families):
    group = cells[cells.city == city]
    total_by_family = {f: int(group[f'count_{f}'].sum()) for f in families}
    total = sum(total_by_family.values())
    result = {}
    for family in display_types:
        count = (sum(total_by_family[f] for f in other_families)
                 if family == 'other' else total_by_family[family])
        result[family] = count / total if total else 0.
    return result


def draw_city(cells, city, families, main_families, other_families, display_types,
              global_max, output_path=None, pdf=None, overview_ax=None):
    group = cells[cells.city == city].copy()
    min_x, max_x = int(group.grid25_x.min()), int(group.grid25_x.max())
    min_y, max_y = int(group.grid25_y.min()), int(group.grid25_y.max())
    x_span, y_span = (max_x - min_x + 1) * 2.5, (max_y - min_y + 1) * 2.5
    single = overview_ax is None
    if single:
        fig_width = float(np.clip(8.4 + 1.4 * x_span / y_span, 10.2, 13.33))
        fig = plt.figure(figsize=(fig_width, 9.0), facecolor='white')
        ax = fig.add_axes([.085, .275, .83, .59])
    else:
        fig, ax = overview_ax.figure, overview_ax

    x = (group.grid25_x.to_numpy() - min_x) * 2.5 + 1.25
    y = (group.grid25_y.to_numpy() - min_y) * 2.5 + 1.25
    ax.set_xlim(0, x_span)
    ax.set_ylim(0, y_span)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axisbelow(True)
    ax.grid(color='#e7ebec', linewidth=.45, zorder=0)
    ax.tick_params(labelsize=7 if single else 5.8, length=2.3, colors='#566164')
    x_step = max(5, int(np.ceil(x_span / 7 / 5) * 5))
    y_step = max(5, int(np.ceil(y_span / 7 / 5) * 5))
    ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.set_xlabel('Local easting (km)', fontsize=8 if single else 6.5, color='#3f4a4d')
    ax.set_ylabel('Local northing (km)', fontsize=8 if single else 6.5, color='#3f4a4d')
    for spine in ax.spines.values():
        spine.set_color('#aeb8bb')
        spine.set_linewidth(.65)
    fig.canvas.draw()
    origin = ax.transData.transform((0, 0))
    one_km = ax.transData.transform((1, 0))
    points_per_km = np.hypot(*(one_km - origin)) * 72 / fig.dpi

    cell_counts = []
    for family in main_families:
        cell_counts.append(group[f'count_{family}'].to_numpy(dtype=np.int32))
    if other_families:
        cell_counts.append(group[[f'count_{f}' for f in other_families]]
                           .sum(axis=1).to_numpy(dtype=np.int32))
    values = np.column_stack(cell_counts)
    totals = values.sum(axis=1)
    radii_points = MAX_RADIUS_POINTS * np.sqrt(totals / max(global_max, 1))
    radii_km = radii_points / points_per_km

    palette = {family: FAMILY_COLORS[family] for family in main_families}
    palette['other'] = FAMILY_COLORS['other']
    for index, family in enumerate(display_types):
        patches = []
        for row in np.flatnonzero((values[:, index] > 0) & (totals > 0)):
            theta = 90.
            for j in range(len(display_types)):
                value = values[row, j]
                if value <= 0:
                    continue
                delta = 360. * value / totals[row]
                if j == index:
                    patches.append(Wedge((x[row], y[row]), radii_km[row],
                                         theta, theta + delta))
                theta += delta
        if patches:
            ax.add_collection(PatchCollection(
                patches, facecolor=palette[family], edgecolor='white', linewidth=.15,
                rasterized=True, zorder=2))

    borders = [Circle((x[i], y[i]), radii_km[i]) for i in np.flatnonzero(totals > 0)]
    if borders:
        ax.add_collection(PatchCollection(
            borders, facecolor='none', edgecolor='white', linewidth=.2,
            rasterized=True, zorder=3))
    empty = totals == 0
    if empty.any():
        ax.scatter(x[empty], y[empty], s=3, facecolors='white', edgecolors='#737d80',
                   linewidths=.45, rasterized=True, zorder=4)

    stats = {
        'grid_cells': int(len(group)),
        'sampled_panoramas': int(group.sampled_panoramas.sum()),
        'source_1km_cells': int(group.source_1km_cells.sum()),
        'landscape_type_incidences': int(totals.sum()),
        'median_landscape_total_per_grid': float(np.median(totals)),
        'citywide_type_shares': type_citywide_shares(cells, city, families,
                                                     display_types, other_families),
    }
    if single:
        ax.set_title(f'{name(city)}  |  {stats["grid_cells"]:,} cells', loc='left',
                     fontsize=12, weight='bold', pad=8, color='#202729')
        fig.text(.5, .949,
                 f'2.5 x 2.5 km grid  |  {stats["sampled_panoramas"]:,} sampled panoramas  |  '
                 f'{stats["landscape_type_incidences"]:,} landscape-family incidences',
                 ha='center', va='top', fontsize=8.6, color='#4f5d60')

        type_handles = []
        for family in display_types:
            pct = stats['citywide_type_shares'][family] * 100
            label = 'Other landscape types' if family == 'other' else PARENTS[family]
            type_handles.append(Patch(facecolor=palette[family], edgecolor='none',
                                      label=f'{label}  ({pct:.1f}% citywide)'))
        fig.legend(handles=type_handles, loc='lower center', bbox_to_anchor=(.5, .04),
                   ncol=3, frameon=False, fontsize=7.7, handlelength=1.0,
                   handletextpad=.4, columnspacing=1.0, labelspacing=.4)

        quantiles = np.unique(np.maximum(1, np.rint(np.quantile(
            cells.landscape_total, [.25, .5, .75, 1])).astype(int)))
        size_handles = [Line2D(
            [0], [0], marker='o', linestyle='None',
            markersize=max(1.3, 2 * MAX_RADIUS_POINTS * np.sqrt(value / max(global_max, 1))),
            markerfacecolor='#657175', markeredgecolor='white', label=f'{value:,}')
            for value in quantiles]
        fig.legend(handles=size_handles,
                   title='Landscape-family incidences per cell (circle area)',
                   loc='lower center', bbox_to_anchor=(.5, .17), ncol=len(size_handles),
                   frameon=False, fontsize=7.8, title_fontsize=8.0,
                   handletextpad=.5, columnspacing=1.25)
        fig.text(.5, .012,
                 'Pie slices show within-cell type shares; legend percentages show each type share of the city total. '
                 'Counts are sampled-panorama family presences in recurrent SAE combinations, not physical object counts.',
                 ha='center', va='bottom', fontsize=7.2, color='#4f5d60', wrap=True)
        if output_path:
            fig.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.12)
        if pdf:
            pdf.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
    else:
        ax.set_title(f'{name(city)}  |  {stats["grid_cells"]:,} cells',
                     loc='left', fontsize=8.6, weight='bold', pad=5, color='#202729')
    return stats


def render(stage=STAGE, output=OUTPUT):
    summary, cells, families, active_families, main_families, other_families, display_types = aggregate(stage)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / 'SixCity_2p5km_PieCells.csv'
    pdf_path = output / 'SixCity_2p5km_PieAtlas.pdf'
    cells.to_csv(csv_path, index=False)
    global_max = int(cells.landscape_total.max())

    city_stats = {}
    with PdfPages(pdf_path) as pdf:
        for city in CITY_ORDER:
            output_path = output / f'City_{city.replace("/", "__")}_2p5km_PieMap.png'
            city_stats[city] = draw_city(
                cells, city, families, main_families, other_families,
                display_types, global_max, output_path, pdf)
            print(f'Rendered {name(city)}', flush=True)

    manifest = {
        'run_identity': summary['run_identity'],
        'cities': CITY_ORDER,
        'source_panorama_count': int(sum(city_stats[c]['sampled_panoramas'] for c in CITY_ORDER)),
        'grid_size_m': 2500,
        'grid_assignment': 'Assign each source 1-km UTM grid by its centroid to a fixed UTM-aligned 2.5-km grid.',
        'landscape_total_definition': 'Sum across active landscape families of sampled panoramas whose recurrent SAE-combination match includes that family; a panorama contributes at most once per family.',
        'bubble_area': 'Proportional to total family incidences; one common point-area scale is used across the six city maps.',
        'pie_slices': 'Within-grid share of the landscape-family incidences.',
        'citywide_type_share': 'Type legend percentages are each family incidence divided by all family incidences in that city.',
        'per_grid_citywide_contribution': 'In the CSV, citywide_share_<family> is the family incidence in a grid divided by that family\'s citywide total.',
        'displayed_types': [PARENTS[f] for f in main_families] + (['Other landscape types'] if other_families else []),
        'other_families': other_families,
        'active_families': active_families,
        'maximum_total_incidences': global_max,
        'city_statistics': city_stats,
        'limitations': [
            'Counts are panorama-level family presences from exploratory, non-human-validated SAE combinations, not object counts.',
            'Each source panorama is included once per family even if several patterns match it.',
            'The 2.5-km cells are approximated from the centroids of the sampled 1-km UTM cells.',
            'Citywide shares use the selected 10-panorama-per-1-km-cell sample and should not be read as exhaustive city inventories.',
        ],
        'files': [pdf_path.name, csv_path.name] + sorted(p.name for p in output.glob('City_*_2p5km_PieMap.png')),
    }
    (output / 'SixCity_2p5km_PieAtlas.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pdf': str(pdf_path), 'csv': str(csv_path),
                      'cities': city_stats, 'grid_cells': int(len(cells)),
                      'types': manifest['displayed_types']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    render(args.stage, args.output)


if __name__ == '__main__':
    main()
