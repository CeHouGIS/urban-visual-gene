"""Render six-city landscape-family maps on city-normalized 30 by 30 grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from .composition_patterns import pattern_mask
from .followup import FOLLOWUP
from .semantic_city_figures import FIGURES, PARENTS, load
from sae_experiments.panorama_vitl30_figures import name


OUTPUT = FIGURES / 'six_city_2p5km_category_maps'
STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
CITY_ORDER = [
    'Argentina/BuenosAires', 'Australia/Melbourne', 'Bangladesh/Dhaka',
    'Portugal/Lisbon', 'SouthAfrica/CapeTown', 'US/Chicago',
]
DETAIL_CITY_ORDER = [
    'Argentina/BuenosAires', 'Portugal/Lisbon', 'SouthAfrica/CapeTown',
]
MAX_MARKER_AREA = 40.0
GRID_SIDE = 30
FAMILY_COLORS = {
    'building': '#327f9d', 'tree': '#438f45', 'low_vegetation': '#91aa35',
    'road': '#bd5c3e', 'pedestrian': '#ca8b53', 'barrier': '#8687b5',
    'water': '#4eacb2', 'terrain': '#a37767', 'transport_structure': '#577bb0',
    'utility': '#d1a344', 'sign': '#a86688', 'furniture': '#808385',
    'construction': '#b96857',
}


def build_family_matrix(stage):
    summary, units, patterns, frame, _scores, assignment = load(stage)
    families = list(PARENTS)
    family_index = {family: index for index, family in enumerate(families)}
    presence = np.zeros((len(frame), len(families)), dtype=bool)
    for pattern_index, pattern in enumerate(patterns):
        matched = pattern_mask(assignment, units, pattern['units'])
        for family in {units[index]['family'] for index in pattern['units']}:
            if family in family_index:
                presence[:, family_index[family]] |= matched
        if pattern_index and pattern_index % 50 == 0:
            print(f'Pattern masks: {pattern_index:,}/{len(patterns):,}', flush=True)

    selected_mask = frame.city.isin(CITY_ORDER).to_numpy()
    selected = frame.loc[selected_mask, ['city', 'grid_x', 'grid_y']].reset_index(drop=True)
    return summary, selected, presence[selected_mask], families


def aggregate_grid(selected, presence, families, grid_size_km):
    source_x = selected.grid_x.to_numpy(dtype=np.int32)
    source_y = selected.grid_y.to_numpy(dtype=np.int32)
    if grid_size_km == 1.0:
        grid_x, grid_y = source_x, source_y
    else:
        grid_x = np.floor((source_x + .5) / grid_size_km).astype(np.int32)
        grid_y = np.floor((source_y + .5) / grid_size_km).astype(np.int32)

    keys = selected[['city']].copy()
    keys['grid_x'] = grid_x
    keys['grid_y'] = grid_y
    cell_id, unique = pd.factorize(pd.MultiIndex.from_frame(keys), sort=True)
    n_cells = len(unique)
    counts = np.column_stack([
        np.bincount(cell_id, weights=presence[:, index], minlength=n_cells)
        for index in range(len(families))
    ]).astype(np.int32)
    cells = pd.DataFrame(list(unique), columns=['city', 'grid_x', 'grid_y'])
    cells['sampled_panoramas'] = np.bincount(cell_id, minlength=n_cells)
    source_cell_rows = np.unique(np.column_stack([cell_id, source_x, source_y]), axis=0)
    cells['source_1km_cells'] = np.bincount(source_cell_rows[:, 0], minlength=n_cells)
    for index, family in enumerate(families):
        cells[f'count_{family}'] = counts[:, index]
    cells['landscape_total'] = counts.sum(axis=1)

    for city in CITY_ORDER:
        mask = cells.city.to_numpy() == city
        city_family_totals = counts[mask].sum(axis=0)
        total_all_families = int(city_family_totals.sum())
        cells.loc[mask, 'share_of_city_total'] = (
            cells.loc[mask, 'landscape_total'].to_numpy() / max(total_all_families, 1))
        for index, family in enumerate(families):
            count = cells.loc[mask, f'count_{family}'].to_numpy()
            cell_total = cells.loc[mask, 'landscape_total'].to_numpy()
            cells.loc[mask, f'within_grid_share_{family}'] = np.divide(
                count, cell_total, out=np.zeros(len(count), dtype=float), where=cell_total > 0)
            cells.loc[mask, f'citywide_share_{family}'] = (
                count / max(int(city_family_totals[index]), 1))

    active = [family for index, family in enumerate(families) if counts[:, index].sum() > 0]
    return cells, active


def aggregate_normalized_grid(selected, presence, families, grid_side=GRID_SIDE):
    """Aggregate panorama family incidences into a fixed-shape grid per city."""
    normalized = selected[['city']].copy()
    for source, target in [('grid_x', 'cell_x'), ('grid_y', 'cell_y')]:
        values = selected[source].to_numpy(dtype=float)
        bins = np.zeros(len(selected), dtype=np.int32)
        for city in CITY_ORDER:
            mask = selected.city.to_numpy() == city
            city_values = values[mask]
            lower = float(city_values.min())
            extent = float(city_values.max() - lower + 1)
            positions = (city_values + .5 - lower) / extent
            bins[mask] = np.clip(
                np.floor(positions * grid_side), 0, grid_side - 1).astype(np.int32)
        normalized[target] = bins

    cell_id, unique = pd.factorize(pd.MultiIndex.from_frame(normalized), sort=True)
    n_cells = len(unique)
    counts = np.column_stack([
        np.bincount(cell_id, weights=presence[:, index], minlength=n_cells)
        for index in range(len(families))
    ]).astype(np.int32)
    cells = pd.DataFrame(list(unique), columns=['city', 'cell_x', 'cell_y'])
    cells['sampled_panoramas'] = np.bincount(cell_id, minlength=n_cells)
    cells['source_1km_cells'] = 0
    source_cells = selected[['city', 'grid_x', 'grid_y']].copy()
    source_cells['cell_x'] = normalized.cell_x
    source_cells['cell_y'] = normalized.cell_y
    unique_source_cells = source_cells.drop_duplicates()
    source_cell_ids = pd.factorize(pd.MultiIndex.from_frame(
        unique_source_cells[['city', 'cell_x', 'cell_y']]), sort=True)[0]
    cells['source_1km_cells'] = np.bincount(source_cell_ids, minlength=n_cells)
    for index, family in enumerate(families):
        cells[f'count_{family}'] = counts[:, index]
        cells[f'share_{family}'] = np.divide(
            counts[:, index], cells.sampled_panoramas.to_numpy(),
            out=np.zeros(n_cells, dtype=float), where=cells.sampled_panoramas.to_numpy() > 0)
    cells['landscape_total'] = counts.sum(axis=1)
    active = [family for index, family in enumerate(families) if counts[:, index].sum() > 0]
    return cells, active


def draw_category(cells, family, output_path, pdf, cities, grid_side=None, grid_size_km=None):
    normalized = grid_side is not None
    value_column = f'share_{family}' if normalized else f'count_{family}'
    global_max = float(cells[value_column].max())
    if global_max <= 0:
        return None

    columns = min(3, len(cities))
    rows = int(np.ceil(len(cities) / columns))
    fig_height = 10.5 if len(cities) > 3 else 6.8
    fig = plt.figure(figsize=(16.5, fig_height), facecolor='white')
    grid_bottom = (.235 if normalized else .19) if len(cities) > 3 else .25
    grid_top = .865 if len(cities) > 3 else .79
    grid = fig.add_gridspec(
        rows, columns, left=.055, right=.985, top=grid_top, bottom=grid_bottom,
        wspace=.20, hspace=.30)
    category_name = PARENTS[family]
    fig.text(.5, .965, category_name, ha='center', va='top',
             fontsize=18, weight='bold', color='#202729')
    grid_label = (f'{grid_side} x {grid_side} normalized city grid' if normalized else
                  f'{grid_size_km:g} x {grid_size_km:g} km cells')
    metric_label = ('circle area = matching-panorama proportion (shared 0-100% scale)' if normalized else
                    'circle area = sampled panorama count')
    fig.text(.5, .927,
             f'Parent semantic family  |  {grid_label}  |  {len(cities)}-city comparison  |  {metric_label}',
             ha='center', va='top', fontsize=9.3, color='#4f5d60')

    city_stats = {}
    for panel, city in enumerate(cities):
        ax = fig.add_subplot(grid[panel // columns, panel % columns])
        group = cells[cells.city == city].copy()
        if normalized:
            x = group.cell_x.to_numpy(dtype=float) + .5
            y = group.cell_y.to_numpy(dtype=float) + .5
            x_span = y_span = float(grid_side)
        else:
            min_x, max_x = int(group.grid_x.min()), int(group.grid_x.max())
            min_y, max_y = int(group.grid_y.min()), int(group.grid_y.max())
            x = (group.grid_x.to_numpy() - min_x) * grid_size_km + grid_size_km / 2
            y = (group.grid_y.to_numpy() - min_y) * grid_size_km + grid_size_km / 2
            x_span = (max_x - min_x + 1) * grid_size_km
            y_span = (max_y - min_y + 1) * grid_size_km
        values = group[value_column].to_numpy(dtype=float if normalized else np.int32)
        samples = group.sampled_panoramas.to_numpy(dtype=np.int32)

        ax.scatter(x, y, s=5, c='#e1e6e7', linewidths=0, rasterized=True, zorder=1)
        active = values > 0
        if active.any():
            sizes = (MAX_MARKER_AREA * values[active] if normalized else
                     MAX_MARKER_AREA * values[active] / global_max)
            ax.scatter(x[active], y[active], s=sizes,
                       c=FAMILY_COLORS[family], edgecolors='white', linewidths=.25,
                       alpha=.92, rasterized=True, zorder=2)

        ax.set_xlim(0, x_span)
        ax.set_ylim(0, y_span)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axisbelow(True)
        if normalized:
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.grid(which='minor', color='#eef1f1', linewidth=.25, zorder=0)
            ax.grid(which='major', color='#e0e5e5', linewidth=.45, zorder=0)
        else:
            ax.grid(color='#e7ebec', linewidth=.4, zorder=0)
        if not normalized:
            x_step = max(10, int(np.ceil(x_span / 5 / 10) * 10))
            y_step = max(10, int(np.ceil(y_span / 5 / 10) * 10))
            ax.xaxis.set_major_locator(MultipleLocator(x_step))
            ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.tick_params(labelsize=6.8, length=2, colors='#566164')
        axis_x = 'Normalized east-west position' if normalized else 'Local easting (km)'
        axis_y = 'Normalized north-south position' if normalized else 'Local northing (km)'
        ax.set_xlabel(axis_x, fontsize=7.5, color='#3f4a4d')
        ax.set_ylabel(axis_y, fontsize=7.5, color='#3f4a4d')
        ax.set_title(f'{name(city)}  |  {len(group):,} cells', loc='left',
                     fontsize=10.3, weight='bold', pad=5, color='#202729')
        for spine in ax.spines.values():
            spine.set_color('#aeb8bb')
            spine.set_linewidth(.6)

        city_stats[city] = {
            'grid_cells': int(len(group)),
            'sampled_panoramas': int(samples.sum()),
            'cells_with_category': int(active.sum()),
            'category_panorama_incidences': int(group[f'count_{family}'].sum()),
        }
        if normalized:
            city_stats[city].update({
                'mean_category_share': float(np.average(values, weights=samples)),
                'median_category_share_across_observed_bins': float(np.median(values)),
                'maximum_category_share': float(values.max()),
            })
        else:
            city_stats[city].update({
                'median_incidences_per_cell': float(np.median(values)),
                'maximum_incidences_per_cell': int(values.max()),
            })

    if normalized:
        reference_shares = [.25, .5, .75, 1.]
        handles = [Line2D(
            [0], [0], marker='o', linestyle='None',
            markersize=np.sqrt(MAX_MARKER_AREA * value),
            markerfacecolor=FAMILY_COLORS[family], markeredgecolor='white',
            label=f'{value:.0%}') for value in reference_shares]
        legend_title = f'Sampled panoramas with {category_name.lower()} / all panoramas in cell'
    else:
        quantiles = np.unique(np.maximum(1, np.rint(
            np.quantile(cells[value_column], [.25, .5, .75, 1])).astype(int)))
        handles = [Line2D(
            [0], [0], marker='o', linestyle='None',
            markersize=max(1.6, np.sqrt(MAX_MARKER_AREA * value / global_max)),
            markerfacecolor=FAMILY_COLORS[family], markeredgecolor='white',
            label=f'{value:,}') for value in quantiles]
        legend_title = f'Sampled panoramas with {category_name.lower()} per cell (circle area)'
    legend_y = .105 if normalized else (.15 if len(cities) > 3 else .14)
    fig.legend(handles=handles,
               title=legend_title,
               loc='lower center', bbox_to_anchor=(.5, legend_y),
               ncol=len(handles), frameon=False, fontsize=8.1,
               title_fontsize=8.3, handletextpad=.5, columnspacing=1.25)
    note = ('Each city is independently mapped to 30 x 30 equal-width bins; physical bin widths vary by city and are not fixed kilometers. '
            'Circle area is proportional to the within-cell share of sampled panoramas matching this family. '
            'This is an SAE-response share, not physical object prevalence.' if normalized else
            'Colored circles count sampled panoramas with a recurrent SAE response in this family, not physical objects. '
            f'The same circle-area scale is used across all {len(cities)} cities on this page.')
    fig.text(.5, .035,
             'Categories group semantically related dimension labels into parent landscape families. '
             'Faint dots mark observed cells. ' + note,
             ha='center', va='bottom', fontsize=7.7, color='#4f5d60', wrap=True)

    fig.savefig(output_path, dpi=240, facecolor='white', bbox_inches='tight', pad_inches=.12)
    for target_pdf in pdf if isinstance(pdf, (list, tuple)) else (pdf,):
        target_pdf.savefig(fig, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return {
        'family': family,
        'category': category_name,
        'maximum_metric_value_across_cities': global_max,
        'metric': 'within-cell matching-panorama share' if normalized else 'matching-panorama count',
        'city_statistics': city_stats,
        'png': output_path.name,
    }


def render(stage=STAGE, output=OUTPUT):
    summary, selected, presence, families = build_family_matrix(stage)
    cells, active_families = aggregate_normalized_grid(selected, presence, families)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / 'SixCity_30x30_CategoryCells.csv'
    pdf_path = output / 'SixCity_30x30_CategoryAtlas.pdf'
    main_pdf_path = output / 'LandscapeCategoryAtlas.pdf'
    cells.to_csv(csv_path, index=False)

    detail_cells, detail_families = aggregate_grid(selected, presence, families, 1.0)
    detail_cells = detail_cells[detail_cells.city.isin(DETAIL_CITY_ORDER)].copy()
    detail_csv = output / 'ThreeCity_1km_CategoryCells.csv'
    detail_pdf = output / 'ThreeCity_1km_CategoryAtlas.pdf'
    detail_manifest_path = output / 'ThreeCity_1km_CategoryAtlas.json'
    detail_cells.to_csv(detail_csv, index=False)
    category_results = []
    detail_results = []
    detail_family_set = set(detail_families)
    with (PdfPages(pdf_path) as pdf,
          PdfPages(detail_pdf) as detail_only_pdf,
          PdfPages(main_pdf_path) as main_pdf):
        for family in active_families:
            output_path = output / f'Category_{family}_30x30_SixCity.png'
            result = draw_category(
                cells, family, output_path, [pdf, main_pdf],
                CITY_ORDER, grid_side=GRID_SIDE)
            if result is not None:
                category_results.append(result)
                print(f"Rendered {result['category']}", flush=True)
            if family in detail_family_set:
                output_path = output / f'Category_{family}_1km_ThreeCity.png'
                result = draw_category(
                    detail_cells, family, output_path, detail_only_pdf,
                    DETAIL_CITY_ORDER, grid_size_km=1.0)
                if result is not None:
                    detail_results.append(result)
                    print(f"Rendered 1-km detail: {result['category']}", flush=True)

    manifest = {
        'run_identity': summary['run_identity'],
        'cities': CITY_ORDER,
        'grid_shape_per_city': [GRID_SIDE, GRID_SIDE],
        'grid_definition': 'Each city extent is independently divided into 30 equal-width bins per axis from the observed 1-km source-cell centroids; physical bin widths differ by city and no fixed-km grid is used.',
        'combined_atlas': main_pdf_path.name,
        'city_count': len(CITY_ORDER),
        'category_count': len(category_results),
        'category_order': [item['family'] for item in category_results],
        'category_definition': 'Each displayed category is a parent landscape family that groups semantically related dimension labels in the current provisional hierarchy.',
        'cell_metric': 'Within-cell share of sampled panoramas with at least one recurrent SAE combination containing an eligible unit from that family; each panorama contributes at most once per category.',
        'bubble_area': 'Proportional to within-cell matching-panorama share; a common 0-100% area scale is used across all six cities and all category pages.',
        'proportion_definition': 'Category-matching sampled panoramas divided by all sampled panoramas assigned to the normalized bin.',
        'sampling': 'Based on the selected panorama sample and assigned to normalized bins by source 1-km cell centroid.',
        'limitations': [
            'Shares are panorama-level family responses, not physical object prevalence.',
            'Categories and recurrent SAE combinations remain exploratory and are not human-validated.',
        ],
        'categories': category_results,
        'files': [main_pdf_path.name, pdf_path.name, csv_path.name] + sorted(
            path.name for path in output.glob('Category_*_30x30_SixCity.png')),
        'higher_resolution_detail': {
            'cities': DETAIL_CITY_ORDER,
            'grid_size_m': 1000,
            'reason': 'Separate 1-km detail views for three cities; these are not included in the six-city main atlas.',
            'directory': '.',
            'cell_counts': {
                city: int((detail_cells.city == city).sum())
                for city in DETAIL_CITY_ORDER
            },
            'pdf': detail_pdf.name,
            'csv': detail_csv.name,
            'categories': [item['family'] for item in detail_results],
        },
    }
    manifest_path = output / 'SixCity_30x30_CategoryAtlas.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    detail_manifest = {
        'run_identity': summary['run_identity'],
        'cities': DETAIL_CITY_ORDER,
        'grid_size_m': 1000,
        'category_definition': manifest['category_definition'],
        'cell_metric': 'Count of sampled panoramas in each original 1-km cell with at least one recurrent SAE combination containing the category; each panorama contributes at most once per category.',
        'bubble_area': 'Proportional to category-specific panorama incidences per cell; a shared scale is used across the three cities on each category page.',
        'categories': detail_results,
        'files': [detail_pdf.name, detail_csv.name] + sorted(
            path.name for path in output.glob('Category_*_1km_ThreeCity.png')),
    }
    detail_manifest_path.write_text(
        json.dumps(detail_manifest, ensure_ascii=False, indent=2) + '\n')
    from .render_landscape_combination_classes import render as render_combination_classes
    render_combination_classes(stage=STAGE, output=output)
    from .render_landscape_combination_city_bubbles import render as render_combination_bubbles
    render_combination_bubbles(stage=STAGE, output=output, append_to_main=True)
    print(json.dumps({
        'pdf': str(pdf_path), 'combined_pdf': str(main_pdf_path),
        'csv': str(csv_path),
        'manifest': str(manifest_path), 'categories': len(category_results),
        'grid_cells': int(len(cells)),
        'detail_pdf': str(detail_pdf),
        'detail_manifest': str(detail_manifest_path),
        'detail_grid_cells': int(len(detail_cells)),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    render(args.stage, args.output)


if __name__ == '__main__':
    main()
