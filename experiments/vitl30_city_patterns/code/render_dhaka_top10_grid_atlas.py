"""Map Dhaka's ten most prevalent replicated combinations in 1-km cells.

Each map cell is one UTM kilometre square. Color represents the fraction of
sampled four-view panoramas in that cell matching a panorama-level combination.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from .composition_patterns import pattern_mask
from .followup import FOLLOWUP
from .semantic_city_figures import UNIT_NAMES


ROOT = Path(__file__).resolve().parents[1]
CITY = 'Bangladesh/Dhaka'
RESULTS = ROOT / 'results/city_internal_compositions'
STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
FRAME = FOLLOWUP / 'stages/candidate_combinations/panorama_elements.parquet'
OUTPUT = ROOT / 'results/chapter5_revision/dhaka_top10_1km_atlas'
STEM = 'Fig_ViTL30_Dhaka_Top10_Combinations_1km'


def load_top_ten():
    table = pd.read_csv(RESULTS / 'city_patterns.csv')
    selected = table.loc[(table.city == CITY) & table.replicated].copy()
    selected = selected.sort_values(['city_rate', 'pattern_id'],
                                    ascending=[False, True]).head(10).reset_index(drop=True)
    if len(selected) != 10 or selected.pattern_id.duplicated().any():
        raise ValueError('Expected ten distinct replicated Dhaka combinations')
    units = json.loads((STAGE / 'units.json').read_text())
    summary = json.loads((STAGE / 'summary.json').read_text())
    metadata = pd.read_parquet(FRAME, columns=['city', 'grid_x', 'grid_y'])
    assignment = np.load(STAGE / 'assignment.npy', mmap_mode='r')
    if len(metadata) != len(assignment) or len(metadata) != summary['panoramas']:
        raise ValueError('Panorama metadata and SAE assignments are misaligned')
    city_positions = np.flatnonzero(metadata.city.to_numpy() == CITY)
    city_metadata = metadata.iloc[city_positions].reset_index(drop=True)
    city_assignment = assignment[city_positions]
    cell_index, index = pd.factorize(
        pd.MultiIndex.from_frame(city_metadata[['grid_x', 'grid_y']]), sort=True)
    cells = index.to_frame(index=False)
    cells.columns = ['grid_x', 'grid_y']
    cells['sampled_panoramas'] = np.bincount(cell_index, minlength=len(cells))
    if len(city_metadata) != 93400 or len(cells) != 9587:
        raise ValueError('Dhaka panorama or observed-cell denominator changed')
    return selected, units, cell_index, cells, city_assignment, summary


def compute_rates(selected, units, cell_index, cells, city_assignment):
    records, long_rows = [], []
    denominator = cells.sampled_panoramas.to_numpy()
    for rank, row in enumerate(selected.itertuples(), 1):
        unit_ids = json.loads(row.unit_ids)
        match = pattern_mask(city_assignment, units, unit_ids)
        counts = np.bincount(cell_index[match], minlength=len(cells))
        rates = counts / denominator
        balanced = float(rates.mean())
        if (int(match.sum()) != row.panoramas or
                int((counts > 0).sum()) != row.cells or
                not np.isclose(balanced, row.city_rate, atol=1e-10)):
            raise ValueError(f'Saved city-internal support changed: {row.pattern_id}')
        labels = [units[unit_id]['label'] for unit_id in unit_ids]
        unknown = [label for label in labels if label not in UNIT_NAMES]
        if unknown:
            raise ValueError(f'No English label for {unknown}')
        english = ' + '.join(UNIT_NAMES[label] for label in labels)
        records.append({'rank': rank, 'pattern_id': row.pattern_id,
                        'label_zh': row.label, 'label_en': english,
                        'unit_ids': unit_ids, 'order': int(row.order),
                        'matching_panoramas': int(match.sum()),
                        'detected_cells': int((counts > 0).sum()),
                        'observed_cells': len(cells),
                        'cell_balanced_rate': balanced,
                        'cities_replicated': int(row.cities_replicated),
                        'rates': rates})
        long_rows.append(pd.DataFrame({
            'rank': rank, 'pattern_id': row.pattern_id,
            'grid_x': cells.grid_x, 'grid_y': cells.grid_y,
            'matching_panoramas': counts,
            'sampled_panoramas': denominator,
            'cell_match_fraction': rates,
        }))
    return records, pd.concat(long_rows, ignore_index=True)


def cell_image(cells, values, x_min, y_min, width, height):
    image = np.full((height, width), np.nan, dtype=np.float32)
    image[cells.grid_y.to_numpy() - y_min,
          cells.grid_x.to_numpy() - x_min] = values
    return image


def render(records, cells):
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 24,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    x_min, x_max = int(cells.grid_x.min()), int(cells.grid_x.max())
    y_min, y_max = int(cells.grid_y.min()), int(cells.grid_y.max())
    width, height = x_max - x_min + 1, y_max - y_min + 1
    extent = (x_min - .5, x_max + .5, y_min - .5, y_max + .5)
    background = cell_image(cells, np.ones(len(cells)),
                            x_min, y_min, width, height)
    background_cmap = matplotlib.colors.ListedColormap(['#E1E5E4'])
    background_cmap.set_bad((1, 1, 1, 0))
    active_cmap = LinearSegmentedColormap.from_list(
        'dhaka_match_share', ['#DFEEE9', '#91C2B5', '#469184', '#1A5C64', '#173A51'])
    active_cmap.set_bad((1, 1, 1, 0))
    norm = Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(2, 5, figsize=(25, 12.2), facecolor='white')
    fig.subplots_adjust(left=.027, right=.985, top=.855, bottom=.135,
                        wspace=.075, hspace=.31)
    for ax, record in zip(axes.flat, records):
        active = np.where(record['rates'] > 0, record['rates'], np.nan)
        response = cell_image(cells, active, x_min, y_min, width, height)
        ax.imshow(background, cmap=background_cmap, vmin=0, vmax=1,
                  origin='lower', interpolation='nearest', extent=extent,
                  rasterized=True)
        ax.imshow(response, cmap=active_cmap, norm=norm,
                  origin='lower', interpolation='nearest', extent=extent,
                  rasterized=True)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_title(f"{record['rank']:02d}  {record['label_en']}",
                     loc='left', fontsize=25, fontweight='bold',
                     color='#273941', pad=26)
        ax.text(0, 1.015,
                f"{record['cell_balanced_rate']:.1%} city mean   |   "
                f"{record['detected_cells']:,} / {record['observed_cells']:,} cells",
                transform=ax.transAxes, ha='left', va='bottom', fontsize=17,
                color='#5B686B')

    # Every panel has the same UTM extent, so a single scale bar applies to all.
    first = axes[0, 0]
    scale_x, scale_y = x_min + 7, y_min + 9
    first.plot([scale_x, scale_x + 20], [scale_y, scale_y],
               color='#26363B', lw=1.6, solid_capstyle='butt')
    for x in (scale_x, scale_x + 20):
        first.plot([x, x], [scale_y - 1.2, scale_y + 1.2],
                   color='#26363B', lw=1)
    first.text(scale_x + 10, scale_y + 3, '20 km', ha='center', va='bottom',
               fontsize=19, color='#26363B')
    first.text(x_max - 7, y_max - 10, 'N ↑', ha='right', va='top',
               fontsize=20, fontweight='bold', color='#26363B')

    fig.text(.03, .97, 'Dhaka | ten most prevalent landscape combinations',
             ha='left', va='top', fontsize=36, fontweight='bold', color='#273941')
    fig.text(.03, .93,
             f'{len(cells):,} observed 1 × 1 km cells · 93,400 sampled four-view panoramas · '
             'same spatial extent in every panel',
             ha='left', va='top', fontsize=22, color='#5B686B')

    color_ax = fig.add_axes((.385, .095, .23, .016))
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=active_cmap),
                          cax=color_ax, orientation='horizontal')
    colorbar.set_ticks([0, .25, .5, .75, 1])
    colorbar.set_ticklabels(['0', '25', '50', '75', '100%'])
    colorbar.ax.tick_params(labelsize=18)
    colorbar.outline.set_visible(False)
    fig.text(.5, .126, 'Matching panoramas / sampled panoramas in an observed cell',
             ha='center', va='bottom', fontsize=18, color='#37474C')
    fig.text(.5, .018,
             'Grey = observed cell with no match; white = no sampled panorama. Panels may overlap because one panorama can match multiple combinations.\n'
             'SAE co-response does not verify object adjacency or road side.',
             ha='center', va='bottom', fontsize=14, color='#647073')

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{STEM}.{extension}', dpi=260,
                    bbox_inches='tight', pad_inches=.09, facecolor='white')
    plt.close(fig)


def main():
    selected, units, cell_index, cells, city_assignment, summary = load_top_ten()
    records, grid_rates = compute_rates(selected, units, cell_index,
                                        cells, city_assignment)
    render(records, cells)
    grid_rates.to_csv(OUTPUT / 'top10_cell_match_rates.csv', index=False)
    result = [{key: value for key, value in record.items() if key != 'rates'}
              for record in records]
    pd.DataFrame(result).to_csv(OUTPUT / 'top10_combinations.csv', index=False)
    (OUTPUT / 'manifest.json').write_text(json.dumps({
        'city': CITY, 'grid_meters': 1000,
        'selection': 'Top 10 Dhaka-replicated city-internal patterns by cell-balanced rate',
        'source_run_identity': summary['run_identity'],
        'panoramas': len(city_assignment), 'observed_cells': len(cells),
        'rate_definition': 'Mean over observed 1-km cells of matching sampled panoramas divided by sampled panoramas in that cell',
        'interpretation': 'Exploratory panorama-level SAE co-response; not verified spatial arrangement',
        'patterns': result,
    }, indent=2, ensure_ascii=False) + '\n')
    print(OUTPUT / f'{STEM}.png')


if __name__ == '__main__':
    main()
