"""Mine within-city SAE-response combinations and compare their city support.

The input is the frozen, globally calibrated semantic-unit assignment from
composition_patterns.py. A match is panorama-level co-response, not a verified
spatial arrangement or a human-confirmed landscape prototype.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .candidate_combinations import ELEMENTS
from .composition_patterns import pattern_mask
from .extract import write_json
from .followup import FOLLOWUP


STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
FRAME = FOLLOWUP / 'stages/candidate_combinations/panorama_elements.parquet'
OUTPUT = Path(__file__).resolve().parents[1] / 'results/city_internal_compositions'


def split_blocks(frame: pd.DataFrame) -> np.ndarray:
    """Hold out entire 5-km blocks, deterministically within each city."""
    blocks = zip(frame.city, frame.grid_x // 5, frame.grid_y // 5)
    return np.fromiter((int.from_bytes(hashlib.blake2b(
        f'{city}/{x}/{y}'.encode(), digest_size=8).digest(), 'little') % 3 == 0
        for city, x, y in blocks), dtype=bool, count=len(frame))


def bitmask(mask: np.ndarray) -> int:
    return int.from_bytes(np.packbits(mask, bitorder='little').tobytes(), 'little')


def unmask(bits: int, size: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bits.to_bytes((size + 7) // 8, 'little'),
                                      dtype=np.uint8), bitorder='little', count=size).astype(bool)


def city_design(frame: pd.DataFrame, holdout: np.ndarray):
    cells = pd.MultiIndex.from_frame(frame[['grid_x', 'grid_y']])
    cell_id, unique_cells = pd.factorize(cells, sort=True)
    cell_sizes = np.bincount(cell_id)
    weight = 1 / (len(unique_cells) * cell_sizes[cell_id])
    split_cells = [len(np.unique(cell_id[~holdout])), len(np.unique(cell_id[holdout]))]
    if min(split_cells) == 0:
        raise ValueError(f'No spatial holdout for {frame.city.iloc[0]}')
    split_weight = [weight * len(unique_cells) / split_cells[0],
                    weight * len(unique_cells) / split_cells[1]]
    return cell_id, weight, split_weight, split_cells


def measure(bits: int, size: int, cell_id: np.ndarray, weights: np.ndarray):
    positive = unmask(bits, size)
    return {'panoramas': int(bits.bit_count()),
            'cells': int(np.unique(cell_id[positive]).size),
            'cell_balanced_rate': float(weights[positive].sum())}


def eligible(stats: dict, min_rate: float, min_panoramas: int, min_cells: int):
    return (stats['cell_balanced_rate'] >= min_rate and
            stats['panoramas'] >= min_panoramas and stats['cells'] >= min_cells)


def mine_city(assignment: np.ndarray, frame: pd.DataFrame, units: list[dict],
              holdout: np.ndarray, min_rate: float = .005, max_order: int = 4,
              min_panoramas: int = 20, min_cells: int = 8) -> list[dict]:
    n = len(frame)
    cell_id, weights, split_weights, split_cells = city_design(frame, holdout)
    discovery = ~holdout
    validation = holdout
    discovery_bits, validation_bits = bitmask(discovery), bitmask(validation)
    family_columns = {family: index for index, family in enumerate(ELEMENTS)}
    item_bits = {u['id']: bitmask(assignment[:, family_columns[u['family']]] == u['id'])
                 for u in units if u.get('landscape_eligible', True) and not u['fallback']}
    level = {}
    for item, bits in item_bits.items():
        selected = bits & discovery_bits
        if selected.bit_count() < min_panoramas:
            continue
        stats = measure(selected, n, cell_id, split_weights[0])
        if eligible(stats, min_rate, min_panoramas, min_cells):
            level[(item,)] = selected

    found = []
    for order in range(2, max_order + 1):
        groups = defaultdict(list)
        for key in sorted(level):
            groups[key[:-1]].append(key)
        next_level = {}
        for members in groups.values():
            for first, second in itertools.combinations(members, 2):
                key = first + (second[-1],)
                if len({units[item]['family'] for item in key}) != order:
                    continue
                if any(key[:i] + key[i + 1:] not in level for i in range(order)):
                    continue
                selected = level[first] & level[second]
                if selected.bit_count() < min_panoramas:
                    continue
                train = measure(selected, n, cell_id, split_weights[0])
                if not eligible(train, min_rate, min_panoramas, min_cells):
                    continue
                next_level[key] = selected
                test_bits = validation_bits
                full_bits = item_bits[key[0]]
                for item in key[1:]:
                    full_bits &= item_bits[item]
                for item in key:
                    test_bits &= item_bits[item]
                test = measure(test_bits, n, cell_id, split_weights[1])
                full = measure(full_bits, n, cell_id, weights)
                label = ' ＋ '.join(units[item]['label'] for item in key)
                unit_key = '+'.join(units[item]['key'] for item in key)
                found.append({'city': frame.city.iloc[0],
                              'pattern_id': hashlib.sha256(unit_key.encode()).hexdigest()[:12],
                              'label': label, 'unit_ids': list(key), 'unit_key': unit_key,
                              'family_signature': '+'.join(sorted(units[item]['family'] for item in key)),
                              'order': order,
                              'discovery_panoramas': train['panoramas'],
                              'discovery_cells': train['cells'],
                              'discovery_rate': train['cell_balanced_rate'],
                              'holdout_panoramas': test['panoramas'],
                              'holdout_cells': test['cells'],
                              'holdout_rate': test['cell_balanced_rate'],
                              'panoramas': full['panoramas'], 'cells': full['cells'],
                              'city_rate': full['cell_balanced_rate'],
                              'replicated': eligible(test, min_rate, max(10, min_panoramas // 2),
                                                     max(4, min_cells // 2))})
        level = next_level
    return found


def compare_cities(rows: pd.DataFrame, cities: list[str]):
    kept = rows[rows.replicated].copy()
    sets = {city: set(kept.loc[kept.city == city, 'pattern_id']) for city in cities}
    support = kept.groupby('pattern_id').city.nunique().to_dict()
    rows['cities_replicated'] = rows.pattern_id.map(support).fillna(0).astype(int)
    rows['cross_city_status'] = np.where(~rows.replicated, 'unreplicated',
                                  np.where(rows.cities_replicated == 1,
                                           'single_city_detected', 'shared'))
    overlap = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for left in cities:
        for right in cities:
            union = sets[left] | sets[right]
            overlap.loc[left, right] = len(sets[left] & sets[right]) / len(union) if union else 0.
    return rows, kept, overlap


def cross_city_rates(assignment: np.ndarray, frame: pd.DataFrame,
                     units: list[dict], kept: pd.DataFrame) -> pd.DataFrame:
    """Measure every retained exact pattern in every city, including below-cutoff hits."""
    city_id, cities = pd.factorize(frame.city, sort=True)
    cell_key = pd.MultiIndex.from_frame(frame[['city', 'grid_x', 'grid_y']])
    cell_id, cells = pd.factorize(cell_key, sort=True)
    cell_city = pd.factorize(cells.get_level_values(0), sort=True)[0]
    cells_per_city = np.bincount(cell_city, minlength=len(cities))
    cell_size = np.bincount(cell_id)
    row_weight = 1 / (cells_per_city[city_id] * cell_size[cell_id])
    patterns = kept.drop_duplicates('pattern_id')
    results = []
    for row in patterns.itertuples():
        positive = pattern_mask(assignment, units, row.unit_ids)
        counts = np.bincount(city_id[positive], minlength=len(cities))
        rates = np.bincount(city_id[positive], weights=row_weight[positive],
                            minlength=len(cities))
        active_cells = np.unique(cell_id[positive])
        cell_counts = np.bincount(cell_city[active_cells], minlength=len(cities))
        results.extend({'pattern_id': row.pattern_id, 'label': row.label,
                        'city': city, 'panoramas': int(counts[index]),
                        'cells': int(cell_counts[index]),
                        'cell_balanced_rate': float(rates[index])}
                       for index, city in enumerate(cities))
    return pd.DataFrame(results)


def parent_overlap(kept: pd.DataFrame, cities: list[str]) -> pd.DataFrame:
    signatures = {city: set(kept.loc[kept.city == city, 'family_signature'])
                  for city in cities}
    return pd.DataFrame({left: {right: (
        len(signatures[left] & signatures[right]) /
        len(signatures[left] | signatures[right])
        if signatures[left] | signatures[right] else 0.)
        for right in cities} for left in cities}).loc[cities, cities]


def plot_overview(rows: pd.DataFrame, overlap: pd.DataFrame, path: Path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    cities = list(overlap.index)
    labels = [city.split('/')[-1] for city in cities]
    totals = rows[rows.replicated].groupby(['city', 'cross_city_status']).size().unstack(fill_value=0)
    totals = totals.reindex(cities, fill_value=0)
    shared = totals.get('shared', pd.Series(0, index=cities))
    single = totals.get('single_city_detected', pd.Series(0, index=cities))
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={'width_ratios': [1, 1.25]})
    y = np.arange(len(cities))
    ax.barh(y, shared, color='#397d76', label='Detected in multiple cities')
    ax.barh(y, single, left=shared, color='#b26e4e', label='Detected in one city')
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Spatially replicated within-city combinations')
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax.set_title('a  Within-city candidates', loc='left', fontweight='bold')
    image = heat.imshow(overlap.to_numpy(), vmin=0, vmax=1, cmap='YlGnBu')
    heat.set_xticks(y, labels, rotation=90, fontsize=6)
    heat.set_yticks(y, labels, fontsize=6)
    heat.set_title('b  Exact subtype overlap (Jaccard)', loc='left', fontweight='bold')
    fig.colorbar(image, ax=heat, shrink=.55, label='Jaccard index')
    fig.suptitle('City-internal SAE-response combinations', x=.06, ha='left', fontsize=14)
    fig.text(.06, .012, 'One globally calibrated unit per semantic family; 5-km-block holdout. '
             'Single-city detection is sample-specific and does not prove exclusivity.', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .96))
    fig.savefig(path, dpi=200, facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)


def run(stage: Path, frame_path: Path, output: Path, min_rate: float = .005,
        max_order: int = 4, min_panoramas: int = 20, min_cells: int = 8):
    units = json.loads((stage / 'units.json').read_text())
    assignment = np.load(stage / 'assignment.npy', mmap_mode='r')
    frame = pd.read_parquet(frame_path, columns=['city', 'panoid', 'grid_x', 'grid_y'])
    if len(frame) != len(assignment) or frame.duplicated(['city', 'panoid']).any():
        raise ValueError('Assignment and panorama metadata do not align')
    stage_summary = json.loads((stage / 'summary.json').read_text())
    if len(frame) != stage_summary['panoramas'] or frame.city.nunique() != stage_summary['cities']:
        raise ValueError('Input counts differ from screened composition stage')
    if [u['id'] for u in units] != list(range(len(units))):
        raise ValueError('Unit IDs must align with assignment values')

    holdout = split_blocks(frame)
    city_rows, split_rows = [], []
    for city, indices in frame.groupby('city', sort=True).indices.items():
        city_frame = frame.iloc[indices]
        city_holdout = holdout[indices]
        if not city_holdout.any() or city_holdout.all():
            raise ValueError(f'No discovery/holdout split for {city}')
        _, _, _, split_cells = city_design(city_frame, city_holdout)
        split_rows.append({'city': city, 'panoramas': len(indices),
                           'cells': sum(split_cells), 'discovery_cells': split_cells[0],
                           'holdout_cells': split_cells[1]})
        found = mine_city(assignment[indices], city_frame, units, city_holdout,
                          min_rate, max_order, min_panoramas, min_cells)
        city_rows.extend(found)
        print(f'{city}: discovered={len(found)} replicated={sum(row["replicated"] for row in found)}',
              flush=True)
    if not city_rows:
        raise ValueError('No city-internal combinations passed discovery rules')
    rows = pd.DataFrame(city_rows)
    cities = sorted(frame.city.unique())
    rows, kept, overlap = compare_cities(rows, cities)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / 'city_patterns.csv', index=False)
    overlap.rename_axis('city').to_csv(output / 'city_overlap_jaccard.csv')
    family_overlap = parent_overlap(kept, cities)
    family_overlap.rename_axis('city').to_csv(output / 'city_parent_overlap_jaccard.csv')
    pd.DataFrame(split_rows).to_csv(output / 'city_sampling.csv', index=False)
    city_summary = kept.groupby('city').agg(
        replicated_patterns=('pattern_id', 'size'),
        two_element=('order', lambda s: int((s == 2).sum())),
        three_element=('order', lambda s: int((s == 3).sum())),
        four_element=('order', lambda s: int((s == 4).sum())),
        parent_signatures=('family_signature', 'nunique')).reindex(cities, fill_value=0)
    city_summary['single_city_detected'] = rows[rows.cross_city_status == 'single_city_detected'].groupby('city').size().reindex(cities, fill_value=0)
    city_summary.rename_axis('city').to_csv(output / 'city_summary.csv')
    shared = kept.groupby(['pattern_id', 'label', 'family_signature', 'order']).agg(
        cities=('city', 'nunique'), panoramas=('panoramas', 'sum'),
        city_names=('city', lambda s: ' | '.join(sorted(s)))).reset_index()
    shared = shared.sort_values(['cities', 'panoramas'], ascending=False)
    shared.to_csv(output / 'pattern_city_support.csv', index=False)
    rates = cross_city_rates(assignment, frame, units, kept)
    rates.to_csv(output / 'all_city_pattern_rates.csv', index=False)
    single = rows[rows.cross_city_status == 'single_city_detected'].copy()
    if not single.empty:
        rate_lookup = rates.pivot(index='pattern_id', columns='city', values='cell_balanced_rate')
        single['highest_other_city_rate'] = [float(rate_lookup.loc[row.pattern_id].drop(row.city).max())
                                             for row in single.itertuples()]
        single['ratio_to_highest_other_city'] = single.city_rate / single.highest_other_city_rate.replace(0, np.nan)
        single = single.sort_values(['ratio_to_highest_other_city', 'city_rate'], ascending=False)
    single.to_csv(output / 'single_city_candidates.csv', index=False)
    plot_overview(rows, overlap, output / 'city_pattern_overlap.png')
    summary = {'source_run_identity': stage_summary['run_identity'],
               'cities': len(cities), 'sampled_panoramas': len(frame),
               'discovered_city_pattern_occurrences': len(rows),
               'replicated_city_pattern_occurrences': len(kept),
               'distinct_replicated_patterns': int(kept.pattern_id.nunique()),
               'shared_in_2plus_cities': int((shared.cities >= 2).sum()),
               'single_city_detected': int((shared.cities == 1).sum()),
               'single_city_with_no_response_elsewhere': int((single.highest_other_city_rate == 0).sum()),
               'parent_signatures': int(kept.family_signature.nunique()),
               'min_rate_per_split': min_rate, 'max_order': max_order,
               'minimum_discovery_panoramas': min_panoramas,
               'minimum_discovery_cells': min_cells,
               'minimum_holdout_panoramas': max(10, min_panoramas // 2),
               'minimum_holdout_cells': max(4, min_cells // 2),
               'spatial_holdout': 'Hash of city and 5-km grid block modulo 3; one third held out',
               'interpretation': 'Exploratory panorama-level SAE co-response. Globally calibrated semantic units; one subtype per family. No verified spatial arrangement, object count, or significance claim. Single-city means only this sample and rule.',
               'files': ['city_patterns.csv', 'city_overlap_jaccard.csv',
                         'city_parent_overlap_jaccard.csv', 'city_sampling.csv',
                         'city_summary.csv', 'pattern_city_support.csv',
                         'all_city_pattern_rates.csv', 'single_city_candidates.csv',
                         'city_pattern_overlap.png', 'city_pattern_overlap.pdf']}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--frame', type=Path, default=FRAME)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--min-rate', type=float, default=.005)
    parser.add_argument('--max-order', type=int, default=4)
    parser.add_argument('--min-panoramas', type=int, default=20)
    parser.add_argument('--min-cells', type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.min_rate < 1 or not 2 <= args.max_order <= 4:
        parser.error('Invalid support or order')
    run(args.stage, args.frame, args.output, args.min_rate, args.max_order,
        args.min_panoramas, args.min_cells)
