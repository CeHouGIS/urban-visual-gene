"""Chapter 5 figures from the current semantic units and recurrent compositions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from .candidate_combinations import ELEMENTS
from .composition_patterns import pattern_mask
from .followup import FOLLOWUP
from sae_experiments.panorama_vitl30_figures import name

ROOT = Path(__file__).resolve().parents[3]
THESIS = Path('/root/code/HKUST-Thesis')
FIGURES = ROOT / 'experiments/vitl30_city_patterns/results/figures'
PARENTS = {
    'building': 'Buildings', 'tree': 'Trees', 'low_vegetation': 'Low vegetation',
    'road': 'Roads', 'pedestrian': 'Pedestrian space', 'barrier': 'Barriers',
    'water': 'Water', 'terrain': 'Terrain', 'transport_structure': 'Transport structures',
    'utility': 'Utilities', 'sign': 'Signs', 'furniture': 'Street furniture',
    'construction': 'Fixed construction',
}
COLORS = ['#327f9d', '#438f45', '#91aa35', '#bd5c3e', '#ca8b53', '#8687b5',
          '#4eacb2', '#a37767', '#577bb0', '#d1a344', '#a86688', '#808385', '#b96857']
UNIT_NAMES = {
    '临时围栏': 'Temporary fences', '围墙': 'Walls', '围栏': 'Fences', '护栏': 'Guardrails',
    '木栅栏': 'Wooden fences', '混凝土护栏': 'Concrete barriers', '石质栏杆': 'Stone railings',
    '路障': 'Road barriers', '铁丝网围栏': 'Wire fences', '传统建筑': 'Traditional buildings',
    '住宅建筑': 'Residential buildings', '工业建筑': 'Industrial buildings', '建筑': 'Buildings',
    '教堂': 'Churches', '混凝土建筑': 'Concrete buildings', '温室': 'Greenhouses',
    '玻璃幕墙建筑': 'Glass facades', '瓦顶建筑': 'Tile-roof buildings', '石质建筑': 'Stone buildings',
    '红色建筑': 'Red buildings', '老屋': 'Old houses', '蓝色建筑': 'Blue buildings',
    '车库': 'Garages', '金属屋顶建筑': 'Metal-roof buildings', '高层建筑': 'High-rise buildings',
    '储罐': 'Storage tanks', '喷泉': 'Fountains', '垃圾桶': 'Street bins', '座椅': 'Seats',
    '篮球架': 'Basketball hoops', '花盆': 'Planters', '邮箱': 'Mailboxes', '雕塑': 'Sculptures',
    '仙人掌': 'Cacti', '灌木': 'Shrubs', '芦苇': 'Reeds', '花卉': 'Flowers', '苔藓': 'Moss',
    '草地': 'Grass', '龙舌兰': 'Agave', '人行横道': 'Crosswalks', '人行道': 'Sidewalks',
    '台阶与楼梯': 'Steps / stairs', '路缘石': 'Curbs', '道路': 'Roads',
    '交通信号灯': 'Traffic lights', '交通标志': 'Traffic signs', '商业招牌': 'Shop signs',
    '广告牌': 'Billboards', '山地': 'Hills / mountains', '岩石': 'Rocks', '沙地': 'Sand',
    '雪地': 'Snow', '人行天桥': 'Footbridges', '公交站台': 'Bus stops', '桥梁': 'Bridges',
    '车站站台': 'Station platforms', '铁路': 'Railways', '隧道': 'Tunnels', '树木': 'Trees',
    '消防栓': 'Fire hydrants', '电线': 'Overhead wires', '电线杆': 'Utility poles',
    '管道': 'Pipes', '路灯': 'Streetlights', '输电塔': 'Transmission towers',
    '配电箱': 'Utility boxes', '风力发电机': 'Wind turbines', '水体': 'Water',
}


def composition_label(pattern, units):
    return ' + '.join(PARENTS[units[i]['family']] + ' (mixed)'
                      if units[i]['fallback'] else UNIT_NAMES[units[i]['label']]
                      for i in pattern['units'])


def save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        fig.savefig(output / f'{stem}.{extension}', dpi=240, facecolor='white',
                    bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def load(folder):
    summary = json.loads((folder / 'summary.json').read_text())
    units = json.loads((folder / 'units.json').read_text())
    patterns = [r for r in json.loads((folder / 'patterns.json').read_text()) if r['recurrent']]
    identity = json.loads((folder / 'scores_identity.json').read_text())
    frame = pd.read_parquet(folder.parent / 'candidate_combinations/panorama_elements.parquet',
                            columns=['city', 'grid_x', 'grid_y'])
    scores = np.load(folder / 'scores.npy', mmap_mode='r')
    assignment = np.load(folder / 'assignment.npy', mmap_mode='r')
    if (len(frame) != summary['panoramas'] or scores.shape != (len(frame), identity['units'])
            or assignment.shape != (len(frame), len(ELEMENTS)) or len(patterns) != summary['recurrent_patterns']
            or identity['rows'] != len(frame)):
        raise ValueError('Incomplete or mismatched current composition run')
    return summary, units, patterns, frame, scores, assignment


def cell_composition(frame, scores, units):
    keys = pd.MultiIndex.from_frame(frame[['city', 'grid_x', 'grid_y']])
    cell_id, unique = pd.factorize(keys, sort=True)
    counts = np.bincount(cell_id)
    cells = pd.DataFrame(list(unique), columns=['city', 'grid_x', 'grid_y'])
    cells['sampled_panoramas'] = counts
    families = [f for f in ELEMENTS if f in PARENTS]
    family_values = []
    for family in families:
        ids = [u['id'] for u in units if not u['fallback']
               and u['landscape_eligible'] and u['family'] == family]
        family_values.append(np.asarray(scores[:, ids]).sum(axis=1, dtype=np.float64))
    parent = np.column_stack(family_values)
    excluded = np.asarray(scores[:, [u['id'] for u in units if not u['fallback']
                                   and not u['landscape_eligible']]]).sum(axis=1, dtype=np.float64)
    other = np.maximum(0, 1 - parent.sum(axis=1) - excluded)
    for i, family in enumerate(families):
        cells[family] = np.bincount(cell_id, weights=parent[:, i]) / counts
    cells['excluded'] = np.bincount(cell_id, weights=excluded) / counts
    cells['unclassified'] = np.bincount(cell_id, weights=other) / counts
    return cells, cell_id, families


def pipeline(output):
    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(12.8, 3.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    stages = [
        ('Four street views', '0 / 90 / 180 / 270 degrees', '#dce9e7'),
        ('Frozen DINOv3-L', '4 x 28 x 28 x 1024 tokens', '#dbe7f0'),
        ('Panorama context', '28 x 112 ring; 3 x 3 mixing', '#e5e5f1'),
        ('BatchTopK SAE', 'W = 4,096; k = 32', '#f2e7d5'),
        ('Landscape analysis', '22 classes; landscape patterns', '#e1edda'),
    ]
    width, gap = .174, .031
    for i, (title, subtitle, color) in enumerate(stages):
        x = .008 + i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, .39), width, .27, boxstyle='round,pad=0.012',
                                    facecolor=color, edgecolor='#65716e', linewidth=.8))
        ax.text(x + width / 2, .565, title, ha='center', va='center', fontsize=12, weight='bold')
        ax.text(x + width / 2, .465, subtitle, ha='center', va='center', fontsize=10.5)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + width + .006, .525),
                                         (x + width + gap - .003, .525),
                                         arrowstyle='-|>', mutation_scale=12, color='#424b4a'))
    ax.text(.5, .22, 'Frozen model inference across 681,645 complete panoramas in 30 cities',
            ha='center', fontsize=12, color='#475653')
    ax.text(.5, .11, 'Activation evidence and Qwen labels remain provisional; city summaries weight observed 1-km cells equally.',
            ha='center', fontsize=10.5, color='#5f6b68')
    save(fig, output, 'Fig_ViTL30_Pipeline_Current')


def city_figures(cells, families, output, labels=None, colors=None, residuals=None):
    labels = PARENTS if labels is None else labels
    colors = COLORS if colors is None else colors
    residuals = ([('excluded', 'Excluded mobile/temporary', '#bdc1c2'),
                  ('unclassified', 'Unclassified SAE mass', '#edf0ef')]
                 if residuals is None else residuals)
    cities = sorted(cells.city.unique())
    city = cells.groupby('city', sort=True)[families + [r[0] for r in residuals]].mean().loc[cities]
    city.to_csv(output / 'semantic_city_composition.csv')
    core = city[families].to_numpy()
    ordered = sorted(range(len(families)), key=lambda j: -core[:, j].mean())
    x = np.arange(len(cities))
    fig, ax = plt.subplots(figsize=(15.8, 7.2), layout='constrained')
    bottom = np.zeros(len(cities))
    for rank, j in enumerate(ordered):
        y = core[:, j] * 100
        ax.bar(x, y, bottom=bottom, color=colors[j], label=labels[families[j]], width=.84)
        bottom += y
    for column, label, color in residuals:
        y = city[column].to_numpy() * 100
        ax.bar(x, y, bottom=bottom, color=color, label=label, width=.84)
        bottom += y
    ax.set_xticks(x, [name(c) for c in cities], rotation=90, fontsize=8)
    ax.set(ylabel='Share of all positive SAE activation mass (%)', ylim=(0, 100))
    ax.legend(ncol=5 if len(families) > 16 else 4, loc='upper center',
              bbox_to_anchor=(.5, 1.29 if len(families) > 16 else 1.23),
              fontsize=7 if len(families) > 16 else 8, frameon=False)
    save(fig, output, 'Fig_ViTL30_Semantic_City_Composition')

    diff = 100 * (core - (core.sum(axis=0) - core) / (len(cities) - 1))
    limit = np.quantile(np.abs(diff), .98)
    fig, ax = plt.subplots(figsize=(12.5, 7), layout='constrained')
    im = ax.imshow(diff[:, ordered], cmap='RdBu_r', aspect='auto',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
    ax.set_xticks(range(len(families)), [labels[families[j]] for j in ordered],
                  rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cities)), [name(c) for c in cities], fontsize=8)
    fig.colorbar(im, ax=ax, label='Difference from other-city mean (percentage points)')
    save(fig, output, 'Fig_ViTL30_Semantic_City_Excess')

    fig, ax = plt.subplots(figsize=(9.5, 8), layout='constrained')
    im = ax.imshow(cosine_similarity(core), cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(30), [name(c) for c in cities], rotation=90, fontsize=7)
    ax.set_yticks(range(30), [name(c) for c in cities], fontsize=7)
    fig.colorbar(im, ax=ax, label='Cosine similarity of semantic-class mass')
    save(fig, output, 'Fig_ViTL30_Semantic_City_Similarity')

    joint = np.zeros((len(families), len(families)))
    marginal = np.zeros(len(families))
    spatial = []
    for city_name in cities:
        group = cells[cells.city == city_name]
        values = group[families].to_numpy()
        present = (values >= .01).astype(float)
        joint += present.T @ present / len(group) / len(cities)
        marginal += present.mean(axis=0) / len(cities)
        norm = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-20)
        lookup = {(int(r.grid_x), int(r.grid_y)): i for i, r in enumerate(group.itertuples())}
        pairs = [(i, lookup[n]) for (gx, gy), i in lookup.items()
                 for n in ((gx + 1, gy), (gx, gy + 1)) if n in lookup]
        global_similarity = (np.square(norm.sum(axis=0)).sum() - np.square(norm).sum()) / (len(norm) * (len(norm) - 1))
        local_similarity = np.mean([norm[i] @ norm[j] for i, j in pairs]) if pairs else np.nan
        spatial.append({'city': city_name, 'cells': len(group), 'adjacent_pairs': len(pairs),
                        'global_similarity': global_similarity, 'local_similarity': local_similarity})
    ppmi = np.maximum(np.log(np.maximum(joint, 1e-15) /
                             np.maximum(np.outer(marginal, marginal), 1e-15)), 0)
    ppmi[joint < .0005] = 0
    np.fill_diagonal(ppmi, 0)
    fig, ax = plt.subplots(figsize=(11, 9) if len(families) > 16 else (7.5, 6.8),
                           layout='constrained')
    im = ax.imshow(ppmi[np.ix_(ordered, ordered)], cmap='viridis', vmin=0)
    ordered_labels = [labels[families[j]] for j in ordered]
    ax.set_xticks(range(len(families)), ordered_labels, rotation=45, ha='right',
                  fontsize=7 if len(families) > 16 else 8)
    ax.set_yticks(range(len(families)), ordered_labels,
                  fontsize=7 if len(families) > 16 else 8)
    fig.colorbar(im, ax=ax, label='Cell-level PPMI')
    save(fig, output, 'Fig_ViTL30_Semantic_Cooccurrence')

    spatial = pd.DataFrame(spatial)
    spatial.to_csv(output / 'semantic_spatial_structure.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 6.5), layout='constrained')
    for row in spatial.itertuples():
        if row.adjacent_pairs < 20:
            continue
        ax.scatter(row.global_similarity, row.local_similarity, s=32, color='#327f9d')
        ax.annotate(name(row.city), (row.global_similarity, row.local_similarity),
                    xytext=(4, 3), textcoords='offset points', fontsize=7)
    ax.set(xlabel='Mean similarity among observed cells', ylabel='Mean adjacent-cell similarity')
    save(fig, output, 'Fig_ViTL30_Semantic_Spatial_Structure')

    import umap
    sampled = []
    for city_name in cities:
        group = cells[cells.city == city_name].copy()
        group['selection_hash'] = [hashlib.sha256(f'{city_name}/{x}/{y}'.encode()).hexdigest()
                                   for x, y in zip(group.grid_x, group.grid_y)]
        sampled.append(group.sort_values('selection_hash').head(100))
    sample = pd.concat(sampled, ignore_index=True)
    coordinates = umap.UMAP(n_neighbors=30, min_dist=.1, metric='cosine', random_state=42).fit_transform(sample[families].to_numpy())
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    palette = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)[:10]
    for i, city_name in enumerate(cities):
        mask = sample.city == city_name
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=.62,
                   color=palette[i], label=name(city_name), rasterized=True)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2, frameon=False)
    save(fig, output, 'Fig_ViTL30_Semantic_Cell_UMAP')
    return cities, core


def choose_patterns(patterns, units, rates):
    score = rates.groupby('pattern_id').cell_weighted_rate.mean().to_dict()
    chosen, seen = [], set()
    for pattern in sorted(patterns, key=lambda r: -score[r['pattern_id']]):
        signature = tuple(sorted(units[i]['family'] for i in pattern['units']))
        if signature in seen:
            continue
        seen.add(signature)
        chosen.append(pattern)
        if len(chosen) == 12:
            break
    if len(chosen) != 12:
        raise ValueError('Fewer than 12 distinct landscape compositions')
    return chosen


def pattern_figures(folder, units, patterns, frame, assignment, cell_id, cells, cities, output):
    rates = pd.read_csv(folder / 'city_distribution.csv')
    if len(rates) != len(patterns) * 30:
        raise ValueError('Incomplete 30-city pattern rates')
    selected = choose_patterns(patterns, units, rates)
    ids = [p['pattern_id'] for p in selected]
    labels = [' + '.join(PARENTS[units[i]['family']] for i in p['units']) for p in selected]
    pd.DataFrame([{'display_id': f'P{i+1:02d}', 'pattern_id': p['pattern_id'],
                  'semantic_families': labels[i], 'provisional_label': p['label'],
                  'english_units': composition_label(p, units)}
                 for i, p in enumerate(selected)]).to_csv(output / 'selected_composition_key.csv', index=False)
    matrix = rates.pivot(index='city', columns='pattern_id', values='cell_weighted_rate').loc[cities, ids].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(11.5, 8), layout='constrained')
    im = ax.imshow(matrix, aspect='auto', cmap='YlGnBu', vmin=0)
    ax.set_xticks(range(len(ids)), [f'P{i+1:02d}  {labels[i]}' for i in range(len(ids))],
                  rotation=50, ha='right', fontsize=7)
    ax.set_yticks(range(30), [name(c) for c in cities], fontsize=8)
    fig.colorbar(im, ax=ax, label='Cell-balanced matched panoramas (%)')
    save(fig, output, 'Fig_CityPatterns_Current_Cities')

    cell_counts = cells.sampled_panoramas.to_numpy()
    rng = np.random.default_rng(42)
    rows = []
    for i, pattern in enumerate(selected[:6]):
        matches = pattern_mask(assignment, units, pattern['units'])
        cell_rates = np.bincount(cell_id, weights=matches, minlength=len(cells)) / cell_counts
        for city_name in cities:
            group = cells[cells.city == city_name]
            indices = group.index.to_numpy()
            block = pd.MultiIndex.from_arrays([group.grid_x.to_numpy() // 5,
                                               group.grid_y.to_numpy() // 5])
            block_id, _ = pd.factorize(block, sort=True)
            block_sum = np.bincount(block_id, weights=cell_rates[indices])
            block_n = np.bincount(block_id)
            draws = rng.integers(0, len(block_n), size=(999, len(block_n)))
            boot = block_sum[draws].sum(axis=1) / block_n[draws].sum(axis=1)
            rows.append({'pattern_id': pattern['pattern_id'], 'display_id': f'P{i+1:02d}',
                         'city': city_name, 'rate': cell_rates[indices].mean(),
                         'low': np.quantile(boot, .025), 'high': np.quantile(boot, .975)})
            stored = rates[(rates.city == city_name) & (rates.pattern_id == pattern['pattern_id'])]
            if len(stored) != 1 or not np.isclose(stored.cell_weighted_rate.iloc[0], rows[-1]['rate'], atol=1e-9):
                raise ValueError('Recomputed pattern rate differs from the completed analysis')
    interval = pd.DataFrame(rows)
    interval.to_csv(output / 'selected_composition_city_intervals.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 11), layout='constrained')
    for i, ax in enumerate(axes.flat):
        group = interval[interval.display_id == f'P{i+1:02d}'].sort_values('rate')
        y = np.arange(len(group))
        ax.hlines(y, group.low * 100, group.high * 100, color='#87a9a3', lw=1)
        ax.scatter(group.rate * 100, y, s=12, color='#287b77')
        ax.set_yticks(y, [name(c) for c in group.city], fontsize=6)
        ax.set(title=f'P{i+1:02d}  {labels[i]}', xlabel='Cell-balanced matches (%)')
    save(fig, output, 'Fig_CityPatterns_Current_Intervals')
    # Replace the old seven-pair maps with the same current compositions used
    # in the comparison plot; retain sampled cells with zero matches.
    for i, pattern in enumerate(selected):
        matches = pattern_mask(assignment, units, pattern['units'])
        values = np.bincount(cell_id, weights=matches, minlength=len(cells)) / cell_counts
        fig, axes = plt.subplots(5, 6, figsize=(18, 13), layout='constrained')
        for ax, city_name in zip(axes.flat, cities):
            group = cells[cells.city == city_name]
            im = ax.scatter(group.grid_x, group.grid_y, c=values[group.index] * 100,
                            cmap='YlGnBu', vmin=0, vmax=100, s=3, marker='s', rasterized=True)
            ax.set_title(name(city_name), fontsize=8)
            ax.set_aspect('equal'); ax.tick_params(labelsize=5)
        fig.suptitle(f'P{i+1:02d}  {composition_label(pattern, units)}', fontsize=13)
        fig.supxlabel('Observed 1-km cells; city-local UTM coordinates (km)', fontsize=9)
        fig.colorbar(im, ax=list(axes.flat), shrink=.65, label='Within-cell matched panoramas (%)')
        save(fig, output, f'Fig_CityPatterns_Current_Map_P{i+1:02d}')
    return selected


def city_atlas(folder, units, patterns, cells, families, output,
               labels=None, colors=None, residuals=None):
    """One inspectable page per city, with a complete key to all retained patterns."""
    labels = PARENTS if labels is None else labels
    colors = COLORS if colors is None else colors
    residuals = ([('excluded', 'Excluded mobile / temporary', '#bdc1c2'),
                  ('unclassified', 'Unclassified SAE mass', '#edf0ef')]
                 if residuals is None else residuals)
    rates = pd.read_csv(folder / 'city_distribution.csv')
    mean_rate = rates.groupby('pattern_id').cell_weighted_rate.mean()
    ordered = sorted(patterns, key=lambda p: (-mean_rate[p['pattern_id']], p['pattern_id']))
    key = pd.DataFrame([{
        'atlas_id': f'C{i+1:03d}', 'pattern_id': p['pattern_id'],
        'provisional_label': p['label'],
        'english_units': composition_label(p, units),
        'families': ' + '.join(PARENTS[units[j]['family']] for j in p['units']),
        'unit_ids': json.dumps(p['units']),
    } for i, p in enumerate(ordered)])
    key.to_csv(output / 'all_composition_key.csv', index=False)
    key = key.set_index('pattern_id')
    city = cells.groupby('city')[families + [r[0] for r in residuals]].mean()
    if not np.allclose(city.sum(axis=1), 1., atol=1e-6):
        raise ValueError('City composition does not sum to one')
    target = output / 'cities'
    target.mkdir(exist_ok=True)
    with PdfPages(output / 'Fig_ViTL30_City_Composition_Atlas.pdf') as atlas:
        for city_name, shares in city.iterrows():
            many_classes = len(families) > 16
            fig, axes = plt.subplots(2, 1, figsize=(8.4, 11.8 if many_classes else 9.8),
                                     layout='constrained',
                                     gridspec_kw={'height_ratios': [1.6 if many_classes else 1.2, 1]})
            ax, other = axes
            values = shares.to_numpy() * 100
            category_labels = [labels[f] for f in families] + [r[1] for r in residuals]
            ax.barh(np.arange(len(values)), values, color=list(colors) + [r[2] for r in residuals])
            ax.set_yticks(range(len(values)), category_labels,
                          fontsize=8 if many_classes else 9)
            ax.invert_yaxis()
            ax.set(xlim=(0, 100), xlabel='Share of all positive SAE mass (%)', title='a  Semantic-class response composition')
            ax.grid(axis='x', color='#d9dedd', linewidth=.55, zorder=0)
            ax.set_axisbelow(True)
            for i, value in enumerate(values):
                if value >= 1:
                    ax.text(value + .6, i, f'{value:.1f}', va='center', fontsize=8.5)
            top = rates[rates.city == city_name].nlargest(8, 'cell_weighted_rate')
            yy = np.arange(len(top))
            other.barh(yy, top.cell_weighted_rate.to_numpy() * 100, color='#327f9d')
            other.set_yticks(yy, [key.loc[p, 'atlas_id'] + '  ' + key.loc[p, 'english_units']
                                  for p in top.pattern_id], fontsize=8.5)
            other.invert_yaxis()
            other.set(xlim=(0, 100), xlabel='Cell-balanced matched panoramas (%)',
                      title='b  Eight most prevalent exploratory combinations')
            other.grid(axis='x', color='#d9dedd', linewidth=.55, zorder=0)
            other.set_axisbelow(True)
            for y, rate in zip(yy, top.cell_weighted_rate):
                other.text(rate * 100 + .6, y, f'{rate*100:.1f}', va='center', fontsize=8.5)
            group = cells[cells.city == city_name]
            fig.suptitle(f'{name(city_name)}  |  {int(group.sampled_panoramas.sum()):,} panoramas; {len(group):,} observed 1-km cells', fontsize=13)
            atlas.savefig(fig, bbox_inches='tight')
            slug = city_name.replace('/', '__')
            save(fig, target, f'City_{slug}_Composition')


def write_manifest(folder, output, summary, families):
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()

    inputs = [folder / f for f in ('summary.json', 'units.json', 'patterns.json',
              'scores_identity.json', 'calibration.json', 'city_distribution.csv',
              'scores.npy', 'assignment.npy', 'examples.json')]
    inputs += [folder.parent / 'candidate_combinations/panorama_elements.parquet',
               folder / 'evidence_review.json', folder / 'protocol.json',
               ROOT / 'experiments/vitl30_city_patterns/results/current_semantic_hierarchy.json']
    manifest = {
        'renderer_sha256': digest(Path(__file__)), 'stage': str(folder), 'summary': summary,
        'families': families, 'aggregation': 'mean panoramas within cell; equal cells within city',
        'denominator': 'all 4096 positive SAE activation dimensions',
        'cooccurrence': {'cell_presence_mass': .01, 'minimum_joint_probability': .0005},
        'umap': {'cells_per_city': 100, 'neighbors': 30, 'min_dist': .1, 'metric': 'cosine', 'seed': 42},
        'intervals': {'block_size_km': 5, 'bootstrap_draws': 999, 'seed': 42},
        'input_sha256': {str(p): digest(p) for p in inputs},
        'output_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*'))
                          if p.is_file() and p.name != 'semantic_figure_manifest.json'},
    }
    (output / 'semantic_figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def example_figure(folder, units, patterns, output):
    review = json.loads((folder / 'evidence_review.json').read_text())
    approved = {(r['pattern_id'], r['city'], r['panoid']) for r in review['examples']
                if r['status'] == 'plausible_spot_check'}
    patterns_by_id = {r['pattern_id']: r for r in patterns}
    examples = json.loads((folder / 'examples.json').read_text())
    pool = [r for r in examples if (r['pattern_id'], r['city'], r['panoid']) in approved
            and r['pattern_id'] in patterns_by_id]
    selected, seen, families_seen = [], set(), set()
    while pool and len(selected) < 4:
        row = max(pool, key=lambda r: (len({units[i]['family'] for i in r['units']} - families_seen),
                                      len(r['units'])))
        selected.append(row)
        seen.add(row['pattern_id'])
        families_seen.update(units[i]['family'] for i in row['units'])
        pool = [r for r in pool if r['pattern_id'] not in seen]
    if len(selected) < 4:
        raise ValueError('Insufficient reviewed current examples')
    fig, axes = plt.subplots(len(selected), 2, figsize=(15, len(selected) * 2.1), layout='constrained')
    for pair, row in zip(axes, selected):
        for ax, field in zip(pair, ('original', 'overlay')):
            with Image.open(row[field]) as im:
                ax.imshow(im)
            ax.axis('off')
        pair[0].set_title(f'{composition_label(patterns_by_id[row["pattern_id"]], units)}  |  {name(row["city"])}', loc='left', fontsize=9)
        color_words = ['amber', 'cyan', 'violet', 'green']
        legend = '; '.join(f'{color_words[k]}: {UNIT_NAMES[units[i]["label"]]}'
                           for k, i in enumerate(row['units']))
        pair[1].set_title(legend, loc='left', fontsize=8)
    save(fig, output, 'Fig_ViTL30_Current_Composition_Examples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=FOLLOWUP / 'stages/composition_patterns_screened')
    parser.add_argument('--output', type=Path, default=FIGURES)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    folder, output = args.stage.resolve(), args.output.resolve()
    summary, units, patterns, frame, scores, assignment = load(folder)
    output.mkdir(parents=True, exist_ok=True)
    cells, cell_id, families = cell_composition(frame, scores, units)
    if len(cells) != summary['cells'] or cells.city.nunique() != 30:
        raise ValueError('Cell sample does not match the current run')
    pipeline(output)
    cities, _ = city_figures(cells, families, output)
    pattern_figures(folder, units, patterns, frame, assignment, cell_id, cells, cities, output)
    city_atlas(folder, units, patterns, cells, families, output)
    example_figure(folder, units, patterns, output)
    write_manifest(folder, output, summary, families)
    print(f'Wrote current semantic and landscape composition figures to {output}')


if __name__ == '__main__':
    main()
