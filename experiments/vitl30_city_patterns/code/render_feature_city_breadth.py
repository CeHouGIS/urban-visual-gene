"""Standalone tree-city heatmap and semantic-class city-breadth boxplot.

The heatmap reproduces the original 30-city panel A response and display order
without dendrograms or extra panels. The boxplot retains its 30-city measure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

from .followup import sha256


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'results/tree_geography'
SOURCE = ROOT / 'results/chapter5_revision/city_feature_space/all_sae/city_feature_shifts.csv'
OUTPUT = ROOT / 'results/chapter5_revision/feature_city_breadth'
CATEGORY_HEATMAP_OUTPUT = OUTPUT / 'category_heatmaps'
TREE_STEM = 'Fig_ViTL30_Tree_City_Clustered_Heatmap'
BOX_STEM = 'Fig_ViTL30_Semantic_Class_City_Breadth_Boxplots'
TEXT = '#303D42'
MUTED = '#687579'
TREE_COLORS = [
    (0.0, '#78909E'),
    (1 / 3, '#C7D2D4'),
    (2 / 3, '#F5F4F0'),
    (5 / 6, '#DEC8BE'),
    (1.0, '#B98779'),
]
TREE_COLOR_LIMITS = (-6.0, 3.0)
BOXPLOT_EXCLUDED_FAMILIES = {
    'unresolved', 'spatial_encoding', 'person', 'vehicle',
}
EXPECTED_BOX_FAMILY_COUNTS = {
    'barrier': 57, 'building': 542, 'construction': 25, 'furniture': 22,
    'ground_unspecified': 526, 'low_vegetation': 93, 'material_unspecified': 118,
    'pedestrian': 28, 'road': 239, 'scene_context': 84,
    'sign': 61, 'sky': 207, 'terrain': 30, 'transport_structure': 54,
    'tree': 300, 'utility': 50, 'vegetation_unspecified': 172, 'water': 18,
}
CLASS_NAMES = {
    'barrier': 'Barriers', 'building': 'Buildings',
    'construction': 'Construction', 'furniture': 'Street furniture',
    'ground_unspecified': 'Unspecified ground',
    'low_vegetation': 'Low vegetation',
    'material_unspecified': 'Unspecified materials',
    'pedestrian': 'Pedestrian space', 'road': 'Roads',
    'scene_context': 'Scene context', 'sign': 'Signs', 'sky': 'Sky',
    'terrain': 'Terrain', 'transport_structure': 'Transport structures',
    'tree': 'Trees', 'utility': 'Utilities',
    'vegetation_unspecified': 'Unspecified vegetation', 'water': 'Water',
}
CITY_NAMES = {
    'Argentina/BuenosAires': 'Buenos Aires',
    'Australia/Melbourne': 'Melbourne', 'Australia/Perth': 'Perth',
    'Australia/Sydney': 'Sydney', 'Austria/Vienna': 'Vienna',
    'Bangladesh/Dhaka': 'Dhaka', 'Belgium/Antwerp': 'Antwerp',
    'Belgium/Brussels': 'Brussels', 'Brazil/Brasília': 'Brasilia',
    'Brazil/Brasilia': 'Brasilia',
    'Brazil/RiodeJaneiro': 'Rio de Janeiro', 'Brazil/SaoPaulo': 'Sao Paulo',
    'China/Taipei': 'Taipei', 'France/Paris': 'Paris',
    'Germany/Hamburg': 'Hamburg', 'Indonesia/Yogyakarta': 'Yogyakarta',
    'Italy/Naples': 'Naples', 'Korea/Seoul': 'Seoul',
    'Mexico/Monterrey': 'Monterrey', 'Nigeria/Lagos': 'Lagos',
    'Portugal/Lisbon': 'Lisbon', 'Russia/Moscow': 'Moscow',
    'Singapore/Singapore': 'Singapore',
    'SouthAfrica/CapeTown': 'Cape Town',
    'SouthAfrica/Johannesburg': 'Johannesburg',
    'Spain/Barcelona': 'Barcelona', 'Sweden/Stockholm': 'Stockholm',
    'UAE/Dubai': 'Dubai', 'US/Chicago': 'Chicago',
    'US/NewYork': 'New York', 'US/Phoenix': 'Phoenix',
}


def cluster_order(values):
    return leaves_list(linkage(pdist(values, metric='euclidean'),
                               method='average', optimal_ordering=True))


def load_profiles():
    table = pd.read_csv(SOURCE, usecols=[
        'city', 'feature', 'label_zh', 'family', 'status', 'city_mass_share'])
    if len(table) != 4096 * 30 or table.feature.nunique() != 4096 or table.city.nunique() != 30:
        raise ValueError('Expected a complete 4096-by-30 city response table')
    if table.duplicated(['feature', 'city']).any():
        raise ValueError('Duplicated feature/city response')
    selected = table.loc[table.status.eq('candidate_ready')].copy()
    profiles = selected.pivot(index='feature', columns='city', values='city_mass_share')
    if profiles.shape != (1535, 30) or profiles.isna().any().any():
        raise ValueError('Unexpected candidate-ready feature response shape')
    breadth = feature_breadth(selected, profiles)

    # Keep the original family assignments and retain every feature in included classes.
    assigned = table.loc[~table.family.isin(BOXPLOT_EXCLUDED_FAMILIES)].copy()
    class_profiles = assigned.pivot(
        index='feature', columns='city', values='city_mass_share').reindex(columns=profiles.columns)
    if class_profiles.shape != (2626, 30) or class_profiles.isna().any().any():
        raise ValueError('Expected 2,626 retained dimensions across 30 cities')
    assigned_breadth = feature_breadth(assigned, class_profiles)
    actual_counts = assigned_breadth.groupby('family').feature.nunique().to_dict()
    if actual_counts != EXPECTED_BOX_FAMILY_COUNTS:
        raise ValueError(f'Existing class membership changed: {actual_counts}')
    if assigned_breadth.feature.nunique() != 2626 or len(actual_counts) != 18:
        raise ValueError('Expected 2,626 dimensions across the 18 retained classes')
    return profiles, breadth, assigned_breadth, class_profiles


def feature_breadth(selected, profiles):
    values = profiles.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Non-finite or negative feature response')
    totals = values.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError('Included feature with no city response')
    shares = values / totals[:, None]
    effective = 1 / np.sum(shares ** 2, axis=1)
    metadata = selected.drop_duplicates('feature').set_index('feature')
    ordered = metadata.loc[profiles.index]
    result = pd.DataFrame({
        'feature': profiles.index.to_numpy(),
        'family': ordered.family.to_numpy(),
        'label_zh': ordered.label_zh.to_numpy(),
        'status': ordered.status.to_numpy(),
        'effective_cities': effective,
        'positive_cities': (values > 0).sum(axis=1),
        'top_city': profiles.columns.to_numpy()[shares.argmax(axis=1)],
        'top_city_response_share': shares.max(axis=1),
    })
    if not np.all((effective >= 1 - 1e-8) & (effective <= 30 + 1e-8)):
        raise ValueError('Invalid effective-city breadth')
    return result


def similarity_order(values, feature_ids):
    """Order features along a one-dimensional seriation of city response profiles."""
    totals = values.sum(axis=1)
    if not np.isfinite(values).all() or (values < 0).any() or (totals <= 0).any():
        raise ValueError('Invalid feature response for category heatmap ordering')
    if len(values) < 3:
        return np.argsort(feature_ids, kind='stable')
    profiles = np.sqrt(values / totals[:, None])
    distances = squareform(pdist(profiles, metric='sqeuclidean'))
    positive = distances[distances > 0]
    if len(positive) == 0:
        return np.argsort(feature_ids, kind='stable')
    affinity = np.exp(-distances / np.median(positive))
    np.fill_diagonal(affinity, 0)
    degree = affinity.sum(axis=1)
    inv_sqrt_degree = 1 / np.sqrt(degree)
    laplacian = np.eye(len(values)) - (
        inv_sqrt_degree[:, None] * affinity * inv_sqrt_degree[None, :])
    _, eigenvectors = eigh(laplacian, subset_by_index=[0, 1])
    position = eigenvectors[:, 1] * inv_sqrt_degree
    order = np.argsort(position, kind='stable')
    if feature_ids[order[0]] > feature_ids[order[-1]]:
        order = order[::-1]
    return order


def tree_heatmap(profiles, breadth):
    tree = profiles.loc[breadth.loc[breadth.family.eq('tree'), 'feature']].sort_index()
    if tree.shape != (280, 30):
        raise ValueError('Expected 280 candidate-ready tree features in 30 cities')
    original = pd.read_csv(TREE / 'city_feature_response.csv')
    eligible = original.loc[original.eligible.astype(str).str.lower().eq('true')]
    check = eligible.pivot(index='feature', columns='city', values='response').reindex_like(tree)
    if check.isna().any().any() or not np.allclose(tree, check, atol=5e-10, rtol=1e-7):
        raise ValueError('Tree heatmap differs from the original panel A response data')

    # Reproduce panel A before selecting the heatmap alone.
    city_response = tree.T
    within_tree = city_response.div(
        city_response.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    city_mean = within_tree.mean(axis=0)
    log_fold = np.log2((within_tree + 1e-6).div(city_mean + 1e-6, axis=1))
    cities = log_fold.index.to_numpy()[cluster_order(log_fold.to_numpy())]
    features = log_fold.columns.to_numpy()[cluster_order(log_fold.to_numpy().T)]
    matrix = log_fold.loc[cities, features].to_numpy().T

    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'font.size': 10})
    fig, heat_ax = plt.subplots(figsize=(14.8, 12.6), facecolor='white')
    fig.subplots_adjust(left=.10, right=.88, top=.88, bottom=.17)
    palette = LinearSegmentedColormap.from_list('muted_tree_response', TREE_COLORS)
    heat = heat_ax.imshow(matrix, aspect='auto', interpolation='nearest',
                          cmap=palette,
                          norm=TwoSlopeNorm(vmin=TREE_COLOR_LIMITS[0], vcenter=0,
                                            vmax=TREE_COLOR_LIMITS[1]),
                          rasterized=True)
    heat_ax.set_xticks(np.arange(len(cities)),
                       [CITY_NAMES.get(city, city.split('/')[-1]) for city in cities],
                       rotation=90, ha='center', fontsize=9)
    heat_ax.set_yticks([])
    heat_ax.set_xlabel('Study cities', fontsize=11, labelpad=8)
    heat_ax.set_ylabel('280 tree-related SAE dimensions', fontsize=11, labelpad=8)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=heat_ax, fraction=.027, pad=.018)
    colorbar.set_label('log2 response / equal-city mean for each dimension', fontsize=10)
    colorbar.set_ticks([-6, -4, -2, 0, 1.5, 3])
    colorbar.outline.set_visible(False)
    fig.text(.10, .962, 'Tree-related SAE responses across 30 study cities',
             va='top', fontsize=20, fontweight='bold', color=TEXT)
    fig.text(.10, .917,
             'Relative within-tree activation of 280 semantically eligible dimensions',
             va='top', fontsize=11.5, color=MUTED)
    fig.text(.10, .035,
             'Each city is balanced by observed 1-km cells. The color shows response relative to each dimension’s '
             'equal-city mean; row and column positions reproduce the original panel A ordering. Labels are provisional.',
             va='bottom', fontsize=9.2, color=MUTED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{TREE_STEM}.{extension}', dpi=260,
                    bbox_inches='tight', pad_inches=.1, facecolor='white')
    plt.close(fig)
    pd.DataFrame({'display_row_from_top': np.arange(1, len(features) + 1),
                  'feature': features}).to_csv(
                      OUTPUT / 'tree_feature_display_order.csv', index=False)
    pd.DataFrame({'display_column': np.arange(1, len(cities) + 1),
                  'city': cities}).to_csv(OUTPUT / 'tree_city_display_order.csv', index=False)
    return {'features': len(features), 'cities': len(cities)}


def breadth_boxplot(breadth):
    class_stats = breadth.groupby('family').effective_cities.agg(['count', 'median'])
    families = class_stats.sort_values(['median', 'count'], ascending=[False, False]).index.tolist()
    if len(families) != 18:
        raise ValueError('Expected all 18 retained semantic classes')
    data = [breadth.loc[breadth.family.eq(family), 'effective_cities'].to_numpy()
            for family in families]
    fig, ax = plt.subplots(figsize=(12.8, 9.4), facecolor='white')
    fig.subplots_adjust(left=.27, right=.965, top=.865, bottom=.13)
    boxes = ax.boxplot(data, positions=np.arange(len(families)), orientation='horizontal',
                       widths=.59, patch_artist=True, showfliers=False,
                       whis=(5, 95),
                       medianprops={'color': '#273941', 'linewidth': 1.7},
                       whiskerprops={'color': '#7C8D8E', 'linewidth': 1},
                       capprops={'color': '#7C8D8E', 'linewidth': 1},
                       boxprops={'edgecolor': '#74898A', 'linewidth': .8})
    rng = np.random.default_rng(42)
    for index, (family, values, box) in enumerate(zip(families, data, boxes['boxes'])):
        tree = family == 'tree'
        box.set_facecolor('#74A392' if tree else '#CEDCD8')
        jitter = rng.uniform(-.22, .22, size=len(values))
        ax.scatter(values, index + jitter, s=7, color='#3F786D' if tree else '#718F90',
                   alpha=.22, linewidths=0, rasterized=True, zorder=2)
    ax.set_yticks(np.arange(len(families)),
                  [f'{CLASS_NAMES[family]}   (n={class_stats.loc[family, "count"]})'
                   for family in families], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(.5, 30.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel('Effective number of responding cities (out of 30)', fontsize=11.5)
    ax.grid(axis='x', color='#DDE4E2', linewidth=.75)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.text(.27, .968, 'Geographic breadth of semantic SAE features',
             va='top', fontsize=19, fontweight='bold', color=TEXT)
    fig.text(.27, .924,
             'Each point is one previously classified dimension (n=2,626); boxes show the median and interquartile range',
             va='top', fontsize=11, color=MUTED)
    fig.text(.27, .048,
             'Breadth = 1 / Σ(city response share)². City response is averaged equally over observed 1-km cells. '
             'Whiskers mark the 5th–95th percentiles. Excludes 1,264 unresolved dimensions, 17 spatial-encoding dimensions, '
             'and all people (46) and vehicle (143) features; all other prior class assignments and review statuses are retained.',
             va='bottom', fontsize=9.2, color=MUTED)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{BOX_STEM}.{extension}', dpi=260,
                    bbox_inches='tight', pad_inches=.1, facecolor='white')
    plt.close(fig)
    class_counts = class_stats['count'].rename('dimensions').reindex(families).rename_axis('family').reset_index()
    class_counts['class_label'] = class_counts.family.map(CLASS_NAMES)
    class_counts.to_csv(OUTPUT / 'semantic_class_breadth_counts.csv', index=False)
    return families, class_counts


def category_heatmaps(class_profiles, breadth, families):
    city_order = pd.read_csv(OUTPUT / 'tree_city_display_order.csv').city.tolist()
    if set(city_order) != set(class_profiles.columns) or len(city_order) != 30:
        raise ValueError('Category heatmaps require the shared 30-city display order')
    CATEGORY_HEATMAP_OUTPUT.mkdir(parents=True, exist_ok=True)
    palette = LinearSegmentedColormap.from_list('muted_class_response', TREE_COLORS)
    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'font.size': 10})
    city_labels = [CITY_NAMES.get(city, city.split('/')[-1]) for city in city_order]
    feature_orders = []
    outputs = []
    atlas_path = CATEGORY_HEATMAP_OUTPUT / 'Fig_ViTL30_All_Class_Heatmaps.pdf'
    atlas = PdfPages(atlas_path)
    for family in families:
        feature_ids = breadth.loc[breadth.family.eq(family), 'feature'].to_numpy(dtype=int)
        values = class_profiles.loc[feature_ids, city_order].to_numpy(dtype=np.float64)
        city_totals = values.sum(axis=0)
        if (city_totals <= 0).any():
            missing = [city_order[i] for i in np.flatnonzero(city_totals <= 0)]
            raise ValueError(f'No {family} response in cities: {missing}')
        within_class = values / city_totals[None, :]
        equal_city_mean = within_class.mean(axis=1)
        matrix = np.log2((within_class + 1e-6) /
                         (equal_city_mean[:, None] + 1e-6))
        order = similarity_order(within_class, feature_ids)
        ordered_features = feature_ids[order]
        matrix = matrix[order]

        count = len(ordered_features)
        height = max(5.4, min(14.0, 4.8 + count * .017))
        fig, ax = plt.subplots(figsize=(14.8, height), facecolor='white')
        fig.subplots_adjust(left=.105, right=.88, top=.84,
                            bottom=max(.20, 2.0 / height))
        heat = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=palette,
                         norm=TwoSlopeNorm(vmin=TREE_COLOR_LIMITS[0], vcenter=0,
                                           vmax=TREE_COLOR_LIMITS[1]),
                         rasterized=True)
        ax.set_xticks(np.arange(len(city_order)), city_labels,
                      rotation=90, ha='center', fontsize=8.5)
        ax.set_yticks([])
        ax.set_xlabel('Study cities · shared order across categories', fontsize=10.5, labelpad=7)
        ax.set_ylabel(f'{count} SAE features', fontsize=10.5, labelpad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(heat, ax=ax, fraction=.027, pad=.018)
        colorbar.set_label('log2 response / equal-city mean within category', fontsize=9.5)
        colorbar.set_ticks([-6, -4, -2, 0, 1.5, 3])
        colorbar.outline.set_visible(False)
        label = CLASS_NAMES[family]
        fig.text(.105, .958, f'{label} · SAE feature responses across 30 study cities',
                 va='top', fontsize=18, fontweight='bold', color=TEXT)
        fig.text(.105, .91,
                 'All previously assigned dimensions in this category; no cluster tree or cluster labels',
                 va='top', fontsize=10.8, color=MUTED)
        fig.text(.105, .035,
                 'Within each city, response is normalized across this category; color is relative to each feature’s '
                 'equal-city mean. Feature rows are arranged by response-profile similarity. Labels are provisional.',
                 va='bottom', fontsize=8.8, color=MUTED)
        stem = f'Fig_ViTL30_Class_Heatmap_{family}'
        for extension in ('png', 'pdf'):
            fig.savefig(CATEGORY_HEATMAP_OUTPUT / f'{stem}.{extension}', dpi=260,
                        bbox_inches='tight', pad_inches=.1, facecolor='white')
        atlas.savefig(fig, bbox_inches='tight', pad_inches=.1, facecolor='white')
        plt.close(fig)
        for position, feature in enumerate(ordered_features, start=1):
            feature_orders.append({
                'family': family, 'class_label': label,
                'display_row_from_top': position, 'feature': int(feature),
            })
        outputs.append({'family': family, 'class_label': label,
                        'dimensions': count, 'stem': stem})
    atlas.close()

    pd.DataFrame(feature_orders).to_csv(
        CATEGORY_HEATMAP_OUTPUT / 'category_feature_similarity_order.csv', index=False)
    (CATEGORY_HEATMAP_OUTPUT / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'classes': outputs, 'city_order': city_order,
        'atlas_pdf': atlas_path.name,
        'feature_order': 'spectral seriation of square-root-normalized within-category city response profiles; no clustering labels or trees',
        'heatmap_definition': 'within each city normalize feature response across the retained class, then log2 relative to each feature equal-city mean (epsilon 1e-6)',
        'palette': TREE_COLORS,
        'color_limits': {'vmin': TREE_COLOR_LIMITS[0], 'vcenter': 0,
                         'vmax': TREE_COLOR_LIMITS[1]},
        'excluded_families': sorted(BOXPLOT_EXCLUDED_FAMILIES),
    }, indent=2, ensure_ascii=False) + '\n')
    return outputs


def main():
    profiles, breadth, boxplot_breadth, class_profiles = load_profiles()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    boxplot_breadth.sort_values(['effective_cities', 'feature'], ascending=[False, True]).to_csv(
        OUTPUT / 'feature_city_breadth.csv', index=False)
    tree_summary = tree_heatmap(profiles, breadth)
    family_order, family_counts = breadth_boxplot(boxplot_breadth)
    category_outputs = category_heatmaps(class_profiles, boxplot_breadth, family_order)
    (OUTPUT / 'manifest.json').write_text(json.dumps({
        'dimensions_total': 4096, 'candidate_ready_dimensions': len(breadth),
        'boxplot_dimensions': len(boxplot_breadth),
        'boxplot_excluded_dimensions': {
            'unresolved': 1264, 'spatial_encoding': 17, 'person': 46, 'vehicle': 143,
        },
        'tree_dimensions': int(breadth.family.eq('tree').sum()),
        'source_cities': profiles.shape[1], 'heatmap_cities': tree_summary['cities'],
        'semantic_classes': len(family_order),
        'boxplot_family_counts': family_counts.set_index('family').dimensions.to_dict(),
        'boxplot_status_counts': boxplot_breadth.status.value_counts().to_dict(),
        'source_sha256': sha256(SOURCE),
        'boxplot_breadth_definition': 'inverse Simpson effective city count: 1 / sum((city_mass_share / feature total across cities)^2)',
        'heatmap_definition': 'original 30-city panel A: within-tree city share, then log2 fold relative to each feature equal-city mean (epsilon 1e-6)',
        'heatmap_palette': TREE_COLORS,
        'heatmap_color_limits': {'vmin': TREE_COLOR_LIMITS[0], 'vcenter': 0,
                                 'vmax': TREE_COLOR_LIMITS[1]},
        'display_order': 'original panel A hierarchical row and column leaf order, with dendrograms and cluster annotations omitted',
        'boxplot_family_order': family_order,
        'boxplot_membership': 'Use the existing family assignment for every dimension except unresolved, spatial_encoding, person, and vehicle; no status-based filtering or within-class selection.',
        'category_heatmaps': category_outputs,
        'interpretation_limit': 'effective cities measures response evenness among the 30 sampled study cities, not the number of observed positive activations',
    }, indent=2, ensure_ascii=False) + '\n')
    print(OUTPUT / f'{TREE_STEM}.png')
    print(OUTPUT / f'{BOX_STEM}.png')


if __name__ == '__main__':
    main()
