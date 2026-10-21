"""Refresh spatial-analysis figures, retain training/semantic provenance, compile.

The larger sample measures activation mass, not the older dominant-patch
assignment statistic. Captions and methods are replaced together with assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from sklearn.metrics.pairwise import cosine_similarity

from .extract import DEFAULT_OUT, OUTPUT, write_json
from .followup import FOLLOWUP, sha256
from sae_experiments.panorama_vitl30_figures import city_map, loss_curve, hierarchy, name, save
from sae_experiments.panorama_vitl30_analysis import checked_groups, model_and_identity

GROUP_COLS = [f'group_{i:02d}' for i in range(64)]
THESIS = Path('/root/code/HKUST-Thesis')


def pair_similarity(vectors):
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-20)
    n = len(normalized)
    return float((np.square(normalized.sum(axis=0)).sum() - np.square(normalized).sum()) /
                 (n * (n - 1))) if n > 1 else float('nan')


def plots(args, figures):
    cells = pd.read_parquet(args.followup / 'cells.parquet')
    cities = sorted(cells.city.unique())
    comp = np.stack([cells.loc[cells.city == c, GROUP_COLS].mean().to_numpy() for c in cities])
    prevalence = np.stack([(cells.loc[cells.city == c, GROUP_COLS] >= .01).mean().to_numpy() for c in cities])
    locations = []
    for city, group in cells.groupby('city'):
        transform = Transformer.from_crs(int(group.utm_epsg.iloc[0]), 4326, always_xy=True)
        lon, lat = transform.transform((group.grid_x.median() + .5) * 1000,
                                       (group.grid_y.median() + .5) * 1000)
        locations.append({'city': city, 'latitude': lat, 'longitude': lon})
    city_map(figures, pd.DataFrame(locations))
    model, model_hash = model_and_identity(OUTPUT, 'cpu')
    groups = checked_groups(OUTPUT, model_hash)
    loss_curve(figures, OUTPUT)
    hierarchy(figures, model, groups)
    order = np.argsort(-comp.mean(axis=0))
    shared = ((prevalence >= .1).sum(axis=0) >= 27) & (comp.mean(axis=0) >= .005)
    excess = 100 * (comp - (comp.sum(axis=0) - comp) / (len(cities) - 1))
    limit = max(np.quantile(abs(excess), .98), .001)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained', gridspec_kw={'height_ratios': [1, 3]})
    axes[0].bar(range(64), comp.mean(axis=0)[order] * 100,
                color=np.where(shared[order], '#247e77', '#bdc6c4'))
    axes[0].set(ylabel='Activation mass (%)', title='Shared groups and city differences; equal weight per 1-km cell')
    im = axes[1].imshow(excess[:, order], aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -limit, limit))
    axes[1].set_yticks(range(30), [name(c) for c in cities], fontsize=7)
    axes[1].set_xlabel('Decoder group (ordered by equal-city mean)')
    fig.colorbar(im, ax=axes[1], label='Percentage-point excess')
    save(fig, figures, 'Fig_ViTL30_Shared_Distinctive')
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
    im = ax.imshow(comp * 100, aspect='auto', cmap='YlGnBu')
    ax.set_yticks(range(30), [name(c) for c in cities], fontsize=7)
    ax.set(xlabel='Decoder group', title='City composition: mean cell-normalized SAE group activation mass')
    fig.colorbar(im, ax=ax, label='Activation mass (%)')
    save(fig, figures, 'Fig_ViTL30_City_Composition')
    fig, ax = plt.subplots(figsize=(9, 8), layout='constrained')
    im = ax.imshow(cosine_similarity(comp), cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(30), [name(c) for c in cities], rotation=90, fontsize=7)
    ax.set_yticks(range(30), [name(c) for c in cities], fontsize=7)
    ax.set_title('Cosine similarity of cell-balanced city compositions')
    fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, figures, 'Fig_ViTL30_City_Similarity')
    # Equal-city cell co-occurrence: distinct from older image-level PPMI.
    joint, marginal = np.zeros((64, 64)), np.zeros(64)
    spatial = []
    for city in cities:
        group = cells[cells.city == city]
        x = group[GROUP_COLS].to_numpy(float)
        present = (x >= .01).astype(float)
        joint += present.T @ present / len(x) / 30
        marginal += present.mean(axis=0) / 30
        normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-20)
        lookup = {(int(r.grid_x), int(r.grid_y)): i for i, r in enumerate(group.itertuples())}
        pairs = [(i, lookup[n]) for (gx, gy), i in lookup.items()
                 for n in ((gx + 1, gy), (gx, gy + 1)) if n in lookup]
        local = float(np.mean([normalized[i] @ normalized[j] for i, j in pairs])) if pairs else np.nan
        spatial.append({'city': city, 'cells': len(group), 'adjacent_pairs': len(pairs),
                        'global_homogeneity': pair_similarity(x), 'local_continuity': local})
    expected = np.outer(marginal, marginal)
    ppmi = np.maximum(np.log(np.maximum(joint, 1e-15) / np.maximum(expected, 1e-15)), 0)
    ppmi[joint < .0005] = 0
    np.fill_diagonal(ppmi, 0)
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    im = ax.imshow(ppmi, cmap='magma', vmin=0)
    ax.set(xlabel='Decoder group', ylabel='Decoder group', title='Equal-city, 1-km-cell group co-occurrence')
    fig.colorbar(im, ax=ax, label='PPMI (natural log)')
    save(fig, figures, 'Fig_ViTL30_Cooccurrence')
    spatial = pd.DataFrame(spatial)
    eligible = spatial[spatial.adjacent_pairs >= 20]
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    ax.scatter(eligible.global_homogeneity, eligible.local_continuity, c='#247e77', s=35)
    for r in eligible.itertuples():
        ax.annotate(name(r.city), (r.global_homogeneity, r.local_continuity), xytext=(4, 4), textcoords='offset points', fontsize=7)
    ax.set(xlabel='Mean similarity across all observed cells', ylabel='Mean adjacent-cell similarity', title='Spatial structure of SAE activation mass')
    save(fig, figures, 'Fig_ViTL30_Spatial_Structure')
    spatial.to_csv(args.output / 'spatial_structure.csv', index=False)
    # Cell-balanced sampling for descriptive embedding; city labels do not enter fitting.
    import umap
    selected = []
    for city in cities:
        group = cells[cells.city == city].copy()
        group['hash'] = [hashlib.sha256(f'{city}/{x}/{y}'.encode()).hexdigest()
                         for x, y in zip(group.grid_x, group.grid_y)]
        selected.append(group.sort_values('hash').head(100))
    sample = pd.concat(selected, ignore_index=True)
    coords = umap.UMAP(n_neighbors=30, min_dist=.1, metric='cosine', random_state=42).fit_transform(sample[GROUP_COLS].to_numpy())
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    palette = plt.get_cmap('tab20', 30)
    for i, city in enumerate(cities):
        mask = sample.city == city
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette(i)], s=8, alpha=.6, label=name(city), rasterized=True)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2', title='1-km-cell SAE composition UMAP (100 cells/city)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2, frameon=False)
    save(fig, figures, 'Fig_ViTL30_Panorama_UMAP')
    sample[['city', 'grid_x', 'grid_y']].assign(umap_x=coords[:, 0], umap_y=coords[:, 1]).to_csv(args.output / 'cell_umap.csv', index=False)
    return {'shared_groups': int(shared.sum()), 'shared_mass': float(comp.mean(axis=0)[shared].sum()),
            'top_four_mass': float(np.sort(comp.mean(axis=0))[-4:].sum()),
            'spatial_plot_cities': len(eligible), 'cities': 30, 'cells': len(cells)}


def candidate_plots(args, figures):
    source = args.followup / 'stages/prototype_city_distribution'
    city = pd.read_csv(source / 'city_prototype_distribution.csv')
    cells = pd.read_parquet(source / 'cell_prototype_distribution.parquet')
    table = city.pivot(index='city', columns='prototype_id', values='cell_weighted_prevalence')
    fig, ax = plt.subplots(figsize=(11, 9), layout='constrained')
    im = ax.imshow(table.to_numpy() * 100, aspect='auto', cmap='YlGnBu', vmin=0)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(30), [name(c) for c in table.index], fontsize=8)
    ax.set_title('Provisional co-presence candidates: equal-cell city prevalence')
    fig.colorbar(im, ax=ax, label='Positive panoramas per cell, city mean (%)')
    save(fig, figures, 'Fig_CityPatterns_Candidate_Cities')
    fig, axes = plt.subplots(2, 4, figsize=(17, 11), layout='constrained')
    for ax, (candidate, group) in zip(axes.flat, city.groupby('prototype_id')):
        group = group.sort_values('cell_weighted_prevalence')
        y = np.arange(len(group))
        mid = group.cell_weighted_prevalence.to_numpy() * 100
        low, high = group.bootstrap_low.to_numpy() * 100, group.bootstrap_high.to_numpy() * 100
        ax.hlines(y, low, high, color='#7aa69c', lw=1)
        ax.scatter(mid, y, c='#247e77', s=10)
        ax.set_yticks(y, [name(c) for c in group.city], fontsize=6)
        ax.set(title=candidate, xlabel='Cell-weighted prevalence (%)')
    axes.flat[-1].axis('off')
    save(fig, figures, 'Fig_CityPatterns_Candidate_Uncertainty')
    for candidate, group in cells.groupby('prototype_id'):
        fig, axes = plt.subplots(5, 6, figsize=(18, 13), layout='constrained')
        maximum = max(float(group.prevalence.quantile(.99)), .01)
        for ax, (city_name, city_cells) in zip(axes.flat, group.groupby('city')):
            im = ax.scatter(city_cells.grid_x, city_cells.grid_y, c=city_cells.prevalence,
                            cmap='YlGnBu', vmin=0, vmax=maximum, s=3, marker='s', rasterized=True)
            ax.set_title(name(city_name), fontsize=8)
            ax.set_aspect('equal'); ax.tick_params(labelsize=5)
        fig.suptitle(f'{candidate}: sampled 1-km cells; coordinates in local UTM kilometres')
        fig.colorbar(im, ax=list(axes.flat), shrink=.6, label='Within-cell panorama co-presence fraction')
        save(fig, figures, f'Fig_CityPatterns_Map_{candidate.replace("+", "_")}')


def image_figure(args, figures):
    examples = json.loads((args.followup / 'stages/road_sides/examples.json').read_text())
    selected = []
    for candidate in dict.fromkeys(r['candidate'] for r in examples):
        pool = [r for r in examples if r['candidate'] == candidate]
        selected.append(max(pool, key=lambda r: r['road_axis_confidence']))
    fig, axes = plt.subplots(len(selected), 2, figsize=(18, len(selected) * 2.6), layout='constrained')
    for pair, row in zip(axes, selected):
        for ax, key in zip(pair, ('original', 'overlay')):
            with Image.open(row[key]) as im:
                ax.imshow(im)
            ax.axis('off')
        pair[0].set_title(f'{row["candidate"]} | {name(row["city"])}', fontsize=9, loc='left')
        pair[1].set_title('SAE families: amber / cyan; independent road: violet; side unverified', fontsize=8)
    save(fig, figures, 'Fig_ViTL30_Activation_Examples')


def revise_writing(thesis, metrics, summary):
    path = thesis / '05_research3.tex'
    text = path.read_text()
    replacements = {
      'At the city level, each city is represented by a visual-composition vector containing the mean panorama-level dominant-response share of each visual gene, with the proportions summing to one. For each panorama, these shares are calculated from the proportion of patch-level dominant latent assignments belonging to each gene. The resulting vector summarizes the relative dominance of learned visual responses across panoramas. Pairwise inter-city similarity is subsequently evaluated from these composition vectors using cosine similarity and distribution-based measures.':
      'For downstream geographical analysis, positive SAE activation mass is summed within each fixed decoder group and normalized within each complete panorama. Panorama vectors are averaged within each observed one-kilometre cell; each cell then receives equal weight in its city composition. These vectors measure relative activation mass rather than dominant-patch assignments. City composition similarity is evaluated using cosine similarity.',
      'where $P(a)$ and $P(b)$ denote marginal occurrence probabilities and $P(a,b)$ denotes joint probability within one directional image. The current experiment reports a global PPMI matrix. Higher values describe stronger co-occurrence relative to marginal frequencies, but are not statistical significance tests.':
      'where $P(a)$ and $P(b)$ denote marginal occurrence probabilities and $P(a,b)$ denotes joint occurrence in an observed one-kilometre cell. Presence requires at least one percent of cell-normalized group activation mass. Cells receive equal weight within cities and cities receive equal weight globally. Higher PPMI describes association relative to marginal frequencies, rather than statistical significance.',
      'The retained four-view panoramas supply the SAE inputs and subsequent composition analyses.':
      'These retained panoramas define the SAE training and holdout cohort. A separate spatial analysis sample, described below, reuses the frozen selected model.',
      'For each contextualized patch, its strongest positive latent response is assigned to one of the fixed groups. The four directional grids are combined within each panorama to obtain a group-share vector whose entries sum to one. City-level composition is the equal-panorama mean of these vectors. City similarities are then calculated from the resulting composition vectors; the groups are never defined from city prevalence.':
      'The downstream spatial sample selects at most ten distinct complete panoramas per occupied $1\\,\\mathrm{km}\\times1\\,\\mathrm{km}$ cell in each city, using local UTM coordinates and a fixed-seed hash. Of 688,174 selected candidates, 681,645 pass image completeness and quality checks, spanning 72,625 observed cells in all 30 cities. The frozen encoder and SAE process all four views of each retained panorama. Positive activation mass is aggregated by the fixed decoder groups, normalized within panorama, averaged within cell, and then averaged equally across observed cells within each city. These are activation-mass shares, not dominant-patch proportions. The original 89,031-panorama SAE training cohort remains distinct.',
      "Co-occurrence is measured within individual directional images. A group is considered present when it accounts for at least one percent of the image's positive dominant-patch assignments. For groups $a$ and $b$, positive pointwise mutual information is defined in Equation~\\ref{eq:element_ppmi}; marginal and joint support are retained alongside it. PPMI describes association relative to marginal prevalence and is not by itself a significance test. A separate permutation analysis would be required for inferential claims about specific combinations.":
      'Descriptive group co-occurrence is measured at the cell level using a one-percent activation-mass presence threshold and Equation~\\ref{eq:element_ppmi}. Separately, provisional Qwen object families define panorama-level pair candidates at a two-percent mass threshold, with all 4,096 dimensions in the denominator. Candidate discovery and confirmation use disjoint city sets. Confirmation compares observed support with 999 marginal-preserving null draws stratified by city and by city plus five-kilometre block. BH-FDR is applied to the larger of the two p-values; retained pairs additionally require lift at least 1.05 under both nulls and minimum panorama, cell and city support in each split. These tests concern provisional family co-presence and do not validate spatial layouts.',
      'Within-city structure is summarized in $1\\,\\mathrm{km}\\times1\\,\\mathrm{km}$ grid cells containing at least five retained panoramas. Each valid cell receives a normalized group-composition vector. Mean cosine similarity among all valid pairs measures global visual homogeneity; mean cosine similarity among adjacent valid cells measures local continuity. Spatial permutation tests and Moran\'s $I$ are outside the presently completed analysis.':
      'Within-city structure uses all observed cells containing at least one quality-screened panorama. Mean cosine similarity among all cell pairs measures global homogeneity, while adjacent-cell mean similarity measures local continuity. City comparison plots require at least 20 adjacent pairs. Provisional pair occurrence rates are averaged equally over cells; 95-percent descriptive intervals resample five-kilometre spatial blocks. They do not account for semantic labelling errors or candidate selection.',
      'Under the stated prevalence rule, 16 of the 64 groups are classified as shared across cities, accounting for 92.47\\% of the equal-city mean dominant-patch share (Figs.~\\ref{fig:vitl30_shared_distinctive} and \\ref{fig:vitl30_city_composition}). These shares describe the strongest latent response at each patch, rather than the full sparse activation vector. The four most prevalent groups account for 72.12\\% of the equal-city mean share, so composition is concentrated in a small part of the operational vocabulary. This concentration and the unequal decoder-group sizes must be considered when interpreting city similarities.':
      f'The expanded spatial sample covers 681,645 panoramas and 72,625 observed cells across all 30 cities. Under the cell-based sharing rule, {metrics["shared_groups"]} of the 64 groups are shared, accounting for {100 * metrics["shared_mass"]:.2f}\\% of equal-city mean activation mass (Figs.~\\ref{{fig:vitl30_shared_distinctive}} and \\ref{{fig:vitl30_city_composition}}). The four groups with the highest mean shares account for {100 * metrics["top_four_mass"]:.2f}\\%. These values concern the full positive sparse activation mass; they replace the earlier dominant-patch summaries.',
      'Geographical analysis uses the 29 cities with verified coordinate mappings; 23 cities meet the plotting requirement of at least 20 adjacent valid grid-cell pairs. These composition and spatial summaries are descriptive. Co-occurrence PPMI indicates association relative to marginal prevalence and does not establish statistical significance (Fig.~\\ref{fig:vitl30_cooccurrence}).':
      f'All 30 cities enter the new spatial sample, with {metrics["spatial_plot_cities"]} meeting the plotting requirement of at least 20 adjacent-cell pairs. The provisional family analysis retains seven of 76 tested co-presence pairs after both null comparisons, FDR correction and lift filtering. Candidate city distributions describe cell-balanced co-presence rather than validated landscape prototypes. Independent road segmentation and frozen-SAE patch inference provide a {summary["examples"]}-example four-view audit; all final side labels remain undetermined pending acquisition calibration and human review. Geometric hypotheses are recorded separately and are not extrapolated to corpus-wide roadside frequencies.',
      'Geographical distribution of the verified image mappings in 29 study cities. Each marker shows the median location of mapped panoramas; marker numbers correspond to the city list on the map. Moscow is retained in feature-space analyses but omitted from geographical and image-evidence analyses because its feature-to-image row mapping could not be verified.':
      'Geographical distribution of the new spatial analysis sample in all 30 study cities. Markers locate the median UTM cell coordinates transformed to latitude and longitude. The new sample resolves original images directly by city and panorama identifier; earlier cache-based image interpretation retains its original 29-city scope.'
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
    refinements = {
      'Generated descriptions are retained with their source images and prompts; they do not enter training, clustering, or geographical analysis.':
      'Generated group descriptions are retained with their source images and prompts and do not enter SAE training or decoder clustering. Separately, provisional dimension-level Qwen family labels define the downstream family co-presence analysis; its results therefore depend on these unvalidated labels.',
      'The 2,993 Moscow panoramas remain in training and feature-composition summaries, while their unverified image mapping is excluded from overlays and spatial analysis.':
      'The 2,993 Moscow panoramas remain in the original training cohort, but their unverified cache-to-image mapping is excluded from historical semantic retrieval. The new spatial sample resolves images directly by city and panorama identifier and includes Moscow in its image audit and geographical analysis.',
      'The SAE run reuses existing half-precision contextualized patch arrays; DINOv3 features are not extracted again.':
      'SAE training reuses existing half-precision contextualized patch arrays. The larger downstream spatial sample is encoded separately using the same frozen DINOv3 backbone, contextualization, and selected SAE checkpoint.',
      'dominant-latent maps support visualization and panorama-level composition summaries.':
      'dominant-latent maps support visualization. The new spatial composition summaries aggregate all positive sparse activation mass.',
      'After the SAE and group assignments are fixed, visual evidence is retrieved from the 29 cities with verified cache-to-image mappings.':
      'After the SAE and group assignments are fixed, the original group-interpretation evidence is retrieved from the 29 cities with verified cache-to-image mappings.',
      'the annotations do not enter SAE training, decoder clustering, or geographical analysis.':
      'these group annotations do not enter SAE training, decoder clustering, or group activation-mass summaries. Separate provisional dimension-level Qwen family labels are used for the downstream pair analysis, which consequently requires semantic review.',
      'Downstream image evidence is linked only to verified cache-to-image mappings.':
      'Historical group-interpretation evidence uses verified cache-to-image mappings; the new four-view audit instead resolves images directly from the spatial sampling manifest and retains their city, panorama identifier, and grid identity.',
      'provide a 84-example four-view audit': 'provide an 84-example four-view audit',
      'Group-specific retrieval searched 3,698 candidate panoramas':
      'The original group-specific semantic retrieval searched 3,698 candidate panoramas',
      'Co-occurrence similarly describes association and requires a separate null model before a specific combination can be called statistically significant.':
      'Group PPMI remains descriptive. The separate provisional-family pair analysis uses city-stratified and spatial-block-stratified nulls with FDR correction; this evidence concerns co-presence and does not validate the family labels, object counts, or roadside layouts.'
    }
    for old, new in refinements.items():
        text = text.replace(old, new)
    text = text.replace('Full-sample activation inference is complete for all 89,031 panoramas. The resulting group-composition, co-occurrence, spatial, and image-evidence figures are derived from the selected checkpoint.',
        'Frozen inference on the original feature cohort and the larger spatial sample uses this selected checkpoint. Geographical figures now use the larger cell-balanced sample; training diagnostics and historical semantic evidence retain their original input identities.')
    text = text.replace('Figure~\\ref{fig:vitl30_activation_examples} shows localized responses with original-image details.',
        'Figure~\\ref{fig:vitl30_activation_examples} shows new four-view family attribution masks alongside independent road segmentation; these are not verified object-instance boundaries or roadside assignments.')
    text = text.replace('Full-sample activation summaries and ten result figures are complete.',
        'Downstream spatial figures have been refreshed using 681,645 quality-screened panoramas. Seven provisional family co-presence candidates have cell-balanced city distributions; their semantic and roadside validation remains pending.')
    if '\\input{chapter5_city_patterns.tex}' not in text:
        text = text.replace('\\input{chapter5_vitl30_semantics.tex}', '\\input{chapter5_vitl30_semantics.tex}\n\\input{chapter5_city_patterns.tex}')
    path.write_text(text)
    captions = {
      'Shared_Distinctive': 'Shared groups and city-specific excess in the spatial sample. Positive sparse activation mass is normalized within panorama and averaged equally across observed one-kilometre cells within each city. Shared groups exceed one percent mass in at least ten percent of cells in at least 27 cities and have equal-city mean share at least 0.5 percent. The lower panel subtracts the mean of the other cities.',
      'City_Composition': 'City composition in all 30 cities using the new 681,645-panorama sample. Colors show normalized positive SAE group activation mass averaged within one-kilometre cells and then equally across observed cells. These are not dominant-patch proportions.',
      'City_Similarity': 'Cosine similarity of cell-balanced city group activation-mass vectors. This descriptive comparison does not test city differences.',
      'Cooccurrence': 'Equal-city, equal-cell group co-occurrence. A group is present at one percent of cell-normalized activation mass. PPMI uses the weighted joint and marginal probabilities, suppresses joint probability below 0.0005, and omits the diagonal; it is not a significance test.',
      'Spatial_Structure': 'Within-city group activation-mass similarity in the observed one-kilometre UTM cells. Global similarity averages all cell pairs; local continuity averages horizontally or vertically adjacent cells. All cells with at least one retained panorama enter analysis; plotted cities require 20 adjacent pairs.',
      'Panorama_UMAP': 'UMAP of one-kilometre cell composition vectors. A deterministic hash selects 100 cells per city; cosine distance, 30 neighbours, minimum distance 0.1, and seed 42 define the projection. City labels are used only for display. Points represent cells, not individual panoramas or SAE dimensions.',
      'Activation_Examples': 'New four-view evidence for seven provisional family co-presence candidates. Left: original views in nominal heading order 0, 90, 180, 270 degrees. Right: amber and cyan show the two SAE family attribution masks; violet shows independently predicted road surface. Masks use 35 percent of each panorama family maximum. Background is dimmed; overlaps follow rendering order. These masks are neither object-instance segmentations nor verified roadside decisions.'}
    figure_text = (thesis / 'chapter5_vitl30_figures.tex').read_text()
    import re
    def update(match):
        block = match.group(0)
        for stem, caption in captions.items():
            if f'Fig_ViTL30_{stem}.pdf' in block:
                start = block.index('    \\caption{')
                end = block.index('\n    \\label', start)
                block = block[:start] + '    \\caption{' + caption + '}' + block[end:]
        return block
    figure_text = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', update, figure_text, flags=re.S)
    (thesis / 'chapter5_vitl30_figures.tex').write_text(figure_text)
    (thesis / 'chapter5_city_patterns.tex').write_text(r'''% Expanded spatial sample; provisional family co-presence, not validated prototypes.
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{img_gene/vitl30_w4096/Fig_CityPatterns_Candidate_Cities.pdf}
\caption{Cell-balanced city occurrence rates of seven provisional family co-presence candidates. Each family must exceed two percent of total positive SAE activation mass in a panorama. The city rate is the equal-cell mean of within-cell positive fractions. Qwen family labels remain provisional; these results do not establish object counts or spatial relations.}
\label{fig:citypattern_candidates}
\end{figure}
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{img_gene/vitl30_w4096/Fig_CityPatterns_Candidate_Uncertainty.pdf}
\caption{Candidate city occurrence with 95-percent percentile intervals from 999 bootstrap resamples of five-kilometre spatial blocks. Ratios preserve equal weight per observed one-kilometre cell. Intervals condition on selected candidates and provisional labels, and are not corrected for semantic errors or multiple city comparisons. Roadside prevalence remains unmeasured.}
\label{fig:citypattern_uncertainty}
\end{figure}
''')


def run(args):
    if (Path(__file__).resolve().parents[1] / 'results/current_semantic_hierarchy.json').exists():
        raise RuntimeError(
            'The 64-group thesis exporter is retired. Use '
            'experiments.vitl30_city_patterns.code.semantic_city_figures with '
            'the screened semantic snapshot; legacy installation would overwrite current results.'
        )
    for stage in ('road_sides', 'prototype_city_distribution'):
        completed = json.loads((args.followup / 'stages' / stage / 'complete.json').read_text())
        if completed.get('status') != 'complete' or completed.get('run_identity') != args.run_identity:
            raise ValueError(f'Unverified prerequisite: {stage}')
    source = json.loads((args.followup / 'complete.json').read_text())
    if source['identity']['extract_run_identity'] != args.run_identity or source['cities'] != 30:
        raise ValueError('Figure inputs do not match the 30-city extraction')
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    metrics = plots(args, figures)
    candidate_plots(args, figures)
    image_figure(args, figures)
    # This figure and semantic evidence use unchanged original retrieved crops.
    vlm = args.thesis / 'img_gene/vitl30_w4096/Fig_ViTL30_VLM_Interpretation.pdf'
    for suffix in ('.pdf', '.png'):
        source = vlm.with_suffix(suffix)
        if source.exists():
            shutil.copy2(source, figures / source.name)
    audit = json.loads((args.followup / 'stages/road_sides/summary.json').read_text())
    revise_writing(args.thesis, metrics, audit)
    target = args.thesis / 'img_gene/vitl30_w4096'
    for source in figures.iterdir():
        if source.suffix in {'.pdf', '.png'}:
            shutil.copy2(source, target / source.name)
    inventory = []
    for figure in sorted(figures.glob('*.pdf')):
        kind = ('original_training_cohort' if any(k in figure.name for k in ('Training_Loss', 'Decoder_Hierarchy'))
                else 'original_29city_semantic_evidence' if 'VLM_Interpretation' in figure.name
                else 'new_spatial_sample')
        inventory.append({'file': str(figure), 'sha256': sha256(figure), 'provenance': kind,
                          'action': 'retained_unchanged' if 'VLM_Interpretation' in figure.name else 'regenerated'})
    # Explicitly inventory unchanged diagrams and semantic embeddings.
    for source in [args.thesis / 'figure/pipeline.jpg', args.thesis / 'figure/panorama-aware.jpg',
                   *sorted(target.glob('*UMAP*.pdf'))]:
        if source.exists() and source.name != 'Fig_ViTL30_Panorama_UMAP.pdf':
            inventory.append({'file': str(source), 'sha256': sha256(source),
                              'provenance': 'original_frozen_representation_or_semantic_evidence',
                              'action': 'retained_unchanged_input_identity'})
    write_json(args.output / 'figure_inventory.json', {'run_identity': args.run_identity, 'figures': inventory,
        'metrics': metrics, 'note': 'Activation-mass statistics replace dominant-patch statistics together with captions.'})
    command = ['latexmk', '-xelatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               '-outdir=build', '000_thesis.tex']
    with (args.output / 'compile.log').open('w') as log:
        result = subprocess.run(command, cwd=args.thesis, stdout=log, stderr=subprocess.STDOUT)
    pdf = args.thesis / 'build/000_thesis.pdf'
    passed = result.returncode == 0 and pdf.is_file() and pdf.stat().st_size > 100000
    write_json(args.output / 'compile_report.json', {'passed': passed, 'returncode': result.returncode,
        'command': command, 'thesis': str(args.thesis), 'independent_semantic_validation': False})
    if not passed:
        raise RuntimeError('Thesis compilation failed; see compile.log')
    shutil.copy2(pdf, args.output / 'thesis_preview.pdf')
    write_json(args.output / 'complete.json', {'schema_version': 1, 'stage': 'chapter5_figures',
        'run_identity': args.run_identity, 'status': 'complete',
        'evidence': {'figures': len(inventory), 'compiled': True, 'semantic_and_roadside_review_pending': True}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    p.add_argument('--followup', type=Path, default=FOLLOWUP)
    p.add_argument('--output', type=Path, default=FOLLOWUP / 'stages/chapter5_figures')
    p.add_argument('--run-identity', required=True)
    p.add_argument('--thesis', type=Path, default=THESIS)
    run(p.parse_args())
