"""Render a paper figure and its complete activation-evidence companion."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

try:
    from .render_semantic_map import activation_thumbnail
except ImportError:
    from render_semantic_map import activation_thumbnail


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiments/vitl30_relational_prototypes/results/figures'
STEM = 'Fig_Qwen_Semantic_Hierarchy'


def check_map(data):
    families = data['families']
    leaves = [feature for family in families for child in family['children']
              for feature in child['features']]
    if (len(families) != data['family_count'] or len(families) != 22
            or data['total'] != 4096 or data['leaf_count'] != 4096
            or sorted(leaves) != list(range(4096))):
        raise ValueError('Paper figure requires each of the 4096 SAE dimensions exactly once')
    if any(len(family['examples']) != 4 or not family['label_en'] for family in families):
        raise ValueError('Every parent requires an English title and four activation examples')
    if sum(len(family['children']) for family in families) != 190:
        raise ValueError('The current Qwen taxonomy must have 190 child groups')
    return families


def family_palette(families):
    # Two review categories have distinct neutrals; the other 20 use discrete colors.
    tab20 = plt.get_cmap('tab20').colors
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    ordinary = iter(tab20[index] for index in order)
    return [('#898f96' if family['id'] == 'unresolved' else
             '#b35c54' if family['id'] == 'spatial_encoding' else next(ordinary))
            for family in families]


def figure_examples(site, family):
    examples = list(family['examples'])
    if family['id'] == 'sky':
        with gzip.open(site / 'semantic/features/97.json.gz', 'rt') as stream:
            cloud = next((item for item in json.load(stream)['examples']
                          if item['image'] == 7602), None)
        cloud_child = next((child for child in family['children']
                            if child['label'] == '云'), None)
        if cloud is None or cloud['activation'] <= 0 or cloud_child is None or 97 not in cloud_child['features']:
            raise ValueError('Replacement must be a positive response from the cloud subclass')
        # The original f3478 example responds to artificial lights, not the sun.
        examples[2] = {
            'label': cloud_child['label'], 'feature': 97,
            'city': cloud['city'], 'heading': cloud['heading'],
            'image': f"../semantic/sources/{cloud['image']}.jpg",
            'heat': cloud['heat'], 'family': family['id'],
        }
    return examples


def dimension_grid(families):
    """Preserve feature ID order so all 4096 individual cells are addressable."""
    grid = np.full(4096, -1, dtype=np.int16)
    for family_index, family in enumerate(families):
        for child in family['children']:
            for feature in child['features']:
                if grid[feature] != -1:
                    raise ValueError(f'Duplicate SAE dimension f{feature}')
                grid[feature] = family_index
    if (grid < 0).any():
        raise ValueError('The figure is missing SAE dimensions')
    return grid.reshape(64, 64)


def save(fig, output_dir, stem):
    png = output_dir / f'{stem}.png'
    pdf = output_dir / f'{stem}.pdf'
    fig.savefig(png, dpi=400, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    return png, pdf


def render_hierarchy(families, palette, output_dir):
    fig = plt.figure(figsize=(14, 7.3), dpi=400, facecolor='white')
    ax = fig.add_axes([.285, .135, .305, .795])
    ax.set_xlim(0, 1660)
    ax.set_ylim(-.65, len(families) - .35)
    ax.invert_yaxis()
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f"{index + 1:02d}  {family['label_en']}"
                        for index, family in enumerate(families)], fontsize=9.5)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.set_xticks([0, 500, 1000])
    ax.tick_params(axis='x', labelsize=8.5, length=3, color='#9da6ae')
    ax.set_xlabel('SAE dimensions', fontsize=9.5, labelpad=7)
    ax.xaxis.grid(True, color='#e5e9eb', linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#919aa2')

    for index, (family, color) in enumerate(zip(families, palette)):
        offset = 0
        for child in family['children']:
            ax.barh(index, child['count'], left=offset, height=.68,
                    color=color, edgecolor='white', linewidth=.32, zorder=2)
            offset += child['count']
        if offset != family['count']:
            raise ValueError(f'Incorrect subclass count for {family["id"]}')
        ax.text(min(offset + 20, 1320), index, f'{offset:,}', va='center',
                fontsize=8.5, color='#273036', fontweight='bold')
        ax.text(1460, index, str(len(family['children'])), va='center',
                fontsize=8.3, color='#4e5860')

    fig.text(.025, .955, 'a', fontweight='bold', fontsize=15, va='center')
    fig.text(.055, .955, 'Parent and subclass sizes', fontsize=12,
             fontweight='bold', va='center')
    fig.text(.545, .955, 'No. of\nsubclasses', fontsize=8,
             ha='center', va='center', color='#56626b')
    fig.text(.025, .045,
             'Bar length = number of SAE dimensions; white separators mark Qwen fine-label groups.',
             fontsize=8, color='#56626b')

    grid = dimension_grid(families)
    rgb = np.asarray([to_rgb(color) for color in palette])[grid]
    grid_ax = fig.add_axes([.652, .208, .303, .581])
    grid_ax.imshow(rgb, origin='upper', interpolation='nearest', aspect='equal',
                   extent=(-.5, 63.5, 63.5, -.5))
    grid_ax.set_xlim(-.5, 63.5)
    grid_ax.set_ylim(63.5, -.5)
    grid_ax.set_xticks([0, 16, 32, 48, 63])
    grid_ax.set_yticks([0, 16, 32, 48, 63])
    grid_ax.set_yticklabels(['0', '1,024', '2,048', '3,072', '4,032'])
    grid_ax.set_xlabel('Column within 64-dimension block', fontsize=9.5, labelpad=8)
    grid_ax.set_ylabel('First dimension ID in block', fontsize=9.5, labelpad=8)
    grid_ax.tick_params(labelsize=8.5, length=2.5, color='#8c959d')
    for position in np.arange(15.5, 64, 16):
        grid_ax.axhline(position, color='white', lw=.55)
        grid_ax.axvline(position, color='white', lw=.55)
    for spine in grid_ax.spines.values():
        spine.set_linewidth(.6)
        spine.set_color('#768189')
    fig.text(.63, .955, 'b', fontweight='bold', fontsize=15, va='center')
    fig.text(.659, .955, 'Complete 4,096-dimension census', fontsize=12,
             fontweight='bold', va='center')
    fig.text(.63, .105,
             'Each cell is one feature (ID = row x 64 + column). Colors match parent classes in a.',
             fontsize=8, color='#56626b')
    fig.text(.63, .073,
             'Positions retain SAE dimension IDs; neither color nor adjacency encodes semantic distance.',
             fontsize=8, color='#56626b')
    return save(fig, output_dir, STEM)


def render_evidence(site, families, palette, output_dir):
    fig = plt.figure(figsize=(14, 8.6), dpi=400, facecolor='white')
    fig.text(.028, .958, 'Activation examples by provisional parent class',
             fontsize=13, fontweight='bold', va='center')
    fig.text(.028, .93, 'Four positive-activation directional views per class; feature IDs refer to the SAE dictionary.',
             fontsize=8.5, color='#59636b', va='center')
    for index, (family, color) in enumerate(zip(families, palette)):
        column, row = divmod(index, 11)
        start_x = .025 + .50 * column
        top = .895 - row * .078
        wrapped = textwrap.fill(f"{index + 1:02d}  {family['label_en']}", width=24)
        fig.text(start_x + .006, top - .008, wrapped, ha='left', va='top',
                 fontsize=9.3, fontweight='bold', linespacing=1.05, color='#242a2e')
        fig.text(start_x + .006, top - .061,
                 f"{family['count']:,} dimensions  /  {len(family['children'])} subclasses",
                 ha='left', va='bottom', fontsize=7.2, color='#66727b')
        fig.add_artist(plt.Line2D([start_x, start_x + .45], [top - .075, top - .075],
                                  transform=fig.transFigure, color='#d8dde0', lw=.5))
        fig.add_artist(plt.Line2D([start_x, start_x], [top - .068, top],
                                  transform=fig.transFigure, color=color, lw=3.2))
        for example_index, example in enumerate(figure_examples(site, family)):
            thumb = np.asarray(activation_thumbnail(example, site, size=144))
            image_ax = fig.add_axes([start_x + .218 + example_index * .057,
                                     top - .068, .047, .068])
            image_ax.imshow(thumb, interpolation='nearest')
            image_ax.set_axis_off()
            image_ax.text(.03, .97, f"f{example['feature']}", transform=image_ax.transAxes,
                          fontsize=6.8, va='top', color='#1f292d',
                          bbox={'facecolor': 'white', 'edgecolor': 'none',
                                'alpha': .87, 'pad': .5})
    fig.text(.028, .02,
             'Yellow overlays show normalized patch-level SAE responses, not object masks. '
             'The four unresolved examples come from weak positive responses.',
             fontsize=8, color='#56626b')
    return save(fig, output_dir, f'{STEM}_Evidence')


def render(site, output_dir=DEFAULT_OUTPUT):
    data = json.loads((site / 'semantic-map/map.json').read_text())
    families = check_map(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    palette = family_palette(families)
    key_path = output_dir / f'{STEM}_Subclass_Key.csv'
    with key_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['parent_number', 'parent_en', 'subclass_number',
                         'subclass_label_zh', 'dimension_count', 'feature_ids'])
        for parent_number, family in enumerate(families, 1):
            for subclass_number, child in enumerate(family['children'], 1):
                writer.writerow([parent_number, family['label_en'], subclass_number,
                                 child['label'], child['count'],
                                 ' '.join(f'f{feature}' for feature in child['features'])])
    return (*render_hierarchy(families, palette, output_dir),
            *render_evidence(site, families, palette, output_dir), key_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in render(args.site_dir.resolve(), args.output_dir.resolve()):
        print(path)
