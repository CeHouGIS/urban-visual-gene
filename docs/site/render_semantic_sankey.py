"""Draw conservative parent-to-child Sankey diagrams for SAE semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

try:
    from .render_semantic_paper_figure import check_map, family_palette
except ImportError:
    from render_semantic_paper_figure import check_map, family_palette


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiments/vitl30_relational_prototypes/results/figures'


def displayed_children(family):
    children = [{'count': child['count']} for child in family['children']]
    if sum(child['count'] for child in children) != family['count']:
        raise ValueError(f'Sankey flows do not conserve {family["id"]}')
    return children


def layout(families, unit, child_gap, parent_gap, top=88, bottom=46):
    parents, flows = [], []
    cursor = float(top)
    for family_index, family in enumerate(families):
        children = displayed_children(family)
        internal_gap = (len(children) - 1) * child_gap
        parent_top = cursor + internal_gap / 2
        source = parent_top
        target = cursor
        for child_index, child in enumerate(children):
            height = child['count'] * unit
            flows.append({'family_index': family_index, 'child_index': child_index,
                          'count': child['count'],
                          'source_top': source, 'target_top': target, 'height': height})
            source += height
            target += height + child_gap
        parents.append({'family_index': family_index, 'top': parent_top,
                        'height': family['count'] * unit,
                        'child_count': len(children)})
        cursor += family['count'] * unit + internal_gap + parent_gap
    if sum(flow['count'] for flow in flows) != 4096:
        raise ValueError('Sankey flows must cover exactly 4096 SAE dimensions')
    return parents, flows, cursor - parent_gap + bottom


def ribbon(x0, x1, source, target, height):
    bend = (x1 - x0) * .42
    vertices = [(x0, source), (x0 + bend, source), (x1 - bend, target),
                (x1, target), (x1, target + height),
                (x1 - bend, target + height),
                (x0 + bend, source + height), (x0, source + height),
                (x0, source)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    return MplPath(vertices, codes)


def render_variant(families, palette, output_dir, *, unit, child_gap,
                   parent_gap, suffix, dpi, label_size, show_child_ids):
    parents, flows, height = layout(families, unit, child_gap, parent_gap)
    width = 920
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=dpi, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    ax.text(25, 28, 'SAE semantic taxonomy: parent to fine label',
            fontsize=14, weight='bold', color='#1d292e', va='center')
    ax.text(25, 49, '22 parent classes  |  190 subclasses  |  4,096 SAE dimensions',
            fontsize=9.3, color='#58656c', va='center')
    ax.text(18, 71, 'QWEN PARENT CLASS', fontsize=8.4, weight='bold', color='#56636b')
    ax.text(610, 71, 'SUBCLASS NODES (NAMES HIDDEN)', fontsize=8.4,
            weight='bold', color='#56636b')
    if show_child_ids:
        ax.text(886, 71, 'DIM.', fontsize=8.4, weight='bold',
                ha='right', color='#56636b')
    ax.plot([20, 897], [77, 77], color='#a5aeb4', lw=.65)

    parent_x, parent_width = 220, 14
    child_x, child_width = 585, 10
    for flow in flows:
        color = palette[flow['family_index']]
        ax.add_patch(PathPatch(ribbon(parent_x + parent_width, child_x,
                                      flow['source_top'], flow['target_top'],
                                      flow['height']),
                               facecolor=color, edgecolor='none', alpha=.42))
    for parent in parents:
        family_index = parent['family_index']
        family = families[family_index]
        middle = parent['top'] + parent['height'] / 2
        ax.add_patch(Rectangle((parent_x, parent['top']), parent_width,
                               parent['height'], facecolor=palette[family_index],
                               edgecolor='none'))
        ax.text(207, middle,
                f"{family_index + 1:02d}  {family['label_en']}  ({family['count']:,})",
                ha='right', va='center', fontsize=label_size,
                color='#263238')
    for flow in flows:
        family_index = flow['family_index']
        middle = flow['target_top'] + flow['height'] / 2
        ax.add_patch(Rectangle((child_x, flow['target_top']), child_width,
                               flow['height'], facecolor=palette[family_index],
                               edgecolor='none'))
        if show_child_ids:
            ax.text(610, middle,
                    f"{family_index + 1:02d}.{flow['child_index'] + 1:02d}",
                    ha='left', va='center', fontsize=label_size, color='#263238')
            ax.text(886, middle, f"{flow['count']:,}", ha='right', va='center',
                    fontsize=label_size, color='#48565d')

    ax.plot([20, 897], [height - 35, height - 35], color='#c7cdd1', lw=.55)
    ax.text(25, height - 19,
            'Ribbon thickness is proportional to SAE dimensions. Subclass names are hidden; no clustering distance is implied.',
            fontsize=8, color='#57636a', va='center')
    stem = 'Fig_Qwen_Parent_Child_Sankey' + suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = output_dir / f'{stem}.png', output_dir / f'{stem}.pdf'
    fig.savefig(png, dpi=dpi, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    return png, pdf, len(flows)


def render(site, output_dir=DEFAULT_OUTPUT):
    data = json.loads((site / 'semantic-map/map.json').read_text())
    families = check_map(data)
    palette = family_palette(families)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    compact = render_variant(families, palette, output_dir,
                             unit=.04, child_gap=1.2, parent_gap=9,
                             suffix='', dpi=260, label_size=9,
                             show_child_ids=False)
    full = render_variant(families, palette, output_dir,
                          unit=.35, child_gap=9.5, parent_gap=14,
                          suffix='_Full', dpi=180, label_size=9,
                          show_child_ids=True)
    return compact, full


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for result in render(args.site_dir.resolve(), args.output_dir.resolve()):
        print(*result)
