"""Plot parent-class size and child-size distributions for the SAE taxonomy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

try:
    from .render_semantic_paper_figure import check_map, family_palette, figure_examples
    from .render_semantic_map import activation_thumbnail
except ImportError:
    from render_semantic_paper_figure import check_map, family_palette, figure_examples
    from render_semantic_map import activation_thumbnail


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'experiments/vitl30_relational_prototypes/results/figures'
STEM = 'Fig_Qwen_Parent_Child_Distribution'


def child_points(family):
    """Keep one point per subclass while separating repeated integer counts."""
    by_count = defaultdict(list)
    for child in family['children']:
        by_count[child['count']].append(child)
    if sum(count * len(children) for count, children in by_count.items()) != family['count']:
        raise ValueError(f'Child counts do not sum to {family["id"]}')

    points = []
    for count, children in sorted(by_count.items()):
        columns = min(5, int(np.ceil(np.sqrt(len(children)))))
        rows = int(np.ceil(len(children) / columns))
        for index, _ in enumerate(children):
            column, row = index % columns, index // columns
            x = count * 10 ** ((column - (columns - 1) / 2) * .041)
            y = (row - (rows - 1) / 2) * .135
            points.append((x, y))
    return points


def render(site, output_dir=DEFAULT_OUTPUT):
    data = json.loads((site / 'semantic-map/map.json').read_text())
    families = check_map(data)
    neutral = np.array(to_rgb('#777d80'))
    palette = [to_hex(.83 * np.array(to_rgb(color)) + .17 * neutral)
               for color in family_palette(families)]
    points_by_family = [child_points(family) for family in families]
    if sum(map(len, points_by_family)) != 190:
        raise ValueError('Expected one plotted point per subclass')

    plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(13.8, 9.4), facecolor='white')
    row_bottom, row_height = .10, .795
    bars = fig.add_axes([.245, row_bottom, .20, row_height])
    dots = fig.add_axes([.50, row_bottom, .28, row_height], sharey=bars)
    positions = np.arange(len(families))

    bars.barh(positions, [family['count'] for family in families],
              height=.52, color=palette, edgecolor='none', zorder=2)
    bars.set_xlim(0, 1460)
    bars.set_ylim(len(families) - .5, -.5)
    bars.set_yticks(positions)
    bars.set_yticklabels([f"{index + 1:02d}  {family['label_en']}"
                          for index, family in enumerate(families)], fontsize=13,
                         color='#23282b')
    bars.tick_params(axis='y', length=0, pad=8)
    bars.set_xticks([0, 500, 1000])
    bars.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    bars.tick_params(axis='x', labelsize=12, length=2.5, colors='#4d5559')
    bars.set_xlabel('SAE dimensions', fontsize=13,
                    color='#343b3f', labelpad=7)
    for y, family in enumerate(families):
        bars.text(family['count'] + 22, y, f"{family['count']:,}",
                  va='center', fontsize=12.5, color='#343b3f')

    dots.set_xscale('log')
    dots.set_xlim(.72, 1650)
    dots.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    dots.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    dots.minorticks_off()
    dots.tick_params(axis='x', labelsize=12, length=2.5, colors='#4d5559')
    dots.tick_params(axis='y', left=False, labelleft=False)
    dots.set_xlabel('Dimensions per subclass (log scale)', fontsize=13,
                    color='#343b3f', labelpad=7)
    image_height = row_height / len(families) * .87
    image_width = image_height * fig.get_figheight() / fig.get_figwidth()
    image_left, image_step = .849, .027
    for y, (family, color, points) in enumerate(zip(families, palette, points_by_family)):
        dots.scatter([x for x, _ in points], [y + offset for _, offset in points],
                     s=16, color=color, edgecolor='white', linewidth=.25,
                     zorder=3, rasterized=False)
        row_center = row_bottom + row_height * (1 - (y + .5) / len(families))
        fig.text(.813, row_center, str(len(points)), ha='center', va='center',
                 fontsize=12.5, color='#343b3f')
        for index, example in enumerate(figure_examples(site, family)):
            image_ax = fig.add_axes([image_left + index * image_step,
                                     row_center - image_height / 2,
                                     image_width, image_height])
            image_ax.imshow(activation_thumbnail(example, site, size=128),
                            interpolation='nearest')
            image_ax.set_xticks([])
            image_ax.set_yticks([])
            for spine in image_ax.spines.values():
                spine.set_color('#aeb5b8')
                spine.set_linewidth(.35)

    for axis in (bars, dots):
        axis.xaxis.grid(True, color='#e6e9e9', linewidth=.55, zorder=0)
        axis.yaxis.grid(True, color='#eff0f0', linewidth=.4, zorder=0)
        axis.set_axisbelow(True)
        for side in ('left', 'right', 'top'):
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color('#81898d')
        axis.spines['bottom'].set_linewidth(.55)

    fig.text(.245, .948, 'a', fontsize=15, weight='bold', color='#1f2528')
    fig.text(.269, .948, 'Parent-class size', fontsize=14, color='#1f2528')
    fig.text(.50, .948, 'b', fontsize=15, weight='bold', color='#1f2528')
    fig.text(.525, .948, 'Subclass sizes', fontsize=14, color='#1f2528')
    fig.text(.849, .948, 'c', fontsize=15, weight='bold', color='#1f2528')
    fig.text(.874, .948, 'Activation examples', fontsize=14, color='#1f2528')
    fig.text(.813, .917, 'No.', fontsize=12, ha='center', color='#596267')
    fig.text(.905, .917, 'Four per parent class', fontsize=12,
             ha='center', color='#596267')
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f'{STEM}.png'
    pdf = output_dir / f'{STEM}.pdf'
    fig.savefig(png, dpi=350, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    return png, pdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in render(args.site_dir.resolve(), args.output_dir.resolve()):
        print(path)
