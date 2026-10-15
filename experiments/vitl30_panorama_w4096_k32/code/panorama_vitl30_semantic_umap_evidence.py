"""Show original images and saved activations for the six annotated UMAP dimensions."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import io
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

from formal.analysis.semantic_feature_umap import digest, write_json
from sae_experiments.activation_visualization import activation_overlay, mark_peak
from sae_experiments.panorama_vitl30 import OUTPUT
from sae_experiments.panorama_vitl30_analysis import checked_groups


def panel(features, examples, labels, report, name):
    height = 3.15 * len(features) + .6
    fig, axes = plt.subplots(len(features), 4, figsize=(14, height), squeeze=False)
    for y, feature in enumerate(features):
        for rank in range(2):
            row = examples[feature][rank]
            city = row['city'].split('/')[-1]
            for offset, kind in enumerate(['original', 'activation']):
                ax = axes[y, rank * 2 + offset]
                with Image.open(report / row[kind]) as image:
                    ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if offset == 0:
                    ax.set_title(f'Top {row["rank"]} | {city} | {row["heading"]} deg\nOriginal', fontsize=10)
                else:
                    ax.set_title(f'Activation | peak={row["activation_peak"]:.4f}', fontsize=10)
                if rank == 0 and offset == 0:
                    ax.set_ylabel(f'f{feature} / G{labels[feature]:02d}', fontsize=12, labelpad=10)
    fig.suptitle('Actual top-activation examples for annotated SAE dimensions', fontsize=15, y=.99)
    fig.subplots_adjust(left=.065, right=.99, top=1 - .85 / height, bottom=.95 / height,
                        hspace=.30, wspace=.06)
    bar_ax = fig.add_axes([.71, .4 / height, .25, .1 / height])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='inferno'), cax=bar_ax,
                      orientation='horizontal', ticks=[0, .5, 1])
    bar.ax.tick_params(labelsize=8)
    bar.set_label('Relative activation within each directional image', fontsize=8)
    fig.text(.065, .3 / height, 'Two highest-ranked distinct panoramas per dimension.\n'
             'Cyan box: region used for the semantic crop; heatmaps use per-image peak normalization.',
             fontsize=9, color='#526071')
    for suffix in ['png', 'pdf']:
        target = report / f'{name}.{suffix}'
        fig.savefig(target, dpi=170)
        shutil.copy2(target, OUTPUT / 'figures' / target.name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=OUTPUT / 'analysis/semantic_umap_top100')
    parser.add_argument('--report', type=Path,
                        default=Path('/root/code/urban-visual-gene/reports/vitl30_semantic_umap'))
    args = parser.parse_args()
    run = json.loads((args.result / 'run.json').read_text())
    summary = json.loads((args.report / 'summary.json').read_text())
    features = summary['illustrated_dimensions']
    labels = checked_groups(args.result.parent.parent, run['checkpoint_sha256'])
    examples = defaultdict(list)
    for line in (args.result / 'crops.jsonl').open():
        row = json.loads(line)
        if row['feature'] in features and row['rank'] <= 2:
            examples[row['feature']].append(row)
    rendered = defaultdict(list)
    manifest = []
    for feature in features:
        selected = sorted(examples[feature], key=lambda row: row['rank'])
        if [row['rank'] for row in selected] != [1, 2] or len({row['pano_index'] for row in selected}) != 2:
            raise ValueError(f'Expected the two distinct highest-ranked panoramas: f{feature}')
        for row in selected:
            folder = args.report / 'annotated_evidence' / f'f{feature}'
            folder.mkdir(parents=True, exist_ok=True)
            with Image.open(row['image']) as source:
                image = source.convert('RGB')
            with Image.open(io.BytesIO(base64.b64decode(row['heatmap_png']))) as source:
                heat = np.asarray(source, dtype=np.float32) / 255
            if list(image.size) != row['native_view_size'] or heat.shape != (run['grid'], run['grid']):
                raise ValueError('Original image and saved heatmap geometry do not match')
            if heat.max() != 1 or heat[tuple(row['peak_patch'])] <= 0:
                raise ValueError('Saved normalized heatmap is invalid')
            original = folder / f'rank{row["rank"]}_original.png'
            activation = folder / f'rank{row["rank"]}_activation.png'
            image.save(original)
            overlay = activation_overlay(image, heat, 1)
            mark_peak(overlay, row['bbox_native_view'])
            overlay.save(activation)
            item = {'feature': feature, 'group': int(labels[feature]), 'rank': row['rank'],
                    'city': row['city'], 'pano': row['pano'], 'heading': row['heading'],
                    'image_id': row['image_id'], 'source': row['image'],
                    'activation_peak': row['activation'], 'bbox_native_view': row['bbox_native_view'],
                    'original': str(original.relative_to(args.report)),
                    'activation': str(activation.relative_to(args.report)),
                    'heatmap_source': 'saved per-dimension sparse 28x28 activation, PNG quantized to uint8',
                    'normalization': 'directional-image peak; color intensity is not comparable across images'}
            rendered[feature].append(item)
            manifest.append(item)
    for part, start in enumerate(range(0, len(features), 3), 1):
        panel(features[start:start + 3], rendered, labels, args.report,
              f'Fig_ViTL30_Annotated_UMAP_Activations_{part}')
    dimension_selection = summary.get('illustration_selection', {})
    replacement_features = [item['feature'] for item in dimension_selection.get('replacements', [])]
    if replacement_features:
        panel(replacement_features, rendered, labels, args.report,
              'Fig_ViTL30_Replaced_UMAP_Activations')
    write_json(args.report / 'annotated_activation_evidence.json', {
        'checkpoint_sha256': run['checkpoint_sha256'],
        'crop_manifest_sha256': digest(args.result / 'crops.jsonl'),
        'features': features, 'selection': 'ranks 1 and 2 per dimension; no manual example selection',
        'dimension_selection': dimension_selection,
        'examples': manifest})
    report_path = args.report / 'REPORT.md'
    section = '\n## Annotated dimensions: original images and activations\n\n'
    section += '每维展示当前候选集激活排名前两位的不同全景，使用已保存的逐维稀疏激活。黄色较强，暗色较弱；青色框为语义裁剪区域。热图按每张方向图的峰值归一化，跨图强度请参照原始 peak。\n\n'
    section += '\n\n'.join(f'![Annotated activation evidence {i}](Fig_ViTL30_Annotated_UMAP_Activations_{i}.png)'
                             for i in range(1, (len(features) + 2) // 3 + 1))
    text = report_path.read_text().split('\n## Annotated dimensions: original images and activations')[0]
    report_path.write_text(text.rstrip() + '\n' + section + '\n')
    print(json.dumps({'dimensions': features, 'examples': len(manifest), 'report': str(args.report)}, indent=2))


if __name__ == '__main__':
    main()
