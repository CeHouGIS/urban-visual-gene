"""Show diverse supported dimensions beside their actual top activation crops."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from formal.analysis.semantic_feature_umap import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    root = args.result
    run = json.loads((root / 'run.json').read_text())
    with np.load(root / 'feature_semantic_umap.npz') as data:
        count, coherence, eligible, xy = (data[k] for k in ('count', 'coherence', 'eligible', 'xy'))
    prototypes = np.load(root / 'feature_semantic_embeddings.f32.npy')
    available = np.flatnonzero(eligible & (count >= run['top']))
    if not len(available):
        available = np.flatnonzero(eligible)
    # Pick separated prototypes without assigning semantic category labels.
    chosen = [int(available[np.argmax(coherence[available])])]
    for _ in range(min(6, len(available)) - 1):
        distance = 1 - prototypes[available] @ prototypes[chosen].T
        minimum = distance.min(axis=1)
        minimum[np.isin(available, chosen)] = -np.inf
        chosen.append(int(available[np.argmax(minimum)]))
    examples = defaultdict(list)
    for line in (root / 'crops.jsonl').open():
        row = json.loads(line)
        if row['feature'] in chosen and len(examples[row['feature']]) < 8:
            examples[row['feature']].append(row)
    fig, axes = plt.subplots(len(chosen), 8, figsize=(16, len(chosen) * 2), squeeze=False)
    for i, feature in enumerate(chosen):
        for j, ax in enumerate(axes[i]):
            ax.set_xticks([]); ax.set_yticks([])
            if j < len(examples[feature]):
                row = examples[feature][j]
                with Image.open(root / row['crop']) as crop:
                    ax.imshow(crop.convert('RGB'))
                ax.set_title(f"#{row['rank']} | {row['city']}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"f{feature}\nn={count[feature]}\nagreement={coherence[feature]:.2f}", fontsize=10)
    fig.suptitle('Six dimensions with separated crop-semantic prototypes\n'
                 'Eight highest-activation regions per dimension; no category labels assigned', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(root / 'semantic_crop_examples.png', dpi=160)
    plt.close(fig)
    support = np.load(root / 'positive_image_support.npy')
    report = {
        'checkpoint': run['checkpoint'], 'width': run['width'], 'k': run['k'],
        'evidence_cities': len(run['cities']), 'evidence_panos': run['images'] // 4,
        'evidence_images': run['images'], 'top_requested': run['top'],
        'dimensions_zero_activation': int((support == 0).sum()),
        'dimensions_with_full_top100': int((count >= run['top']).sum()),
        'dimensions_with_20_to_99': int(((count >= 20) & (count < run['top'])).sum()),
        'dimensions_projected': int(eligible.sum()),
        'dimensions_excluded': int((~eligible).sum()),
        'valid_crops': int(count.sum()),
        'crop_errors': len(json.loads((root / 'crop_errors.json').read_text())),
        'finite_prototypes': bool(np.isfinite(prototypes).all()),
        'finite_projected_coordinates': bool(np.isfinite(xy[eligible]).all()),
        'illustrated_dimensions': chosen,
        'illustration_selection': 'Start with highest agreement, then farthest cosine prototypes with full Top100; diagnostic only',
    }
    write_json(root / 'test_summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
