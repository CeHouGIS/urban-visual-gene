"""Publishable UMAP atlas: shared original views and per-feature lazy data.

Export only original-view WebP previews and normalized crop boxes. The browser
renders exact crop content from the shared images; no need to publish 400k crops.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from formal.analysis.semantic_feature_umap import source_image, digest


def save_view(task):
    index, row, output = task
    image = source_image(row)
    size = list(image.size)
    for directory, bound, quality in [('images', 960, 82), ('thumbs', 320, 76)]:
        target = Path(output) / directory / f'{index:04d}.webp'
        if not target.exists():
            preview = image.copy()
            preview.thumbnail((bound, bound), Image.Resampling.LANCZOS)
            preview.save(target, quality=quality, method=4)
    return size


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shared-assets', type=Path,
                   help='Reuse an existing semantic site image catalog and image files')
    p.add_argument('--image-base', default='',
                   help='Relative URL prefix for shared images, e.g. ../semantic/')
    args = p.parse_args()
    root, out = args.result.resolve(), args.out.resolve()
    for directory in ['features'] + ([] if args.shared_assets else ['images', 'thumbs']):
        (out / directory).mkdir(parents=True, exist_ok=True)
    run = json.loads((root / 'run.json').read_text())
    images = [json.loads(line) for line in (root / 'images.jsonl').read_text().splitlines()]
    if args.shared_assets:
        shared = args.shared_assets.resolve()
        previous = json.loads((shared / 'atlas.json').read_text())
        old_images = previous['images']
        identities = [(r['city'], r['pano'], int(r['heading'])) for r in images]
        old_identities = [(r['city'], r['pano'], int(r['heading'])) for r in old_images]
        if identities != old_identities:
            raise ValueError('Shared image catalog does not match the semantic evaluation manifest')
        sizes = [r['size'] for r in old_images]
    else:
        with ProcessPoolExecutor(args.workers) as executor:
            sizes = []
            for size in executor.map(save_view, ((i, r, str(out)) for i, r in enumerate(images)), chunksize=8):
                sizes.append(size)
                if len(sizes) % 400 == 0:
                    print(f'Exported {len(sizes)}/{len(images)} original views', flush=True)
    by_feature = defaultdict(list)
    for line in (root / 'crops.jsonl').open():
        row = json.loads(line)
        width, height = row['native_view_size']
        bbox = [round(v / (width if k % 2 == 0 else height), 7) for k, v in enumerate(row['bbox_native_view'])]
        by_feature[row['feature']].append({'image': row['image_id'], 'rank': row['rank'],
                                         'activation': round(row['activation'], 6), 'bbox': bbox})
    with np.load(root / 'feature_semantic_umap.npz') as data:
        xy, count, coherence, eligible = (data[k] for k in ['xy', 'count', 'coherence', 'eligible'])
    points = []
    for feature in range(run['width']):
        examples = sorted(by_feature[feature], key=lambda r: r['rank'])
        save_json(out / 'features' / f'{feature}.json', {'feature': feature, 'examples': examples})
        points.append({'feature': feature, 'x': float(xy[feature, 0]) if eligible[feature] else None,
                       'y': float(xy[feature, 1]) if eligible[feature] else None,
                       'count': int(count[feature]), 'coherence': round(float(coherence[feature]), 5),
                       'preview': examples[:3]})
    catalog = [{'city': r['city'], 'pano': r['pano'], 'heading': r['heading'], 'size': s,
                'source': 'Baidu' if r.get('view_bbox') else 'Google'} for r, s in zip(images, sizes)]
    save_json(out / 'atlas.json', {'version': f"w{run['width']}-k{run['k']}-top{run['top']}",
                                 'width': run['width'], 'k': run['k'], 'top': run['top'],
                                 'cities': len(run['cities']), 'panos': len(images) // 4,
                                 'crop_count': int(count.sum()), 'points': points, 'images': catalog})
    save_json(out / 'provenance.json', {
        'checkpoint_sha256': run['checkpoint_sha256'],
        'manifest_sha256': run['manifest_sha256'], 'crops_sha256': digest(root / 'crops.jsonl'),
        'training_cities': 304, 'evidence_cities': len(run['cities']),
        'directional_views': len(images), 'width': run['width'], 'k': run['k'],
        'ranking': run['ranking'], 'activation': run['activation'],
        'crop_fraction': run['crop_fraction'], 'grid': run['grid'], 'top': run['top'],
        'encoder': 'SigLIP base patch16 224', 'embedding_precision': run['embedding_precision'],
        'web_images': 'Original directional content resized to max 960 px; thumbnail max 320 px; no activation overlay burned in',
        'crop_display': 'Native-view bbox normalized; rendered from the same original-view WebP',
        'scope': 'Top100 among 4720 directional views, not an exhaustive NAS search',
    })
    template = Path(__file__).with_name('semantic_site')
    for name in ('index.html', 'style.css', 'app.js'):
        target = out / name
        shutil.copyfile(template / name, target)
        if name == 'index.html' and args.image_base:
            html = target.read_text()
            target.write_text(html.replace('<body>', f'<body data-image-base="{args.image_base}">', 1))
    shutil.copyfile(root / 'feature_semantic_umap.csv', out / 'feature_semantic_umap.csv')
    print(f'Saved online atlas: {len(points)} dimensions, {len(images)} original views, {int(count.sum())} examples', flush=True)


if __name__ == '__main__':
    main()
