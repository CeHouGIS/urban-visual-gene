"""Resolve a bounded held-out pano list through the verified QC image order.

Reads existing sample/QC metadata only; never searches the image directory.
Supports native Baidu view crops with the feature extractor's exact geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from formal.analysis.plot_feature_activation_maps import load_image_catalog
from formal.analysis.semantic_feature_umap import digest, write_json, write_jsonl


def build_manifest(anchors, kept, paths, qc, source_root):
    selected = sorted({(r['city'], int(r['source_pano'])) for r in anchors})
    arrays, rows = {}, []
    for city, position in selected:
        info = qc['cities'][city]
        if not info.get('verified') or len(kept[city]) != info['cached_panos']:
            raise ValueError(f'Unverified image/feature order for {city}')
        if position in info.get('bad_kept_positions', []):
            raise ValueError(f'Held-out sample contains a QC-excluded pano: {city}/{position}')
        pid = kept[city][position]
        images = paths[city][pid]
        feature_path = source_root / f'dict_{city}_p784.f16.npy'
        if city not in arrays:
            arrays[city] = np.load(feature_path, mmap_mode='r')
        array = arrays[city]
        if array.shape != (info['cached_panos'] * 3136, 1024):
            raise ValueError(f'Incompatible source features for {city}: {array.shape}')
        for view, heading in enumerate((0, 90, 180, 270)):
            image = images.get(heading)
            bbox = None
            if image is None and images.get(None):
                image = images[None]
                with Image.open(image) as original:
                    width, height = original.size
                side = min(width // 4, height)
                top = max(0, (height - side) // 2)
                left = int((view + .5) * width / 4) - side // 2
                if left < 0 or left + side > width:
                    raise ValueError('Wrapped Baidu view requires explicit native-view export')
                bbox = [left, top, left + side, top + side]
            if not image or not Path(image).is_file():
                raise ValueError(f'Missing original view for {city}/{pid}/{heading}')
            row = {'city': city, 'pano': pid, 'heading': heading, 'image': image,
                   'source_pano': position, 'feature_path': str(feature_path),
                   'row_start': position * 3136 + view * 784}
            if bbox:
                row['view_bbox'] = bbox
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchors', type=Path, required=True)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--qc-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    qc_path = args.qc_root / 'feature_qc_map.json'
    qc = json.loads(qc_path.read_text())
    training = json.loads(args.training_manifest.read_text())
    anchors = json.loads(args.anchors.read_text())
    print('Resolving existing sample and QC metadata; no image-directory scan', flush=True)
    kept, paths = load_image_catalog(args.qc_root)
    rows = build_manifest(anchors, kept, paths, qc, Path(training['source_out']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    write_json(args.out.with_suffix('.provenance.json'), {
        'anchors': str(args.anchors), 'anchor_sha256': digest(args.anchors),
        'qc_map': str(qc_path), 'qc_sha256': digest(qc_path),
        'training_manifest': str(args.training_manifest),
        'training_manifest_sha256': digest(args.training_manifest),
        'images': len(rows), 'panoramas': len(rows) // 4,
        'cities': len({r['city'] for r in rows}),
        'ranking_scope': 'All 784 patches in each of four views for listed held-out panoramas',
        'baidu_views': sum('view_bbox' in r for r in rows),
    })
    print(f'Saved {len(rows):,} views / {len(rows)//4:,} panos / '
          f'{len({r["city"] for r in rows})} cities to {args.out}', flush=True)


if __name__ == '__main__':
    main()
