"""Make four-view original-image evidence for held-out feature combinations."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from formal.analysis.test_feature_combinations import OUT


def main():
    records = json.loads((OUT / 'records.json').read_text())
    thresholds = np.load(OUT / 'analysis_arrays.npz')['thresholds']
    validation = set(np.load(OUT / 'analysis_arrays.npz')['validation_city'].tolist())
    counts = np.concatenate([np.load(OUT / 'city_counts' / f'{city}.npy')
                             for city in sorted({r['city'] for r in records})])
    pair_rows = list(csv.DictReader((OUT / 'pair_results.csv').open()))
    image_rows = defaultdict(dict)
    manifest = Path('reports/data/semantic_umap_w2048_k64_top100/images.jsonl')
    for line in manifest.open():
        row = json.loads(line)
        image_rows[(row['city'], row['source_pano'])][row['heading']] = row
    selected = []
    for pair in pair_rows:
        a, b = int(pair['feature_a']), int(pair['feature_b'])
        candidates = []
        for i, rec in enumerate(records):
            key = (rec['city'], rec['source_pano'])
            if rec['city'] not in validation or len(image_rows.get(key, {})) != 4:
                continue
            if counts[i, a] >= thresholds[a] and counts[i, b] >= thresholds[b]:
                score = min(counts[i, a] / thresholds[a], counts[i, b] / thresholds[b])
                candidates.append((float(score), rec, image_rows[key]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen, seen_cities = [], set()
        for score, rec, images in candidates:
            if rec['city'] in seen_cities:
                continue
            chosen.append((score, rec, images))
            seen_cities.add(rec['city'])
            if len(chosen) == 3:
                break
        if len(chosen) == 3:
            selected.append((pair, chosen))
        if len(selected) == 5:
            break
    if not selected:
        raise ValueError('No validation pano examples with four original views')
    font = ImageFont.load_default()
    tile = 192
    header = 25
    row_height = tile + 25
    canvas = Image.new('RGB', (tile * 4, header + len(selected) * 3 * row_height), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), 'Validation panoramas: four original headings in order 0, 90, 180, 270', fill='#111111', font=font)
    evidence = []
    for pair_index, (pair, chosen) in enumerate(selected):
        for pano_index, (score, rec, images) in enumerate(chosen):
            y = header + (pair_index * 3 + pano_index) * row_height
            for view, heading in enumerate((0, 90, 180, 270)):
                row = images[heading]
                with Image.open(row['image']) as original:
                    if row.get('view_bbox'):
                        original = original.crop(tuple(row['view_bbox']))
                    im = ImageOps.fit(original.convert('RGB'), (tile, tile))
                canvas.paste(im, (view * tile, y))
            label = (f"f{pair['feature_a']} + f{pair['feature_b']} | {rec['city']} | "
                     f"validation lift {float(pair['validation_lift']):.2f} | score {score:.2f}")
            draw.text((5, y + tile + 4), label, fill='#111111', font=font)
            evidence.append({'feature_a': int(pair['feature_a']), 'feature_b': int(pair['feature_b']),
                             'city': rec['city'], 'source_pano': rec['source_pano'],
                             'score': score, 'images': [images[h]['image'] for h in (0, 90, 180, 270)]})
    destination = OUT / 'validation_original_examples.jpg'
    canvas.save(destination, quality=88)
    (OUT / 'validation_original_examples.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(f'{destination}: {len(selected)} pairs, {len(evidence)} panoramas')


if __name__ == '__main__':
    main()
