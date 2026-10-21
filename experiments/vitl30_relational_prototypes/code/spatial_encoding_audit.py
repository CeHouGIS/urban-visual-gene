"""Screen SAE heatmaps for fixed vertical bands and export original-image review sheets."""
from __future__ import annotations

import argparse
import base64
import csv
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'experiments/vitl30_panorama_w4096_k32/reports/vlm_dimensions_top10_masked/evidence.json'
SCREENING = ROOT / 'experiments/vitl30_panorama_w4096_k32/results/dimension_screening/screening.json'
HIERARCHY = ROOT / 'experiments/vitl30_relational_prototypes/results/hierarchy.json'


def heat_array(crop):
    return np.asarray(Image.open(BytesIO(base64.b64decode(crop['heatmap_png']))).convert('L'))


def longest_run(values):
    indices = np.flatnonzero(values)
    if not len(indices):
        return 0
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    return max(len(part) for part in np.split(indices, splits))


def stripe_metrics(heat):
    if heat.shape != (28, 28) or heat.max() == 0:
        return {'vertical_run': 0, 'horizontal_run': 0, 'band_fraction': 0.0,
                'center_x': None, 'stripe': False}
    mask = heat >= max(32, round(float(heat.max()) * .45))
    total = max(1, int(mask.sum()))
    bands = []
    for x in range(26):
        band = mask[:, x:x + 3]
        bands.append((longest_run(band.any(axis=1)), float(band.sum()) / total, x + 1))
    vertical, fraction, center = max(bands, key=lambda item: (item[0], item[1]))
    horizontal = max(longest_run(mask[y:y + 3, :].any(axis=0)) for y in range(26))
    stripe = vertical >= 17 and vertical >= horizontal + 4 and fraction >= .48
    return {'vertical_run': vertical, 'horizontal_run': horizontal,
            'band_fraction': round(fraction, 4), 'center_x': center, 'stripe': stripe}


def overlay(crop, size=112):
    with Image.open(crop['image']) as original:
        base = original.convert('RGBA').resize((size, size), Image.Resampling.LANCZOS)
    heat = Image.fromarray(heat_array(crop), 'L')
    rgba = Image.new('RGBA', heat.size)
    rgba.putdata([(255, round(210 * min(1, value / 255 * 1.5)), 30,
                   round(225 * max(0, (value / 255 - .08) / .92))) for value in heat.tobytes()])
    return Image.alpha_composite(base, rgba.resize((size, size), Image.Resampling.NEAREST)).convert('RGB')


def review_sheet(rows, evidence, output, start, count=16):
    selected = rows[start:start + count]
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    small = ImageFont.truetype(font_path, 13)
    label = ImageFont.truetype(font_path, 16)
    width, cell_w, cell_h = 1080, 520, 158
    canvas = Image.new('RGB', (width, 52 + ((len(selected) + 1) // 2) * cell_h), '#f7f9f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 13), f'Vertical-band review  |  rank {start + 1}-{start + len(selected)}',
              font=label, fill='#263b32')
    for index, row in enumerate(selected):
        x = 20 + index % 2 * (cell_w + 20)
        y = 52 + index // 2 * cell_h
        draw.line((x, y, x + cell_w, y), fill='#c9d9ce', width=1)
        draw.text((x, y + 7), f'f{row["feature"]}  stripe {row["stripe_fraction"]:.2f}  '
                  f'run {row["median_vertical_run"]:.0f}  xSD {row["stripe_center_sd"]:.1f}',
                  font=label, fill='#1f4637')
        for crop_index, crop in enumerate(evidence[row['feature']]['crops'][:4]):
            thumb = overlay(crop)
            left = x + crop_index * 122
            canvas.paste(thumb, (left, y + 32))
            draw.text((left + 2, y + 145), crop['city'].split('/')[-1][:13],
                      font=small, fill='#53665a')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def run(output):
    evidence = json.loads(REPORT.read_text())['dimensions']
    screening = json.loads(SCREENING.read_text())['dimensions']
    hierarchy = json.loads(HIERARCHY.read_text())['dimensions']
    if len(evidence) != len(screening) or len(evidence) != len(hierarchy) or len(evidence) != 4096:
        raise ValueError('SAE evidence is incomplete')
    rows = []
    for feature, item in enumerate(evidence):
        crops = item['crops'][:10]
        metrics = [stripe_metrics(heat_array(crop)) for crop in crops]
        matching = [metric for metric in metrics if metric['stripe']]
        centers = [metric['center_x'] for metric in matching]
        rows.append({
            'feature': feature, 'family': hierarchy[feature]['family'],
            'qwen_label': hierarchy[feature].get('label_zh', ''),
            'positive_panoramas': screening[feature]['positive_panoramas'],
            'position_review': screening[feature]['position_review'],
            'examples': len(metrics), 'cities': len({crop['city'] for crop in crops}),
            'stripe_examples': len(matching),
            'stripe_fraction': round(len(matching) / len(metrics), 4) if metrics else 0.0,
            'median_vertical_run': float(np.median([m['vertical_run'] for m in metrics])) if metrics else 0.0,
            'stripe_center_sd': float(np.std(centers)) if centers else 28.0,
            'median_band_fraction': float(np.median([m['band_fraction'] for m in metrics])) if metrics else 0.0,
        })
        if feature % 512 == 0:
            print(f'checked {feature + 1}/4096 dimensions', flush=True)
    ranked = sorted((row for row in rows if row['examples'] >= 4),
                    key=lambda row: (-row['stripe_fraction'], row['stripe_center_sd'],
                                     -row['median_vertical_run'], -row['cities']))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'all_dimensions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'ranked.json').write_text(json.dumps({
        'status': 'screened_pending_visual_review', 'dimensions': 4096,
        'rule': 'Top10 heatmaps: >=17-row 3-column band, 4 more rows than any horizontal band, >=48% high-response pixels inside band. Fixed x and different visual content require visual review.',
        'candidates': ranked,
    }, ensure_ascii=False, indent=2))
    for start in range(0, min(48, len(ranked)), 16):
        review_sheet(ranked, evidence, output / f'review_{start // 16 + 1}.png', start)
    print(json.dumps({'output': str(output), 'ranked': len(ranked),
                      'stripe_8_of_10': sum(row['stripe_fraction'] >= .8 for row in ranked)},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT /
                        'experiments/vitl30_relational_prototypes/results/spatial_encoding_audit')
    args = parser.parse_args()
    run(args.output.resolve())
