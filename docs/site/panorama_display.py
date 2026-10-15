"""Render website panoramas from saved selections, preserving original aspect.

This only reads local image/ranking caches and completed reports. It does not
fit models, change analysis results, or infer a seamless spherical projection.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def panorama_row(source, images, pid, caption, marks=()):
    """Four separately labelled perspective views; mark coordinates are patches."""
    views = images[pid * 4:pid * 4 + 4]
    assert len(views) == 4
    assert len({(v['city'], v['pano']) for v in views}) == 1
    assert [v['heading'] for v in views] == [0, 90, 180, 270]
    cell_width, header, footer = 320, 22, 26
    resized = []
    for offset in range(4):
        with Image.open(source / 'sources' / f'{pid * 4 + offset}.jpg') as image:
            resized.append(ImageOps.contain(image.convert('RGB'), (cell_width, 240), Image.Resampling.LANCZOS))
    height = max(im.height for im in resized)
    canvas = Image.new('RGB', (cell_width * 4, header + height + footer), '#fff')
    draw = ImageDraw.Draw(canvas)
    bounds = []
    for offset, image in enumerate(resized):
        x = offset * cell_width + (cell_width - image.width) // 2
        y = header + (height - image.height) // 2
        bounds.append((x, y, image.width, image.height))
        canvas.paste(image, (x, y))
        draw.rectangle((offset * cell_width, 0, (offset + 1) * cell_width - 1, header - 1), fill='#e8efe7')
        draw.text((offset * cell_width + 8, 4), f'{offset * 90} degrees', fill='#18322d')
        if offset:
            draw.line((offset * cell_width, 0, offset * cell_width, header + height), fill='#d8e2dd', width=2)
    colors = ['#ffdd00', '#00e5ff', '#ff55dd']
    for (feature, patch), color in zip(marks, colors):
        assert 0 <= patch < 4 * 28 * 28
        heading, inner = divmod(patch, 784)
        py, px = divmod(inner, 28)
        x, y, width, view_height = bounds[heading]
        cx, cy = x + (px + .5) * width / 28, y + (py + .5) * view_height / 28
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline='black', width=5)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=color, width=3)
        tx = min(x + width - 55, max(x + 2, cx + 9))
        ty = min(y + view_height - 15, max(y + 2, cy - 8))
        draw.rectangle((tx, ty, tx + 53, ty + 14), fill='black')
        draw.text((tx + 2, ty), f'f{feature}', fill=color)
    draw.text((8, header + height + 5), caption, fill='#18322d')
    return canvas


def save_rows(rows, path):
    assert rows
    canvas = Image.new('RGB', (1280, sum(row.height for row in rows)), '#fff')
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def motif_examples(motif, sample, rows, peaks, thresholds):
    """Use the report's original deterministic selection rule and saved split."""
    features = motif['features']
    candidates = []
    for pid, split in sample.items():
        values = peaks[pid, features]
        if np.all((values > 0) & (values >= thresholds[features])):
            strength = float(np.min(values / thresholds[features]))
            candidates.append((pid, split != 'confirmation', strength))
    candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
    chosen, cities = [], set()
    for distinct in (True, False):
        for pid, _, _ in candidates:
            if pid in chosen or (distinct and rows[pid]['city'] in cities):
                continue
            chosen.append(pid)
            cities.add(rows[pid]['city'])
            if len(chosen) == 6:
                return chosen
    return chosen


def render_panorama_galleries(out: Path, source: Path, reports: Path):
    images = [json.loads(line) for line in (source / 'images.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (source / 'panoramas.jsonl').read_text().splitlines()]
    assert len(images) == len(rows) * 4
    for pid, row in enumerate(rows):
        assert row['pano_index'] == pid
        for heading in range(4):
            image = images[pid * 4 + heading]
            assert (image['city'], image['pano'], image['heading']) == (row['city'], row['pano'], heading * 90)
    by_key = {r['city'] + '/' + r['pano']: r['pano_index'] for r in rows}
    panels, previews = {}, []
    nmf = reports / 'landscape_prototypes'
    with np.load(nmf / 'nmf_w32.npz') as saved:
        weights = saved['weights']
    assert weights.shape == (len(rows), 32)
    shares = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    dominant = shares.argmax(axis=1)
    for component in read_csv(nmf / 'prototypes.csv'):
        number = int(component['component'])
        ids = [by_key[key] for key in component['example_panoramas'].split()]
        strips = []
        for pid in ids:
            kind = 'dominant' if dominant[pid] == number else 'secondary'
            caption = f'{rows[pid]["city"]} | {rows[pid]["pano"]} | share {shares[pid, number]:.3f} | {kind}'
            strips.append(panorama_row(source, images, pid, caption))
        name = f'panoramas/prototype_{number:02d}.jpg'
        save_rows(strips, out / 'combinations' / name)
        panels[name] = ids
        if component['status'] == 'supported candidate':
            # Keep the four-view image region; thumbnail without changing aspect.
            preview = strips[0].crop((0, 0, 1280, strips[0].height - 26))
            previews.append((number, component['dominant_panoramas'], preview))
    overview = Image.new('RGB', (2000, 160 * ((len(previews) + 3) // 4)), '#fff')
    draw = ImageDraw.Draw(overview)
    for position, (number, count, preview) in enumerate(previews):
        preview.thumbnail((480, 126), Image.Resampling.LANCZOS)
        x, y = position % 4 * 500, position // 4 * 160
        overview.paste(preview, (x, y + 20))
        draw.text((x + 4, y + 2), f'P{number:02d} | {count} panoramas', fill='#18322d')
    (out / 'assets/figures').mkdir(parents=True, exist_ok=True)
    overview.save(out / 'assets/figures/Fig_ViTL30_Landscape_Prototype_Overview.jpg', quality=92)

    motifs = reports / 'unit_motifs'
    summary = read_json(motifs / 'summary.json')
    assert read_json(source / 'run.json')['checkpoint_sha256'] == summary['checkpoint_sha256']
    sample = {int(r['pano_index']): r['split'] for r in read_csv(motifs / 'analysis_sample.csv')}
    for record in read_csv(motifs / 'analysis_sample.csv'):
        row = rows[int(record['pano_index'])]
        assert (row['city'], row['pano']) == (record['city'], record['pano'])
    peaks = np.zeros((len(rows), 4096), np.float32)
    patches = np.zeros((len(rows), 4096), np.int16)
    for city in sorted({r['city'] for r in rows}):
        selected = [r for r in rows if r['city'] == city]
        with np.load(source / 'ranking' / f'{city.replace("/", "__")}.npz') as saved:
            assert saved['ids'].tolist() == [r['cache_pano_index'] for r in selected]
            ids = [r['pano_index'] for r in selected]
            peaks[ids] = saved['peaks']
            patches[ids] = saved['patches']
    thresholds = np.load(motifs / 'presence_thresholds.npy')
    records = {r['motif']: r for r in read_json(motifs / 'motifs.json')}
    for number in summary['illustrated_motifs']:
        record = records[number]
        ids = motif_examples(record, sample, rows, peaks, thresholds)
        strips = []
        for pid in ids:
            marks = [(f, int(patches[pid, f])) for f in record['features']]
            caption = f'{rows[pid]["city"]} | {rows[pid]["pano"]} | {sample[pid].upper()}'
            strips.append(panorama_row(source, images, pid, caption, marks))
        name = f'motifs/m{number}.jpg'
        save_rows(strips, out / 'combinations' / name)
        panels[name] = ids
    metadata = {'view_order': [0, 90, 180, 270], 'preserves_original_aspect': True,
                'projection': 'separate perspective views; no seamless spherical reprojection',
                'source': 'semantic/sources/{4 * pano_index + heading_index}.jpg',
                'panorama_indices_by_panel': panels}
    (out / 'combinations/panorama_layout.json').write_text(json.dumps(metadata, ensure_ascii=False))
    return metadata
