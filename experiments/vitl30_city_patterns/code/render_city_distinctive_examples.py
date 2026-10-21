"""Render four-view evidence for selected city-enriched SAE co-responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

from .composition_evidence import COLORS, infer, render
from .composition_patterns import pattern_mask
from .extract import image_paths, load_models, write_json
from .followup import FOLLOWUP


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/city_internal_compositions'
STAGE = FOLLOWUP / 'stages/composition_patterns_screened'
FRAME = FOLLOWUP / 'stages/candidate_combinations/panorama_elements.parquet'
FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

# City-enriched three-element examples spanning several semantic families.
EXAMPLES = [
    ('Bangladesh/Dhaka', 'cd8d335c382b', '老屋 + 廣告牌 + 樹木'),
    ('UAE/Dubai', '867d8c3032f1', '道路 + 沙地 + 水體'),
    ('Korea/Seoul', '99a28f773552', '混凝土建築 + 橋梁 + 電線'),
    ('US/Phoenix', 'e938ca62801c', '人行道 + 道路 + 沙地'),
    ('Sweden/Stockholm', 'a9e43f224ca2', '草地 + 岩石 + 樹木'),
]

OTHER_CITY_EXAMPLES = [
    ('Argentina/BuenosAires', '0dc381db3bd1', '围栏 + 草地 + 人行道 + 电线'),
    ('Australia/Sydney', 'b972badb694e', '围栏 + 草地 + 树木 + 电线杆'),
    ('Brazil/RiodeJaneiro', '627c338c01db', '围墙 + 瓦顶建筑 + 山地'),
    ('Indonesia/Yogyakarta', '3861f3bfb801', '老屋 + 人行道 + 树木'),
    ('Italy/Naples', '1a3979e2407c', '围栏 + 建筑 + 台阶与楼梯 + 路灯'),
    ('Mexico/Monterrey', 'b28d78f0e2df', '道路 + 广告牌 + 山地'),
    ('Singapore/Singapore', 'd7d9b447e254', '灌木 + 道路 + 桥梁 + 树木'),
    ('US/Chicago', '8cde7f13e09e', '车库 + 草地 + 人行道 + 树木'),
]


def choose_indices(frame, assignment, scores, thresholds, units, city, unit_ids, count):
    matching = pattern_mask(assignment, units, unit_ids) & (frame.city.to_numpy() == city)
    positions = np.flatnonzero(matching)
    if len(positions) < count:
        raise ValueError(f'Not enough panoramas for {city}: {len(positions)}')
    unit_ids = np.asarray(unit_ids)
    ratios = scores[positions[:, None], unit_ids[None, :]] / thresholds[unit_ids][None, :]
    strength = ratios.min(axis=1)
    order = np.argsort(-strength, kind='stable')
    chosen, blocks = [], set()
    for index in order:
        position = int(positions[index])
        row = frame.iloc[position]
        block = int(row.grid_x) // 5, int(row.grid_y) // 5
        if block in blocks or not all(path.is_file() for path in image_paths(city, row.panoid)):
            continue
        chosen.append((position, float(strength[index])))
        blocks.add(block)
        if len(chosen) == count:
            return chosen
    raise ValueError(f'Could not find {count} distinct-block examples in {city}')


def make_panel(original: Path, overlay: Path, title: str, labels: list[str],
               city: str, rank: int, city_rate: float,
               other_rate: float, target: Path):
    with Image.open(original) as source, Image.open(overlay) as activation:
        source = source.convert('RGB')
        activation = activation.convert('RGB')
    margin, header, between, footer = 24, 106, 12, 58
    width = source.width + margin * 2
    height = header + source.height + between + activation.height + footer
    canvas = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.truetype(FONT, 25)
    regular = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 14)
    draw.text((margin, 14), f'{city.split("/")[-1]} | {title}', font=heading, fill='#202729')
    ratio = city_rate / other_rate if other_rate > 0 else float('inf')
    draw.text((margin, 52),
              f'实例 {rank} | 城内格网平衡检出率 {city_rate:.2%} | 其他城市最高 {other_rate:.2%} | 富集 {ratio:.1f} 倍',
              font=regular, fill='#5d686a')
    x = margin
    for color, label in zip(COLORS, labels):
        draw.rectangle((x, 82, x + 14, 96), fill=color)
        draw.text((x + 20, 80), label, font=small, fill='#283034')
        x += 28 + draw.textlength(label, font=small)
    canvas.paste(source, (margin, header))
    canvas.paste(activation, (margin, header + source.height + between))
    draw.text((margin, height - 42), '上：四方向原图。下：SAE 激活叠加，颜色表示各要素的相对响应。',
              font=small, fill='#4f5d60')
    draw.text((margin, height - 23), '这是候选共响应，不代表经过验证的物体分割或空间关系。',
              font=small, fill='#4f5d60')
    canvas.save(target)
    return canvas


def make_contact_sheet(panels: list[Image.Image], target: Path):
    width, margin, gap = 760, 24, 20
    thumbs = []
    for panel in panels:
        height = round(panel.height * width / panel.width)
        thumbs.append(panel.resize((width, height), Image.Resampling.LANCZOS))
    tile_height = max(panel.height for panel in thumbs)
    columns, rows = 2, (len(thumbs) + 1) // 2
    canvas = Image.new('RGB', (margin * 2 + columns * width + gap,
                               margin * 2 + rows * tile_height + (rows - 1) * gap), '#eef1f0')
    for index, panel in enumerate(thumbs):
        x = margin + (index % columns) * (width + gap)
        y = margin + (index // columns) * (tile_height + gap)
        canvas.paste(panel, (x, y))
    canvas.save(target)


def run(stage: Path, frame_path: Path, result_dir: Path, device: str, count: int,
        limit: int | None, group: str = 'initial'):
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GPU required for DINOv3 + SAE attribution')
    rows = pd.read_csv(result_dir / 'single_city_candidates.csv')
    units = json.loads((stage / 'units.json').read_text())
    calibration = json.loads((stage / 'calibration.json').read_text())
    frame = pd.read_parquet(frame_path, columns=['city', 'panoid', 'grid_x', 'grid_y'])
    assignment = np.load(stage / 'assignment.npy', mmap_mode='r')
    scores = np.load(stage / 'scores.npy', mmap_mode='r')
    if len(frame) != len(assignment) or len(frame) != len(scores):
        raise ValueError('Corpus row alignment changed')
    thresholds = np.asarray(calibration['unit_thresholds'], dtype=float)
    candidates = EXAMPLES if group == 'initial' else OTHER_CITY_EXAMPLES
    selections = candidates[:limit] if limit else candidates
    planned = []
    for city, pattern_id, title in selections:
        selected = rows[(rows.city == city) & (rows.pattern_id == pattern_id)]
        if len(selected) != 1 or not bool(selected.iloc[0].replicated):
            raise ValueError(f'City-enriched candidate not found: {city} {pattern_id}')
        row = selected.iloc[0]
        ids = json.loads(row.unit_ids)
        positions = choose_indices(frame, assignment, scores, thresholds, units,
                                   city, ids, count)
        planned.append((row, title, ids, positions))

    torch.set_num_threads(4)
    dino, sae, _ = load_models(device)
    output = result_dir / ('examples' if group == 'initial' else 'examples_other_cities')
    output.mkdir(parents=True, exist_ok=True)
    manifest, atlas = [], []
    for row, title, ids, positions in planned:
        labels = [units[item]['label'] for item in ids]
        for rank, (position, score) in enumerate(positions, 1):
            sample = frame.iloc[position]
            image_list = []
            for path in image_paths(row.city, sample.panoid):
                with Image.open(path) as image:
                    image_list.append(image.convert('RGB'))
            maps = infer(image_list, dino, sae, ids, units, device)
            if maps.shape != (len(ids), 4, 28, 28) or not np.isfinite(maps).all():
                raise ValueError(f'Invalid activation map for {row.city} {sample.panoid}')
            folder = output / f'{row.city.split("/")[-1]}_{row.pattern_id}_{rank}'
            render(image_list, maps, folder)
            panel_path = output / f'{row.city.split("/")[-1]}_{row.pattern_id}_{rank}.png'
            panel = make_panel(folder / 'original.webp', folder / 'overlay.webp',
                               title, labels, row.city, rank, float(row.city_rate),
                               float(row.highest_other_city_rate), panel_path)
            atlas.append(panel)
            manifest.append({'city': row.city, 'pattern_id': row.pattern_id,
                             'labels': labels, 'title': title,
                             'city_rate': float(row.city_rate),
                             'highest_other_city_rate': float(row.highest_other_city_rate),
                             'panoid': str(sample.panoid), 'grid_x': int(sample.grid_x),
                             'grid_y': int(sample.grid_y), 'rank': rank,
                             'minimum_unit_score_ratio': score,
                             'original': str((folder / 'original.webp').resolve()),
                             'overlay': str((folder / 'overlay.webp').resolve()),
                             'panel': str(panel_path.resolve()),
                             'interpretation': 'Unverified Qwen labels and SAE co-response; no object or spatial-relation validation.'})
            print(f'{row.city} {row.pattern_id} example {rank}/{count}', flush=True)
            write_json(output / 'manifest.json', {'examples': manifest, 'complete': False})
    atlas[0].save(output / 'CityDistinctiveCandidates.pdf', save_all=True,
                  append_images=atlas[1:])
    make_contact_sheet(atlas[::count], output / 'CityDistinctiveCandidates_Overview.png')
    write_json(output / 'manifest.json', {'examples': manifest, 'complete': True,
               'selection': 'Top minimum unit-score ratio among panorama-level matches; distinct 5-km blocks per city and pattern.',
               'overlay': 'Frozen DINOv3 ViT-L + BatchTopK SAE; relative activation threshold 0.35 per unit, not segmentation.',
               'overview': 'CityDistinctiveCandidates_Overview.png',
               'pdf': 'CityDistinctiveCandidates.pdf'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=STAGE)
    parser.add_argument('--frame', type=Path, default=FRAME)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--per-pattern', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--group', choices=('initial', 'other'), default='initial')
    args = parser.parse_args()
    run(args.stage, args.frame, args.results, args.device, args.per_pattern,
        args.limit, args.group)
