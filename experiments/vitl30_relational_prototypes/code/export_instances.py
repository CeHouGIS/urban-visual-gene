"""Export traceable development examples for the trained relation slots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from sae_experiments.io_utils import digest
from .hierarchy import EXPERIMENT, read
from .model import LandscapeDictionary
from .regions import OUT, region_boxes
from .train import load_arrays

HEADINGS = (0, 90, 180, 270)
FAMILY_LABELS = {
    'building': '建筑', 'road': '道路', 'pedestrian': '步行空间', 'tree': '乔木',
    'utility': '市政杆线', 'sign': '标识设施', 'water': '水体',
    'low_vegetation': '低矮植被', 'vehicle': '车辆', 'person': '行人',
    'transport_structure': '交通结构', 'barrier': '围合设施',
    'furniture': '街道家具', 'terrain': '地形', 'construction': '施工设施',
}
FAMILY_COLORS = {
    'building': '#e45756', 'road': '#4c78a8', 'pedestrian': '#f58518',
    'tree': '#54a24b', 'utility': '#72b7b2', 'sign': '#eeca3b',
    'water': '#5f9ed1', 'low_vegetation': '#8cd17d', 'vehicle': '#ff9da6',
    'person': '#b279a2', 'transport_structure': '#9d755d', 'barrier': '#79706e',
    'furniture': '#d37295', 'terrain': '#b6992d', 'construction': '#f28e2b',
}


def position_name(values):
    grid = np.asarray([(i // 4 / 3, i % 4 / 3) for i in range(16)], np.float32)
    center = (values[:, None] * grid).sum(0) / max(float(values.sum()), 1e-8)
    vertical = '上部' if center[0] < .38 else '下部' if center[0] > .62 else '中部'
    horizontal = '左侧' if center[1] < .38 else '右侧' if center[1] > .62 else '中央'
    return '中央' if vertical == '中部' and horizontal == '中央' else vertical + horizontal


def describe_layout(template, families):
    means = template.mean(-1)
    leaders = np.argsort(means)[::-1][:5]
    maximum = float(template.max())
    dominant = template.argmax(0)
    strength = template.max(0) / max(maximum, 1e-8)
    return {
        'grid': [4, 4],
        'encoding': 'dominant parent family per cell; opacity is learned template strength',
        'leaders': [{'family': families[i], 'label_zh': FAMILY_LABELS[families[i]],
                     'color': FAMILY_COLORS[families[i]], 'mean': float(means[i]),
                     'position_zh': position_name(template[i])} for i in leaders],
        'dominant_families': [families[i] for i in dominant.tolist()],
        'relative_strengths': strength.tolist(),
    }


def select_examples(records, slots=32, per_slot=4):
    selected = []
    for slot in range(slots):
        candidates = sorted((r for r in records if r['slot'] == slot),
                            key=lambda r: (-r['probability'], r['mse']))
        cities, panoramas, chosen = set(), set(), []
        for record in candidates:
            if record['city'] in cities or record['pano_index'] in panoramas:
                continue
            chosen.append(record)
            cities.add(record['city'])
            panoramas.add(record['pano_index'])
            if len(chosen) == per_slot:
                break
        if len(chosen) != per_slot:
            raise ValueError(f'Slot {slot} has fewer than {per_slot} distinct-city examples')
        selected.extend(chosen)
    return selected


def render_example(row, record, layout, destination):
    size, top, left, bottom, right = region_boxes()[record['region_index']]
    tile, bar = 224, 26
    canvas = Image.new('RGB', (tile * 6, tile + bar), '#1a2525')
    draw = ImageDraw.Draw(canvas)
    selected = None
    selected_box = None
    for position, heading in enumerate(HEADINGS):
        path = Path(row[f'image_{heading}'])
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            original = source.convert('RGB')
        canvas.paste(original.resize((tile, tile), Image.Resampling.LANCZOS),
                     (position * tile, bar))
        if heading == record['heading']:
            width, height = original.size
            box = (round(left * width / 28), round(top * height / 28),
                   round(right * width / 28), round(bottom * height / 28))
            selected, selected_box = original, box
            crop = original.crop(box)
            if not crop.width or not crop.height:
                raise ValueError(f'Empty region crop for {record}')
            canvas.paste(crop.resize((tile, tile), Image.Resampling.LANCZOS), (tile * 5, bar))
            draw.rectangle((position * tile + round(box[0] * tile / width),
                            bar + round(box[1] * tile / height),
                            position * tile + round(box[2] * tile / width) - 1,
                            bar + round(box[3] * tile / height) - 1),
                           outline='#f7c948', width=4)
        draw.text((position * tile + 9, 7), f'{heading} deg', fill='white')
    if selected is None:
        raise ValueError('Selected heading was not rendered')
    width, height = selected.size
    masked = selected.resize((tile, tile), Image.Resampling.LANCZOS).convert('RGBA')
    overlay = Image.new('RGBA', masked.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(overlay)
    mapped = tuple(round(value * tile / source) for value, source in
                   zip(selected_box, (width, height, width, height)))
    x0, y0, x1, y1 = mapped
    mask_draw.rectangle((0, 0, tile, y0), fill=(0, 0, 0, 145))
    mask_draw.rectangle((0, y1, tile, tile), fill=(0, 0, 0, 145))
    mask_draw.rectangle((0, y0, x0, y1), fill=(0, 0, 0, 145))
    mask_draw.rectangle((x1, y0, tile, y1), fill=(0, 0, 0, 145))
    for cell, family in enumerate(layout['dominant_families']):
        row_index, column = divmod(cell, 4)
        xa = round(x0 + column * (x1 - x0) / 4)
        xb = round(x0 + (column + 1) * (x1 - x0) / 4)
        ya = round(y0 + row_index * (y1 - y0) / 4)
        yb = round(y0 + (row_index + 1) * (y1 - y0) / 4)
        color = FAMILY_COLORS[family].lstrip('#')
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        alpha = round(55 + 125 * layout['relative_strengths'][cell])
        mask_draw.rectangle((xa, ya, xb, yb), fill=(*rgb, alpha), outline=(255, 255, 255, 135), width=1)
    masked = Image.alpha_composite(masked, overlay).convert('RGB')
    canvas.paste(masked, (tile * 4, bar))
    draw.text((tile * 4 + 9, 7), 'learned 4 x 4 layout mask', fill='white')
    draw.text((tile * 5 + 9, 7), f'{size} x {size} patch crop', fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, 'WEBP', quality=85, method=4)
    record['box_patch'] = [left, top, right, bottom]
    record['region_size'] = size


@torch.inference_mode()
def export(out=OUT, device='cuda:1', per_slot=4):
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Relation inference requires CUDA; CPU fallback is disabled')
    out = Path(out)
    x, context, _, dev_ids, meta = load_arrays(out)
    frame = pd.read_parquet(out / 'pilot_panoramas.parquet')
    summary = read(EXPERIMENT / 'results/relational_pilot_summary.json')
    config = read(out / 'models/relation/config.json')
    complete = read(out / 'models/relation/complete.json')
    checkpoint_path = out / 'models/relation/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint['identity'] != config['identity'] or complete['identity'] != config['identity']
            or complete['test_used'] or summary['pilot_identity'] != config['pilot_identity']
            or summary['region_identity'] != meta['region_identity']):
        raise ValueError('Model, pilot, or region identity differs')
    model = LandscapeDictionary(families=x.shape[1], slots=config['slots'], mode='relation').to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    templates = F.softplus(model.dictionary).detach().cpu().numpy()
    layouts = [describe_layout(template, meta['families']) for template in templates]
    records = []
    for start in range(0, len(dev_ids), 256):
        chosen = dev_ids[start:start + 256]
        target = torch.from_numpy(x[chosen]).to(device)
        pano = torch.from_numpy(context[chosen].copy()).to(device)
        prediction, probabilities, winner = model(target, pano)
        errors = (prediction - target).square().mean((1, 2)).cpu().numpy()
        probs = probabilities.cpu().numpy()
        assigned = winner.argmax(-1).cpu().numpy()
        for i, region_id in enumerate(chosen):
            pano_index, remainder = divmod(int(region_id), 4 * len(region_boxes()))
            heading_index, region_index = divmod(remainder, len(region_boxes()))
            source = frame.iloc[pano_index]
            if source.prototype_split != 'dev':
                raise ValueError('Selected non-development panorama')
            slot = int(assigned[i])
            records.append({'slot': slot, 'region_id': int(region_id),
                            'pano_index': pano_index, 'city': source.city, 'panoid': source.panoid,
                            'heading': HEADINGS[heading_index], 'region_index': region_index,
                            'probability': float(probs[i, slot]), 'mse': float(errors[i])})
    counts = np.bincount([r['slot'] for r in records], minlength=config['slots'])
    if counts.tolist() != [slot['development_regions'] for slot in summary['slot_usage']]:
        raise ValueError('Fresh GPU assignments differ from audited slot counts')
    selected = select_examples(records, config['slots'], per_slot)
    image_dir = out / 'instance_gallery'
    for index, record in enumerate(selected):
        filename = f"slot{record['slot']:02d}_example{index % per_slot:02d}.webp"
        record['image'] = filename
        render_example(frame.iloc[record['pano_index']], record, layouts[record['slot']], image_dir / filename)
        del record['pano_index']
    payload = {'schema_version': 1, 'partition': 'dev', 'test_used': False,
               'pilot_identity': summary['pilot_identity'], 'region_identity': meta['region_identity'],
               'model_identity': config['identity'], 'model_sha256': digest(checkpoint_path),
               'development_regions': len(dev_ids), 'examples_per_slot': per_slot,
               'mask_definition': ('The fifth panel projects the learned slot dictionary onto the selected '
                                   'direction. Each of 16 cells uses its strongest parent-family color; opacity '
                                   'is proportional to the learned template strength. Outside the selected '
                                   'region is darkened. This is not a segmentation mask.'),
               'family_palette': {family: {'label_zh': FAMILY_LABELS[family],
                                           'color': FAMILY_COLORS[family]}
                                  for family in meta['families']},
               'slots': [{**slot, 'layout': layouts[slot['slot']],
                          'examples': [r for r in selected if r['slot'] == slot['slot']]}
                         for slot in summary['slot_usage']]}
    (image_dir / 'index.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return {'slots': len(payload['slots']), 'examples': len(selected), 'image_dir': str(image_dir)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--examples', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(export(device=args.device, per_slot=args.examples), ensure_ascii=False))
