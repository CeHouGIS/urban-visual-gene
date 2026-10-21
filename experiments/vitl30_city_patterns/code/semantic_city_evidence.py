"""Refresh previously inspected four-view examples using screened unit membership.

No old grouped heatmap is reused. Frozen DINO/SAE inference requires a GPU.
Earlier automated plausibility flags select images; they are not human validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .composition_evidence import choose_examples, infer, render
from .composition_patterns import pattern_mask
from .extract import image_paths, load_models, write_json
from .followup import FOLLOWUP, sha256

ROOT = Path(__file__).resolve().parents[3]


def run(folder, device):
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU inference fallback')
    torch.set_num_threads(4)
    marker = json.loads((folder / 'complete.json').read_text())
    if marker['status'] != 'complete':
        raise ValueError('Screened composition run is incomplete')
    previous = json.loads((ROOT / 'experiments/vitl30_city_patterns/results/composition_evidence_review.json').read_text())
    patterns = {p['pattern_id']: p for p in json.loads((folder / 'patterns.json').read_text())}
    units = json.loads((folder / 'units.json').read_text())
    assignment = np.load(folder / 'assignment.npy', mmap_mode='r')
    frame = pd.read_parquet(folder.parent / 'candidate_combinations/panorama_elements.parquet',
                            columns=['city', 'panoid', 'grid_x', 'grid_y'])
    chosen = []
    replacements = {}
    for example in previous['examples']:
        if example['status'] != 'plausible_spot_check' or example['pattern_id'] not in patterns:
            continue
        indices = np.flatnonzero(((frame.city == example['city']) & (frame.panoid == example['panoid'])).to_numpy())
        if len(indices) != 1:
            raise ValueError('Example panorama is missing or duplicated')
        index = indices[0]
        pattern = patterns[example['pattern_id']]
        if not pattern_mask(assignment[index:index+1], units, pattern['units'])[0]:
            replacements[pattern['pattern_id']] = pattern
            continue
        row = frame.iloc[index]
        chosen.append({'pattern_id': example['pattern_id'], 'city': row.city, 'panoid': row.panoid,
                       'grid_x': int(row.grid_x), 'grid_y': int(row.grid_y), 'units': pattern['units']})
    # Changed semantic membership can invalidate an old image match. Retrieve
    # replacements with the frozen current rule and review their new maps.
    for pattern_id, pattern in replacements.items():
        if pattern_id not in {e['pattern_id'] for e in chosen}:
            chosen.extend(choose_examples(frame, assignment, units, [pattern], 3))
    # Include the strongest two-family vegetation matches as fresh inspection
    # candidates, so examples are not limited to built structures and wires.
    for family in ('tree', 'low_vegetation'):
        pool = [p for p in patterns.values() if p['order'] == 2
                and any(units[i]['family'] == family for i in p['units'])]
        if pool:
            pattern = max(pool, key=lambda p: min(p['discovery_rate'], p['evaluation_rate']))
            if pattern['pattern_id'] not in {e['pattern_id'] for e in chosen}:
                chosen.extend(choose_examples(frame, assignment, units, [pattern], 3))
    if len({e['pattern_id'] for e in chosen}) < 4:
        raise ValueError('Fewer than four retained, previously inspected compositions')
    identity = hashlib.sha256(json.dumps({'protocol': sha256(folder / 'protocol.json'),
        'code': sha256(Path(__file__)), 'examples': chosen}, sort_keys=True).encode()).hexdigest()
    dino, sae, _ = load_models(device)
    rows = []
    for index, example in enumerate(chosen):
        key = hashlib.sha256(f'{example["pattern_id"]}/{example["city"]}/{example["panoid"]}'.encode()).hexdigest()[:20]
        target = folder / 'evidence' / identity[:12] / key
        target.mkdir(parents=True, exist_ok=True)
        images = []
        for path in image_paths(example['city'], example['panoid']):
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        cache = target / 'activation.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['identity']) != identity:
                    raise ValueError('Evidence cache identity differs')
                maps = saved['maps']
        else:
            maps = infer(images, dino, sae, example['units'], units, device)
            np.savez_compressed(cache, identity=identity, maps=maps)
        render(images, maps, target)
        rows.append({**example, 'original': str(target / 'original.webp'),
                     'overlay': str(target / 'overlay.webp'), 'mask_is_object_segmentation': False,
                     'semantic_review': 'pending refreshed visual inspection; no independent human validation'})
        print(f'Refreshed four-view masks: {index+1}/{len(chosen)} {example["city"]}', flush=True)
    write_json(folder / 'examples.json', rows)
    write_json(folder / 'evidence_review.json', {
        'identity': identity, 'source': 'old inspection selects candidates; current matching and screened masks require fresh inspection',
        'human_validated': False, 'examples': [{**e, 'status': 'pending_visual_review'} for e in rows]})
    write_json(folder / 'evidence_complete.json', {'status': 'complete', 'identity': identity,
               'protocol_sha256': sha256(folder / 'protocol.json'), 'examples': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=FOLLOWUP / 'stages/composition_patterns_screened')
    parser.add_argument('--device', default='cuda:3')
    args = parser.parse_args()
    run(args.stage.resolve(), args.device)
