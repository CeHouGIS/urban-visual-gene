"""GPU-only, four-view attribution audit of representative composition matches.

Selection is deterministic and spans distinct cities/cells. These are retrieval
examples, not a random prevalence audit or object segmentation ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

from .candidate_combinations import ELEMENTS
from .composition_patterns import landscape_children, pattern_mask
from .extract import HEADINGS, PATCHES, WIDTH, image_paths, load_models, preprocess, write_json
from .followup import FOLLOWUP, sha256
from experiments.vitl30_multiscale_prototypes.code.data import sparse_encode
from sae_experiments.models.panorama_context import mix_panorama_context

COLORS = ('#ffb700', '#00dcf0', '#e772ff', '#95ee5b')


def choose_patterns(patterns, count):
    """Spread the small audit over order and semantic units, without altering mining."""
    ranked = sorted(patterns, key=lambda r: -min(r['discovery_rate'], r['evaluation_rate']))
    selected, used = [], set()
    quota = max(1, count // 3)
    for order in (2, 3, 4):
        pool = [r for r in ranked if r['order'] == order]
        for slot in range(quota):
            if not pool:
                break
            # Include the most recurrent composition at each order, then spread
            # the remaining audit over units. Selection never uses image quality.
            best = pool[0] if slot == 0 else max(pool, key=lambda r: (len(set(r['units']) - used),
                        min(r['discovery_rate'], r['evaluation_rate'])))
            selected.append(best)
            used.update(best['units'])
            pool.remove(best)
    # Include the water-related compositions that motivated this revision.
    remaining = [r for r in ranked if r not in selected]
    remaining.sort(key=lambda r: ('water:' not in r['key'], -min(r['discovery_rate'], r['evaluation_rate'])))
    selected.extend(remaining[:max(0, count - len(selected))])
    return selected[:count]


def choose_examples(frame, assignment, units, patterns, count):
    examples = []
    for pattern in patterns:
        matches = np.flatnonzero(pattern_mask(assignment, units, pattern['units']))
        subset = frame.iloc[matches]
        hashes = [hashlib.sha256(f'{pattern["pattern_id"]}/{c}/{p}'.encode()).digest()
                  for c, p in zip(subset.city, subset.panoid)]
        ranked = matches[sorted(range(len(matches)), key=hashes.__getitem__)]
        used_cities = set()
        for i in ranked:
            row = frame.iloc[i]
            if row.city in used_cities:
                continue
            examples.append({'pattern_id': pattern['pattern_id'], 'units': pattern['units'],
                             'city': row.city, 'panoid': row.panoid,
                             'grid_x': int(row.grid_x), 'grid_y': int(row.grid_y)})
            used_cities.add(row.city)
            if len(used_cities) == count:
                break
    return examples


@torch.inference_mode()
def infer(images, dino, sae, unit_ids, units, device):
    pixels = torch.stack([preprocess()(im) for im in images]).to(device, dtype=torch.bfloat16)
    prefix = 1 + int(getattr(dino.config, 'num_register_tokens', 0))
    tokens = torch.nn.functional.normalize(dino(pixel_values=pixels).last_hidden_state[:, prefix:].float(), dim=-1)
    context = mix_panorama_context(tokens.reshape(1, 4, PATCHES, 1024), grid_size=28)
    values, indices = sparse_encode(sae, context)
    values, indices = values[0], indices[0]
    feature, patch = indices % WIDTH, indices // WIDTH
    maps = []
    for uid in unit_ids:
        unit = units[uid]
        dims = unit['dimensions']
        if unit['fallback']:
            dims = [d for u in landscape_children(units, unit['family']) for d in u['dimensions']]
        lookup = torch.zeros(WIDTH, device=device, dtype=torch.bool)
        lookup[torch.as_tensor(dims, device=device)] = True
        valid = lookup[feature] & (values > 0)
        flat = torch.zeros(4 * PATCHES, device=device)
        flat.scatter_add_(0, patch[valid], values[valid])
        maps.append(flat.reshape(4, 28, 28).cpu().numpy())
    return np.stack(maps)


def render(images, maps, folder):
    folder.mkdir(parents=True, exist_ok=True)
    size = 336
    original = Image.new('RGB', (size * 4, size + 24), '#141920')
    overlay = original.copy()
    normalized = maps / np.maximum(maps.max(axis=(1, 2, 3), keepdims=True), 1e-8)
    for h, source in enumerate(images):
        rgb = np.asarray(source.resize((size, size)), dtype=float)
        shown = rgb * .35
        fields = np.stack([np.asarray(Image.fromarray(a[h]).resize((size, size), Image.Resampling.NEAREST))
                           for a in normalized])
        winner, strength = fields.argmax(axis=0), fields.max(axis=0)
        for i, color in enumerate(COLORS[:len(maps)]):
            selected = (winner == i) & (strength >= .35)
            color_rgb = np.array([int(color[j:j+2], 16) for j in (1, 3, 5)])
            shown[selected] = .5 * rgb[selected] + .5 * color_rgb
        original.paste(Image.fromarray(rgb.astype('uint8')), (h * size, 24))
        overlay.paste(Image.fromarray(shown.clip(0, 255).astype('uint8')), (h * size, 24))
        for canvas in (original, overlay):
            ImageDraw.Draw(canvas).text((h * size + 8, 6), f'{HEADINGS[h]} degrees', fill='white')
    original.save(folder / 'original.webp', quality=88)
    overlay.save(folder / 'overlay.webp', quality=90)


def run(args):
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU inference fallback')
    torch.set_num_threads(4)
    folder = args.output
    marker = json.loads((folder / 'complete.json').read_text())
    if marker['status'] != 'complete' or marker['run_identity'] != args.run_identity:
        raise ValueError('Composition corpus is incomplete or mismatched')
    units = json.loads((folder / 'units.json').read_text())
    patterns = json.loads((folder / 'patterns.json').read_text())
    selected = choose_patterns(patterns, args.patterns)
    frame = pd.read_parquet(args.followup / 'stages/candidate_combinations/panorama_elements.parquet')
    assignment = np.load(folder / 'assignment.npy', mmap_mode='r')
    examples = choose_examples(frame, assignment, units, selected, args.per_pattern)
    identity = hashlib.sha256(json.dumps({'run': args.run_identity,
        'protocol': sha256(folder / 'protocol.json'), 'threshold': .35,
        'selection': 'sha256 panorama order, distinct cities per pattern'}, sort_keys=True).encode()).hexdigest()
    write_json(folder / 'evidence_selection.json', {'identity': identity, 'examples': examples})
    dino, sae, _ = load_models(args.device)
    result = []
    for i, example in enumerate(examples):
        key = hashlib.sha256(f'{example["pattern_id"]}/{example["city"]}/{example["panoid"]}'.encode()).hexdigest()[:20]
        target = folder / 'evidence' / key
        cache = target / 'activation.npz'
        images = []
        for path in image_paths(example['city'], example['panoid']):
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['identity']) != identity:
                    raise ValueError('Evidence cache identity changed')
                maps = saved['maps']
        else:
            maps = infer(images, dino, sae, example['units'], units, args.device)
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, identity=identity, maps=maps)
        render(images, maps, target)
        result.append({**example, 'labels': [units[j]['label'] for j in example['units']],
                       'colors': list(COLORS[:len(example['units'])]),
                       'selection': 'deterministic distinct-city retrieval audit; not prevalence sampling',
                       'semantic_review': 'pending', 'mask_is_object_segmentation': False,
                       'original': str(target / 'original.webp'), 'overlay': str(target / 'overlay.webp')})
        write_json(folder / 'examples.json', result)
        write_json(folder / 'evidence_progress.json', {'processed': i + 1, 'total': len(examples)})
        print(f'Four-view evidence: {i+1}/{len(examples)} {example["city"]}', flush=True)
    write_json(folder / 'evidence_complete.json', {'status': 'complete', 'identity': identity,
               'run_identity': args.run_identity, 'protocol_sha256': sha256(folder / 'protocol.json'),
               'examples': len(result), 'patterns': len(selected)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup', type=Path, default=FOLLOWUP)
    parser.add_argument('--output', type=Path, default=FOLLOWUP / 'stages/composition_patterns')
    parser.add_argument('--run-identity', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--patterns', type=int, default=14)
    parser.add_argument('--per-pattern', type=int, default=3)
    run(parser.parse_args())
