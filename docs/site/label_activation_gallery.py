"""Choose traceable panorama activation examples for every Qwen fine label."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image

REPORT_NAMES = ('vlm_dimensions_top10_masked', 'vlm_dimensions_top10_expanded')
HEADINGS = (0, 90, 180, 270)


def select_examples(candidates, limit=5):
    """Use peak response, with distinct dimensions before repeat examples."""
    ranked = sorted((row for row in candidates if row['activation'] > 0),
                    key=lambda row: (-row['activation'], row['feature'], row['rank']))
    chosen, panoramas, features = [], set(), set()
    for distinct_feature in (True, False):
        for row in ranked:
            key = (row['city'], row['pano'])
            if key in panoramas or (distinct_feature and row['feature'] in features):
                continue
            chosen.append(row)
            panoramas.add(key)
            features.add(row['feature'])
            if len(chosen) == limit:
                return chosen
    return chosen


def _source_views(crop):
    active = Path(crop['image'])
    return [active.with_name(f'{crop["pano"]}_{heading}.jpg') for heading in HEADINGS]


def install(site, target, graph, reports, source_identities):
    evidence = {}
    for name in REPORT_NAMES:
        payload = json.loads((reports / name / 'evidence.json').read_text())
        if payload['identity'] != source_identities[name]['identity']:
            raise ValueError(f'Qwen activation evidence identity changed: {name}')
        records = payload['dimensions']
        if len(records) != graph['total']:
            raise ValueError(f'Unexpected Qwen evidence length: {name}')
        evidence[name] = records
    expanded_dir = target / 'label-sources'
    expanded_dir.mkdir(exist_ok=True)
    extra_cache = {}
    stats = defaultdict(int)
    stats['expanded_examples'] = 0
    for family in graph['families']:
        for label in family['labels']:
            if not label['eligible']:
                label['examples'] = []
                continue
            candidates = []
            for feature in label['features']:
                for source in REPORT_NAMES:
                    for crop in evidence[source][feature]['crops']:
                        if crop['feature'] != feature:
                            raise ValueError('Qwen crop points to another SAE dimension')
                        if source == REPORT_NAMES[0]:
                            image_id = crop['image_id']
                            start = image_id - image_id % 4
                            paths = [site / 'semantic/sources' / f'{start+i}.jpg' for i in range(4)]
                        else:
                            paths = _source_views(crop)
                        if all(path.exists() for path in paths):
                            candidates.append({**crop, 'source': source, 'paths': paths})
            chosen = select_examples([row for row in candidates if row['source'] == REPORT_NAMES[0]])
            if len(chosen) < 5:
                seen = {(row['city'], row['pano']) for row in chosen}
                expanded = [row for row in candidates if row['source'] == REPORT_NAMES[1]]
                for row in select_examples(expanded, len(expanded)):
                    key = (row['city'], row['pano'])
                    if key not in seen:
                        chosen.append(row)
                        seen.add(key)
                    if len(chosen) == 5:
                        break
            label['examples'] = []
            for crop in chosen:
                if crop['source'] == REPORT_NAMES[0]:
                    image_id = crop['image_id']
                    start = image_id - image_id % 4
                    views = [f'../semantic/sources/{start+i}.jpg' for i in range(4)]
                    heat = crop['heatmap_png']
                else:
                    paths = crop['paths']
                    key = (crop['city'], crop['pano'])
                    if key not in extra_cache:
                        digest = hashlib.sha256(str(paths[0]).encode()).hexdigest()[:16]
                        names = [f'{digest}_{heading}.webp' for heading in HEADINGS]
                        for source, name in zip(paths, names):
                            destination = expanded_dir / name
                            if not destination.exists() or destination.stat().st_mtime < source.stat().st_mtime:
                                with Image.open(source) as image:
                                    image.convert('RGB').save(destination, 'WEBP', quality=86, method=4)
                        extra_cache[key] = [f'label-sources/{name}' for name in names]
                    views = extra_cache[key]
                    heat = crop['heatmap_png']
                    stats['expanded_examples'] += 1
                label['examples'].append({
                    'feature': crop['feature'], 'rank': crop['rank'],
                    'activation': crop['activation'], 'city': crop['city'],
                    'pano': crop['pano'], 'heading': crop['heading'],
                    'source': crop['source'], 'views': views, 'heat': heat,
                })
            stats['labels'] += 1
            stats['examples'] += len(chosen)
            stats['labels_with_five'] += len(chosen) == 5
            stats['labels_without_examples'] += len(chosen) == 0
    stats['expanded_panoramas'] = len(extra_cache)
    used = {Path(view).name for views in extra_cache.values() for view in views}
    for path in expanded_dir.iterdir():
        if path.name not in used:
            path.unlink()
    return dict(stats)
