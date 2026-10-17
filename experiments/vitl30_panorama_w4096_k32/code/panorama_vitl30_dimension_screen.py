"""Screen SAE dimensions before treating Qwen descriptions as landscape labels.

The fixed validation sample and the expanded interpretation cache have different
roles. Expanded support is available only for dimensions absent from the fixed
sample. Spatial concentration is a review cue, not evidence of positional coding.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = EXPERIMENT / 'outputs/analysis/semantic_umap_top100'
EXPANDED = EXPERIMENT / 'outputs/analysis/interpretation_expanded_top10/top.npz'
AUDIT = EXPERIMENT / 'results/unit_motifs/dimension_audit.csv'
OUT = EXPERIMENT / 'results/dimension_screening'
WIDTH = 4096
MIN_SUPPORT = 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen_one(feature, fixed_support, expanded_support, audit, positions, expanded_cities=()):
    fixed_support = int(fixed_support)
    expanded_support = int(expanded_support) if fixed_support == 0 else None
    support = fixed_support if fixed_support else expanded_support
    if support is None:
        raise ValueError('Missing expanded support for an unobserved dimension')
    scope = 'fixed_validation' if fixed_support else 'expanded_posthoc'
    if support == 0:
        status = 'no_positive_activation'
    elif support < MIN_SUPPORT:
        status = 'insufficient_support'
    else:
        status = 'reviewable'
    bins = Counter((min(y // 7, 3), min(x // 7, 3)) for y, x in positions)
    concentration = max(bins.values()) / len(positions) if positions else None
    return {
        'feature': feature, 'status': status, 'label_eligible': status == 'reviewable',
        'support_scope': scope, 'positive_panoramas': support,
        'fixed_validation_positive_panoramas': fixed_support,
        'expanded_positive_panoramas': expanded_support,
        'top_crop_count': (int(audit['top_crop_count']) if fixed_support else len(expanded_cities)),
        'top_crop_cities': (int(audit['top_crop_cities']) if fixed_support else len(set(expanded_cities))),
        'embedding_screen': audit['status'],
        'peak_position_examples': len(positions),
        'peak_4x4_max_bin_fraction': concentration,
        'position_review': bool(len(positions) >= MIN_SUPPORT and concentration >= .7),
        'position_coding_confirmed': False,
        'semantic_validation': 'pending',
    }


def build(source=SOURCE, expanded=EXPANDED, audit_path=AUDIT, out=OUT):
    source, expanded, audit_path, out = map(Path, (source, expanded, audit_path, out))
    run = json.loads((source / 'run.json').read_text())
    if run['checkpoint_sha256'] != json.loads((EXPERIMENT / 'outputs/analysis/interpretation_expanded_top10/run.json').read_text())['checkpoint_sha256']:
        raise ValueError('Ranking checkpoint mismatch')
    fixed = np.load(source / 'positive_panorama_support.npy')
    with np.load(expanded) as cache:
        wider = cache['support'].copy()
    if fixed.shape != (WIDTH,) or wider.shape != (WIDTH,):
        raise ValueError('Expected 4,096 support counts')
    with audit_path.open(newline='') as stream:
        audit = {int(row['feature']): row for row in csv.DictReader(stream)}
    if set(audit) != set(range(WIDTH)):
        raise ValueError('Incomplete dimension audit')
    positions = [[] for _ in range(WIDTH)]
    with (source / 'top_images.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            feature, patch = int(row['feature']), int(row['peak_patch']) % 784
            if not 0 <= feature < WIDTH:
                raise ValueError('Invalid ranked dimension')
            positions[feature].append(divmod(patch, 28))
    expanded_cities = [[] for _ in range(WIDTH)]
    with (expanded.parent / 'crops.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            feature = int(row['feature'])
            expanded_cities[feature].append(row['city'])
            if fixed[feature] == 0:
                positions[feature].append(divmod(int(row['peak_patch']), 28))
    rows = [screen_one(f, fixed[f], wider[f], audit[f], positions[f], expanded_cities[f])
            for f in range(WIDTH)]
    protocol = {
        'checkpoint_sha256': run['checkpoint_sha256'],
        'fixed_validation_panoramas': run['panoramas'],
        'expanded_posthoc_panoramas': json.loads((expanded.parent / 'run.json').read_text())['panoramas'],
        'expanded_support_only_for_fixed_zero': True,
        'min_positive_panoramas_for_label': MIN_SUPPORT,
        'position_rule': 'At least 20 ranked peaks and >=70% in one 7x7 block of a 4x4 view grid; review cue only',
        'position_coding_established': False,
        'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [source / 'run.json', source / 'positive_panorama_support.npy',
             source / 'top_images.jsonl', expanded, expanded.parent / 'crops.jsonl', audit_path]},
    }
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    summary = {'identity': identity, 'dimensions': WIDTH,
               'status_counts': dict(Counter(r['status'] for r in rows)),
               'position_review_candidates': sum(r['position_review'] for r in rows),
               'label_eligible': sum(r['label_eligible'] for r in rows),
               'semantic_validation': 'pending'}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'screening.json').write_text(json.dumps({'identity': identity, 'protocol': protocol,
        'summary': summary, 'dimensions': rows}, ensure_ascii=False, indent=2) + '\n')
    with (out / 'screening.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(build(out=args.out), ensure_ascii=False, indent=2))
