"""Freeze the small, auditable website gallery from server-side experiment outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_patterns(patterns):
    chosen, seen = [], set()
    for item in patterns:
        relation = item['relation']
        signature = (tuple(sorted(item['families'])),
                     'cross' if relation.startswith('cross_') else 'same',
                     relation.rsplit('_', 1)[-1])
        if signature not in seen:
            seen.add(signature)
            chosen.append(item)
    return chosen


def export(source=HERE / 'outputs', destination=HERE / 'results'):
    source, destination = Path(source), Path(destination)
    original = source / 'results.json'
    data = json.loads(original.read_text())
    if data['status'] != 'complete' or data['scope'] != 'exploratory balanced city sample; sealed test not accessed':
        raise ValueError('Only the completed, sealed-test exploratory result may be published')
    selected = unique_patterns(data['patterns'])
    names = {example['image'] for item in selected for example in item['examples']}
    if len(selected) != 5 or len(names) != 30:
        raise ValueError('Unexpected gallery count; inspect the new experiment before publication')
    if any(Path(name).name != name or not name.endswith('.webp') for name in names):
        raise ValueError('Unsafe gallery filename')
    # Resolve and hash every source before replacing an existing snapshot.
    hashes = {name: digest(source / 'gallery' / name) for name in sorted(names)}
    compact = {**data, 'patterns': selected}
    encoded = json.dumps(compact, ensure_ascii=False, indent=2) + '\n'
    destination.mkdir(parents=True, exist_ok=True)
    gallery = destination / 'gallery'
    gallery.mkdir(exist_ok=True)
    for stale in gallery.glob('*.webp'):
        if stale.name not in names:
            stale.unlink()
    for name in names:
        shutil.copy2(source / 'gallery' / name, gallery / name)
    result = destination / 'results.json'
    result.write_text(encoded)
    provenance = {
        'checkpoint_sha256': data['checkpoint_sha256'],
        'split_identity': data['split_identity'],
        'hierarchy_sha256': data['hierarchy_sha256'],
        'source_results_sha256': digest(original),
        'results_sha256': digest(result),
        'source_displayed': data['findings']['displayed'],
        'snapshot_displayed': len(selected),
        'gallery_sha256': hashes,
    }
    (destination / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'outputs')
    parser.add_argument('--destination', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination), ensure_ascii=False))
