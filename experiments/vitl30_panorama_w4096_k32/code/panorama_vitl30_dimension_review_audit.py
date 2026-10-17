"""Audit cross-group agreement before promoting a Qwen description to a candidate."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .panorama_vitl30_dimension_review_v3 import EXPERIMENT, atomic_json

REPORT = EXPERIMENT / 'reports/vlm_dimensions_top40_masked_only_v5'
PUBLIC_SUMMARY = EXPERIMENT / 'results/dimension_screening/review_pilot.json'


def assess(record):
    parts = record.get('parts', [])
    observations = [part.get('interpretation') for part in parts]
    final = record.get('interpretation')
    if len(parts) != 4 or any(value is None for value in observations) or final is None:
        return 'incomplete_visual_evidence'
    if any(value['consistency'] != 'mostly_consistent' or value['type'] in {'mixed', 'unclear'}
           for value in observations):
        return 'mixed_or_unclear'
    types = {value['type'] for value in observations}
    if len(types) != 1 or final['type'] not in types:
        return 'cross_group_type_conflict'
    if final['consistency'] != 'mostly_consistent' or final['confidence'] == 'low':
        return 'mixed_or_unclear'
    return 'candidate_needs_independent_validation'


def audit(report=REPORT):
    report = Path(report)
    run = json.loads((report / 'identity.json').read_text())
    records = [json.loads(path.read_text()) for path in sorted((report / 'records').glob('f*.json'))]
    rows = []
    for record in records:
        if record['identity'] != run['identity'] or record['status'] != 'completed':
            continue
        decision = assess(record)
        rows.append({'feature': record['feature'], 'decision': decision,
                     'instance_count': record['instance_count'],
                     'part_labels': [part['interpretation']['label'] if part['interpretation'] else ''
                                     for part in record['parts']],
                     'part_types': [part['interpretation']['type'] if part['interpretation'] else ''
                                    for part in record['parts']],
                     'final_label': record['interpretation']['label'] if record['interpretation'] else '',
                     'independently_validated': False})
    result = {'review_identity': run['identity'], 'pilot_only': True,
              'semantic_validation': 'pending', 'status_counts': dict(Counter(r['decision'] for r in rows)),
              'dimensions': rows}
    atomic_json(report / 'audit.json', result)
    atomic_json(PUBLIC_SUMMARY, result)
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False))
