"""Build a conservative, reversible semantic hierarchy from existing VLM records.

This is text bookkeeping, not a new VLM prediction or semantic ground truth.
Every dimension survives, including textures, ambiguous labels and no evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'experiments/vitl30_panorama_w4096_k32/reports'
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'
WIDTH = 4096
SOURCES = ('vlm_dimensions_top10_masked', 'vlm_dimensions_top10_expanded', 'vlm_dimensions')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def label_family(label, taxonomy):
    """Only explicit label aliases; no guessing from incidental objects in cues."""
    aliases = {alias: item['id'] for item in taxonomy['families'] for alias in item['aliases']}
    return aliases.get(label.strip(), 'unresolved')


def quality_flags(row, family, taxonomy):
    if row['status'] != 'interpreted':
        return ['no_interpretation']
    reasons = []
    if row['crop_count'] < taxonomy['minimum_examples']:
        reasons.append('fewer_than_five_examples')
    if row['consistency'] != 'mostly_consistent':
        reasons.append('mixed_or_unclear')
    # A screening rule only: the VLM's confidence is NOT calibrated accuracy.
    if row['confidence_self_reported'] not in {'high', 'medium'}:
        reasons.append('low_self_reported_confidence')
    if family == 'unresolved':
        reasons.append('ambiguous_or_unmapped_label')
    if row.get('semantic_type') in {'camera_artifact', 'color_light', 'mixed', 'unclear'}:
        reasons.append('non_element_or_unclear_response')
    if row.get('embedding_evidence_status') == 'semantic_mixture_candidate':
        reasons.append('embedding_mixture_flag')
    return reasons


def combine(masked, expanded, reference, taxonomy):
    """Primary masked evidence wins; expanded evidence fills missing dimensions only."""
    tables = [{int(r['feature']): r for r in rows} for rows in (masked, expanded, reference)]
    if any(set(table) != set(range(WIDTH)) for table in tables):
        raise ValueError('Each source must contain every dimension exactly once')
    if any(len(rows) != WIDTH for rows in (masked, expanded, reference)):
        raise ValueError('Duplicate/missing dimension records')
    family_info = {r['id']: r for r in taxonomy['families']}
    result = []
    for feature in range(WIDTH):
        original, extra, clean = [t[feature] for t in tables]
        if original['status'] == 'interpreted':
            if extra['status'] == 'interpreted':
                raise ValueError(f'Expanded evidence unexpectedly overwrites f{feature}')
            row, source = original, SOURCES[0]
        elif extra['status'] == 'interpreted':
            row, source = extra, SOURCES[1]
        else:
            row, source = extra if extra['status'] != 'not_requested' else original, SOURCES[1]
        family = label_family(row.get('label_zh', ''), taxonomy)
        reference_family = label_family(clean.get('label_zh', ''), taxonomy) if clean['status'] == 'interpreted' else 'unresolved'
        flags = quality_flags(row, family, taxonomy)
        # Same-object parts may disagree in names but agree at the parent level.
        disagreement = (family != 'unresolved' and reference_family != 'unresolved'
                        and family != reference_family and clean.get('crop_count', 0) >= 5
                        and clean.get('consistency') == 'mostly_consistent')
        if disagreement:
            flags.append('masked_vs_unmasked_family_disagreement')
        info = family_info[family]
        if row['status'] != 'interpreted':
            status = 'unobserved'
        elif 'fewer_than_five_examples' in flags:
            status = 'insufficient_evidence'
        elif flags:
            status = 'needs_review'
        elif not info['counts_as_element']:
            status = 'context_or_material'
        else:
            status = 'candidate_ready'
        result.append({
            'feature': feature, 'label_zh': row.get('label_zh', ''),
            'fine_label_zh': row.get('label_zh', ''), 'family': family,
            'family_zh': info['label_zh'], 'semantic_role': info['role'],
            'counts_as_element': info['counts_as_element'], 'status': status,
            'provisional_graph_candidate': status == 'candidate_ready',
            'human_validated': False, 'independently_validated': False,
            'source': source, 'crop_count': row.get('crop_count', 0),
            'semantic_type': row.get('semantic_type', ''),
            'consistency': row.get('consistency', ''),
            'confidence_self_reported': row.get('confidence_self_reported', ''),
            'cues_zh': row.get('cues_zh', []), 'caveats_zh': row.get('caveats_zh', []),
            'quality_flags': flags, 'reference_label_zh': clean.get('label_zh', ''),
            'reference_family': reference_family, 'family_disagreement': disagreement,
            'review_priority': 0 if disagreement else 1 if status == 'needs_review' else 2 if status == 'candidate_ready' else 3,
            'main_evidence_url': f'../semantic/feature.html?f={feature}' if source == SOURCES[0] else None,
        })
    return result


def build(reports=REPORTS, out=EXPERIMENT / 'results'):
    reports, out = Path(reports), Path(out)
    taxonomy_path = EXPERIMENT / 'taxonomy.json'
    taxonomy = read(taxonomy_path)
    aliases = [a for f in taxonomy['families'] for a in f['aliases']]
    if len(set(aliases)) != len(aliases):
        raise ValueError('Ambiguous taxonomy aliases')
    inputs, sources = [], {}
    for name in SOURCES:
        path = reports / name
        identity = read(path / 'identity.json')
        table = read(path / 'dimension_interpretations.json')
        if identity['checkpoint_sha256'] != CHECKPOINT or table['identity'] != identity['identity']:
            raise ValueError(f'SAE/source identity mismatch: {name}')
        sources[name] = {'identity': identity['identity'],
            'records_sha256': sha(path / 'dimension_interpretations.json'),
            'identity_sha256': sha(path / 'identity.json')}
        inputs.append(table['dimensions'])
    rows = combine(*inputs, taxonomy)
    protocol = {'version': 1, 'checkpoint_sha256': CHECKPOINT, 'sources': sources,
        'taxonomy_sha256': sha(taxonomy_path), 'code_sha256': sha(Path(__file__)),
        'algorithm': 'explicit fine-label aliases + parent consolidation + conservative evidence screening',
        'new_model_inference': False, 'semantic_ground_truth': False,
        'test_activation_access': False,
        'prior_evidence_scope': 'Existing post-hoc VLM reports include images from prototype train/dev/test. Semantic priors therefore are not independent of the sealed test; no new test images or activations are opened.',
        'intended_use': 'Provisional diagnostic grouping; semantic validation is still required before scientific landscape-combination claims'}
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    summary = {'identity': identity, 'checkpoint_sha256': CHECKPOINT, 'dimensions': WIDTH,
        'interpreted': sum(r['status'] != 'unobserved' for r in rows),
        'source_counts': dict(Counter(r['source'] for r in rows if r['status'] != 'unobserved')),
        'status_counts': dict(Counter(r['status'] for r in rows)),
        'family_counts': dict(Counter(r['family'] for r in rows)),
        'graph_candidate_family_counts': dict(Counter(r['family'] for r in rows if r['provisional_graph_candidate'])),
        'family_disagreements': sum(r['family_disagreement'] for r in rows),
        'semantic_validation': 'pending', 'graph_training': 'not_started',
        'test_activations_opened': False, 'prior_semantic_evidence_includes_test': True}
    write(out / 'hierarchy.json', {'identity': identity, 'protocol': protocol,
        'families': taxonomy['families'], 'summary': summary, 'dimensions': rows})
    write(out / 'summary.json', summary)
    out.mkdir(parents=True, exist_ok=True)
    columns = ['feature', 'fine_label_zh', 'family', 'family_zh', 'semantic_role', 'status',
        'provisional_graph_candidate', 'crop_count', 'consistency', 'confidence_self_reported',
        'source', 'reference_label_zh', 'reference_family', 'family_disagreement',
        'quality_flags', 'human_validated', 'independently_validated']
    for name, selected in [('dimension_elements.csv', rows),
                           ('review_queue.csv', sorted(rows, key=lambda r: (r['review_priority'], r['feature'])))]:
        with (out / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows({**r, 'quality_flags': ';'.join(r['quality_flags'])} for r in selected)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=EXPERIMENT / 'results')
    args = parser.parse_args()
    print(json.dumps(build(out=args.out), ensure_ascii=False, indent=2))
