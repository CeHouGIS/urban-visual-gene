"""Roadside publication keeps the retrieval audit distinct from verified rates."""
import json

import pytest

from docs.site.composition_roadside import checked_audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_roadside_publication_requires_exact_composition_protocol_and_no_final_claim(tmp_path):
    composition = tmp_path / 'stages/composition_patterns'
    put(composition / 'summary.json', {'run_identity': 'corpus', 'recurrent_patterns': 10})
    (composition / 'protocol.json').write_text('source protocol')
    import hashlib
    digest = hashlib.sha256(b'source protocol').hexdigest()
    folder = tmp_path / 'stages/composition_roadside_audit'
    put(folder / 'protocol.json', {'identity': 'audit', 'composition_protocol_sha256': digest})
    put(folder / 'complete.json', {'status': 'complete', 'run_identity': 'corpus',
                                   'identity': 'audit', 'examples': 24})
    summary = {'run_identity': 'corpus', 'protocol_identity': 'audit',
        'patterns_total': 10, 'patterns_side_eligible': 4, 'patterns_not_side_eligible': 6,
        'sampled_pattern_instances': 24,
        'tentative': {'same_side': 1, 'opposite_sides': 2, 'undetermined': 21},
        'final': {'same_side': 0, 'opposite_sides': 0, 'undetermined': 24},
        'human_validated': False, 'calibration_verified': False,
        'corpus_roadside_prevalence_estimable': False}
    put(folder / 'summary.json', summary)
    assert checked_audit(tmp_path)[1] == summary
    for delta in ({'run_identity': 'other'}, {'patterns_total': 11},
                  {'corpus_roadside_prevalence_estimable': True},
                  {'final': {'same_side': 1, 'opposite_sides': 0, 'undetermined': 23}}):
        put(folder / 'summary.json', {**summary, **delta})
        with pytest.raises(ValueError):
            checked_audit(tmp_path)
