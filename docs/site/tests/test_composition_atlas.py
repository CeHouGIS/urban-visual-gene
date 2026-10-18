"""The exploratory atlas cannot acquire validated claims through publication."""
import json

import pytest

from docs.site.composition_patterns import checked_compositions
from docs.site.project_progress import city_pattern_progress


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_publication_requires_matching_complete_corpus_and_exploratory_claims(tmp_path):
    put(tmp_path / 'complete.json', {'processed': 100, 'valid_cells': 10,
                                     'identity': {'extract_run_identity': 'correct'}})
    root = tmp_path / 'stages/composition_patterns'
    put(root / 'complete.json', {'status': 'complete', 'run_identity': 'correct'})
    summary = {'run_identity': 'correct', 'panoramas': 100, 'cells': 10,
               'statistical_confirmation': False, 'layout_verified': False,
               'labels_human_validated': False, 'landscape_filter_applied': True}
    put(root / 'summary.json', summary)
    assert checked_compositions(tmp_path)[1] == summary
    for change, message in [({'run_identity': 'other'}, 'Mismatched'),
                            ({'panoramas': 90}, 'Incomplete'),
                            ({'layout_verified': True}, 'validated'),
                            ({'landscape_filter_applied': False}, 'scope filter')]:
        put(root / 'summary.json', {**summary, **change})
        with pytest.raises(ValueError, match=message):
            checked_compositions(tmp_path)


def test_revision_compute_completion_leaves_semantics_geometry_and_writing_pending(tmp_path):
    put(tmp_path / 'task_queue.json', {'phase': 'complete', 'run_identity': 'correct', 'tasks': []})
    put(tmp_path / 'complete.json', {'processed': 100, 'valid_cells': 10,
                                     'identity': {'extract_run_identity': 'correct'}})
    root = tmp_path / 'stages/composition_patterns'
    put(root / 'complete.json', {'status': 'complete', 'run_identity': 'correct'})
    put(root / 'summary.json', {'run_identity': 'correct', 'canonical_units': 103, 'recurrent_patterns': 42})
    value = city_pattern_progress(tmp_path)
    assert value['rows'][-1]['status'] == 'complete'
    assert value['milestones']['patterns']['status'] == 'partial'
    assert value['milestones']['road_sides']['status'] == 'pending'
    assert value['milestones']['figure_refresh']['status'] == 'partial'
    assert '人工校准' in value['next_step']
