"""The exploratory atlas cannot acquire validated claims through publication."""
import json

import pytest

from docs.site.composition_patterns import checked_compositions, checked_roadside_rows
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


def test_roadside_catalogue_counts_are_complete_and_tentative(tmp_path, monkeypatch):
    from docs.site import composition_roadside

    folder = tmp_path / 'stages/composition_roadside_audit'
    folder.mkdir(parents=True)
    summary = {'patterns_side_eligible': 1, 'sampled_pattern_instances': 2,
               'tentative': {'same_side': 1, 'opposite_sides': 0, 'undetermined': 1}}
    monkeypatch.setattr(composition_roadside, 'checked_audit', lambda _: (folder, summary))
    csv_file = folder / 'pattern_side_audit.csv'
    csv_file.write_text('pattern_id,eligible_for_side_audit,sampled_panoramas,same_side_hypotheses,'
                        'opposite_sides_hypotheses,undetermined_hypotheses,final_same_side,'
                        'final_opposite_sides,final_undetermined\n'
                        'a,True,2,1,0,1,0,0,2\n'
                        'b,False,0,0,0,0,0,0,0\n')
    checked_summary, rows = checked_roadside_rows(tmp_path, [{'pattern_id': 'a'}, {'pattern_id': 'b'}])
    assert checked_summary == summary
    assert rows['a'] == {'eligible': True, 'sampled': 2,
                         'tentative': {'same_side': 1, 'opposite_sides': 0, 'undetermined': 1}}
    with pytest.raises(ValueError, match='catalogue'):
        checked_roadside_rows(tmp_path, [{'pattern_id': 'a'}])
    csv_file.write_text(csv_file.read_text().replace('a,True,2,1,0,1,0,0,2',
                                                      'a,True,2,1,0,1,1,0,1'))
    with pytest.raises(ValueError, match='unverified'):
        checked_roadside_rows(tmp_path, [{'pattern_id': 'a'}, {'pattern_id': 'b'}])
