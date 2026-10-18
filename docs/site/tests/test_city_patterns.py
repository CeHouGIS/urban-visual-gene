"""Publication and progress must preserve run identity and review uncertainty."""
import json

import pytest

from docs.site.city_patterns import checked_inputs
from docs.site.project_progress import city_pattern_progress


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_cannot_publish_mixed_runs_or_failed_thesis(tmp_path):
    put(tmp_path / 'complete.json', {'cities': 30, 'identity': {'extract_run_identity': 'correct'}})
    names = ('candidate_combinations', 'road_sides', 'prototype_city_distribution', 'chapter5_figures')
    for stage in names:
        put(tmp_path / 'stages' / stage / 'complete.json', {'status': 'complete', 'run_identity': 'correct'})
    report = tmp_path / 'stages/chapter5_figures/compile_report.json'
    put(report, {'passed': False})
    with pytest.raises(ValueError, match='compile gate'):
        checked_inputs(tmp_path)
    put(report, {'passed': True})
    assert checked_inputs(tmp_path)[0] == 'correct'
    put(tmp_path / 'stages/road_sides/complete.json', {'status': 'complete', 'run_identity': 'other'})
    with pytest.raises(ValueError, match='mismatched'):
        checked_inputs(tmp_path)


def test_compute_completion_does_not_complete_scientific_road_review(tmp_path):
    stages = ('extract', 'validate_chunks', 'city_cell_screening', 'candidate_combinations',
              'road_sides', 'prototype_city_distribution', 'chapter5_figures')
    put(tmp_path / 'task_queue.json', {'phase': 'complete', 'run_identity': 'correct',
        'tasks': [{'id': stage, 'status': 'complete'} for stage in stages]})
    put(tmp_path / 'complete.json', {'processed': 681645, 'valid_cells': 72625,
                                     'identity': {'extract_run_identity': 'correct'}})
    put(tmp_path / 'stages/candidate_combinations/summary.json', {'candidate_count': 7})
    value = city_pattern_progress(tmp_path)
    assert value['terminal']
    assert value['milestones']['road_sides']['status'] == 'partial'
    assert value['milestones']['geography']['status'] == 'partial'
    assert value['milestones']['figure_refresh']['status'] == 'complete'
    assert '语义' in value['focus']
