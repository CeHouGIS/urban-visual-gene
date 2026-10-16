"""Publishing must distinguish running calculations from achieved coverage."""
import json

import pytest

from docs.site.project_progress import patch_coverage_status, response_refit_status


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_completed_pipeline_retains_actual_failures_and_semantic_review(tmp_path):
    out = tmp_path / 'experiments/vitl30_patch_coverage/outputs'
    put(out / 'protocol.json', {
        'split_counts': {'train': 10, 'dev': 2, 'test': 3}, 'cities': 1, 'epochs': 8,
        'policy': {'target_fraction': .8, 'maximum_patch_nmse': .5}})
    put(out / 'pipeline_status.json', {'status': 'complete', 'stage': 'semantic_review_pending'})
    put(out / 'model/complete.json', {})
    summary = {'images': 8, 'image_pass_fraction': .25, 'mean_image_patch_coverage': .6,
               'panorama_all_four_pass_fraction': 0., 'images_below_target': 6,
               'invalid_patches': 5, 'all_images_meet_target': False, 'coverage_quantiles': {'0': .1}}
    put(out / 'evaluation/dev/complete.json', summary)
    row = patch_coverage_status(tmp_path)
    assert row['status'] == 'complete'
    assert row['semantic_review'] == 'pending' and row['test_sealed']
    assert row['metrics']['dev']['all_images_meet_target'] is False
    assert '达标图25.000%' in row['detail']
    assert '语义待核查' in row['detail']
    assert row['runs'][0]['status'] == 'complete'
    assert row['runs'][1]['status'] == 'pending'


def test_missing_pipeline_is_pending_not_running_or_achieved(tmp_path):
    assert patch_coverage_status(tmp_path) is None
    out = tmp_path / 'experiments/vitl30_patch_coverage/outputs'
    put(out / 'protocol.json', {
        'split_counts': {'train': 10, 'dev': 2, 'test': 3}, 'cities': 1, 'epochs': 8,
        'policy': {'target_fraction': .8, 'maximum_patch_nmse': .5}})
    row = patch_coverage_status(tmp_path)
    assert row['status'] == 'pending'
    assert row['progress']['current'] == 0
    assert row['metrics'] == {}


def test_response_diagnostic_cannot_publish_a_different_experiment(tmp_path):
    out = tmp_path / 'experiments/vitl30_patch_coverage/outputs'
    put(out / 'protocol.json', {'identity': 'expected'})
    put(out / 'response_refit_v1/complete.json', {'protocol': {'experiment_identity': 'other'}})
    with pytest.raises(ValueError, match='another coverage protocol'):
        response_refit_status(tmp_path)


def test_response_diagnostic_preserves_failures_and_sealed_test(tmp_path):
    out = tmp_path / 'experiments/vitl30_patch_coverage/outputs'
    put(out / 'protocol.json', {'identity': 'expected'})
    before = {'panoramas': 29, 'images': 116, 'mean_image_patch_coverage': .478,
              'images_meeting_target': 0}
    after = {**before, 'mean_image_patch_coverage': .504}
    put(out / 'response_refit_v1/complete.json', {
        'protocol': {'experiment_identity': 'expected', 'test_sealed': True},
        'test_used': False, 'summaries': {'encoder': before, 'coverage': after}})
    row = response_refit_status(tmp_path)
    assert row['status'] == 'complete' and row['test_sealed']
    assert row['semantic_review'] == 'pending'
    assert row['metrics']['coverage']['images_meeting_target'] == 0
    assert '47.8%→50.4%' in row['detail'] and '0/116' in row['detail']
