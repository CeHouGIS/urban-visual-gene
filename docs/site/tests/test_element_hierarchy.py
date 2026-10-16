"""Stage-one metadata must not be published as trained or verified science."""
import json

import pytest

from docs.site import element_hierarchy
from docs.site.project_progress import CHECKPOINT, relational_status, research_progress


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fixture(tmp_path):
    result = tmp_path / 'experiments/vitl30_relational_prototypes/results'
    summary = {'identity': 'hierarchy-a', 'checkpoint_sha256': CHECKPOINT, 'dimensions': 4096,
               'status_counts': {'candidate_ready': 1561}, 'graph_candidate_family_counts': {'building': 513},
               'family_disagreements': 426, 'semantic_validation': 'pending'}
    pilot = {'hierarchy_identity': summary['identity'], 'training_started': False,
             'stage': 'manifest_prepared', 'cities': 29, 'split_counts': {'train': 928, 'dev': 232}}
    put(result / 'summary.json', summary)
    put(result / 'pilot_manifest_summary.json', pilot)
    return result, summary, pilot


def test_candidate_table_is_calculated_but_semantics_and_training_are_pending(tmp_path):
    fixture(tmp_path)
    hierarchy, open_panorama = relational_status(tmp_path)
    assert hierarchy['status'] == 'complete' and hierarchy['semantic_review'] == 'pending'
    assert open_panorama['status'] == 'pending' and open_panorama['test_sealed']
    tasks = [{'id': name, 'status': 'complete', 'detail': 'old model finished'}
             for name in ['activations', 'sae', 'dimension_vlm', 'prototypes', 'prototype_vlm', 'evaluation']]
    progress = research_progress(tasks + [hierarchy, open_panorama], {'panoramas': 86038})
    milestones = {m['id']: m for m in progress['milestones']}
    assert milestones['element_families']['status'] == 'partial'
    assert milestones['local_compositions']['status'] == 'pending'
    assert milestones['patterns']['status'] == 'pending'
    assert milestones['validity']['status'] == 'pending'
    assert '局部要素' in progress['focus']


def test_open_panorama_status_links_current_method(tmp_path):
    fixture(tmp_path)
    put(tmp_path / 'experiments/vitl30_open_panorama_prototypes/outputs/status.json', {
        'status': 'running', 'detail': '第2轮生成阶段', 'metrics': {'active_prototypes': 11}})
    rows = relational_status(tmp_path)
    assert rows[-1]['status'] == 'running' and rows[-1]['test_sealed']
    assert rows[-1]['detail'] == '第2轮生成阶段'
    assert rows[-1]['metrics']['active_prototypes'] == 11
    assert rows[-1]['link'] == 'elements/#open-method'


def test_progress_rejects_stale_pilot_hierarchy(tmp_path):
    result, summary, pilot = fixture(tmp_path)
    pilot['hierarchy_identity'] = 'other'
    put(result / 'pilot_manifest_summary.json', pilot)
    with pytest.raises(ValueError, match='another element hierarchy'):
        relational_status(tmp_path)


def test_publication_rejects_changed_vlm_records_before_copying_images(tmp_path, monkeypatch):
    experiment = tmp_path / 'experiment'
    reports = tmp_path / 'reports'
    put(experiment / 'results/hierarchy.json', {'identity': 'a', 'summary': {},
         'protocol': {'checkpoint_sha256': CHECKPOINT,
                      'sources': {'masked': {'records_sha256': 'old-hash'}}}})
    put(experiment / 'results/pilot_manifest_summary.json', {'hierarchy_identity': 'a'})
    put(reports / 'masked/dimension_interpretations.json', {'dimensions': ['changed']})
    monkeypatch.setattr(element_hierarchy, 'EXPERIMENT', experiment)
    monkeypatch.setattr(element_hierarchy, 'REPORTS', reports)
    with pytest.raises(ValueError, match='VLM evidence changed'):
        element_hierarchy.install(tmp_path / 'site', None, 'today')
    assert not (tmp_path / 'site').exists()
