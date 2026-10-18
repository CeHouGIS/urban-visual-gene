import argparse
import json

import pytest

from experiments.vitl30_city_patterns.code import auto_continue, followup


RUN_ID = 'test-extraction'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def ready_run(tmp_path):
    extraction = tmp_path / 'extraction'
    followup_dir = tmp_path / 'followup'
    write_json(extraction / 'run.json', {'identity': RUN_ID})
    write_json(followup_dir / 'complete.json', {
        'identity': {'extract_run_identity': RUN_ID}, 'cities': 30})
    write_json(followup_dir / 'status.json', {'status': 'complete'})
    tasks = [{'id': stage, 'status': 'complete', 'depends_on': []}
             for stage in ('extract', 'validate_chunks', 'city_cell_screening')]
    tasks += [{**task, 'status': 'pending'} for task in followup.NEXT_TASKS]
    write_json(followup_dir / 'task_queue.json', {
        'run_identity': RUN_ID, 'phase': 'complete', 'tasks': tasks})
    return argparse.Namespace(followup=followup_dir, extraction=extraction,
                              interval=10, retry_delay=10, attempts=2, once=True)


def make_artifacts(stage, stage_dir):
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == 'candidate_combinations':
        write_json(stage_dir / 'protocol.json', {'run_identity': RUN_ID})
        write_json(stage_dir / 'summary.json', {'cities': 30, 'candidate_count': 0})
        (stage_dir / 'candidates.csv').write_text('candidate_id\n')
    elif stage == 'road_sides':
        write_json(stage_dir / 'protocol.json', {'run_identity': RUN_ID})
        write_json(stage_dir / 'summary.json', {
            'same_side': 0, 'opposite_side': 0, 'undetermined': 0})
        (stage_dir / 'side_examples.csv').write_text('panoid,city,side_label\n')
    elif stage == 'prototype_city_distribution':
        write_json(stage_dir / 'protocol.json', {'run_identity': RUN_ID})
        write_json(stage_dir / 'summary.json', {'cities': 30})
        (stage_dir / 'city_prototype_distribution.csv').write_text(
            'city,prototype_id\n' + ''.join(f'city_{i},none\n' for i in range(30)))
        (stage_dir / 'cell_prototype_distribution.parquet').write_bytes(b'PAR1')
    elif stage == 'chapter5_figures':
        write_json(stage_dir / 'figure_inventory.json', {'figures': [{}] * 11})
        write_json(stage_dir / 'compile_report.json', {'passed': True})
        (stage_dir / 'thesis_preview.pdf').write_bytes(b'%PDF' + b'x' * 100000)
    write_json(stage_dir / 'complete.json', {
        'schema_version': 1, 'stage': stage,
        'run_identity': RUN_ID, 'status': 'complete'})


def test_waits_for_completed_queue_even_after_screening_marker(tmp_path, monkeypatch):
    args = ready_run(tmp_path)
    queue_file = args.followup / 'task_queue.json'
    queue = json.loads(queue_file.read_text())
    queue['phase'] = 'waiting'
    write_json(queue_file, queue)
    monkeypatch.setattr(auto_continue, 'run_agent', lambda *unused: pytest.fail('started early'))
    auto_continue.run(args)
    status = json.loads((args.followup / 'auto_continue_status.json').read_text())
    assert status['status'] == 'waiting'


def test_runs_each_stage_once_and_resumes_verified_results(tmp_path, monkeypatch):
    args = ready_run(tmp_path)
    called = []

    def agent(stage, stage_dir, *unused):
        called.append(stage)
        make_artifacts(stage, stage_dir)
        return 0

    monkeypatch.setattr(auto_continue, 'run_agent', agent)
    auto_continue.run(args)
    assert called == [task['id'] for task in followup.NEXT_TASKS]
    queue = json.loads((args.followup / 'task_queue.json').read_text())
    assert queue['phase'] == 'complete'
    assert all(task['status'] == 'complete' for task in queue['tasks'])
    auto_continue.run(args)
    assert called == [task['id'] for task in followup.NEXT_TASKS]


def test_stops_on_failed_stage_without_running_successors(tmp_path, monkeypatch):
    args = ready_run(tmp_path)
    called = []
    monkeypatch.setattr(auto_continue.time, 'sleep', lambda unused: None)

    def failing_agent(stage, *unused):
        called.append(stage)
        return 1

    monkeypatch.setattr(auto_continue, 'run_agent', failing_agent)
    with pytest.raises(RuntimeError, match='candidate_combinations'):
        auto_continue.run(args)
    assert called == ['candidate_combinations', 'candidate_combinations']
    queue = json.loads((args.followup / 'task_queue.json').read_text())
    statuses = {task['id']: task['status'] for task in queue['tasks']}
    assert statuses['candidate_combinations'] == 'error'
    assert statuses['road_sides'] == 'pending'
    assert statuses['prototype_city_distribution'] == 'pending'
    called.clear()

    def succeeding_agent(stage, stage_dir, *unused):
        called.append(stage)
        make_artifacts(stage, stage_dir)
        return 0

    monkeypatch.setattr(auto_continue, 'run_agent', succeeding_agent)
    auto_continue.run(args)
    assert called == [task['id'] for task in followup.NEXT_TASKS]


def test_agent_runtime_exception_is_recorded_and_retried(tmp_path, monkeypatch):
    args = ready_run(tmp_path)
    called = []
    monkeypatch.setattr(auto_continue.time, 'sleep', lambda unused: None)

    def agent(stage, stage_dir, *unused):
        called.append(stage)
        if len(called) == 1:
            raise OSError('temporary runner error')
        make_artifacts(stage, stage_dir)
        return 0

    monkeypatch.setattr(auto_continue, 'run_agent', agent)
    auto_continue.run(args)
    assert called[:2] == ['candidate_combinations'] * 2
    assert len(called) == len(followup.NEXT_TASKS) + 1
