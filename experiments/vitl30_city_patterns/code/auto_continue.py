#!/usr/bin/env python3
"""Resume the remaining research tasks sequentially after city screening."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from .extract import DEFAULT_OUT, write_json
from .followup import FOLLOWUP, NEXT_TASKS

ROOT = Path(__file__).resolve().parents[3]
THESIS = Path('/root/code/HKUST-Thesis')
PROMPTS = Path(__file__).with_name('prompts')
ARTIFACTS = {
    'candidate_combinations': ('protocol.json', 'summary.json', 'candidates.csv'),
    'road_sides': ('protocol.json', 'summary.json', 'side_examples.csv'),
    'prototype_city_distribution': ('protocol.json', 'summary.json',
                                    'city_prototype_distribution.csv',
                                    'cell_prototype_distribution.parquet'),
    'chapter5_figures': ('figure_inventory.json', 'compile_report.json', 'thesis_preview.pdf'),
}


def stage_manifest(stage_dir, stage, run_identity):
    path = stage_dir / 'complete.json'
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if (result.get('schema_version') != 1 or result.get('stage') != stage
            or result.get('run_identity') != run_identity or result.get('status') != 'complete'):
        raise ValueError(f'Invalid completion identity: {path}')
    for name in ARTIFACTS[stage]:
        artifact = stage_dir / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(f'Missing stage artifact: {artifact}')
    if stage == 'candidate_combinations':
        summary = json.loads((stage_dir / 'summary.json').read_text())
        if summary.get('cities') != 30 or summary.get('candidate_count', -1) < 0:
            raise ValueError('Candidate summary lacks 30-city accounting')
    elif stage == 'road_sides':
        summary = json.loads((stage_dir / 'summary.json').read_text())
        if not {'same_side', 'opposite_side', 'undetermined'} <= summary.keys():
            raise ValueError('Roadside result lacks three-way uncertainty accounting')
        if any(not isinstance(summary[key], int) or summary[key] < 0
               for key in ('same_side', 'opposite_side', 'undetermined')):
            raise ValueError('Invalid roadside counts')
    elif stage == 'prototype_city_distribution':
        with (stage_dir / 'city_prototype_distribution.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'city', 'prototype_id'} <= set(reader.fieldnames or []):
                raise ValueError('City distribution lacks city/prototype columns')
            if len({row['city'] for row in reader}) != 30:
                raise ValueError('City distribution does not cover 30 cities')
    elif stage == 'chapter5_figures':
        report = json.loads((stage_dir / 'compile_report.json').read_text())
        if report.get('passed') is not True or (stage_dir / 'thesis_preview.pdf').stat().st_size < 100000:
            raise ValueError('Chapter 5 compile did not produce a valid PDF')
        inventory = json.loads((stage_dir / 'figure_inventory.json').read_text())
        if not isinstance(inventory.get('figures'), list) or len(inventory['figures']) < 11:
            raise ValueError('Figure inventory does not cover existing Chapter 5 figures')
    return result


def update_queue(queue_file, stage, status, detail):
    queue = json.loads(queue_file.read_text())
    found = False
    for task in queue['tasks']:
        if task['id'] == stage:
            task['status'] = status
            task['detail'] = detail
            task['execution'] = 'automatic sequential runner'
            found = True
        elif task['status'] == 'blocked' and all(
                any(previous['id'] == required and previous['status'] == 'complete'
                    for previous in queue['tasks']) for required in task.get('depends_on', [])):
            task['status'] = 'pending'
    if not found:
        raise ValueError(f'Missing queue task: {stage}')
    queue['phase'] = 'complete' if all(task['status'] == 'complete' for task in queue['tasks']) else 'research_running'
    queue['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(queue_file, queue)


def make_prompt(stage, stage_dir, run_identity, screening):
    instructions = (PROMPTS / f'{stage}.md').read_text()
    contract = f'''\n\nExecution contract:\n- Main repository: {ROOT}\n- Thesis repository: {THESIS}\n- Frozen extraction identity: {run_identity}\n- Completed city-cell screening: {screening}\n- Write all stage result artifacts beneath: {stage_dir}\n- Do not overwrite unrelated user files or silently use an earlier experiment.\n- Only after all inputs, computation, evidence, and tests are verified, write\n  {stage_dir / 'complete.json'} as JSON with schema_version=1, stage="{stage}",\n  run_identity="{run_identity}", status="complete", and a concise evidence field.\n- The required artifacts are: {', '.join(ARTIFACTS[stage])}.\n- If a requirement cannot be completed, do not write complete.json. Report\n  the exact blocker in your final message; the supervisor will retain your log.\n'''
    return instructions + contract


def run_agent(stage, stage_dir, run_identity, screening, attempt):
    stage_dir.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(stage, stage_dir, run_identity, screening)
    command = ['codex', 'exec', '-C', str(ROOT), '-s', 'danger-full-access',
               '-c', 'model_reasoning_effort="medium"',
               '--output-last-message', str(stage_dir / f'attempt_{attempt}.txt'), '-']
    with (stage_dir / f'attempt_{attempt}.log').open('w') as log:
        return subprocess.run(command, input=prompt, text=True, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode


def run(args):
    args.followup.mkdir(parents=True, exist_ok=True)
    lock_file = (args.followup / 'auto_continue.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        state_file = args.followup / 'status.json'
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        if state.get('status') == 'error':
            raise RuntimeError('City-cell screening failed; see followup/status.json')
        queue_file = args.followup / 'task_queue.json'
        queue = json.loads(queue_file.read_text()) if queue_file.is_file() else {}
        if (state.get('status') == 'complete'
                and (args.followup / 'complete.json').is_file()
                and queue.get('phase') in {'complete', 'research_running'}
                and all(any(task['id'] == required and task['status'] == 'complete'
                            for task in queue['tasks']) for required in
                        ('extract', 'validate_chunks', 'city_cell_screening'))):
            break
        write_json(args.followup / 'auto_continue_status.json',
                   {'status': 'waiting', 'detail': 'Waiting for verified city-cell screening'})
        if args.once:
            return
        time.sleep(args.interval)
    screening = json.loads((args.followup / 'complete.json').read_text())
    extraction = json.loads((args.extraction / 'run.json').read_text())
    run_identity = screening['identity']['extract_run_identity']
    if run_identity != extraction['identity'] or screening['cities'] != 30:
        raise ValueError('Screening does not belong to the complete 30-city extraction')
    if queue.get('run_identity') != run_identity:
        raise ValueError('Task queue belongs to another extraction run')
    completed = {task['id'] for task in queue['tasks'] if task['status'] == 'complete'}
    if not {'extract', 'validate_chunks', 'city_cell_screening'} <= completed:
        raise ValueError('Task queue does not confirm completed screening')
    queued = [task['id'] for task in queue['tasks'] if task['id'] in ARTIFACTS]
    if queued != [task['id'] for task in NEXT_TASKS]:
        raise ValueError('Task queue does not match the ordered research stages')
    for task in NEXT_TASKS:
        stage = task['id']
        stage_dir = args.followup / 'stages' / stage
        try:
            if stage_manifest(stage_dir, stage, run_identity):
                update_queue(queue_file, stage, 'complete', 'Verified existing completion marker')
                continue
        except ValueError:
            raise
        for attempt in range(1, args.attempts + 1):
            update_queue(queue_file, stage, 'running', f'Attempt {attempt} / {args.attempts}')
            write_json(args.followup / 'auto_continue_status.json',
                       {'status': 'running', 'stage': stage, 'attempt': attempt})
            try:
                code = run_agent(stage, stage_dir, run_identity,
                                 args.followup / 'complete.json', attempt)
                verified = stage_manifest(stage_dir, stage, run_identity)
            except Exception as exc:
                code = -1
                verified = None
                reason = f'{type(exc).__name__}: {exc}'
            else:
                reason = 'Missing verified complete.json' if verified is None else 'Verified artifacts'
            if code == 0 and verified is not None:
                update_queue(queue_file, stage, 'complete', reason)
                break
            update_queue(queue_file, stage, 'retrying' if attempt < args.attempts else 'error',
                         f'Agent exit {code}; {reason}; see attempt_{attempt}.log')
            if attempt == args.attempts:
                write_json(args.followup / 'auto_continue_status.json',
                           {'status': 'error', 'stage': stage, 'detail': reason})
                raise RuntimeError(f'{stage} did not pass completion checks: {reason}')
            time.sleep(args.retry_delay)
    write_json(args.followup / 'auto_continue_status.json',
               {'status': 'complete', 'detail': 'All queued research stages passed their artifact gates'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup', type=Path, default=FOLLOWUP)
    parser.add_argument('--extraction', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--retry-delay', type=int, default=120)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 10 or args.retry_delay < 10 or args.attempts < 1:
        parser.error('interval and retry delay must be >=10; attempts >=1')
    run(args)
