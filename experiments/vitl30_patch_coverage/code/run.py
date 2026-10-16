"""Persistent GPU-only queue for per-image patch coverage, with sealed test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.run import free_gpu_mib
from .experiment import OUT


def execute_worker(stage, device):
    from .experiment import prepare, train, evaluate, report
    if stage == 'prepare':
        prepare()
    elif stage == 'train':
        train(device=device)
    elif stage.startswith('evaluate_'):
        evaluate(device=device, split=stage.removeprefix('evaluate_'))
    elif stage == 'report':
        report()
    else:
        raise ValueError(stage)


def run(device='cuda:3'):
    if not device.startswith('cuda:'):
        raise ValueError('Real experiment requires CUDA; no CPU fallback')
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'device': device,
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'semantic_review': 'pending', 'test_sealed': True}
    child = None

    def update(**changes):
        state.update(changes)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUT / 'pipeline_status.json', state)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        update(status='stopped')
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stages = [('prepare', OUT / 'protocol.json'), ('train', OUT / 'model/complete.json'),
              ('evaluate_dev', OUT / 'evaluation/dev/complete.json'),
              ('evaluate_train', OUT / 'evaluation/train/complete.json'),
              ('report', OUT.parent / 'results/summary.json')]
    try:
        # Always validate the pinned protocol, even when resuming a later stage.
        from .experiment import prepare
        prepare()
        for stage, marker in stages:
            if marker.exists():
                continue
            for attempt in range(1, 11):
                if stage not in ('prepare', 'report'):
                    minimum = int(os.environ.get('UVG_PROTOTYPE_MIN_FREE_MIB', '1600'))
                    while True:
                        free = free_gpu_mib(device)
                        if free >= minimum:
                            break
                        update(status='waiting_for_gpu', stage=stage, attempt=attempt,
                               free_mib=free, required_free_mib=minimum)
                        time.sleep(30)
                update(status='running', stage=stage, attempt=attempt)
                log_path = OUT / f'{stage}.log'
                with log_path.open('ab') as log:
                    start_offset = log.tell()
                    child = subprocess.Popen([sys.executable, '-u', '-m',
                        'experiments.vitl30_patch_coverage.code.run', '--worker', stage,
                        '--device', device], stdout=log, stderr=subprocess.STDOUT)
                    update(child_pid=child.pid)
                    code = child.wait()
                child = None
                if code == 0:
                    if not marker.exists():
                        raise RuntimeError(f'{stage} returned without completion marker')
                    break
                # Read this attempt only, so an older OOM cannot hide a new error.
                with log_path.open('rb') as log:
                    log.seek(max(start_offset, log_path.stat().st_size - 12000))
                    recent = log.read().decode(errors='replace').lower()
                if 'out of memory' in recent and attempt < 10:
                    update(status='retrying_gpu_memory', stage=stage, attempt=attempt)
                    time.sleep(30)
                    continue
                raise RuntimeError(f'{stage} failed with code {code}; see {log_path}')
        update(status='complete', stage='semantic_review_pending', child_pid=None)
    except Exception as error:
        update(status='failed', error=str(error), child_pid=None)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['prepare', 'train', 'evaluate_dev', 'evaluate_train', 'report'])
    parser.add_argument('--device', default='cuda:3')
    args = parser.parse_args()
    if args.worker:
        execute_worker(args.worker, args.device)
    else:
        run(args.device)
