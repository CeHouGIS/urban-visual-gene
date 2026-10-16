"""Persistent, resumable pilot queue. Never fall back to CPU inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

from .protocol import OUT
from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.run import free_gpu_mib


def execute_worker(stage, arm, device):
    from .protocol import prepare
    from .trial import diagnostic, train, evaluate, report
    if stage == 'prepare':
        prepare()
    elif stage == 'diagnostic':
        diagnostic(device=device)
    elif stage == 'train':
        train(arm, device=device)
    elif stage == 'evaluate':
        evaluate(arm, device=device)
    elif stage == 'report':
        report()


def run(device='cuda:2'):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'device': device, 'started_utc': datetime.now(timezone.utc).isoformat(),
             'semantic_audit': 'pending', 'test_sealed': True}
    child = None

    def update(**changes):
        state.update(changes)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUT / 'pipeline_status.json', state)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        update(status='stopped')
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stages = [('prepare', '', OUT / 'calibration.json'),
              ('diagnostic', '', OUT / 'diagnostic/complete.json')]
    for arm in ('balanced_reference', 'energy_warmup'):
        stages += [('train', arm, OUT / 'models' / arm / 'complete.json'),
                   ('evaluate', arm, OUT / 'development' / arm / 'complete.json')]
    stages += [('report', '', OUT.parent / 'results/pilot_summary.json')]
    try:
        for stage, arm, marker in stages:
            if marker.exists():
                continue
            for attempt in range(1, 11):
                if stage not in ('prepare', 'report'):
                    minimum = int(os.environ.get('UVG_PROTOTYPE_MIN_FREE_MIB', '1600'))
                    while free_gpu_mib(device) < minimum:
                        update(status='waiting_for_gpu', stage=stage, arm=arm,
                               free_mib=free_gpu_mib(device), required_free_mib=minimum)
                        time.sleep(30)
                update(status='running', stage=stage, arm=arm, attempt=attempt)
                log_path = OUT / f'{stage}{"_" + arm if arm else ""}.log'
                with log_path.open('a') as log:
                    child = subprocess.Popen([sys.executable, '-u', '-m',
                        'experiments.vitl30_prototype_coverage.code.run', '--worker', stage,
                        '--arm', arm, '--device', device], stdout=log, stderr=subprocess.STDOUT)
                    update(child_pid=child.pid)
                    code = child.wait()
                child = None
                if code == 0:
                    if not marker.exists():
                        raise RuntimeError(f'{stage}/{arm} returned without completion marker')
                    break
                recent = log_path.read_text()[-12000:]
                if 'out of memory' in recent.lower() and attempt < 10:
                    update(status='retrying_gpu_memory', stage=stage, arm=arm, attempt=attempt)
                    time.sleep(30)
                    continue
                raise RuntimeError(f'{stage}/{arm} failed with code {code}; see {log_path}')
        update(status='complete', stage='semantic_audit_pending', arm='')
    except Exception as error:
        update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['prepare', 'diagnostic', 'train', 'evaluate', 'report'])
    parser.add_argument('--arm', default='')
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    if args.worker:
        execute_worker(args.worker, args.arm, args.device)
    else:
        run(args.device)
