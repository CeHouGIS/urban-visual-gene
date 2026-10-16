"""Resume dimension interpretations when the requested GPUs have free memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sae_experiments.panorama_vitl30_dimension_vlm import (
    MODEL, REPORT, SOURCE, atomic_json, prepare, report,
)


def free_memory():
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits',
    ], check=True, capture_output=True, text=True, timeout=20)
    return {int(parts[0]): int(parts[1]) for line in result.stdout.splitlines()
            if (parts := line.split(','))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--devices', default='1,3')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--input-mode', choices=['masked-top10', 'clean-top100'], default='masked-top10')
    parser.add_argument('--mask-fraction', type=float, default=.5)
    parser.add_argument('--min-free-mib', type=int, default=10000)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    lock = (args.report / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.report / 'background.pid').write_text(str(os.getpid()) + '\n')
    prepare(args)
    devices = [int(device) for device in args.devices.split(',')]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Expected distinct GPU indices')
    shards = [dict(shard=i, device=device, status='waiting_gpu', attempts=0)
              for i, device in enumerate(devices)]
    running = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process, _, _ in running.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            for shard, (process, log, path) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[shard]
                state = shards[shard]
                state['last_exit_code'] = code
                if code == 0:
                    state['status'] = 'finished'
                else:
                    tail = path.read_text(errors='replace')[-12000:]
                    if 'OutOfMemoryError' in tail or 'CUDA out of memory' in tail:
                        state['status'] = 'waiting_gpu'
                        state['retry_after'] = time.time() + 60
                        print('GPU_MEMORY_RETRY', shard, flush=True)
                    else:
                        state['status'] = 'failed'
                        print('WORKER_FAILED', shard, str(path), flush=True)
            memory = free_memory()
            for state in shards:
                device, shard = state['device'], state['shard']
                state['free_mib'] = memory.get(device, 0)
                if (state['status'] != 'waiting_gpu' or state['free_mib'] < args.min_free_mib
                        or time.time() < state.get('retry_after', 0)):
                    continue
                state['attempts'] += 1
                path = args.report / f'worker_{shard}_attempt{state["attempts"]}.log'
                log = path.open('a', buffering=1)
                command = [sys.executable, '-u', '-m', 'sae_experiments.panorama_vitl30_dimension_vlm',
                    '--stage', 'worker', '--device', f'cuda:{device}', '--shard', str(shard),
                    '--shards', str(len(devices)), '--batch-size', str(args.batch_size),
                    '--max-new-tokens', str(args.max_new_tokens), '--source', str(args.source),
                    '--input-mode', args.input_mode, '--mask-fraction', str(args.mask_fraction),
                    '--report', str(args.report), '--model', str(args.model)]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                running[shard] = (process, log, path)
                state.update(status='running', pid=process.pid, log=str(path))
                print('WORKER_STARTED', shard, process.pid, 'GPU', device, flush=True)
            report(args)
            final = all(s['status'] in {'finished', 'failed'} for s in shards)
            summary = json.loads((args.report / 'summary.json').read_text())
            status = ('failed' if any(s['status'] == 'failed' for s in shards) else
                      ('complete' if summary['completed_or_explicitly_unobserved']
                       == summary.get('requested_dimensions', 4096)
                       else 'incomplete')) if final else (
                           'running' if running else 'waiting_gpu')
            atomic_json(args.report / 'runner_status.json', {
                'status': status, 'pid': os.getpid(), 'shards': shards,
                'updated_utc': datetime.now(timezone.utc).isoformat(),
            })
            if final:
                if status != 'complete':
                    raise SystemExit('Interpretations incomplete; inspect failed/invalid records')
                break
            time.sleep(30)
    finally:
        for process, log, _ in running.values():
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=30)
            log.close()
        if stopping:
            atomic_json(args.report / 'runner_status.json', {
                'status': 'stopped', 'pid': os.getpid(), 'shards': shards,
                'updated_utc': datetime.now(timezone.utc).isoformat(),
            })


if __name__ == '__main__':
    main()
