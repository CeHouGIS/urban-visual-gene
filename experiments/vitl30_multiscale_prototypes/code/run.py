"""Run the full cache -> prototypes -> controls -> evidence pipeline."""
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

from .data import OUT, prepare, extract, finish_cache, write_json
from sae_experiments.io_utils import digest


def completed_cache(out, corpus):
    """Verify existing outputs before skipping frozen-SAE loading entirely."""
    path = out / 'cache_complete.json'
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    if saved['identity'] != corpus['identity']:
        raise ValueError('Completed cache identity mismatch')
    if digest(out / 'positive_rms.npy') != saved['scale_sha256']:
        raise ValueError('Completed cache normalization changed')
    import numpy as np
    composition = np.load(out / 'composition.npy', mmap_mode='r')
    if composition.shape != (corpus['panoramas'], 4096):
        raise ValueError('Completed cache composition shape mismatch')
    count = chunks = nnz = 0
    for city in corpus['cities']:
        for start in range(0, city['panoramas'], corpus['chunk_panos']):
            folder = out / 'activations' / city['city'].replace('/', '__') / f'{start:06d}'
            marker = json.loads((folder / 'complete.json').read_text())
            if (marker['identity'] != corpus['identity'] or marker['start'] != start
                    or marker['stop'] != min(start + corpus['chunk_panos'], city['panoramas'])):
                raise ValueError(f'Completed activation chunk identity/extent changed: {folder}')
            for name, info in marker['files'].items():
                if (folder / f'{name}.npy').stat().st_size != info['bytes']:
                    raise ValueError(f'Completed activation chunk truncated: {folder}/{name}')
            count += marker['stop'] - marker['start']
            nnz += marker['nnz']
            chunks += 1
    if (count, chunks, nnz) != (saved['panoramas'], saved['chunks'], saved['nnz']):
        raise ValueError('Completed activation cache totals changed')
    return True


def free_gpu_mib(device):
    index = int(device.split(':')[-1])
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    return next(int(row.split(',')[1]) for row in rows.splitlines()
                if int(row.split(',')[0]) == index)


def status(out):
    result = {}
    for name in ['corpus.json', 'pipeline_status.json', 'extraction_shard0.json',
                 'extraction_shard1.json', 'cache_complete.json']:
        p = out / name
        if p.exists():
            value = json.loads(p.read_text())
            if name == 'corpus.json':
                value = {k: value[k] for k in ['identity', 'panoramas', 'directional_images',
                    'train_panoramas', 'validation_panoramas', 'excluded_unmapped_cities']}
            result[name] = value
    for p in sorted((out / 'models').glob('*/status.json')):
        result[str(p.relative_to(out))] = json.loads(p.read_text())
    if (out / 'corpus.json').exists():
        corpus = json.loads((out / 'corpus.json').read_text())
        completed, nnz = 0, 0
        for city in corpus['cities']:
            for start in range(0, city['panoramas'], corpus['chunk_panos']):
                marker = out / 'activations' / city['city'].replace('/', '__') / f'{start:06d}' / 'complete.json'
                if marker.exists():
                    saved = json.loads(marker.read_text())
                    if saved['identity'] != corpus['identity']:
                        raise ValueError('Mismatched activation chunk identity')
                    completed += saved['stop'] - saved['start']
                    nnz += saved['nnz']
        result['activation_progress'] = {'completed_panoramas': completed, 'total_panoramas': corpus['panoramas'],
                                          'percent': round(100 * completed / corpus['panoramas'], 2), 'nnz': nnz}
    for p in sorted((out / 'evaluation').glob('*/status.json')):
        result[str(p.relative_to(out))] = json.loads(p.read_text())
    if (out / 'interpretation/status.json').exists():
        result['interpretation/status.json'] = json.loads((out / 'interpretation/status.json').read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))


def pipeline(args):
    from .train import RUNS
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.out / 'background.pid').write_text(str(os.getpid()) + '\n')
    running = []

    def stop(signum, frame):
        for child in running:
            if child.poll() is None:
                child.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def state(stage, **kwargs):
        write_json(args.out / 'pipeline_status.json', {'stage': stage, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(), **kwargs})
        print('PIPELINE', stage, json.dumps(kwargs), flush=True)

    def jobs(stage, commands):
        state(stage, status='running')
        pending = list(commands)
        active = {}
        log_starts = {}
        retries = {}
        not_before = {}
        minimum_free = int(os.environ.get('UVG_PROTOTYPE_MIN_FREE_MIB', '1600'))
        while pending or active:
            for lane, job in enumerate(list(pending)):
                name, device, options = job
                if any(item[2] == device for item in active.values()):
                    continue
                if time.monotonic() < not_before.get(name, 0):
                    continue
                if device.startswith('cuda') and free_gpu_mib(device) < minimum_free:
                    continue
                log = (args.out / f'{name}.log').open('a', buffering=1)
                log_starts[name] = log.tell()
                child = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name,
                    '--out', str(args.out), '--stage', stage, '--device', device,
                    '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), *options],
                    stdout=log, stderr=subprocess.STDOUT)
                running.append(child)
                active[name] = (child, log, device, job)
                pending.remove(job)
                print('STARTED', name, child.pid, device, flush=True)
            for name, (child, log, device, job) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                log.close()
                del active[name]
                if code:
                    with (args.out / f'{name}.log').open() as failed_log:
                        failed_log.seek(log_starts[name])
                        tail = failed_log.read()[-6000:]
                    if 'CUDA out of memory' in tail or 'torch.OutOfMemoryError' in tail:
                        retries[name] = retries.get(name, 0) + 1
                        not_before[name] = time.monotonic() + min(300 * retries[name], 1800)
                        pending.append(job)
                        print('RETRY_AFTER_OOM', name, retries[name], flush=True)
                        continue
                    raise RuntimeError(f'{name} exited {code}; inspect {args.out / (name + ".log")}')
                print('FINISHED', name, flush=True)
            state(stage, status='running' if active else 'waiting_for_gpu',
                  jobs={name: {'pid': child.pid, 'device': device}
                        for name, (child, _, device, _) in active.items()},
                  pending=[{'run': name, 'device': device, 'oom_retries': retries.get(name, 0)}
                           for name, device, _ in pending], minimum_free_mib=minimum_free)
            if active or pending:
                time.sleep(15)

    try:
        state('prepare', status='running')
        corpus = prepare(args.out)
        print('CORPUS', corpus['panoramas'], corpus['train_panoramas'], corpus['validation_panoramas'], flush=True)
        if completed_cache(args.out, corpus):
            state('cache_reused', status='complete', panoramas=corpus['panoramas'])
        else:
            jobs('extract', [(f'extract_{i}', device, ['--shard', str(i), '--shards', str(len(args.devices))])
                             for i, device in enumerate(args.devices)])
            state('normalization', status='running')
            finish_cache(args.out)
        jobs('train', [(name, args.devices[i % len(args.devices)], ['--run-name', name])
                       for i, name in enumerate(RUNS)])
        state('evaluate', status='running')
        from .evaluate import evaluate
        evaluate(args.out, args.devices[0], args.batch_size, args.null_repeats)
        import torch
        torch.cuda.empty_cache()
        # Qwen's weights need more memory than prototype training. Keep the
        # completed training/evaluation intact while waiting for that capacity.
        vlm_device = args.devices[-1]
        while free_gpu_mib(vlm_device) < 7000:
            state('interpret', status='waiting_for_gpu', device=vlm_device, minimum_free_mib=7000)
            time.sleep(15)
        torch.cuda.set_per_process_memory_fraction(1., vlm_device)
        state('interpret', status='running', device=vlm_device)
        from .interpret import interpret
        interpret(args.out, args.devices[-1])
        state('complete', status='complete')
    except BaseException as exc:
        for child in running:
            if child.poll() is None:
                child.terminate()
        state('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed',
              status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'prepare', 'extract', 'finish', 'train', 'evaluate',
                                          'interpret', 'status'], default='status')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--devices', default='0,2')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--run-name', default='real_seed42')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--null-repeats', type=int, default=39)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.devices = [f'cuda:{int(d)}' for d in args.devices.split(',')]
    if args.stage == 'all':
        pipeline(args)
    elif args.stage == 'prepare':
        print(json.dumps(prepare(args.out), ensure_ascii=False, indent=2))
    elif args.stage == 'extract':
        extract(args.out, args.device, args.shard, args.shards)
    elif args.stage == 'finish':
        finish_cache(args.out)
    elif args.stage == 'train':
        from .train import train
        train(args.out, args.run_name, args.device, args.epochs, args.batch_size)
    elif args.stage == 'evaluate':
        from .evaluate import evaluate
        evaluate(args.out, args.device, args.batch_size, args.null_repeats)
    elif args.stage == 'interpret':
        from .interpret import interpret
        interpret(args.out, args.device)
    else:
        status(args.out)


if __name__ == '__main__':
    main()
