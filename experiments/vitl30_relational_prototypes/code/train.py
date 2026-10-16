"""GPU-only matched training of composition, context, and relation encoders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.resources import configure_device
from sae_experiments.io_utils import digest, save_torch
from .hierarchy import EXPERIMENT, read, sha
from .model import LandscapeDictionary, loss_terms
from .regions import OUT

ARMS = ('composition', 'context', 'relation')


def load_arrays(out=OUT):
    out = Path(out)
    pilot = read(out / 'protocol.json')
    region = read(out / 'regions/complete.json')
    region_identity = read(out / 'regions/identity.json')
    if region['identity'] != region_identity['identity'] or region_identity['pilot_identity'] != pilot['identity']:
        raise ValueError('Region or pilot identity changed')
    frame = pd.read_parquet(out / 'pilot_panoramas.parquet')
    if frame.prototype_split.value_counts().to_dict() != pilot['split_counts']:
        raise ValueError('Pilot partition changed')
    raw = np.asarray(np.load(out / 'regions/pooled.npy', mmap_mode='r'), dtype=np.float32)
    if raw.shape != tuple(region['shape']):
        raise ValueError('Region shape differs')
    train_pano = frame.prototype_split.to_numpy() == 'train'
    masses = raw.mean(-1)
    scale = np.quantile(masses[train_pano].reshape(-1, raw.shape[-2]), .9, axis=0).astype(np.float32)
    scale = np.maximum(scale, .005)
    x = np.log1p(raw / scale[None, None, None, :, None]).clip(0, 4).astype(np.float32)
    # The 28x28 region supplies a common four-heading context. Rotate the
    # order relative to the current heading; no pixels are joined at seams.
    heading = np.log1p(masses[:, :, -1] / scale[None, None, :]).clip(0, 4).astype(np.float32)
    context = np.stack([np.roll(heading, -h, axis=1) for h in range(4)], 1)
    context = np.broadcast_to(context[:, :, None], (len(frame), 4, raw.shape[2], 4, raw.shape[3]))
    strength = x.mean(-1)
    largest = strength.max(-1, keepdims=True)
    strong = (strength >= .5 * largest) & (strength >= .5)
    eligible = strong.sum(-1) >= 2
    split = np.broadcast_to(train_pano[:, None, None], eligible.shape)
    train_id = np.flatnonzero((eligible & split).ravel())
    dev_id = np.flatnonzero((eligible & ~split).ravel())
    if not len(train_id) or not len(dev_id):
        raise ValueError('No multi-element regions in train or dev')
    meta = {'region_identity': region['identity'], 'families': region_identity['families'],
            'family_scale': scale.tolist(), 'eligibility': 'at least two parent families with mean normalized response >= max(0.5, 0.5 * strongest family)',
            'train_regions': len(train_id), 'dev_regions': len(dev_id),
            'all_train_regions': int(split.sum()), 'all_dev_regions': int((~split).sum()),
            'sealed_test_used': False, 'prior_semantic_evidence_includes_test': True}
    return x.reshape(-1, raw.shape[-2], 16), context.reshape(-1, 4, raw.shape[-2]), train_id, dev_id, meta


def initial_dictionary(x, ids, slots, seed):
    rng = np.random.default_rng(seed)
    selected = rng.choice(ids, slots, replace=False)
    values = np.clip(x[selected], .02, 3.)
    return np.log(np.expm1(values)).astype(np.float32)


@torch.inference_mode()
def evaluate(model, x, context, ids, device, batch_size):
    model.eval()
    total, count = 0., 0
    usage = np.zeros(model.slots, np.int64)
    for begin in range(0, len(ids), batch_size):
        chosen = ids[begin:begin + batch_size]
        target = torch.from_numpy(x[chosen]).to(device)
        pano = torch.from_numpy(context[chosen].copy()).to(device)
        prediction, _, winner = model(target, pano)
        error = (prediction - target).square().mean(dim=(1, 2))
        total += error.sum().item()
        count += len(chosen)
        usage += winner.sum(0).cpu().numpy().astype(np.int64)
    return {'mean_mse': total / count, 'regions': count,
            'used_slots': int((usage > 0).sum()), 'slot_counts': usage.tolist()}


def train_arm(arm, device='cuda:1', epochs=10, batch_size=256, out=OUT, seed=42):
    if arm not in ARMS:
        raise ValueError(arm)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Real prototype training requires CUDA; CPU fallback is disabled')
    torch.set_num_threads(4)
    runtime = configure_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    x, context, train_ids, dev_ids, data_meta = load_arrays(out)
    torch.manual_seed(seed)
    model = LandscapeDictionary(families=x.shape[1], slots=32, mode=arm).to(device)
    config = {'version': 1, 'pilot_identity': read(Path(out) / 'protocol.json')['identity'],
              'data': data_meta, 'mode': arm, 'seed': seed, 'epochs': epochs,
              'batch_size': batch_size, 'slots': 32, 'optimizer': 'AdamW', 'lr': .001,
              'model_code_sha256': sha(Path(__file__).with_name('model.py')),
              'train_code_sha256': sha(Path(__file__)),
              'training_scope': 'exploratory; multi-family regions only; development split excludes training 1-km blocks; frozen SAE and prior VLM saw some images'}
    config['identity'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    folder = Path(out) / 'models' / arm
    folder.mkdir(parents=True, exist_ok=True)
    existing = read(folder / 'config.json') if (folder / 'config.json').exists() else None
    if existing and existing != config:
        raise ValueError(f'{arm}: model/data/code identity changed; use another output directory')
    write_json(folder / 'config.json', config)
    if (folder / 'complete.json').exists():
        return read(folder / 'complete.json')
    with torch.no_grad():
        model.dictionary.copy_(torch.from_numpy(initial_dictionary(x, train_ids, 32, seed)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    first_epoch, best = 0, math.inf
    checkpoint = folder / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['identity'] != config['identity']:
            raise ValueError('Checkpoint belongs to another run')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch, best = saved['epoch'], saved['best']
    write_json(folder / 'runtime.json', runtime)
    started = time.monotonic()
    for epoch in range(first_epoch, epochs):
        rng = np.random.default_rng(seed + epoch * 1009)
        order = rng.permutation(train_ids)
        model.train()
        total = 0.
        for begin in range(0, len(order), batch_size):
            chosen = order[begin:begin + batch_size]
            target = torch.from_numpy(x[chosen]).to(device)
            pano = torch.from_numpy(context[chosen].copy()).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, probabilities, _ = model(target, pano)
            objective, terms = loss_terms(prediction, target, probabilities)
            if not torch.isfinite(objective):
                raise FloatingPointError(f'Non-finite {arm} loss')
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += objective.detach().item() * len(chosen)
        validation = evaluate(model, x, context, dev_ids, device, batch_size)
        baseline = total / len(order)
        if validation['mean_mse'] < best:
            best = validation['mean_mse']
            save_torch(folder / 'best.pt', {'identity': config['identity'], 'model': model.state_dict(),
                                          'config': config, 'epoch': epoch + 1, 'development': validation})
        save_torch(checkpoint, {'identity': config['identity'], 'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'best': best})
        status = {'status': 'training', 'arm': arm, 'epoch': epoch + 1, 'epochs': epochs,
                  'train_objective': baseline, 'development_mse': validation['mean_mse'],
                  'best_development_mse': best, 'used_slots': validation['used_slots'],
                  'elapsed_seconds': round(time.monotonic() - started, 2),
                  'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
                  'updated_utc': datetime.now(timezone.utc).isoformat()}
        write_json(folder / 'status.json', status)
        print('EPOCH', json.dumps(status), flush=True)
    result = {'identity': config['identity'], 'arm': arm, 'epochs': epochs,
              'best_development_mse': best, 'best_sha256': digest(folder / 'best.pt'),
              'test_used': False, 'semantic_review': 'pending',
              'interpretation': 'reconstruction of family-layout summaries, not verified landscape coverage'}
    write_json(folder / 'complete.json', result)
    write_json(folder / 'status.json', {'status': 'complete', **result})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(train_arm(args.arm, args.device, args.epochs, args.batch_size), ensure_ascii=False, indent=2))
