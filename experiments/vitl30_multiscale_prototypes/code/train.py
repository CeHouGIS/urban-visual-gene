"""Resumable, city-balanced prototype training and matched null controls."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from sae_experiments.io_utils import digest, save_torch
from .data import OUT, SparseViews, write_json
from .model import PrototypeAutoencoder, balanced_loss, spatial_shuffle, feature_shuffle
from .resources import configure_device

RUNS = {
    'real_seed42': ('real', 42),
    'real_seed43': ('real', 43),
    'spatial_shuffle_seed42': ('spatial_shuffle', 42),
    'feature_shuffle_seed42': ('feature_shuffle', 42),
}


def perturb(x, control, generator):
    if control == 'spatial_shuffle':
        return spatial_shuffle(x, generator)
    if control == 'feature_shuffle':
        return feature_shuffle(x, generator)
    return x


def training_order(frame, epoch, seed, batch_size):
    by_city = {city: group.index.to_numpy() for city, group in
               frame[frame.prototype_split == 'train'].groupby('city', sort=True)}
    length = math.ceil(max(map(len, by_city.values())) / batch_size) * batch_size
    rng = np.random.default_rng(seed + epoch * 1009)
    cities = list(by_city)
    rng.shuffle(cities)
    orders = {}
    for city in cities:
        chunks = []
        while sum(len(x) for x in chunks) < length:
            chunks.append(rng.permutation(by_city[city]))
        orders[city] = np.concatenate(chunks)[:length].reshape(-1, batch_size)
    return cities, orders, length // batch_size * len(cities)


@torch.inference_mode()
def monitor(model, data, batch_size, control, seed):
    """Fixed 16 panoramas/city, four headings; full heldout scoring follows training."""
    model.eval()
    generator = torch.Generator(device=data.device).manual_seed(seed)
    total, seen = 0., 0
    for city, group in data.frame[data.frame.prototype_split == 'validation'].groupby('city', sort=True):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(city.encode()).digest()[:4], 'little'))
        ids = np.sort(rng.choice(group.index.to_numpy(), min(16, len(group)), replace=False))
        for heading in range(4):
            for start in range(0, len(ids), batch_size):
                selected = ids[start:start + batch_size]
                x = perturb(data.load(selected, [heading] * len(selected)), control, generator)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(data.device).startswith('cuda')):
                    pred, maps = model(x)
                    loss, _ = balanced_loss(pred, x, maps, zero_weight=32.)
                total += float(loss) * len(selected)
                seen += len(selected)
    model.train()
    return total / max(seen, 1)


def train(out=OUT, run_name='real_seed42', device='cuda:1', epochs=8, batch_size=8):
    torch.set_num_threads(4)
    control, seed = RUNS[run_name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = configure_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = SparseViews(out, device)
    folder = Path(out) / 'models' / run_name
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'runtime.json', runtime)
    model = PrototypeAutoencoder().to(device)
    configuration = {'corpus_identity': data.spec['identity'], 'model': model.config,
        'control': control, 'seed': seed, 'epochs': epochs, 'batch_size': batch_size,
        'sampling': 'city-balanced panorama visits; heading=epoch%4, two visits/heading in eight epochs',
        'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .00001,
        'warmup_epochs': 2, 'masked_patch_fraction_after_warmup': .2,
        'zero_weight': 32., 'response_l1_weight': .01, 'composition_l1_weight': .02,
        'positive_rms_sha256': digest(Path(out) / 'positive_rms.npy'),
        'model_code_sha256': digest(Path(__file__).with_name('model.py')),
        'training_code_sha256': digest(Path(__file__))}
    configuration['identity'] = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    config_path = folder / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('Training settings or code changed; use a different run directory')
    write_json(config_path, configuration)
    if (folder / 'complete.json').exists():
        if json.loads((folder / 'complete.json').read_text())['identity'] != configuration['identity']:
            raise ValueError('Completed model identity mismatch')
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    generator = torch.Generator(device=device).manual_seed(seed + 2003)
    start_epoch, start_step, best, global_step = 0, 0, float('inf'), 0
    checkpoint_path = folder / 'last.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint['identity'] != configuration['identity']:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        generator.set_state(checkpoint['generator'].cpu())
        start_epoch, start_step = checkpoint['epoch'], checkpoint['step']
        global_step, best = checkpoint['global_step'], checkpoint['best']
    started = time.monotonic()

    def save(epoch, step):
        save_torch(checkpoint_path, {'identity': configuration['identity'], 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
            'epoch': epoch, 'step': step, 'global_step': global_step, 'best': best,
            'config': model.config})

    for epoch in range(start_epoch, epochs):
        cities, orders, steps = training_order(data.frame, epoch, seed, batch_size)
        model.train()
        for step in range(start_step if epoch == start_epoch else 0, steps):
            city = cities[step % len(cities)]
            ids = orders[city][step // len(cities)]
            x = data.load(ids, [epoch % 4] * len(ids))
            x = perturb(x, control, generator)
            hidden = (torch.rand(len(ids), 1, 28, 28, device=device, generator=generator) < .2
                      if epoch >= 2 else None)
            fraction = (epoch + step / steps) / epochs
            lr = .001 * (.1 + .9 * (1 + math.cos(math.pi * fraction)) / 2)
            lr *= min(1., (global_step + 1) / 100)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                pred, maps = model(x, hidden)
                loss, metrics = balanced_loss(pred, x, maps, hidden, zero_weight=32.)
                loss = loss + .02 * model.composition_penalty()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite prototype loss: {run_name}/{epoch}/{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            model.project_()
            global_step += 1
            if global_step % 100 == 0 or step == steps - 1:
                row = {'run': run_name, 'epoch': epoch + 1, 'step': step + 1,
                    'steps_per_epoch': steps, 'global_step': global_step, 'loss': float(loss.detach()),
                    **metrics, 'lr': lr, 'elapsed_seconds': round(time.monotonic() - started, 2)}
                if device.startswith('cuda'):
                    row['gpu_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved(device) / 2**20, 2)
                with (folder / 'training.jsonl').open('a') as log:
                    log.write(json.dumps(row) + '\n')
                write_json(folder / 'status.json', {'status': 'training', **row})
                save(epoch, step + 1)
                print('TRAIN', json.dumps(row), flush=True)
            del x, pred, maps, loss
        validation = monitor(model, data, batch_size, control, seed + 3007)
        improved = validation < best
        if improved:
            best = validation
            save_torch(folder / 'best.pt', {'identity': configuration['identity'], 'model': model.state_dict(),
                'config': model.config, 'epoch': epoch + 1, 'sampled_validation_loss': validation})
        write_json(folder / f'epoch_{epoch + 1:02d}.json', {'epoch': epoch + 1,
            'sampled_validation_loss': validation, 'best': best, 'improved': improved})
        print('VALIDATION', run_name, epoch + 1, validation, flush=True)
        save(epoch + 1, 0)
    write_json(folder / 'complete.json', {'identity': configuration['identity'], 'epochs': epochs,
        'global_steps': global_step, 'sampled_validation_loss': best,
        'best_sha256': digest(folder / 'best.pt')})
    write_json(folder / 'status.json', {'status': 'complete', 'run': run_name, 'epochs': epochs,
                                      'global_steps': global_step, 'sampled_validation_loss': best})


def load_model(out, run_name, device):
    path = Path(out) / 'models' / run_name / 'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = PrototypeAutoencoder(**checkpoint['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    return model, digest(path)
