"""GPU-only response diagnostic and matched, bounded prototype pilot."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.vitl30_multiscale_prototypes.code.data import SparseViews, write_json
from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder, balanced_loss
from experiments.vitl30_multiscale_prototypes.code.resources import configure_device
from experiments.vitl30_multiscale_prototypes.code.train import load_model, training_order
from sae_experiments.io_utils import digest, save_torch
from .protocol import OUT, SOURCE, panel

ARMS = ('balanced_reference', 'energy_warmup')


def runtime(device):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Real experiment requires CUDA; CPU fallback is disabled')
    torch.set_num_threads(4)
    configure_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True


def dataset(out, device, old_scale=False):
    protocol = json.loads((Path(out) / 'protocol.json').read_text())
    data = SparseViews(Path(protocol['source']), device)
    if data.spec['identity'] != protocol['source_corpus_identity']:
        raise ValueError('Source corpus identity differs')
    data.frame = pd.read_parquet(Path(out) / 'panoramas.parquet')
    if not old_scale:
        scale = Path(out) / 'positive_rms.npy'
        calibration = json.loads((Path(out) / 'calibration.json').read_text())
        if calibration['identity'] != protocol['identity'] or digest(scale) != calibration['scale_sha256']:
            raise ValueError('Training-only calibration differs')
        data.scale = torch.from_numpy(np.load(scale)).to(device)
    return data


def decode(model, coefficients):
    return sum(bank(c) for bank, c in zip(model.banks, coefficients.split(model.per_scale, 1)))


def energy_loss(prediction, target):
    error = (prediction.float() - target.float()).square().flatten(1).sum(1)
    energy = target.float().square().flatten(1).sum(1)
    return error / energy.clamp_min(1e-12)


def refit_responses(model, target, steps=150, lr=.15):
    """Exploratory unrestricted NNLS approximation; retain the warm start if better.

    This is an optimization diagnostic, not a certified bound and not a sparse,
    semantically validated explanation. The dictionary remains frozen.
    """
    model.eval().requires_grad_(False)
    with torch.no_grad():
        prediction, initial = model(target)
        initial_nmse = energy_loss(prediction, target)
    coefficients = torch.nn.Parameter(initial.detach().float().clone())
    optimizer = torch.optim.Adam([coefficients], lr=lr)
    best = initial_nmse.clone()
    best_maps = coefficients.detach().clone()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        objective = energy_loss(decode(model, coefficients), target).mean()
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite response-refit objective')
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(min=0)
        if (step + 1) % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                error = energy_loss(decode(model, coefficients), target)
                improved = error < best
                best_maps[improved] = coefficients[improved]
                best = torch.minimum(best, error)
    return initial_nmse.detach(), best.detach(), initial.detach(), best_maps.detach()


def diagnostic(out=OUT, device='cuda:2', steps=150):
    runtime(device)
    out = Path(out)
    folder = out / 'diagnostic'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    data = dataset(out, device, old_scale=True)
    model, checksum = load_model(Path(json.loads((out / 'protocol.json').read_text())['source']), 'real_seed42', device)
    # The protocol source is the existing prototype output directory, not the SAE output.
    ids = panel(data.frame, per_city=1)
    path = folder / 'views.json'
    records = json.loads(path.read_text()) if path.exists() else []
    done = {(r['row_id'], r['heading']) for r in records}
    started = time.monotonic()
    for row_id in ids:
        for heading in range(4):
            if (int(row_id), heading) in done:
                continue
            x = data.load([row_id], [heading])
            initial, best, old_maps, maps = refit_responses(model, x, steps=steps)
            row = data.frame.iloc[row_id]
            records.append({'row_id': int(row_id), 'city': row.city, 'panoid': row.panoid,
                'heading': heading, 'before_nmse': float(initial[0]), 'refit_nmse': float(best[0]),
                'before_active_templates': int((old_maps.amax((2, 3))[0] > 1e-6).sum()),
                'refit_active_templates': int((maps.amax((2, 3))[0] > 1e-6).sum())})
            write_json(path, records)
            state = {'views': len(records), 'total': len(ids) * 4,
                'before_mean_nmse': float(np.mean([r['before_nmse'] for r in records])),
                'refit_mean_nmse': float(np.mean([r['refit_nmse'] for r in records])),
                'elapsed_seconds': round(time.monotonic() - started, 2)}
            write_json(folder / 'status.json', state)
            print('REFIT', json.dumps(state), flush=True)
            del x, initial, best, old_maps, maps
    write_json(folder / 'complete.json', {**state, 'model_sha256': checksum, 'steps': steps,
        'old_normalization': True, 'exploratory': True,
        'note': 'Existing dictionary has seen some new dev blocks; no sparsity penalty; finite optimization does not certify an upper bound; not semantic coverage.'})


@torch.no_grad()
def initialize_from_train(model, data, seed=42):
    """Same sparse, data-based template initialization for both loss arms.

    Four local patch vectors provide channel factors; Gaussian layouts place
    them at four offsets. No labels, dev or test examples are used.
    """
    rng = np.random.default_rng(seed)
    train = data.frame[data.frame.prototype_split == 'train']
    cities = sorted(train.city.unique())
    coordinates = []
    for bank_index, bank in enumerate(model.banks):
        yy, xx = torch.meshgrid(torch.arange(bank.size, device=data.device),
                                torch.arange(bank.size, device=data.device), indexing='ij')
        offset = bank.size // 4
        offsets = [(-offset, -offset), (-offset, offset), (offset, -offset), (offset, offset)]
        for m in range(bank.count):
            city = cities[(m + bank_index * bank.count) % len(cities)]
            row_id = int(rng.choice(train[train.city == city].index.to_numpy()))
            heading = (m + bank_index) % 4
            x = data.load([row_id], [heading])[0]
            power = x.square().sum(0)
            margin = bank.size // 2
            power[:margin] = -1; power[-margin:] = -1
            power[:, :margin] = -1; power[:, -margin:] = -1
            candidates = power.flatten().topk(min(32, (28 - 2 * margin) ** 2)).indices
            center = int(candidates[int(rng.integers(len(candidates)))])
            cy, cx = divmod(center, 28)
            for r, (dy, dx) in enumerate(offsets):
                bank.channels[m, r].copy_(x[:, cy + dy, cx + dx])
                gaussian = torch.exp(-((yy - (margin + dy)) ** 2 + (xx - (margin + dx)) ** 2)
                                     / (2 * max(.65, bank.size / 8) ** 2))
                bank.layout[m, r].copy_(gaussian)
            coordinates.append({'scale': bank.size, 'template': m, 'row_id': row_id,
                                'heading': heading, 'patch_y': cy, 'patch_x': cx})
    model.project_()
    return coordinates


def objective(arm, prediction, x, maps, model, progress, hidden=None):
    if arm == 'balanced_reference':
        value, _ = balanced_loss(prediction, x, maps, hidden, zero_weight=32., sparsity=.01)
        return value + .02 * model.composition_penalty()
    if arm != 'energy_warmup':
        raise ValueError(arm)
    value = energy_loss(prediction, x).mean()
    # Warm up response and composition sparsity over the first full epoch.
    warmup = min(1., progress)
    if hidden is not None:
        hidden_error = ((prediction.float() - x.float()).square() * hidden).flatten(1).sum(1)
        hidden_energy = (x.float().square() * hidden).flatten(1).sum(1)
        value = value + (hidden_error / hidden_energy.clamp_min(1e-12)).mean()
    return value + warmup * (.01 * maps.float().mean() + .02 * model.composition_penalty())


@torch.inference_mode()
def monitor(model, data, ids, batch_size=4):
    model.eval()
    scores = []
    for heading in range(4):
        for at in range(0, len(ids), batch_size):
            selected = ids[at:at + batch_size]
            x = data.load(selected, [heading] * len(selected))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, _ = model(x)
            scores.extend(energy_loss(prediction, x).cpu().tolist())
    model.train()
    return float(np.mean(scores))


def train(arm, out=OUT, device='cuda:2', epochs=4, batch_size=8):
    runtime(device)
    out = Path(out)
    folder = out / 'models' / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    data = dataset(out, device)
    torch.manual_seed(42)
    model = PrototypeAutoencoder().to(device)
    initialization = initialize_from_train(model, data)
    write_json(folder / 'initialization.json', initialization)
    protocol = json.loads((out / 'protocol.json').read_text())
    config = {'protocol_identity': protocol['identity'], 'model': model.config, 'arm': arm,
        'seed': 42, 'epochs': epochs, 'batch_size': batch_size, 'lr': .001,
        'initialization': 'training-only four local patch vectors / template; matched across arms',
        'sampling': 'city-balanced visits; heading=epoch%4; four epochs = one visit per direction',
        'selection': 'lowest mean directional energy NMSE on fixed 4 dev panoramas / city',
        'masked_training': '20% hidden patches after two epochs; explicit mask channel',
        'scale_sha256': digest(out / 'positive_rms.npy'),
        'code_sha256': digest(Path(__file__)),
        'model_code_sha256': digest(Path(__file__).parents[2] / 'vitl30_multiscale_prototypes/code/model.py')}
    if (folder / 'config.json').exists() and json.loads((folder / 'config.json').read_text()) != config:
        raise ValueError('Training config changed; use another output folder')
    write_json(folder / 'config.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(2045)
    checkpoint_path = folder / 'last.pt'
    start_epoch = start_step = global_step = 0
    best = float('inf')
    elapsed_before = 0.
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint config differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        generator.set_state(checkpoint['generator'].cpu())
        start_epoch, start_step, global_step = checkpoint['epoch'], checkpoint['step'], checkpoint['global_step']
        best, elapsed_before = checkpoint['best'], checkpoint['elapsed_seconds']
    dev_ids = panel(data.frame, per_city=4)
    started = time.monotonic()

    def save(epoch, step):
        save_torch(checkpoint_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'generator': generator.get_state(), 'config': config, 'epoch': epoch, 'step': step,
            'global_step': global_step, 'best': best, 'elapsed_seconds': elapsed_before + time.monotonic() - started})

    for epoch in range(start_epoch, epochs):
        cities, orders, steps = training_order(data.frame, epoch, 42, batch_size)
        model.train()
        for step in range(start_step if epoch == start_epoch else 0, steps):
            city = cities[step % len(cities)]
            ids = orders[city][step // len(cities)]
            x = data.load(ids, [epoch % 4] * len(ids))
            hidden = torch.rand(len(ids), 1, 28, 28, device=device, generator=generator) < .2 if epoch >= 2 else None
            fraction = (epoch + step / steps) / epochs
            lr = .001 * (.1 + .9 * (1 + math.cos(math.pi * fraction)) / 2) * min(1., (global_step + 1) / 100)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, maps = model(x, hidden)
                loss = objective(arm, prediction, x, maps, model, epoch + step / steps, hidden)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {arm} loss at {epoch}/{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            model.project_()
            global_step += 1
            if global_step % 100 == 0 or step == steps - 1:
                state = {'status': 'training', 'arm': arm, 'epoch': epoch + 1, 'epochs': epochs,
                    'step': step + 1, 'steps_per_epoch': steps, 'global_step': global_step,
                    'loss': float(loss.detach()), 'training_view_nmse': float(energy_loss(prediction.detach(), x).mean()),
                    'batch_active_templates': int((maps.detach().amax((0, 2, 3)) > 1e-6).sum()),
                    'lr': lr, 'elapsed_seconds': round(elapsed_before + time.monotonic() - started, 2),
                    'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 2**20, 2)}
                write_json(folder / 'status.json', state)
                with (folder / 'training.jsonl').open('a') as log:
                    log.write(json.dumps(state) + '\n')
                save(epoch, step + 1)
                print('TRAIN', json.dumps(state), flush=True)
            del x, prediction, maps, loss
        score = monitor(model, data, dev_ids)
        if score < best:
            best = score
            save_torch(folder / 'best.pt', {'model': model.state_dict(), 'config': model.config,
                'experiment_config': config, 'epoch': epoch + 1, 'dev_nmse': best})
        write_json(folder / f'epoch_{epoch + 1:02d}.json', {'epoch': epoch + 1, 'dev_nmse': score, 'best': best})
        print('DEV', arm, epoch + 1, score, flush=True)
        save(epoch + 1, 0)
    state = {'status': 'complete', 'arm': arm, 'epochs': epochs, 'global_step': global_step,
             'best_dev_nmse': best, 'elapsed_seconds': round(elapsed_before + time.monotonic() - started, 2),
             'best_sha256': digest(folder / 'best.pt')}
    write_json(folder / 'complete.json', state)
    write_json(folder / 'status.json', state)


def reconstruction_curve(nmse, thresholds=(0., .1, .2, .3, .5, .8)):
    """Each row is one panorama, each column one direction; zero-energy views use NaN."""
    nmse = np.asarray(nmse)
    if nmse.ndim != 2 or nmse.shape[1] != 4:
        raise ValueError('Need panorama x four directions')
    rows = []
    for threshold in thresholds:
        passed = np.isfinite(nmse) & ((1 - nmse) >= threshold)
        rows.append({'minimum_energy_error_reduction': threshold,
            'direction_fraction': float(passed.mean()), 'all_four_fraction': float(passed.all(1).mean()),
            'any_direction_fraction': float(passed.any(1).mean())})
    return rows


@torch.inference_mode()
def evaluate(arm, out=OUT, device='cuda:2', batch_size=4):
    runtime(device)
    out = Path(out)
    folder = out / 'development' / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    data = dataset(out, device)
    checkpoint = torch.load(out / 'models' / arm / 'best.pt', map_location='cpu', weights_only=False)
    model = PrototypeAutoencoder(**checkpoint['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    tables = []
    for city, group in data.frame[data.frame.prototype_split == 'dev'].groupby('city', sort=True):
        path = folder / f'{city.replace("/", "__")}.parquet'
        if path.exists():
            tables.append(pd.read_parquet(path))
            continue
        records = []
        ids = group.index.to_numpy()
        for heading in range(4):
            for at in range(0, len(ids), batch_size):
                selected = ids[at:at + batch_size]
                x = data.load(selected, [heading] * len(selected))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction, maps = model(x)
                sse = (prediction.float() - x).square().flatten(1).sum(1)
                energy = x.square().flatten(1).sum(1)
                mass = maps.float().sum((2, 3))
                # Effective number (participation ratio), not a count of semantic objects.
                effective = mass.sum(1).square() / mass.square().sum(1).clamp_min(1e-12)
                peak = maps.amax((2, 3))
                for j, row_id in enumerate(selected):
                    records.append({'row_id': int(row_id), 'city': city, 'heading': heading,
                        'sse': float(sse[j]), 'energy': float(energy[j]),
                        'effective_templates': float(effective[j]),
                        'active_templates': int((peak[j] > 1e-6).sum())})
        table = pd.DataFrame(records)
        table.to_parquet(path.with_suffix('.partial.parquet'), index=False)
        path.with_suffix('.partial.parquet').replace(path)
        tables.append(table)
        write_json(folder / 'status.json', {'cities': len(tables), 'total_cities': data.frame.city.nunique(),
            'views': sum(len(t) for t in tables), 'total_views': int((data.frame.prototype_split == 'dev').sum()) * 4})
        print('EVALUATE', arm, city, sum(len(t) for t in tables), flush=True)
    table = pd.concat(tables, ignore_index=True)
    table['nmse'] = table.sse / table.energy.where(table.energy > 1e-12)
    nmse = table.pivot(index='row_id', columns='heading', values='nmse').reindex(columns=range(4)).to_numpy()
    summary = {'arm': arm, 'partition': 'dev', 'panoramas': len(nmse), 'views': len(table),
        'global_nmse': float(table.sse.sum() / max(table.energy.sum(), 1e-12)),
        'mean_direction_nmse': float(table.nmse.mean()), 'effective_templates_mean': float(table.effective_templates.mean()),
        'active_templates_mean': float(table.active_templates.mean()), 'curve': reconstruction_curve(nmse),
        'semantic_coverage': None, 'semantic_audit': 'pending', 'test_used': False,
        'note': 'Fractions refer ONLY to reconstruction error reduction, not semantic coverage; dev is used for model selection.'}
    write_json(folder / 'complete.json', summary)


def report(out=OUT):
    """Make an honest first report, curves and an unlabeled development audit sheet."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    reports = out.parent / 'results'
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(out / 'panoramas.parquet')
    summaries = [json.loads((out / 'development' / arm / 'complete.json').read_text()) for arm in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for summary in summaries:
        curve = summary['curve']
        x = [r['minimum_energy_error_reduction'] for r in curve]
        axes[0].plot(x, [r['direction_fraction'] for r in curve], 'o-', label=summary['arm'])
        axes[1].plot(x, [r['all_four_fraction'] for r in curve], 'o-', label=summary['arm'])
    for ax, title in zip(axes, ['Directional views', 'Panoramas: all four directions']):
        ax.set(xlabel='Required reconstruction error reduction', ylabel='Fraction meeting requirement', title=title, ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Development set reconstruction diagnostic (not semantic coverage)')
    fig.savefig(reports / 'development_quality_curve.png', dpi=180)
    plt.close(fig)
    audit = frame.loc[panel(frame, per_city=10)].copy()
    for column in ['rater_1_four_directions_pass', 'rater_2_four_directions_pass', 'semantic_consistency',
                   'missing_content', 'prototype_redundancy', 'notes']:
        audit[column] = ''
    audit.to_csv(out / 'development_audit_pending.csv', index=False)
    result = {'protocol': json.loads((out / 'protocol.json').read_text()),
        'diagnostic': json.loads((out / 'diagnostic/complete.json').read_text()),
        'arms': summaries, 'audit_panoramas': len(audit), 'semantic_coverage': None,
        'status': 'computational_pilot_complete; semantic audit pending; test sealed'}
    write_json(reports / 'pilot_summary.json', result)
    lines = ['# 原型覆盖率第一轮试验', '', '测试集保持封存。以下均为开发集重建诊断，不代表语义覆盖率。', '',
             '| 训练设置 | 开发集 NMSE | 平均有效模板数 |', '| --- | ---: | ---: |']
    for row in summaries:
        lines.append(f'| {row["arm"]} | {row["global_nmse"]:.4f} | {row["effective_templates_mean"]:.2f} |')
    lines += ['', '![开发集质量曲线](development_quality_curve.png)', '',
              f'人工核查表已准备 {len(audit)} 个开发全景，标签为空；80% 语义覆盖目标尚未验证。',
              '本轮每尺度32个模板、每组4轮，只用于筛选后续完整训练的设置。尚未完成多种子稳定性、独立语义核查和最终测试。']
    (reports / 'README.md').write_text('\n'.join(lines) + '\n')
