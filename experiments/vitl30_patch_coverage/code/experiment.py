"""Train and evaluate a per-image patch coverage constrained prototype model."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder
from experiments.vitl30_multiscale_prototypes.code.train import training_order
from experiments.vitl30_prototype_coverage.code.protocol import OUT as BASE, panel
from experiments.vitl30_prototype_coverage.code.trial import dataset, initialize_from_train, runtime
from sae_experiments.io_utils import digest, save_torch
from .metrics import CoveragePolicy, patch_statistics, coverage_objective, coverage_summary

OUT = Path(__file__).resolve().parents[1] / 'outputs'
POLICY = CoveragePolicy()


def prepare(out=OUT, base=BASE):
    out, base = Path(out), Path(base)
    source = json.loads((base / 'protocol.json').read_text())
    calibration = json.loads((base / 'calibration.json').read_text())
    if source['identity'] != calibration['identity'] or digest(base / 'positive_rms.npy') != calibration['scale_sha256']:
        raise ValueError('Base split/calibration identity differs')
    protocol = {'version': 1, 'goal': 'Every directional image has >=80% eligible patches reconstructed within a fixed quality threshold',
        'base': str(base.resolve()), 'base_identity': source['identity'],
        'manifest_sha256': digest(base / 'panoramas.parquet'), 'scale_sha256': calibration['scale_sha256'],
        'checkpoint_sha256': source['checkpoint_sha256'], 'split_counts': source['split_counts'],
        'cities': source['cities'], 'policy': POLICY.record(),
        'model': {'dimensions': 4096, 'scales': [3, 7, 15], 'per_scale': 32, 'rank': 4, 'hidden': 64},
        'quality_threshold_status': 'fixed exploratory reconstruction criterion; not calibrated semantic acceptability',
        'valid_patch_rule': 'normalized/clipped target SAE energy > 1e-8; eligibility fixed from target; report all exclusions',
        'image_rule': 'covered_patch_count >= ceil(0.8 * eligible_patch_count); no eligible patches => FAIL',
        'panorama_rule': 'report all four directions passing separately; never substitute panorama/global mean for image pass',
        'test_policy': 'sealed; train/dev only in this pipeline',
        'frozen_SAE_caveat': 'new prototype-layer holdout only; frozen SAE saw some dev/test images',
        'semantic_coverage': None, 'semantic_review': 'pending',
        'epochs': 8, 'batch_size': 8, 'seed': 42, 'lr': .001,
        'sampling': 'city-balanced; epoch%4 heading; two visits per heading in eight epochs',
        'masking': 'no added hidden patches in this first coverage trial',
        'selection': 'dev image pass fraction, then lower mean per-image hard shortfall, then higher mean coverage, then lower NMSE',
        'scope': 'fixed finite 96-template capacity; feasibility is measured, not guaranteed'}
    protocol['identity'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (out / 'protocol.json').exists() and json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('Patch coverage protocol changed; use a different output folder')
    write_json(out / 'protocol.json', protocol)
    return protocol


def load_data(out, device):
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    base = Path(protocol['base'])
    if (digest(base / 'panoramas.parquet') != protocol['manifest_sha256'] or
            digest(base / 'positive_rms.npy') != protocol['scale_sha256'] or
            json.loads((base / 'protocol.json').read_text())['identity'] != protocol['base_identity']):
        raise ValueError('Bound split or training-only scale changed')
    return dataset(base, device)


@torch.inference_mode()
def monitor(model, data, ids, batch_size=4):
    model.eval()
    coverages, passes, errors = [], [], []
    for heading in range(4):
        for at in range(0, len(ids), batch_size):
            selected = ids[at:at + batch_size]
            x = data.load(selected, [heading] * len(selected))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, _ = model(x)
            stats = patch_statistics(prediction, x, POLICY)
            coverages.extend(stats['coverage'].cpu().tolist())
            passes.extend(stats['image_pass'].cpu().tolist())
            errors.extend(stats['image_nmse'].cpu().tolist())
    model.train()
    coverage = np.array(coverages)
    return {'images': len(coverage), 'image_pass_fraction': float(np.mean(passes)),
            'mean_image_patch_coverage': float(coverage.mean()),
            'mean_hard_shortfall': float(np.maximum(POLICY.target_fraction - coverage, 0).mean()),
            'p10_image_patch_coverage': float(np.quantile(coverage, .1)),
            'minimum_image_patch_coverage': float(coverage.min()), 'mean_image_nmse': float(np.mean(errors))}


def selection_score(metrics):
    return (metrics['image_pass_fraction'], -metrics['mean_hard_shortfall'],
            metrics['mean_image_patch_coverage'], -metrics['mean_image_nmse'])


def train(out=OUT, device='cuda:3'):
    runtime(device)
    out = Path(out)
    protocol = prepare(out)
    folder = out / 'model'
    folder.mkdir(parents=True, exist_ok=True)
    data = load_data(out, device)
    config = {'protocol_identity': protocol['identity'], 'policy': POLICY.record(),
        'code_sha256': digest(Path(__file__)), 'metrics_code_sha256': digest(Path(__file__).with_name('metrics.py')),
        'source_code_sha256': digest(Path(__file__).parents[2] / 'vitl30_prototype_coverage/code/trial.py'),
        'model_code_sha256': digest(Path(__file__).parents[2] / 'vitl30_multiscale_prototypes/code/model.py')}
    if (folder / 'config.json').exists() and json.loads((folder / 'config.json').read_text()) != config:
        raise ValueError('Training code/config changed; use a new run')
    write_json(folder / 'config.json', config)
    if (folder / 'complete.json').exists():
        return
    torch.manual_seed(protocol['seed'])
    model = PrototypeAutoencoder(**protocol['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['lr'], weight_decay=1e-5)
    path = folder / 'last.pt'
    epoch_start = step_start = global_step = 0
    best = (-float('inf'),) * 4
    elapsed_before = 0.
    if path.exists():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint['experiment_config'] != config:
            raise ValueError('Checkpoint identity differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start, step_start, global_step = checkpoint['epoch'], checkpoint['step'], checkpoint['global_step']
        best, elapsed_before = tuple(checkpoint['best']), checkpoint['elapsed_seconds']
    else:
        initialization = initialize_from_train(model, data, protocol['seed'])
        write_json(folder / 'initialization.json', initialization)
    epochs, batch_size = protocol['epochs'], protocol['batch_size']
    dev_ids = panel(data.frame, per_city=4)
    started = time.monotonic()
    if not (folder / 'initial_dev.json').exists():
        initial = monitor(model, data, dev_ids)
        write_json(folder / 'initial_dev.json', initial)
        print('INITIAL_DEV', json.dumps(initial), flush=True)

    def save(epoch, step):
        save_torch(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'config': model.config, 'experiment_config': config, 'epoch': epoch, 'step': step,
            'global_step': global_step, 'best': list(best),
            'elapsed_seconds': elapsed_before + time.monotonic() - started})

    for epoch in range(epoch_start, epochs):
        cities, orders, steps = training_order(data.frame, epoch, protocol['seed'], batch_size)
        model.train()
        for step in range(step_start if epoch == epoch_start else 0, steps):
            city = cities[step % len(cities)]
            ids = orders[city][step // len(cities)]
            x = data.load(ids, [epoch % 4] * len(ids))
            progress = epoch + step / steps
            lr = protocol['lr'] * (.1 + .9 * (1 + math.cos(math.pi * progress / epochs)) / 2)
            lr *= min(1., (global_step + 1) / 100)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, maps = model(x)
                loss, metrics = coverage_objective(prediction, x, maps, model.composition_penalty(), progress, POLICY)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite coverage loss at {epoch}/{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            model.project_()
            global_step += 1
            if global_step % 100 == 0 or step == steps - 1:
                state = {'status': 'training', 'epoch': epoch + 1, 'epochs': epochs, 'step': step + 1,
                    'steps_per_epoch': steps, 'global_step': global_step, 'loss': float(loss.detach()),
                    **{k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()},
                    'lr': lr, 'elapsed_seconds': round(elapsed_before + time.monotonic() - started, 2),
                    'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 2**20, 2),
                    'semantic_review': 'pending'}
                write_json(folder / 'status.json', state)
                with (folder / 'training.jsonl').open('a') as log:
                    log.write(json.dumps(state) + '\n')
                save(epoch, step + 1)
                print('TRAIN', json.dumps(state), flush=True)
            del x, prediction, maps, loss, metrics
        dev = monitor(model, data, dev_ids)
        score = selection_score(dev)
        if score > best:
            best = score
            save_torch(folder / 'best.pt', {'model': model.state_dict(), 'config': model.config,
                'experiment_config': config, 'epoch': epoch + 1, 'dev': dev})
        write_json(folder / f'epoch_{epoch + 1:02d}.json', {'epoch': epoch + 1, **dev, 'best_score': list(best)})
        print('DEV', json.dumps({'epoch': epoch + 1, **dev}), flush=True)
        save(epoch + 1, 0)
    state = {'status': 'complete', 'epochs': epochs, 'global_step': global_step,
        'best_score': list(best), 'best_sha256': digest(folder / 'best.pt'),
        'elapsed_seconds': round(elapsed_before + time.monotonic() - started, 2), 'semantic_review': 'pending'}
    write_json(folder / 'complete.json', state)
    write_json(folder / 'status.json', state)


@torch.inference_mode()
def evaluate(out=OUT, device='cuda:3', split='dev'):
    if split not in {'train', 'dev'}:
        raise ValueError('Final test remains sealed')
    runtime(device)
    out = Path(out)
    data = load_data(out, device)
    folder = out / 'evaluation' / split
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / 'model/best.pt'
    model_hash = digest(checkpoint_path)
    if (folder / 'complete.json').exists():
        if json.loads((folder / 'complete.json').read_text())['model_sha256'] != model_hash:
            raise ValueError('Scored checkpoint changed')
        return
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = PrototypeAutoencoder(**checkpoint['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    tables = []
    for city, group in data.frame[data.frame.prototype_split == split].groupby('city', sort=True):
        name = city.replace('/', '__')
        path = folder / f'{name}.parquet'
        marker_path = folder / f'{name}.json'
        marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
        if marker:
            if marker['model_sha256'] != model_hash or marker['policy'] != POLICY.record():
                raise ValueError('Stale per-city scores')
            if digest(path) != marker['table_sha256'] or digest(folder / f'{name}.npz') != marker['masks_sha256']:
                raise ValueError('Per-city evaluation files changed')
            tables.append(pd.read_parquet(path))
            continue
        ids = group.index.to_numpy()
        records, coverage_masks, eligibility_masks = [], [], []
        for heading in range(4):
            for at in range(0, len(ids), 4):
                selected = ids[at:at + 4]
                x = data.load(selected, [heading] * len(selected))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction, maps = model(x)
                stats = patch_statistics(prediction, x, POLICY)
                response_mass = maps.float().sum((2, 3))
                effective = response_mass.sum(1).square() / response_mass.square().sum(1).clamp_min(1e-12)
                covered = stats['covered_mask'].cpu().numpy()
                eligible = stats['valid_mask'].cpu().numpy()
                for j, row_id in enumerate(selected):
                    row = data.frame.loc[row_id]
                    records.append({'row_id': int(row_id), 'city': city, 'panoid': row.panoid,
                        'heading': heading, 'heading_degrees': heading * 90, 'split': split,
                        'image_path': row[f'image_{heading * 90}'], 'spatial_block': row.spatial_block,
                        'patch_count': 784, 'valid_patch_count': int(stats['valid_count'][j]),
                        'covered_patch_count': int(stats['covered_count'][j]),
                        'required_patch_count': int(stats['required_count'][j]),
                        'patch_coverage': float(stats['coverage'][j]), 'image_pass': bool(stats['image_pass'][j]),
                        'image_nmse': float(stats['image_nmse'][j]), 'effective_templates': float(effective[j])})
                coverage_masks.append(np.packbits(covered.reshape(len(selected), -1), axis=1))
                eligibility_masks.append(np.packbits(eligible.reshape(len(selected), -1), axis=1))
        table = pd.DataFrame(records)
        temp = path.with_suffix('.partial.parquet')
        table.to_parquet(temp, index=False)
        temp.replace(path)
        mask_path = folder / f'{name}.npz'
        temp_mask = mask_path.with_suffix('.partial.npz')
        np.savez_compressed(temp_mask, covered=np.concatenate(coverage_masks), valid=np.concatenate(eligibility_masks),
                            row_ids=table.row_id.to_numpy(), headings=table.heading.to_numpy(), grid=28)
        temp_mask.replace(mask_path)
        write_json(marker_path, {'model_sha256': model_hash, 'policy': POLICY.record(),
            'table_sha256': digest(path), 'masks_sha256': digest(mask_path), 'summary': coverage_summary(table, POLICY)})
        tables.append(table)
        write_json(folder / 'status.json', {'cities': len(tables), 'total_cities': data.frame.city.nunique(),
            'images': sum(len(t) for t in tables), 'total_images': int((data.frame.prototype_split == split).sum()) * 4})
        print('EVALUATE', split, city, len(table), flush=True)
    full = pd.concat(tables, ignore_index=True)
    summary = coverage_summary(full, POLICY)
    summary.update({'split': split, 'model_sha256': model_hash, 'test_used': False,
                    'city_summary': {city: coverage_summary(t, POLICY) for city, t in full.groupby('city')}})
    failures = full[~full.image_pass].sort_values(['patch_coverage', 'city', 'row_id', 'heading'])
    failures.to_csv(folder / 'images_below_target.csv', index=False)
    write_json(folder / 'complete.json', summary)


def report(out=OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    out = Path(out)
    results = out.parent / 'results'
    results.mkdir(parents=True, exist_ok=True)
    summary = {'protocol': json.loads((out / 'protocol.json').read_text()), 'partitions': {},
               'semantic_review': 'pending', 'test_used': False}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    evidence = []
    for split, color in [('train', '#2274a5'), ('dev', '#e07a32')]:
        folder = out / 'evaluation' / split
        result = json.loads((folder / 'complete.json').read_text())
        summary['partitions'][split] = result
        paths = sorted(folder.glob('*.parquet'))
        table = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        x = np.sort(table.patch_coverage.to_numpy())
        axes[0].plot(x, np.arange(1, len(x) + 1) / len(x), label=split, color=color)
        axes[1].bar(split, result['image_pass_fraction'], color=color)
        if split == 'dev':
            sorted_rows = table.sort_values('patch_coverage').reset_index(drop=True)
            for quantile in [.05, .25, .5, .75, .95]:
                record = sorted_rows.iloc[int(round((len(sorted_rows) - 1) * quantile))]
                city_table = pd.read_parquet(folder / f'{record.city.replace("/", "__")}.parquet')
                selected = city_table.index[(city_table.row_id == record.row_id) & (city_table.heading == record.heading)][0]
                with np.load(folder / f'{record.city.replace("/", "__")}.npz') as masks:
                    valid = np.unpackbits(masks['valid'][selected])[:784].reshape(28, 28).astype(bool)
                    covered = np.unpackbits(masks['covered'][selected])[:784].reshape(28, 28).astype(bool)
                try:
                    with Image.open(record.image_path) as image:
                        original = image.convert('RGB')
                except OSError as error:
                    evidence.append({'quantile': quantile, 'city': record.city,
                                     'row_id': int(record.row_id), 'heading': int(record.heading),
                                     'patch_coverage': float(record.patch_coverage),
                                     'figure': None, 'image_error': str(error)})
                    continue
                original.thumbnail((600, 400))
                rgba = np.zeros((28, 28, 4), dtype=np.uint8)
                rgba[valid & covered] = [22, 184, 83, 95]
                rgba[valid & ~covered] = [239, 49, 49, 140]
                rgba[~valid] = [110, 110, 110, 140]
                overlay = Image.fromarray(rgba).resize(original.size, Image.Resampling.NEAREST)
                annotated = Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')
                canvas = Image.new('RGB', (original.width * 2, original.height + 42), 'white')
                canvas.paste(original, (0, 0)); canvas.paste(annotated, (original.width, 0))
                ImageDraw.Draw(canvas).text((8, original.height + 9),
                    f'{record.city} / {record.heading_degrees}deg | patch coverage {record.patch_coverage:.1%} | green: pass, red: fail, gray: invalid', fill='black')
                name = f'dev_coverage_q{int(quantile * 100):02d}.jpg'
                canvas.save(results / name, quality=90)
                evidence.append({'quantile': quantile, 'city': record.city, 'row_id': int(record.row_id),
                                 'heading': int(record.heading), 'patch_coverage': float(record.patch_coverage), 'figure': name})
    axes[0].axvline(.8, color='black', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Eligible patch coverage per image', ylabel='Cumulative fraction of images', xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    axes[1].set(ylabel='Fraction of images with >=80% patch coverage', ylim=(0, 1))
    fig.suptitle('Fixed patch NMSE <= 0.5; reconstruction proxy, semantic review pending')
    fig.savefig(results / 'per_image_patch_coverage.png', dpi=180)
    plt.close(fig)
    summary['evidence'] = evidence
    write_json(results / 'summary.json', summary)
    lines = ['# 逐图80%有效patch覆盖实验', '', '以下是SAE激活重建覆盖，尚未完成独立语义核查。测试集保持封存。', '',
             '| 划分 | 方向图像 | 达到逐图80%要求的比例 | 最差图像覆盖率 |', '| --- | ---: | ---: | ---: |']
    for split, s in summary['partitions'].items():
        lines.append(f'| {split} | {s["images"]:,} | {s["image_pass_fraction"]:.2%} | {s["coverage_quantiles"]["0"]:.2%} |')
    lines += ['', '![逐图覆盖分布](per_image_patch_coverage.png)', '',
              '绿色patch达到固定质量门槛，红色未达到，灰色无有效目标激活。实例按开发覆盖率分位数选择。']
    for item in evidence:
        if item['figure']:
            lines += ['', f'![开发集实例 {item["quantile"]} 分位]({item["figure"]})']
    (results / 'README.md').write_text('\n'.join(lines) + '\n')
