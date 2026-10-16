"""GPU-only, fixed-dictionary response refit on a city-balanced dev panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.vitl30_multiscale_prototypes.code.data import write_json
from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder
from experiments.vitl30_prototype_coverage.code.protocol import panel
from experiments.vitl30_prototype_coverage.code.trial import decode, runtime
from sae_experiments.io_utils import digest
from .experiment import OUT, POLICY, load_data
from .metrics import coverage_summary, patch_statistics

RESULTS = Path(__file__).resolve().parents[1] / 'results'
ARMS = ('energy', 'coverage')


def measure(prediction, target):
    stats = patch_statistics(prediction, target, POLICY)
    return {'valid_patch_count': int(stats['valid_count'][0]),
            'covered_patch_count': int(stats['covered_count'][0]),
            'patch_coverage': float(stats['coverage'][0]),
            'image_pass': bool(stats['image_pass'][0]),
            'image_nmse': float(stats['image_nmse'][0])}


def prefer(candidate, previous):
    """Keep the best observed hard coverage, breaking ties by energy error."""
    return (candidate['covered_patch_count'], -candidate['image_nmse']) > (
        previous['covered_patch_count'], -previous['image_nmse'])


def refit(model, target, warm, arm, steps=150, lr=.15):
    if arm not in ARMS:
        raise ValueError(arm)
    coefficients = torch.nn.Parameter(warm.detach().float().clone())
    optimizer = torch.optim.Adam([coefficients], lr=lr)
    with torch.no_grad():
        baseline = measure(decode(model, coefficients), target)
    best = dict(baseline)
    best_step = 0
    checkpoints = [{'step': 0, **baseline}]
    milestones = {25, 50, 100, steps}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = decode(model, coefficients)
        stats = patch_statistics(prediction, target, POLICY, temperature=POLICY.temperature_end)
        if arm == 'energy':
            loss = stats['image_nmse'].mean()
        else:
            deficit = torch.relu(POLICY.target_fraction - stats['soft_coverage'])
            loss = (stats['image_nmse'].mean() +
                    POLICY.robust_patch_weight * stats['robust_patch_error'].mean() +
                    POLICY.coverage_weight * deficit.square().mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite {arm} refit loss at step {step}')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            coefficients.clamp_(min=0)
            if step % 10 == 0 or step in milestones:
                current = measure(decode(model, coefficients), target)
                if prefer(current, best):
                    best, best_step = current, step
                if step in milestones:
                    checkpoints.append({'step': step, **current})
    return {'best_step': best_step, 'best': best, 'final': current, 'checkpoints': checkpoints,
            'steps': steps, 'lr': lr,
            'constraint': 'nonnegative response maps only; no response sparsity penalty'}


def describe(rows):
    table = pd.DataFrame(rows)
    summary = coverage_summary(table.assign(patch_count=784), POLICY)
    summary['mean_image_nmse'] = float(table.image_nmse.mean())
    return summary


def summarize(records, protocol):
    rows = []
    for record in records:
        for stage in ('encoder', *ARMS):
            score = record[stage] if stage == 'encoder' else record[stage]['best']
            rows.append({'row_id': record['row_id'], 'heading': record['heading'],
                         'city': record['city'], 'stage': stage, **score})
    frame = pd.DataFrame(rows)
    summaries = {stage: describe(frame[frame.stage == stage]) for stage in ('encoder', *ARMS)}
    before = frame[frame.stage == 'encoder'].sort_values(['row_id', 'heading']).reset_index(drop=True)
    changes = {}
    for stage in ARMS:
        after = frame[frame.stage == stage].sort_values(['row_id', 'heading']).reset_index(drop=True)
        changes[stage] = {'improved_images': int((after.patch_coverage > before.patch_coverage).sum()),
                          'worsened_images': int((after.patch_coverage < before.patch_coverage).sum()),
                          'newly_passing_images': int((after.image_pass & ~before.image_pass).sum()),
                          'mean_coverage_gain': float((after.patch_coverage - before.patch_coverage).mean())}
    return {'status': 'complete', 'protocol': protocol, 'summaries': summaries, 'changes': changes,
            'interpretation': 'Finite-step fixed-dictionary response search, not a certified maximum or semantic coverage',
            'test_used': False}


def run(device='cuda:1', per_city=1, steps=150, lr=.15):
    if per_city < 1 or steps < 1 or lr <= 0:
        raise ValueError('Invalid diagnostic size or optimizer settings')
    runtime(device)
    checkpoint = OUT / 'model/best.pt'
    training_protocol = json.loads((OUT / 'protocol.json').read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['experiment_config']['protocol_identity'] != training_protocol['identity']:
        raise ValueError('Refit model and activation protocol differ')
    model = PrototypeAutoencoder(**saved['config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    data = load_data(OUT, device)
    ids = panel(data.frame, split='dev', per_city=per_city)
    if len(ids) != training_protocol['cities'] * per_city:
        raise ValueError('Incomplete city-balanced dev panel')
    protocol = {'version': 1, 'experiment_identity': training_protocol['identity'],
                'model_sha256': digest(checkpoint), 'code_sha256': digest(Path(__file__)),
                'panel': ids.tolist(), 'partition': 'dev', 'headings': [0, 1, 2, 3],
                'per_city': per_city, 'steps': steps, 'lr': lr, 'policy': POLICY.record(),
                'arms': list(ARMS), 'no_sparsity_penalty': True, 'test_sealed': True}
    protocol['identity'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    out = OUT / 'response_refit_v1'
    out.mkdir(parents=True, exist_ok=True)
    existing = out / 'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != protocol:
        raise ValueError('Refit protocol changed; use a distinct output folder')
    write_json(existing, protocol)
    if (out / 'complete.json').exists():
        return json.loads((out / 'complete.json').read_text())
    records = []
    start = time.monotonic()
    for row_id in ids:
        row = data.frame.iloc[int(row_id)]
        if row.prototype_split != 'dev':
            raise ValueError('Non-dev panorama selected')
        for heading in range(4):
            path = out / 'views' / f'row_{int(row_id):06d}_h{heading}.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record['protocol_identity'] != protocol['identity']:
                    raise ValueError('Stale refit view')
            else:
                target = data.load([row_id], [heading])
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction, warm = model(target)
                baseline = measure(prediction, target)
                record = {'protocol_identity': protocol['identity'], 'row_id': int(row_id),
                          'city': row.city, 'panoid': row.panoid, 'heading': heading,
                          'encoder': baseline}
                for arm in ARMS:
                    record[arm] = refit(model, target, warm, arm, steps=steps, lr=lr)
                write_json(path, record)
                del target, prediction, warm
            records.append(record)
            if len(records) % 4 == 0:
                state = {'status': 'running', 'views': len(records), 'total_views': len(ids) * 4,
                         'cities': len(records) // 4, 'total_cities': training_protocol['cities'],
                         'elapsed_seconds': round(time.monotonic() - start, 1)}
                write_json(out / 'status.json', state)
                print('REFIT', json.dumps(state), flush=True)
    result = summarize(records, protocol)
    result['elapsed_seconds'] = round(time.monotonic() - start, 1)
    if digest(checkpoint) != protocol['model_sha256']:
        raise ValueError('Model checkpoint changed during refit')
    write_json(out / 'complete.json', result)
    write_json(out / 'status.json', {'status': 'complete', 'views': len(records),
                                   'total_views': len(records), 'elapsed_seconds': result['elapsed_seconds']})
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / 'response_refit_summary.json', result)
    lines = ['# 冻结模板的逐图响应优化诊断', '',
             '开发集29城市各1全景，四方向共116张图；冻结96个模板，只优化非负响应。',
             '同一批图分别记录编码器起点、仅优化全图能量误差、直接优化逐图覆盖损失的结果。',
             '每种路线保留优化中观测到的最佳硬覆盖；不加响应稀疏约束，因此是宽松的探索诊断。',
             '有限步优化不能给出模板表达能力的理论上限；NMSE达标也不是语义解释。测试集未使用。', '',
             '| 设置 | 达标图像 | 平均patch覆盖 | P10覆盖 | 全景四方向全达标 | 平均图像NMSE |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, label in [('encoder', '现有编码器'), ('energy', '只降重建误差'), ('coverage', '直接优化逐图覆盖')]:
        item = result['summaries'][name]
        lines.append(f'| {label} | {item["images_meeting_target"]}/{item["images"]} '
                     f'({item["image_pass_fraction"]:.1%}) | {item["mean_image_patch_coverage"]:.1%} '
                     f'| {item["coverage_quantiles"]["0.1"]:.1%} '
                     f'| {item["panoramas_all_four_pass"]}/{item["panoramas"]} '
                     f'| {item["mean_image_nmse"]:.3f} |')
    lines += ['', f'每张图每条路线运行{steps}步，学习率{lr}；详细协议和逐图记录见服务器'
              f' `outputs/response_refit_v1/`。']
    (RESULTS / 'response_refit_README.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--per-city', type=int, default=1)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--lr', type=float, default=.15)
    args = parser.parse_args()
    run(args.device, args.per_city, args.steps, args.lr)
