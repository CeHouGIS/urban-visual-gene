"""Heldout reconstruction, permutation controls, seed matching and native exemplars.

All outputs remain in this experiment. Validation covers every heldout panorama;
the repeated permutation test uses a fixed, disjoint 16-panorama/city panel.
Neither candidate templates nor VLM names are treated as validated scene classes.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import heapq
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment

from sae_experiments.io_utils import digest, save_torch
from .data import OUT, TOKENS, SparseViews, write_json
from .train import RUNS, load_model, perturb, training_order
from .resources import configure_device

STAT_NAMES = ['sse', 'energy', 'positive_sse', 'positive_count', 'zero_sse', 'zero_count']


class SourceImageError(OSError):
    """A requested native image is missing or unreadable, not an output-write error."""


def derived_statistics(stats):
    """Ratios of summed sufficient statistics, never means of batch ratios."""
    sse, energy, pos, npos, zero, nzero = np.asarray(stats, dtype=np.float64)
    return {'nmse': float(sse / max(energy, 1e-12)),
            'positive_mse': float(pos / max(npos, 1)),
            'zero_mse': float(zero / max(nzero, 1)),
            'balanced_error': float(pos / max(npos, 1) + 32 * zero / max(nzero, 1)),
            'zero_prediction_balanced_error': float(energy / max(npos, 1)),
            'zero_prediction_nmse': 1.}


def sufficient_statistics(pred, target):
    error = (pred.float() - target).square()
    positive = target > 0
    axes = (1, 2, 3)
    pos = error.masked_fill(~positive, 0).sum(axes)
    sse = error.sum(axes)
    count = positive.sum(axes)
    return torch.stack([sse, target.square().sum(axes), pos, count,
                        sse - pos, target[0].numel() - count], 1)


def local_peaks(maps, scales=(3, 7, 15), per_scale=32, maximum=2):
    """Within-template NMS; retain at most two half-peak responses per view.

    Locations are candidate template centres, not segmentation boundaries.
    Adjacent plateaus are deterministically broken before maximum pooling.
    """
    batch, _, height, width = maps.shape
    tie = torch.arange(height * width, device=maps.device, dtype=torch.float64).reshape(1, 1, height, width)
    values, positions = [], []
    for group, size in enumerate(scales):
        response = maps[:, group * per_scale:(group + 1) * per_scale].float()
        # Pooling float64 is supported by torch on CUDA and prevents tie jitter
        # from changing distinct float32 responses at this small grid size.
        order = response.double() + tie * torch.finfo(torch.float64).eps
        peaks = order == F.max_pool2d(order, size, stride=1, padding=size // 2)
        threshold = response.flatten(2).max(2).values[:, :, None, None] * .5
        sparse = response.masked_fill(~peaks | (response < threshold), 0)
        score, index = sparse.flatten(2).topk(maximum, dim=2)
        values.append(score); positions.append(index)
    return torch.cat(values, 1), torch.cat(positions, 1)


def view_batches(frame, batch_size, ids=None):
    selected = frame if ids is None else frame.loc[ids]
    for _, group in selected.groupby('city', sort=True):
        order = group.index.to_numpy()
        for heading in range(4):
            for start in range(0, len(order), batch_size):
                yield order[start:start + batch_size], heading


def panel_ids(frame, per_city=16):
    """Exclude the epoch-monitoring panoramas before choosing the null-test panel."""
    result = []
    for city, group in frame[frame.prototype_split == 'validation'].groupby('city', sort=True):
        seed = int.from_bytes(hashlib.sha256(city.encode()).digest()[:4], 'little')
        rng = np.random.default_rng(seed)
        monitor_ids = rng.choice(group.index.to_numpy(), min(16, len(group)), replace=False)
        available = np.setdiff1d(group.index.to_numpy(), monitor_ids)
        if len(available) < 2:
            raise ValueError(f'Insufficient non-monitoring heldout panoramas in {city}')
        result.extend(np.sort(rng.choice(available, min(per_city, len(available)), replace=False)).tolist())
    return result


def scaled_composition(data):
    """Exact mean of the same clipped, train-RMS-scaled target used by the CNN."""
    path = data.out / 'scaled_composition.npy'
    if path.exists():
        return np.load(path, mmap_mode='r')
    scale = data.scale.cpu().numpy()
    target = np.lib.format.open_memmap(path.with_name('scaled_composition.partial.npy'), mode='w+',
                                      dtype=np.float32, shape=(len(data.frame), len(scale)))
    for city, group in data.frame.groupby('city', sort=False):
        first = group.index[0]
        for start in range(0, len(group), data.spec['chunk_panos']):
            (value, dim, ptr), _ = data.chunk(city, start)
            length = min(data.spec['chunk_panos'], len(group) - start)
            pano = np.repeat(np.arange(length), np.diff(ptr[::TOKENS]).astype(np.int64))
            normalized = np.minimum(value.astype(np.float32) / scale[dim], 8.)
            target[first + start:first + start + length] = np.bincount(
                pano * len(scale) + dim, weights=normalized,
                minlength=length * len(scale)).reshape(length, -1) / TOKENS
        target.flush()
        print('COMPOSITION', city, flush=True)
    del target
    path.with_name('scaled_composition.partial.npy').replace(path)
    return np.load(path, mmap_mode='r')


def select_diverse(rows, frame, count=8, max_per_city=2):
    """Highest scores, unique panoramas, city cap first; then fill remaining slots."""
    ranked = sorted(rows, reverse=True)
    chosen, used, city_count = [], set(), defaultdict(int)
    for cap in [max_per_city, count]:
        for row in ranked:
            _, pano, *_ = row
            city = frame.loc[pano, 'city']
            if pano in used or city_count[city] >= cap:
                continue
            chosen.append(row); used.add(pano); city_count[city] += 1
            if len(chosen) == count:
                return chosen
    return chosen


def cluster_interval(values, repeats=2000, seed=42):
    """City bootstrap of a city-balanced mean (cities, not individual views)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'independent_units': 'cities', 'cities': len(values), 'bootstrap_repeats': repeats}


@torch.inference_mode()
def score_model(data, run_name, batch_size):
    folder = data.out / 'evaluation' / run_name
    folder.mkdir(parents=True, exist_ok=True)
    model, model_hash = load_model(data.out, run_name, data.device)
    if (folder / 'complete.json').exists():
        saved = json.loads((folder / 'complete.json').read_text())
        if saved['model_sha256'] != model_hash or saved['corpus_identity'] != data.spec['identity']:
            raise ValueError('Evaluation model/data identity changed')
        return
    frame = data.frame[data.frame.prototype_split == 'validation']
    stats = np.zeros((len(data.frame), 6), np.float64)
    coefficients = np.zeros((len(data.frame), 96), np.float32)
    composition = np.lib.format.open_memmap(folder / 'reconstructed_composition.partial.npy',
        mode='w+', dtype=np.float32, shape=(len(data.frame), 4096))
    composition[:] = 0
    heaps = [[] for _ in range(96)]
    detection_writer, detection_buffer = None, []
    usage = np.zeros((96, 3), np.float64)  # sum, number of positive responses, maximum
    control, seed = RUNS[run_name]
    generator = torch.Generator(device=data.device).manual_seed(seed + 7001)
    seen = 0
    for ids, heading in view_batches(frame, batch_size):
        target = perturb(data.load(ids, [heading] * len(ids)), control, generator)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(data.device).startswith('cuda')):
            pred, maps = model(target)
        stats[ids] += sufficient_statistics(pred, target).cpu().numpy().astype(np.float64)
        means = maps.float().mean((2, 3)).cpu().numpy()
        coefficients[ids] += means / 4
        composition[ids] += pred.float().mean((2, 3)).cpu().numpy() / 4
        maxima = maps.float().flatten(2).max(2).values.cpu().numpy()
        usage[:, 0] += means.sum(0) * 784
        usage[:, 1] += (maps > 0).sum((0, 2, 3)).cpu().numpy()
        usage[:, 2] = np.maximum(usage[:, 2], maxima.max(0))
        if control == 'real':
            peak_values, peak_indices = local_peaks(maps)
            peak_values, peak_indices = peak_values.cpu().numpy(), peak_indices.cpu().numpy()
            batch_index, prototype_index, slot = np.nonzero(peak_values > 1e-6)
            location = peak_indices[batch_index, prototype_index, slot]
            detection_buffer.append(pd.DataFrame({'pano_row': ids[batch_index],
                'heading': np.full(len(slot), heading * 90, np.int16),
                'prototype': prototype_index.astype(np.int16), 'y': (location // 28).astype(np.int8),
                'x': (location % 28).astype(np.int8),
                'coefficient': peak_values[batch_index, prototype_index, slot]}))
            if len(detection_buffer) >= 32:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(pd.concat(detection_buffer, ignore_index=True), preserve_index=False)
                if detection_writer is None:
                    detection_writer = pq.ParquetWriter(folder / 'detections.partial.parquet', table.schema)
                detection_writer.write_table(table)
                detection_buffer.clear()
            for b, pano in enumerate(ids):
                for prototype in range(96):
                    score = float(peak_values[b, prototype, 0])
                    if score <= 0:
                        continue
                    location = int(peak_indices[b, prototype, 0])
                    row = (score, int(pano), heading, location // 28, location % 28)
                    heap = heaps[prototype]
                    if len(heap) < 128:
                        heapq.heappush(heap, row)
                    elif row > heap[0]:
                        heapq.heapreplace(heap, row)
        seen += len(ids)
        if seen % 1024 < batch_size or seen == len(frame) * 4:
            composition.flush()
            write_json(folder / 'status.json', {'status': 'scoring', 'directional_images': seen,
                'total_directional_images': len(frame) * 4, 'run': run_name})
    if detection_buffer:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(pd.concat(detection_buffer, ignore_index=True), preserve_index=False)
        if detection_writer is None:
            detection_writer = pq.ParquetWriter(folder / 'detections.partial.parquet', table.schema)
        detection_writer.write_table(table)
    if detection_writer is not None:
        detection_writer.close()
        (folder / 'detections.partial.parquet').replace(folder / 'detections.parquet')
    composition.flush(); del composition
    (folder / 'reconstructed_composition.partial.npy').replace(folder / 'reconstructed_composition.npy')
    np.save(folder / 'statistics.npy', stats)
    np.save(folder / 'coefficients.npy', coefficients)
    np.save(folder / 'usage.npy', usage)
    write_json(folder / 'candidates.json', [sorted(heap, reverse=True) for heap in heaps])
    rows = []
    for (city, block), group in frame.groupby(['city', 'spatial_block'], sort=True):
        sums = stats[group.index].sum(0)
        rows.append({'city': city, 'block': block, 'panoramas': len(group),
                     **dict(zip(STAT_NAMES, sums)), **derived_statistics(sums)})
    pd.DataFrame(rows).to_csv(folder / 'spatial_blocks.csv', index=False)
    cities = []
    for city, group in frame.groupby('city', sort=True):
        cities.append({'city': city, 'panoramas': len(group),
                       **derived_statistics(stats[group.index].sum(0))})
    pd.DataFrame(cities).to_csv(folder / 'cities.csv', index=False)
    summary = {'run': run_name, 'target_distribution': control,
        'model_sha256': model_hash, 'corpus_identity': data.spec['identity'],
        'panoramas': len(frame), 'views': len(frame) * 4,
        'global': derived_statistics(stats.sum(0)),
        'city_balanced_nmse': cluster_interval([r['nmse'] for r in cities]),
        'active_templates': int((usage[:, 2] > 1e-6).sum()),
        'response_detection_threshold': 1e-6,
        'note': 'Null-trained models are scored on their respective shuffled targets; all views of every heldout panorama are included.'}
    summary['quality_flags'] = []
    if summary['global']['nmse'] >= 1:
        summary['quality_flags'].append('Spatial SSE is at least that of predicting zero; inspect reconstruction quality before interpreting templates.')
    if summary['global']['balanced_error'] >= summary['global']['zero_prediction_balanced_error']:
        summary['quality_flags'].append('Balanced reconstruction is no better than predicting zero.')
    if summary['active_templates'] < 96:
        summary['quality_flags'].append(f'{96 - summary["active_templates"]} templates had no response above the detection threshold.')
    write_json(folder / 'complete.json', summary)
    write_json(folder / 'status.json', {'status': 'complete', 'run': run_name,
                                       'directional_images': seen, 'total_directional_images': len(frame) * 4})
    print('SCORED', run_name, summary['global'], flush=True)


@torch.inference_mode()
def score_panel(model, data, ids, batch_size, control, seed):
    generator = torch.Generator(device=data.device).manual_seed(seed)
    by_city = defaultdict(lambda: np.zeros(6, np.float64))
    for selected, heading in view_batches(data.frame, batch_size, ids):
        x = perturb(data.load(selected, [heading] * len(selected)), control, generator)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=str(data.device).startswith('cuda')):
            pred, _ = model(x)
        by_city[data.frame.loc[int(selected[0]), 'city']] += sufficient_statistics(pred, x).sum(0).cpu().numpy()
    return float(np.mean([derived_statistics(v)['balanced_error'] for v in by_city.values()]))


def permutation_test(observed, randomized):
    """Lower reconstruction error is better; plus-one Monte Carlo p-value."""
    null = np.asarray(randomized, dtype=np.float64)
    if not len(null):
        raise ValueError('At least one permutation is required')
    return {'observed_balanced_error': observed, 'randomized_balanced_errors': null.tolist(),
            'mean_shuffle_minus_real': float(null.mean() - observed),
            'p_monte_carlo': float((1 + (null <= observed).sum()) / (len(null) + 1)),
            'p_resolution': 1 / (len(null) + 1), 'repeats': len(null)}


def permutation_controls(data, batch_size, repeats):
    model, model_hash = load_model(data.out, 'real_seed42', data.device)
    folder = data.out / 'evaluation' / 'permutations'
    folder.mkdir(parents=True, exist_ok=True)
    ids = panel_ids(data.frame)
    settings = {'ids': ids, 'batch_size': batch_size, 'model_sha256': model_hash,
                'corpus_identity': data.spec['identity'], 'repeats': repeats,
                'statistic': 'city-balanced positive MSE + 32*zero MSE, lower is better',
                'scope': 'fixed 16 validation panoramas/city, excluding epoch-monitoring panoramas',
                'shuffle_scope': 'same city, same heading, fixed batches; feature maps reassigned independently per dimension'}
    old = folder / 'settings.json'
    if old.exists() and json.loads(old.read_text()) != settings:
        raise ValueError('Permutation settings changed')
    write_json(old, settings)
    original_path = folder / 'original.json'
    if original_path.exists():
        observed = json.loads(original_path.read_text())['balanced_error']
    else:
        observed = score_panel(model, data, ids, batch_size, 'real', 42)
        write_json(original_path, {'balanced_error': observed})
    result = {}
    for control in ['spatial_shuffle', 'feature_shuffle']:
        scores = []
        for repeat in range(repeats):
            path = folder / f'{control}_{repeat:03d}.json'
            if path.exists():
                score = json.loads(path.read_text())['balanced_error']
            else:
                score = score_panel(model, data, ids, batch_size, control, 8009 + repeat)
                write_json(path, {'repeat': repeat, 'balanced_error': score})
            scores.append(score)
            write_json(folder / 'status.json', {'control': control, 'completed_repeats': repeat + 1,
                                                'total_repeats_per_control': repeats})
            print('PERMUTATION', control, repeat + 1, score, flush=True)
        result[control] = permutation_test(observed, scores)
    write_json(folder / 'summary.json', {**settings, 'results': result,
        'limits': 'Conditional randomization within fixed city/heading batches. Not a global spatial independence test; minimum p-value is set by repeat count.'})


@torch.inference_mode()
def match_seeds(data):
    a, _ = load_model(data.out, 'real_seed42', data.device)
    b, _ = load_model(data.out, 'real_seed43', data.device)
    ca = json.loads((data.out / 'evaluation/real_seed42/candidates.json').read_text())
    cb = json.loads((data.out / 'evaluation/real_seed43/candidates.json').read_text())
    ua = np.load(data.out / 'evaluation/real_seed42/usage.npy')[:, 2]
    ub = np.load(data.out / 'evaluation/real_seed43/usage.npy')[:, 2]
    rows = []
    for index, (ba, bb) in enumerate(zip(a.banks, b.banks)):
        # Exact cosine of the full unit-norm factored templates, without materializing them.
        gc = torch.einsum('ird,jsd->ijrs', ba.channels, bb.channels)
        gs = torch.einsum('iruv,jsuv->ijrs', ba.layout, bb.layout)
        cosine = (gc * gs).sum((2, 3)).cpu().numpy()
        left, right = linear_sum_assignment(-cosine)
        ac = (ba.channels * ba.layout.sum((2, 3))[:, :, None]).sum(1)
        bc = (bb.channels * bb.layout.sum((2, 3))[:, :, None]).sum(1)
        for i, j in zip(left, right):
            p, q = index * a.per_scale + int(i), index * b.per_scale + int(j)
            da = set(ac[i].topk(20).indices.cpu().tolist())
            db = set(bc[j].topk(20).indices.cpu().tolist())
            ia = {r[1] for r in select_diverse(ca[p], data.frame, 20, 20)}
            ib = {r[1] for r in select_diverse(cb[q], data.frame, 20, 20)}
            rows.append({'prototype_seed42': p, 'prototype_seed43': q, 'scale': ba.size,
                'template_cosine': float(cosine[i, j]), 'top20_dimension_jaccard': len(da & db) / len(da | db),
                'top20_panorama_jaccard': len(ia & ib) / max(len(ia | ib), 1),
                'both_observed': bool(ua[p] > 1e-6 and ub[q] > 1e-6)})
    pd.DataFrame(rows).to_csv(data.out / 'evaluation/seed_stability.csv', index=False)


def nmf_baseline(data, target):
    """NMF and convolutional models are compared ONLY on panorama mean vectors."""
    from sklearn.decomposition import MiniBatchNMF
    from threadpoolctl import threadpool_limits
    import joblib
    folder = data.out / 'evaluation/nmf'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    train_ids = data.frame.index[data.frame.prototype_split == 'train'].to_numpy()
    val_ids = data.frame.index[data.frame.prototype_split == 'validation'].to_numpy()
    model_path = folder / 'model.joblib'
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        # All training panoramas, no validation fitting. The number of components
        # matches the prototype capacity, but NMF has no spatial reconstruction task.
        model = MiniBatchNMF(n_components=96, batch_size=256, init='random', random_state=42,
                             max_iter=8, max_no_improvement=None, tol=0., transform_max_iter=100)
        with threadpool_limits(limits=4):
            for epoch in range(8):
                cities, orders, steps = training_order(data.frame, epoch, 42, 256)
                for step in range(steps):
                    city = cities[step % len(cities)]
                    ids = orders[city][step // len(cities)]
                    model.partial_fit(np.asarray(target[ids]))
                write_json(folder / 'status.json', {'status': 'fitting', 'epoch': epoch + 1, 'epochs': 8})
                print('NMF_EPOCH', epoch + 1, flush=True)
        joblib.dump(model, folder / 'model.partial.joblib')
        (folder / 'model.partial.joblib').replace(model_path)
    stats = np.zeros((len(data.frame), 2), np.float64)
    with threadpool_limits(limits=4):
        for start in range(0, len(val_ids), 256):
            ids = val_ids[start:start + 256]
            x = np.asarray(target[ids])
            reconstructed = model.transform(x) @ model.components_
            stats[ids, 0] = np.square(reconstructed - x).sum(1)
            stats[ids, 1] = np.square(x).sum(1)
            write_json(folder / 'status.json', {'scored_panoramas': min(start + len(ids), len(val_ids)),
                                                'total_validation_panoramas': len(val_ids)})
    np.save(folder / 'statistics.npy', stats)
    write_json(folder / 'complete.json', {'components': 96, 'train_panoramas': len(train_ids),
        'validation_panoramas': len(val_ids), 'global_composition_nmse': float(stats[:, 0].sum() / stats[:, 1].sum()),
        'target': 'mean over all 3136 normalized, clipped patch activation vectors',
        'training': 'MiniBatchNMF, eight city-balanced panorama passes, no validation fitting',
        'limitation': 'NMF models panorama composition only; its vector error must not be compared to patch-map error.'})


def patch_box(y, x, scale, grid=28):
    radius = scale // 2
    return [max(0, x - radius), max(0, y - radius),
            min(grid, x + radius + 1), min(grid, y + radius + 1)]


@torch.inference_mode()
def exemplar_evidence(model, target, prototype, y, x):
    """Spatial footprint and exact increase in SSE on removing one response."""
    scale_index, local = divmod(prototype, model.per_scale)
    bank = model.banks[scale_index]
    pred, maps = model(target)
    coefficient = maps[0, prototype, y, x].float()
    # Only one template is materialized, in float32 for auditable attribution.
    template = torch.einsum('rd,ryx->dyx', bank.channels[local], bank.layout[local])
    x0, y0, x1, y1 = patch_box(y, x, bank.size)
    ty, tx = y0 - y + bank.size // 2, x0 - x + bank.size // 2
    component = coefficient * template[:, ty:ty + y1 - y0, tx:tx + x1 - x0]
    residual = target[0, :, y0:y1, x0:x1] - pred[0, :, y0:y1, x0:x1]
    removal_delta_sse = (2 * residual * component + component.square()).sum()
    attribution = np.zeros((28, 28), np.float32)
    attribution[y0:y1, x0:x1] = component.square().sum(0).sqrt().cpu().numpy()
    return attribution, {'coefficient_fp32': float(coefficient), 'box_patch': [x0, y0, x1, y1],
                         'removing_response_delta_sse': float(removal_delta_sse)}


def native_evidence(image_path, attribution, box, target_stem):
    """Read just this image. Preserve native geometry and distinguish attribution from segmentation."""
    try:
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
    except (OSError, ValueError) as exc:
        raise SourceImageError(f'{image_path}: {exc}') from exc
    width, height = image.size
    x0, y0, x1, y1 = box
    native_box = [round(x0 * width / 28), round(y0 * height / 28),
                  round(x1 * width / 28), round(y1 * height / 28)]
    image.crop(native_box).save(str(target_stem) + '_crop.jpg', quality=92)
    import matplotlib
    heat = attribution / max(float(attribution.max()), 1e-12)
    heat = np.asarray(Image.fromarray(heat).resize(image.size, Image.Resampling.BILINEAR))
    color = matplotlib.colormaps['turbo'](heat)[..., :3] * 255
    alpha = np.where(heat > 0, .25 + .5 * heat, 0)[..., None]
    overlay = Image.fromarray(np.clip(np.asarray(image) * (1 - alpha) + color * alpha, 0, 255).astype('uint8'))
    overlay.thumbnail((1120, 700))
    overlay.save(str(target_stem) + '_overlay.jpg', quality=92)
    support = attribution > max(float(attribution.max()) * .25, 1e-12)
    mask = Image.fromarray(support.astype('uint8') * 255).resize(image.size, Image.Resampling.NEAREST)
    masked = Image.composite(image, Image.new('RGB', image.size, '#7f7f7f'), mask).crop(native_box)
    masked.save(str(target_stem) + '_masked.png')
    thumb = image.copy(); thumb.thumbnail((560, 350)); thumb.save(str(target_stem) + '_original.jpg', quality=90)
    return {'native_size': [width, height], 'native_box': native_box,
            'mask': 'quarter-peak template contribution footprint; not object segmentation'}


def make_sheet(folder, rows):
    # Both dimensions are multiples of Qwen3-VL's patch_size*merge_size (32).
    panel = Image.new('RGB', (4 * 256, 2 * 288), '#dddddd')
    draw = ImageDraw.Draw(panel)
    for i, row in enumerate(rows[:8]):
        with Image.open(folder / row['crop']) as opened:
            image = ImageOps.contain(opened.convert('RGB'), (248, 248))
        x, y = (i % 4) * 256, (i // 4) * 288
        panel.paste(image, (x + (256 - image.width) // 2, y + 32 + (248 - image.height) // 2))
        draw.text((x + 4, y + 3), str(i + 1), fill='black')
    panel.save(folder / 'vlm_sheet.jpg', quality=94)


def template_figure(bank, local, folder, weights):
    """Auditable factors and a compact composition/layout figure for each template."""
    channels = bank.channels[local].detach().cpu().numpy()
    layouts = bank.layout[local].detach().cpu().numpy()
    gram = channels @ channels.T
    energy = np.maximum(np.einsum('rs,ryx,syx->yx', gram, layouts, layouts), 0) ** .5
    np.savez_compressed(folder / 'template_factors.npz', channels=channels, layouts=layouts,
                        spatial_l2_energy=energy, dimension_mass=weights)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), gridspec_kw={'width_ratios': [1, 1.4]})
    image = axes[0].imshow(energy, cmap='magma')
    axes[0].set(title=f'{bank.size}×{bank.size} template footprint', xlabel='Relative patch x', ylabel='Relative patch y')
    fig.colorbar(image, ax=axes[0], shrink=.8, label='Channel L2 norm')
    top = np.argsort(-weights)[:12][::-1]
    axes[1].barh(np.arange(len(top)), weights[top], color='#31688e')
    axes[1].set(yticks=np.arange(len(top)), yticklabels=[f'f{d}' for d in top],
                xlabel='Spatially summed template weight', title='Leading SAE dimensions')
    fig.tight_layout()
    fig.savefig(folder / 'template.png', dpi=150)
    plt.close(fig)


def gallery(data):
    model, model_hash = load_model(data.out, 'real_seed42', data.device)
    candidates = json.loads((data.out / 'evaluation/real_seed42/candidates.json').read_text())
    folder = data.out / 'gallery'
    folder.mkdir(parents=True, exist_ok=True)
    prototypes = []
    for prototype in range(96):
        sub = folder / f'p{prototype:03d}'
        sub.mkdir(exist_ok=True)
        record_path = sub / 'prototype.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['model_sha256'] != model_hash:
                raise ValueError('Exemplar checkpoint changed')
            prototypes.append(record)
            continue
        bank = model.banks[prototype // 32]
        local = prototype % 32
        weights = (bank.channels[local] * bank.layout[local].sum((1, 2))[:, None]).sum(0).detach().cpu().numpy()
        top = np.argsort(-weights)[:20]
        template_figure(bank, local, sub, weights)
        rows, image_failures = [], []
        for score, pano, heading, y, x in select_diverse(candidates[prototype], data.frame, count=40):
            frame_row = data.frame.loc[pano]
            stem = sub / f'r{len(rows) + 1:02d}'
            target = data.load([pano], [heading])
            with torch.inference_mode():
                heat, evidence = exemplar_evidence(model, target, prototype, y, x)
            if evidence['coefficient_fp32'] <= 0:
                continue
            image_path = frame_row[f'image_{heading * 90}']
            try:
                native = native_evidence(image_path, heat, evidence['box_patch'], stem)
            except SourceImageError as exc:
                image_failures.append({'pano_row': pano, 'image': image_path, 'error': str(exc)})
                continue
            np.save(stem.with_suffix('.npy'), heat)
            rows.append({'rank': len(rows) + 1, 'pano_row': pano, 'panoid': frame_row.panoid,
                'city': frame_row.city, 'heading': heading * 90, 'y': y, 'x': x,
                'source_image': image_path, 'ranking_response_bf16': score, **evidence, **native,
                'crop': stem.name + '_crop.jpg', 'overlay': stem.name + '_overlay.jpg',
                'masked': stem.name + '_masked.png', 'original': stem.name + '_original.jpg'})
            if len(rows) == 8:
                break
        record = {'prototype': prototype, 'scale': bank.size, 'status': 'candidate' if rows else 'no_positive_response',
                  'model_sha256': model_hash, 'top_dimensions': [{'dimension': int(d), 'weight': float(weights[d])} for d in top],
                  'effective_dimensions': float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12)),
                  'top20_weight_fraction': float(weights[top].sum() / max(float(weights.sum()), 1e-12)),
                  'instances': rows, 'image_failures': image_failures,
                  'scope': 'all prototype-validation panoramas, unique panorama examples; city diversity preferred'}
        if not rows and image_failures:
            record['status'] = 'source_images_unavailable'
        write_json(record_path, record)
        if rows:
            make_sheet(sub, rows)
        prototypes.append(record)
        print('GALLERY', prototype, len(rows), flush=True)
    write_json(folder / 'prototypes.json', prototypes)
    cards = []
    for p in prototypes:
        links = ''.join(f'<a href="p{p["prototype"]:03d}/{r["overlay"]}"><img loading="lazy" src="p{p["prototype"]:03d}/{r["crop"]}" title="{html.escape(r["city"])} · {r["heading"]}°"></a>' for r in p['instances'])
        cards.append(f'<section id="p{p["prototype"]}"><h2>P{p["prototype"]:03d} · {p["scale"]}×{p["scale"]} patches</h2><p>候选模板；点击裁剪查看原图上的贡献热图。不是已验证的语义类别。</p><img class="template" loading="lazy" src="p{p["prototype"]:03d}/template.png"><div>{links or "无正响应"}</div></section>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Landscape prototype candidates</title><style>body{max-width:1400px;margin:auto;font:16px system-ui;background:#f8f8f8}section{background:white;padding:20px;margin:20px 0}img{width:160px;height:130px;object-fit:contain;margin:4px}.template{width:650px;max-width:100%;height:auto}h1{margin:30px}</style><h1>多尺度景观原型候选库</h1>' + ''.join(cards))


def report(data, target):
    folder = data.out / 'evaluation'
    frames, rows = {}, []
    val_ids = data.frame.index[data.frame.prototype_split == 'validation'].to_numpy()
    for run in RUNS:
        summary = json.loads((folder / run / 'complete.json').read_text())
        frames[run] = pd.read_csv(folder / run / 'cities.csv').set_index('city')
        item = {'run': run, 'target': summary['target_distribution'], **summary['global'],
                'active_templates': summary['active_templates']}
        if RUNS[run][0] == 'real':
            pred = np.load(folder / run / 'reconstructed_composition.npy', mmap_mode='r')
            sse, energy = 0., 0.
            for start in range(0, len(val_ids), 256):
                ids = val_ids[start:start + 256]
                sse += float(np.square(pred[ids] - target[ids]).sum(dtype=np.float64))
                energy += float(np.square(target[ids]).sum(dtype=np.float64))
            item['panorama_composition_nmse'] = sse / energy
        rows.append(item)
    pd.DataFrame(rows).to_csv(folder / 'model_comparison.csv', index=False)
    effects = {}
    for run in ['spatial_shuffle_seed42', 'feature_shuffle_seed42']:
        diff = frames[run].nmse - frames['real_seed42'].nmse
        effects[run] = cluster_interval(diff.to_numpy())
    write_json(folder / 'trained_null_effects.json', {'metric': 'null-minus-real NMSE, city balanced',
        'effects': effects, 'limitation': 'Null models reconstruct shuffled targets; this tests learnable regularity under the specified shuffles.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for run in RUNS:
        training = data.out / 'models' / run / 'training.jsonl'
        logs = pd.read_json(training, lines=True)
        axes[0].plot(logs.global_step, logs.loss.rolling(8, min_periods=1).mean(), label=run, alpha=.85)
    axes[0].set(xlabel='Optimizer step', ylabel='Training objective', title='Training diagnostic')
    axes[0].legend(fontsize=7)
    axes[1].bar(np.arange(4), [r['nmse'] for r in rows], color=['#31688e', '#35b779', '#e3a73b', '#a05a90'])
    axes[1].axhline(1, ls='--', color='grey', lw=1, label='Zero reconstruction')
    axes[1].set(xticks=np.arange(4), xticklabels=['Real 42', 'Real 43', 'Spatial null', 'Pairing null'],
                ylabel='SSE / target energy', title='Full validation; native target per model')
    axes[1].tick_params(axis='x', labelsize=8)
    stable = pd.read_csv(folder / 'seed_stability.csv')
    for size in [3, 7, 15]:
        axes[2].hist(stable[(stable.scale == size) & stable.both_observed].template_cosine,
                     bins=np.linspace(0, 1, 16), alpha=.5, label=f'{size}×{size}')
    axes[2].set(xlabel='Matched template cosine', ylabel='Candidate count', title='Seed 42 ↔ seed 43')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(folder / 'prototype_diagnostics.png', dpi=180)
    fig.savefig(folder / 'prototype_diagnostics.pdf')
    plt.close(fig)
    nmf = json.loads((folder / 'nmf/complete.json').read_text())
    text = f'''# Multi-scale prototype experiment

Corpus: {data.spec['panoramas']:,} panoramas in {len(data.spec['cities'])} cities.
Prototype train/validation: {data.spec['train_panoramas']:,}/{data.spec['validation_panoramas']:,}.
All four views of every validation panorama were scored. This split validates the prototype layer;
the frozen SAE was previously fitted using some of these panoramas.

See model_comparison.csv for exact reconstruction ratios and trained_null_effects.json for city-bootstrap effects.
NMF panorama-vector NMSE: {nmf['global_composition_nmse']:.6f}; compare only against the
panorama_composition_nmse column, never against spatial patch-map errors.
Permutation-test evidence, its restricted randomization scope, and attainable p-value resolution are in permutations/summary.json.
All 96 templates remain candidates. Inspect ../gallery/index.html, signed response-removal errors,
seed_stability.csv, and subsequent Qwen descriptions before assigning landscape meanings.
'''
    (folder / 'README.md').write_text(text)


def evaluate(out=OUT, device='cuda:1', batch_size=8, null_repeats=39):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    runtime = configure_device(device)
    data = SparseViews(out, device)
    folder = Path(out) / 'evaluation'
    folder.mkdir(exist_ok=True)
    write_json(folder / 'runtime.json', runtime)
    settings = {'code_sha256': digest(Path(__file__)), 'corpus_identity': data.spec['identity'],
                'batch_size': batch_size, 'null_repeats': null_repeats}
    if (folder / 'settings.json').exists() and json.loads((folder / 'settings.json').read_text()) != settings:
        raise ValueError('Evaluation settings/code changed; use a new evaluation directory')
    write_json(folder / 'settings.json', settings)
    for run in RUNS:
        score_model(data, run, batch_size)
    match_seeds(data)
    gallery(data)
    target = scaled_composition(data)
    nmf_baseline(data, target)
    permutation_controls(data, batch_size, null_repeats)
    report(data, target)
    write_json(folder / 'complete.json', settings)
