"""Project individual top-activation image crops of the ViT-L BatchTopK SAE."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from formal.analysis.semantic_feature_umap import digest, write_json, write_jsonl
from sae_experiments.panorama_vitl30 import OUTPUT, WIDTH
from sae_experiments.panorama_vitl30_analysis import checked_groups
from sae_experiments.panorama_vitl30_semantic_umap_report import group_palette

SOURCE = OUTPUT / 'analysis/semantic_umap_top100'
REPORT = Path('/root/code/urban-visual-gene/reports/vitl30_instance_umap')


def select_instances(source: Path, top: int, width: int):
    """Keep true activation ranks; preserve their row indices in the embedding file."""
    selected, count = [], np.zeros(width, np.int32)
    total = 0
    seen = set()
    with (source / 'crops.jsonl').open() as manifest:
        for index, line in enumerate(manifest):
            total = index + 1
            row = json.loads(line)
            feature, rank = int(row['feature']), int(row['rank'])
            if feature < 0 or feature >= width or rank < 1:
                raise ValueError(f'Invalid feature/rank at crop row {index}')
            if rank > top:
                continue
            key = (feature, row['city'], row['pano'])
            if key in seen or rank != count[feature] + 1 or row['activation'] <= 0:
                raise ValueError(f'Non-distinct or nonconsecutive positive examples for f{feature}')
            seen.add(key)
            count[feature] += 1
            selected.append({'embedding_row': index, 'feature': feature, 'rank': rank,
                             'activation': float(row['activation']), 'city': row['city'],
                             'pano': row['pano'], 'heading': int(row['heading']),
                             'image_id': int(row['image_id']), 'crop': row['crop']})
    return selected, count, total


def prepare(args):
    source, report = args.source, args.report
    report.mkdir(parents=True, exist_ok=True)
    run = json.loads((source / 'run.json').read_text())
    embedding_run = json.loads((source / 'embedding_run.json').read_text())
    if run['width'] != WIDTH or embedding_run['crop_manifest_sha256'] != digest(source / 'crops.jsonl'):
        raise ValueError('Crop embeddings do not match the current SAE crop manifest')
    selected, counts, total = select_instances(source, args.top, WIDTH)
    embeddings = np.load(source / 'crop_embeddings.f32.npy', mmap_mode='r')
    if len(embeddings) != total or embedding_run['rows'] != total:
        raise ValueError('Embedding rows do not match crop manifest')
    if not selected:
        raise ValueError('No positive activation instances')
    coords_input = np.asarray(embeddings[[row['embedding_row'] for row in selected]], dtype=np.float32)
    norms = np.linalg.norm(coords_input, axis=1)
    if not np.isfinite(coords_input).all() or np.any(norms < .99) or np.any(norms > 1.01):
        raise ValueError('Expected finite unit-normalized SigLIP crop embeddings')
    prepared = {'checkpoint_sha256': run['checkpoint_sha256'],
        'crop_manifest_sha256': embedding_run['crop_manifest_sha256'],
        'encoder': embedding_run['model'], 'encoder_dimension': int(coords_input.shape[1]),
        'top_per_dimension': args.top, 'dimensions': WIDTH,
        'dimensions_with_instances': int(np.count_nonzero(counts)),
        'dimensions_without_instances': int(np.count_nonzero(counts == 0)),
        'dimensions_with_full_top': int(np.count_nonzero(counts == args.top)),
        'instances': len(selected), 'source_panoramas': run['panoramas'],
        'source_images': run['images'], 'source_cities': len(run['cities']),
        'selection': 'highest positive sparse activation per distinct panorama; best heading only',
        'input': 'individual L2-normalized SigLIP embeddings of clean activation crops',
        'umap': {'metric': 'cosine', 'n_neighbors': args.neighbors,
                 'min_dist': args.min_dist, 'random_state': args.seed},
        'limitations': ['Top ranks are within the fixed held-out sample, not the whole NAS',
                        'Dimensions with no positive activation have no image-semantic coordinate',
                        'UMAP distances are descriptive, not statistical significance']}
    state_path = report / 'run.json'
    if state_path.exists() and json.loads(state_path.read_text()) != prepared:
        raise ValueError('Prepared run differs; use a new report directory')
    np.save(report / 'instance_embeddings.f32.npy', coords_input)
    write_jsonl(report / 'instances.jsonl', selected)
    write_json(state_path, prepared)
    print(f'Prepared {len(selected):,} individual images from {np.count_nonzero(counts):,} dimensions', flush=True)


def reduce(args):
    import umap

    report = args.report
    settings = json.loads((report / 'run.json').read_text())['umap']
    if settings != {'metric': 'cosine', 'n_neighbors': args.neighbors,
                    'min_dist': args.min_dist, 'random_state': args.seed}:
        raise ValueError('UMAP settings differ from prepared run')
    embeddings = np.load(report / 'instance_embeddings.f32.npy', mmap_mode='r')
    reducer = umap.UMAP(metric='cosine', n_neighbors=args.neighbors,
                        min_dist=args.min_dist, random_state=args.seed, n_jobs=1,
                        init='random', low_memory=True)
    xy = reducer.fit_transform(np.asarray(embeddings))
    if xy.shape != (len(embeddings), 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid image-instance UMAP coordinates')
    np.savez_compressed(report / 'instance_umap.npz', xy=xy.astype(np.float32))
    print(f'Projected {len(xy):,} image instances', flush=True)


def render(args):
    source, report = args.source, args.report
    run = json.loads((report / 'run.json').read_text())
    source_run = json.loads((source / 'run.json').read_text())
    if run['checkpoint_sha256'] != source_run['checkpoint_sha256'] or run['crop_manifest_sha256'] != digest(source / 'crops.jsonl'):
        raise ValueError('Source data changed since instance selection')
    labels = checked_groups(source.parent.parent, run['checkpoint_sha256'])
    rows = [json.loads(line) for line in (report / 'instances.jsonl').open()]
    with np.load(report / 'instance_umap.npz') as data:
        xy = data['xy']
    if xy.shape != (len(rows), 2) or not np.isfinite(xy).all():
        raise ValueError('Coordinates and image instances do not align')
    palette = group_palette()
    features = np.array([row['feature'] for row in rows], dtype=np.int32)
    colors = palette[labels[features]] / 255
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c=colors, alpha=.55, linewidths=0, rasterized=True)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2', title='Individual top-activation image crops')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.09, .025, f'{len(rows):,} image instances | {run["dimensions_with_instances"]:,}/{WIDTH:,} SAE dimensions '
             f'| {run["source_panoramas"]:,} held-out panoramas\n'
             'Top 10 positive activations per dimension where available; SigLIP image embeddings, cosine UMAP. '
             'Color: 64 decoder groups.', fontsize=9, color='#536273')
    fig.subplots_adjust(left=.09, right=.98, top=.91, bottom=.12)
    fig.savefig(report / 'Fig_ViTL30_Top10_Image_Instance_UMAP.png', dpi=200)
    fig.savefig(report / 'Fig_ViTL30_Top10_Image_Instance_UMAP.pdf')
    plt.close(fig)
    with (report / 'instance_umap.csv').open('w', newline='') as dest:
        writer = csv.writer(dest)
        writer.writerow(['feature', 'rank', 'group', 'umap_1', 'umap_2', 'city', 'pano',
                         'heading', 'activation_peak', 'crop'])
        for row, point in zip(rows, xy):
            writer.writerow([row['feature'], row['rank'], int(labels[row['feature']]),
                             float(point[0]), float(point[1]), row['city'], row['pano'],
                             row['heading'], row['activation'], row['crop']])
    relative = os.path.relpath(source, report).replace(os.sep, '/')
    cities = sorted({row['city'] for row in rows})
    city_index = {city: i for i, city in enumerate(cities)}
    points = [[round(float(pos[0]), 5), round(float(pos[1]), 5), row['feature'], row['rank'],
               city_index[row['city']], row['image_id'], row['crop'], round(row['activation'], 5),
               row['heading']]
              for row, pos in zip(rows, xy)]
    payload = {'points': points, 'cities': cities, 'colors': palette.tolist(),
               'groups': labels.tolist(), 'source': relative, 'top': run['top_per_dimension'],
               'dimensions': WIDTH, 'absent': run['dimensions_without_instances']}
    template = Path(__file__).with_name('panorama_vitl30_instance_umap.html').read_text()
    (report / 'index.html').write_text(template.replace('__DATA__', json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')))
    write_json(report / 'summary.json', {**run, 'coordinate_sha256': digest(report / 'instance_umap.npz'),
               'group_labels_sha256': digest(source.parent / 'decoder_groups.npy'),
               'instance_csv': 'instance_umap.csv', 'interactive_page': 'index.html'})
    (report / 'REPORT.md').write_text(
        '# Top10 image-instance semantic UMAP\n\n'
        f'每个散点是一张图像中的一个高激活裁剪实例，而不是一个 SAE 维度。每维按不同全景的最大正激活排名，'
        f'最多取前 {run["top_per_dimension"]} 名：{len(rows):,} 个点来自 {run["dimensions_with_instances"]:,} 个维度。'
        f'其中 {run["dimensions_with_full_top"]:,} 维有完整 Top10；'
        f'{run["dimensions_without_instances"]:,} 维在当前 {run["source_panoramas"]:,} 个验证全景里没有正激活，无法产生真实图像点。\n\n'
        '每个裁剪使用已保存的 SigLIP 图像 embedding，直接以余弦距离重新拟合 UMAP。'
        '颜色沿用 64 个 decoder 聚类组，组标签和城市不参与拟合。'
        '同一图像可能为多个 SAE 维度提供不同的激活区域；每个维度与全景组合最多保留一个方向。'
        '激活排名只针对固定验证样本，不代表全量 NAS。UMAP 二维距离不构成统计显著性证据。\n\n'
        '[交互式实例图](index.html) · [逐点坐标与来源](instance_umap.csv)\n\n'
        '![Top10 image-instance semantic UMAP](Fig_ViTL30_Top10_Image_Instance_UMAP.png)\n')
    print(f'Wrote interactive instance atlas to {report / "index.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--stage', choices=['all', 'prepare', 'reduce', 'render'], default='all')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--neighbors', type=int, default=30)
    parser.add_argument('--min-dist', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.top < 1 or args.neighbors < 2:
        parser.error('top must be positive and neighbors at least two')
    for name, function in [('prepare', prepare), ('reduce', reduce), ('render', render)]:
        if args.stage in ('all', name):
            function(args)


if __name__ == '__main__':
    main()
