"""Discover and audit panorama-level compositions of SAE dimensions."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from formal.analysis.semantic_feature_umap import digest, write_json
from sae_experiments.panorama_vitl30 import OUTPUT, WIDTH
from sae_experiments.panorama_vitl30_analysis import checked_groups

SOURCE = OUTPUT / 'analysis/semantic_umap_top100'
REPORT = Path('/root/code/urban-visual-gene/reports/vitl30_landscape_prototypes')
RANKS = (16, 32, 64)
# Manual review of the four-heading exemplars for this fixed checkpoint/seed.
VISUAL_QC_FLAGS = {24: 'underpass/tunnel examples remain after the image-quality gate'}


def load_peaks(source: Path):
    run = json.loads((source / 'run.json').read_text())
    rows = [json.loads(line) for line in (source / 'panoramas.jsonl').open()]
    if len(rows) != run['panoramas'] or run['width'] != WIDTH:
        raise ValueError('Panorama manifest and SAE width differ from run')
    values = np.zeros((len(rows), WIDTH), np.float32)
    for city in run['cities']:
        city_rows = [row for row in rows if row['city'] == city]
        path = source / 'ranking' / f'{city.replace("/", "__")}.npz'
        with np.load(path) as data:
            if data['peaks'].shape != (len(city_rows), WIDTH) or data['ids'].tolist() != [r['cache_pano_index'] for r in city_rows]:
                raise ValueError(f'Ranking cache does not match {city} manifest')
            values[[r['pano_index'] for r in city_rows]] = data['peaks']
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Ranking cache has invalid activation peaks')
    return rows, values, run


def fit_scale(train: np.ndarray, min_support: int):
    support = np.count_nonzero(train > 0, axis=0)
    keep = support >= min_support
    if not keep.any():
        raise ValueError('No dimensions meet the support threshold')
    # Positive-activation p95 reduces domination by a few high-amplitude dimensions.
    cap = np.array([np.quantile(train[train[:, j] > 0, j], .95) for j in np.flatnonzero(keep)], np.float32)
    return np.flatnonzero(keep), np.maximum(cap, 1e-6)


def transform(peaks: np.ndarray, columns: np.ndarray, cap: np.ndarray):
    scaled = np.sqrt(np.clip(peaks[:, columns] / cap, 0, 1))
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.asarray(scaled / np.maximum(norm, 1e-12), dtype=np.float32)


def score(model, x):
    weights = model.transform(x)
    reconstructed = weights @ model.components_
    return float(1 - np.sum((x - reconstructed) ** 2) / max(float(np.sum(x ** 2)), 1e-12))


def fit(x, rank, seed, max_iter, init='nndsvda'):
    model = NMF(n_components=rank, init=init, solver='cd', random_state=seed,
                max_iter=max_iter, tol=.002, alpha_W=0, alpha_H=0)
    weights = model.fit_transform(x)
    return model, weights


def shuffled_within_city(peaks, city_ids, rng):
    """Keep each city's per-feature prevalence; remove panorama-level pairing."""
    shuffled = peaks.copy()
    for city in np.unique(city_ids):
        ix = np.flatnonzero(city_ids == city)
        order = rng.random((len(ix), peaks.shape[1])).argsort(axis=0)
        shuffled[ix] = peaks[ix[order], np.arange(peaks.shape[1])]
    return shuffled


def city_folds(rows, folds, seed):
    city_names = sorted({r['city'] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(city_names)
    for held_cities in np.array_split(city_names, folds):
        held = set(held_cities)
        test = np.array([i for i, r in enumerate(rows) if r['city'] in held])
        train = np.array([i for i, r in enumerate(rows) if r['city'] not in held])
        yield train, test, sorted(held)


def evaluate(peaks, rows, ranks, folds, min_support, seed, max_iter, null_repeats):
    cities = np.array([r['city'] for r in rows])
    comparisons, null_details = [], []
    for fold, (train, test, held) in enumerate(city_folds(rows, folds, seed), 1):
        columns, cap = fit_scale(peaks[train], min_support)
        x_train = transform(peaks[train], columns, cap)
        x_test = transform(peaks[test], columns, cap)
        for rank in ranks:
            model, _ = fit(x_train, rank, seed, max_iter)
            real = score(model, x_test)
            comparisons.append({'fold': fold, 'rank': rank, 'heldout_cities': held,
                                'heldout_panoramas': len(test), 'dimensions': len(columns),
                                'real_explained_fraction': real, 'iterations': model.n_iter_})
            print('FOLD', fold, 'RANK', rank, 'explained', round(real, 4), flush=True)
            if rank == 32:
                rng = np.random.default_rng(seed + 1009 * fold)
                null_scores = []
                for repeat in range(null_repeats):
                    shuffled = shuffled_within_city(peaks[test, :][:, columns], cities[test], rng)
                    null_scores.append(score(model, transform(shuffled, np.arange(len(columns)), cap)))
                null_details.append({'fold': fold, 'real': real, 'null_scores': null_scores,
                                     'mean_null': float(np.mean(null_scores)),
                                     'permutation_p': (1 + int(np.sum(np.asarray(null_scores) >= real))) / (null_repeats + 1)})
                print('FOLD', fold, 'NULL', round(float(np.mean(null_scores)), 4), flush=True)
    return comparisons, null_details


def component_similarity(first, other):
    left = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    right = other / np.maximum(np.linalg.norm(other, axis=1, keepdims=True), 1e-12)
    pair = left @ right.T
    indices, matches = linear_sum_assignment(-pair)
    return pair[indices, matches]


def exemplar_ids(weights, rows, count=4):
    """Prefer dominant-component scenes from different cities."""
    share = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    dominant = share.argmax(axis=1)
    examples = []
    for component in range(weights.shape[1]):
        order = np.argsort(-share[:, component])
        chosen, cities = [], set()
        for require_dominant in (True, False):
            for i in order:
                if (require_dominant and dominant[i] != component) or rows[i]['city'] in cities:
                    continue
                chosen.append(int(i));cities.add(rows[i]['city'])
                if len(chosen) == count:
                    break
            if len(chosen) == count:
                break
        examples.append(chosen)
    return examples, dominant, share


def panorama_images(source, row):
    # Source image IDs are four consecutive headings per panorama.
    index = int(row['pano_index'])
    images = []
    for heading in range(4):
        path = source / 'sources' / f'{index * 4 + heading}.jpg'
        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            images.append(image.convert('RGB').resize((240, 240), Image.Resampling.LANCZOS))
    panorama = Image.new('RGB', (960, 240))
    for heading, image in enumerate(images):
        panorama.paste(image, (240 * heading, 0))
    return panorama


def write_gallery(source, report, rows, examples, share, dominant, labels, top_features, counts, statuses):
    figures = report / 'panoramas'
    figures.mkdir(parents=True, exist_ok=True)
    cards = []
    for component, ids in enumerate(examples):
        montage = Image.new('RGB', (960, max(1, len(ids)) * 265), 'white')
        draw = ImageDraw.Draw(montage)
        for line, i in enumerate(ids):
            montage.paste(panorama_images(source, rows[i]), (0, line * 265))
            kind = 'dominant' if dominant[i] == component else 'secondary'
            draw.text((8, line * 265 + 242), f'{rows[i]["city"]}  |  share {share[i, component]:.3f}  |  {kind}', fill='#26333d')
        path = figures / f'prototype_{component:02d}.jpg'
        montage.save(path, quality=91)
        links = ', '.join(f'<a target="_blank" rel="noopener" href="{html.escape(str((source / "features" / f"{j}.html").resolve()))}">'
                          f'f{j} / G{labels[j]:02d}</a>' for j in top_features[component][:8])
        article_class = 'candidate' if statuses[component] == 'supported candidate' else 'limited'
        cards.append(f'<article class="{article_class}" id="p{component}"><h2>Component {component:02d} '
                     f'<small>{counts[component]} dominant panoramas · {html.escape(statuses[component])}</small></h2>'
                     f'<p>Strongest distinctive SAE dimensions: {links}</p>'
                     f'<img loading="lazy" src="panoramas/{path.name}" alt="Four-heading panorama exemplars for prototype {component}"></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Landscape prototypes</title><style>body{margin:0;color:#1c2930;background:#f5f7f7;font:14px system-ui,sans-serif}header{padding:20px 25px;background:white;border-bottom:1px solid #d7dede}h1{margin:0;font-size:24px}p{line-height:1.5}main{max-width:1150px;margin:auto;padding:15px 20px}article{padding:20px 0;border-bottom:1px solid #cbd6d9}h2{font-size:19px}small{font-size:12px;color:#60707a}img{display:block;width:100%;height:auto}a{color:#176c83}input,button{padding:8px;border:1px solid #abbac2;background:#fff;color:#1c2930;border-radius:4px;font:inherit}input{width:95px}button{cursor:pointer}button[aria-pressed=true]{background:#164f56;color:white}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}body.filtered article.limited{display:none}article:target{outline:2px solid #164f56;outline-offset:5px}@media(max-width:600px){header{padding:15px}main{padding:10px}}</style>'
    page += f'<header><h1>Landscape prototype candidates</h1><p>Each component is a weighted composition of SAE dimensions across a complete four-heading panorama. Each row is one real panorama, ordered 0°, 90°, 180°, 270°. Secondary rows are weaker evidence. {statuses.count("supported candidate")} components currently meet the support and QC review criteria.</p><div class="controls"><button id="candidates" aria-pressed="true">Supported candidates</button><button id="all" aria-pressed="false">All 32 components</button><label for="search">Component</label><input id="search" type="number" min="0" max="31" placeholder="0–31"><button id="find">Find</button></div></header><main>'
    page += ''.join(cards) + '</main><script>const show=only=>{document.body.classList.toggle("filtered",only);document.querySelector("#candidates").setAttribute("aria-pressed",only);document.querySelector("#all").setAttribute("aria-pressed",!only)};document.querySelector("#candidates").onclick=()=>show(true);document.querySelector("#all").onclick=()=>show(false);document.querySelector("#find").onclick=()=>{const el=document.querySelector("#p"+Number(document.querySelector("#search").value));if(el){show(false);el.scrollIntoView();location.hash=el.id}};document.querySelector("#search").addEventListener("keydown",e=>{if(e.key==="Enter")document.querySelector("#find").click()});show(!location.hash)</script></html>'
    (report / 'index.html').write_text(page)
    included = [i for i, status in enumerate(statuses) if status == 'supported candidate']
    overview = Image.new('RGB', (2000, 160 * ((len(included) + 3) // 4)), 'white')
    painter = ImageDraw.Draw(overview)
    for position, component in enumerate(included):
        with Image.open(figures / f'prototype_{component:02d}.jpg') as montage:
            preview = montage.crop((0, 0, 960, 240)).resize((480, 120), Image.Resampling.LANCZOS)
        x, y = position % 4 * 500, position // 4 * 160
        overview.paste(preview, (x, y + 20))
        painter.text((x + 4, y + 2), f'P{component:02d} | {counts[component]} panoramas', fill='#26333d')
    overview.save(report / 'Fig_ViTL30_Landscape_Prototype_Overview.jpg', quality=92)


def write_report(report, summary):
    cv = summary['heldout_by_city']
    null = summary['null']
    stability = summary['stability']
    status = summary['component_status']
    report.joinpath('REPORT.md').write_text('# Panorama landscape prototypes\n\n'
        f'基于 {summary["panoramas"]:,} 个验证全景、{summary["cities"]} 城的 4,096 维 SAE 最大正激活，'
        f'保留至少在 {summary["min_positive_panoramas_for_dimension"]} 个全景出现的 {summary["dimensions_modeled"]:,} 维。'
        '使用每维正激活第 95 百分位截断、平方根变换与逐全景 L2 归一化。'
        '主实验用 NMF 分解成 32 个可叠加的景观组合，16 和 64 作为敏感性分析。\n\n'
        f'城市留出 {len(null["fold_details"])} 折的 32 原型平均解释率为 {cv["32"]["mean_explained_fraction"]:.3f}；'
        f'城市内独立打乱每维激活后，基线为 {null["mean_null_explained_fraction"]:.3f}，'
        f'差为 {null["mean_difference"]:.3f}，置换检验 p={null["pooled_permutation_p"]:.4f}（{null["repeats"]} 次）。'
        f'两次独立随机初始化原型匹配余弦均值为 {stability["mean_cosine"]:.3f}。'
        f'按至少 20 个主导全景、5 座城市及匹配余弦不低于 0.8，'
        f'有 {status.count("supported candidate")} 个成分达到候选原型门槛并通过当前目视 QC 复核。'
        '成分 24 的代表图中仍有隧道／下穿通道，已标记为 QC 待复核，不计入候选数。\n\n'
        '图中每个原型的行是一个真实四方向全景，特征编号链接到原始激活证据。'
        'NMF 只证明维度组合的重复性，不检验相邻关系或固定空间排布。'
        '这些编号是候选原型，尚未做逐个语义命名。\n\n'
        '[查看原型与四方向实例](index.html) · [原型表](prototypes.csv) · '
        '[逐全景分配](panorama_assignments.csv) · [完整统计](summary.json)\n\n'
        '![Supported landscape prototype overview](Fig_ViTL30_Landscape_Prototype_Overview.jpg)\n\n'
        '![Cross-city rank comparison](heldout_rank_comparison.png)\n')


def render_saved(args):
    summary_path = args.report / 'summary.json'
    summary = json.loads(summary_path.read_text())
    rows, _, run = load_peaks(args.source)
    if summary['source_checkpoint_sha256'] != run['checkpoint_sha256']:
        raise ValueError('Saved prototypes and SAE checkpoint differ')
    labels = checked_groups(args.source.parent.parent, run['checkpoint_sha256'])
    with np.load(args.report / 'nmf_w32.npz') as data:
        weights, columns, components = data['weights'], data['columns'], data['components']
    if weights.shape != (len(rows), 32) or components.shape != (32, len(columns)):
        raise ValueError('Saved NMF factors do not match panorama manifest')
    full = np.zeros((32, WIDTH), np.float32)
    full[:, columns] = components
    specificity = full / np.maximum(full.sum(axis=0, keepdims=True), 1e-12)
    top_features = np.argsort(-(full * specificity), axis=1)[:, :12]
    examples, dominant, share = exemplar_ids(weights, rows)
    support = np.bincount(dominant, minlength=32)
    cities = np.array([len({rows[i]['city'] for i in np.flatnonzero(dominant == j)}) for j in range(32)])
    stability = np.asarray(summary['stability']['per_final_component'])
    statuses = ['QC review required' if j in VISUAL_QC_FLAGS else
                'supported candidate' if support[j] >= 20 and cities[j] >= 5 and stability[j] >= .8
                else 'limited evidence' for j in range(32)]
    with (args.report / 'prototypes.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['component', 'status', 'dominant_panoramas', 'dominant_cities',
                                                 'random_start_matched_cosine', 'top_features', 'top_groups', 'example_panoramas'])
        for j in range(32):
            writer.writerow([j, statuses[j], int(support[j]), int(cities[j]), float(stability[j]),
                             ' '.join(map(str, top_features[j])), ' '.join(map(str, labels[top_features[j]])),
                             ' '.join(rows[i]['city'] + '/' + rows[i]['pano'] for i in examples[j])])
    write_gallery(args.source, args.report, rows, examples, share, dominant, labels, top_features, support, statuses)
    summary['component_status'] = statuses
    summary['manual_visual_qc_flags'] = VISUAL_QC_FLAGS
    write_json(summary_path, summary)
    write_report(args.report, summary)
    print(f'Rendered {statuses.count("supported candidate")} supported candidates from saved factors', flush=True)


def run(args):
    args.report.mkdir(parents=True, exist_ok=True)
    rows, peaks, source_run = load_peaks(args.source)
    if digest(args.source.parent.parent / 'model_best.pt') != source_run['checkpoint_sha256']:
        raise ValueError('SAE checkpoint differs from saved ranking')
    labels = checked_groups(args.source.parent.parent, source_run['checkpoint_sha256'])
    comparisons, null = evaluate(peaks, rows, RANKS, args.folds, args.min_support,
                                 args.seed, args.max_iter, args.null_repeats)
    cols, cap = fit_scale(peaks, args.min_support)
    x = transform(peaks, cols, cap)
    components, stability = {}, {}
    for rank in RANKS:
        model, weights = fit(x, rank, args.seed, args.max_iter)
        components[rank] = (model, weights)
        np.savez_compressed(args.report / f'nmf_w{rank}.npz', weights=weights.astype(np.float32),
                            components=model.components_.astype(np.float32), columns=cols)
        if rank == 32:
            first_random, _ = fit(x, rank, args.seed + 1, args.max_iter, init='random')
            second_random, _ = fit(x, rank, args.seed + 2, args.max_iter, init='random')
            similarities = component_similarity(first_random.components_, second_random.components_)
            stability = {'matched_cosines': similarities.tolist(),
                         'mean_cosine': float(similarities.mean()),
                         'min_cosine': float(similarities.min()),
                         'seeds': [args.seed + 1, args.seed + 2],
                         'initialization': 'two independently randomized NMF starts'}
    model, weights = components[32]
    full = np.zeros((32, WIDTH), np.float32)
    full[:, cols] = model.components_
    # Distinctive features discount dimensions appearing in every component.
    specificity = full / np.maximum(full.sum(axis=0, keepdims=True), 1e-12)
    distinct = full * specificity
    top_features = np.argsort(-distinct, axis=1)[:, :12]
    examples, dominant, share = exemplar_ids(weights, rows)
    support = np.bincount(dominant, minlength=32)
    city_counts = np.array([len({rows[i]['city'] for i in np.flatnonzero(dominant == j)}) for j in range(32)])
    # Align randomized bases to the final deterministic basis before assigning per-component stability.
    random_basis = first_random.components_
    left = model.components_ / np.maximum(np.linalg.norm(model.components_, axis=1, keepdims=True), 1e-12)
    right = random_basis / np.maximum(np.linalg.norm(random_basis, axis=1, keepdims=True), 1e-12)
    index, match = linear_sum_assignment(-(left @ right.T))
    stability_by_component = np.zeros(32, np.float32)
    stability_by_component[index] = similarities[match]
    statuses = ['QC review required' if i in VISUAL_QC_FLAGS else
                'supported candidate' if support[i] >= 20 and city_counts[i] >= 5 and stability_by_component[i] >= .8
                else 'limited evidence' for i in range(32)]
    with (args.report / 'prototypes.csv').open('w', newline='') as f:
        writer = csv.writer(f);writer.writerow(['component', 'status', 'dominant_panoramas', 'dominant_cities',
                                                'random_start_matched_cosine', 'top_features', 'top_groups', 'example_panoramas'])
        for i in range(32):
            writer.writerow([i, statuses[i], int(support[i]), int(city_counts[i]), float(stability_by_component[i]),
                             ' '.join(map(str, top_features[i])),
                             ' '.join(map(str, labels[top_features[i]])),
                             ' '.join(rows[j]['city'] + '/' + rows[j]['pano'] for j in examples[i])])
    with (args.report / 'panorama_assignments.csv').open('w', newline='') as f:
        writer = csv.writer(f);writer.writerow(['city','pano','prototype','share','cache_pano_index'])
        for i, row in enumerate(rows):
            writer.writerow([row['city'],row['pano'],int(dominant[i]),float(share[i,dominant[i]]),row['cache_pano_index']])
    write_gallery(args.source, args.report, rows, examples, share, dominant, labels, top_features, support, statuses)
    cv = {str(rank): {'mean_explained_fraction': float(np.mean([r['real_explained_fraction'] for r in comparisons if r['rank']==rank])),
                       'fold_scores': [r['real_explained_fraction'] for r in comparisons if r['rank']==rank]} for rank in RANKS}
    pooled = float(np.mean([r['real'] - r['mean_null'] for r in null]))
    combined = np.mean([[r['null_scores'][j] for r in null] for j in range(args.null_repeats)], axis=1)
    observed = float(np.mean([r['real'] for r in null]))
    permutation_p = (1 + int(np.sum(combined >= observed))) / (args.null_repeats + 1)
    summary = {'source_checkpoint_sha256': source_run['checkpoint_sha256'],
               'ranking_manifest_sha256': digest(args.source / 'panoramas.jsonl'),
               'panoramas': len(rows), 'cities': len({r['city'] for r in rows}),
               'dimensions_total': WIDTH, 'dimensions_modeled': len(cols),
               'min_positive_panoramas_for_dimension': args.min_support,
               'main_prototypes': 32, 'sensitivity_ranks': list(RANKS),
               'heldout_by_city': cv, 'fold_details': comparisons,
               'null': {'method': 'independently permute each feature within each held-out city',
                        'repeats': args.null_repeats, 'fold_details': null,
                        'mean_observed_explained_fraction': observed,
                        'mean_null_explained_fraction': float(np.mean(combined)),
                        'mean_difference': pooled, 'pooled_permutation_p': permutation_p},
               'stability': {**stability, 'per_final_component': stability_by_component.tolist()},
               'dominant_panorama_counts': support.tolist(), 'dominant_city_counts': city_counts.tolist(),
               'component_status': statuses,
               'manual_visual_qc_flags': VISUAL_QC_FLAGS,
               'method': 'sqrt of p95-capped positive panorama maxima, row L2 norm, nonnegative matrix factorization',
               'limitation': 'Maxima capture feature co-presence, not angular or spatial arrangement; prototypes require visual review'}
    write_json(args.report / 'summary.json', summary)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([str(rank) for rank in RANKS], [cv[str(rank)]['mean_explained_fraction'] for rank in RANKS], color=['#417d72','#d06c42','#586ca6'])
    ax.set(xlabel='NMF prototypes', ylabel='Explained fraction on held-out cities', title='Cross-city reconstruction of panorama feature compositions')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(args.report / 'heldout_rank_comparison.png',dpi=180);plt.close(fig)
    write_report(args.report, summary)
    print(json.dumps({'panoramas':len(rows),'modeled_dimensions':len(cols),
                      'cv':cv,'null_difference':pooled,'permutation_p':permutation_p,
                      'stability_mean_cosine':stability['mean_cosine']},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--min-support', type=int, default=20)
    parser.add_argument('--null-repeats', type=int, default=39)
    parser.add_argument('--max-iter', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', choices=['all', 'render'], default='all')
    args = parser.parse_args()
    if args.folds < 2 or args.min_support < 1 or args.null_repeats < 1 or args.max_iter < 10:
        parser.error('Invalid experiment settings')
    if args.stage == 'render':
        render_saved(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
