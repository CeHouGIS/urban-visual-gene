#!/usr/bin/env python3
"""Publish existing prototype coverage results without rebuilding image evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'


def load_results(root=ROOT):
    experiment = root / 'experiments/vitl30_patch_coverage'
    report = json.loads((experiment / 'results/summary.json').read_text())
    refit = json.loads((experiment / 'results/response_refit_summary.json').read_text())
    protocol = report['protocol']
    if protocol['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Prototype results use a different SAE')
    if refit['protocol']['experiment_identity'] != protocol['identity']:
        raise ValueError('Response refit uses a different coverage experiment')
    model_hashes = {part['model_sha256'] for part in report['partitions'].values()}
    if model_hashes != {refit['protocol']['model_sha256']}:
        raise ValueError('Response refit and full evaluation use different templates')
    if report['test_used'] or refit['test_used'] or set(report['partitions']) != {'train', 'dev'}:
        raise ValueError('Only the train/dev results are approved for this page')
    if refit['status'] != 'complete':
        raise ValueError('Response diagnostic is not complete')
    return experiment, report, refit


def fig(filename, caption):
    path = f'../assets/prototypes/{html.escape(filename, quote=True)}'
    return (f'<figure><a href="{path}"><img loading="lazy" src="{path}" '
            f'alt="{html.escape(caption, quote=True)}"></a>'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def install(site, render_page, snapshot):
    try:
        from .prototype_gallery import install as install_gallery
    except ImportError:
        from prototype_gallery import install as install_gallery
    experiment, report, refit = load_results()
    results = experiment / 'results'
    target = site / 'prototypes'
    target.mkdir(parents=True, exist_ok=True)
    images = ['per_image_patch_coverage.png', 'response_refit_diagnostic.png']
    images += [item['figure'] for item in report['evidence']]
    assets = site / 'assets/prototypes'
    assets.mkdir(parents=True, exist_ok=True)
    for name in images:
        shutil.copyfile(results / name, assets / name)

    protocol = report['protocol']
    gallery_content, gallery = install_gallery(site, experiment, protocol, refit['protocol']['model_sha256'])
    public = {
        'experiment': 'vitl30_patch_coverage', 'checkpoint_sha256': CHECKPOINT,
        'protocol': {k: protocol[k] for k in ['identity', 'split_counts', 'cities',
                     'policy', 'model', 'epochs', 'batch_size', 'seed', 'lr']},
        'partitions': report['partitions'], 'evidence': report['evidence'],
        'response_refit': {k: refit[k] for k in ['status', 'summaries', 'changes',
                           'elapsed_seconds', 'interpretation', 'test_used']},
        'response_refit_protocol': {k: refit['protocol'][k] for k in ['identity',
            'experiment_identity', 'model_sha256', 'partition', 'per_city',
            'steps', 'lr', 'no_sparsity_penalty', 'test_sealed']},
        'test_used': False, 'semantic_review': 'pending', 'published_at': snapshot,
    }
    data = site / 'data'
    data.mkdir(exist_ok=True)
    (data / 'prototype-coverage.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(results / 'response_refit_convergence.json', data / 'prototype-response-convergence.json')

    dev = report['partitions']['dev']
    final = refit['summaries']['coverage']
    baseline = refit['summaries']['encoder']
    gain = refit['changes']['coverage']['mean_coverage_gain'] * 100
    content = f'''<div class="eyebrow">Main SAE · landscape prototype evaluation</div>
<h1>景观原型：实例与覆盖评估</h1>
<p class="lead">96个共享局部模板，能否让每张街景至少80%的patch达到重建质量要求？</p>
<div class="note"><strong>本轮计算已完成，80%目标尚未达到。</strong>完整开发集平均patch覆盖为{dev['mean_image_patch_coverage']:.1%}，
仅{dev['images_meeting_target']:,}/{dev['images']:,}张方向图达标。冻结模板后，116张开发图的平均覆盖从{baseline['mean_image_patch_coverage']:.1%}提高到{final['mean_image_patch_coverage']:.1%}，仍为0/116达标。</div>
<p>本页复用当前DINOv3 ViT-L、panorama-aware、W4096/K32主SAE的完整稀疏激活。
原型层使用29城市、86,038个全景；Moscow原图映射未核实，暂未纳入。它与语义地图中固定的3,712个验证全景分析范围不同。</p>
<div class="controls"><a href="#prototype-gallery">逐原型实例</a><a href="#full-evaluation">完整评分</a><a href="#response-refit">冻结模板诊断</a><a href="#examples">覆盖率实例</a><a href="#next">待办</a></div>
{gallery_content}
<section class="section" id="criterion"><h2>什么叫一个patch被覆盖？</h2>
<p>每个方向是一张28×28×4096的激活图。输入按训练集每维正激活RMS标准化并截断至8，
在目标激活能量&gt;10<sup>−8</sup>的patch上计算 <code>NMSE = ||重建激活 − 目标激活||² / ||目标激活||²</code>。
只有NMSE≤0.5才计为覆盖；一张图至少80%的有效patch覆盖才算达标。784个patch全部有效时，需要至少628个达标。</p>
<p class="muted">这是预先固定的探索性重建判据；重建达标尚不等于语义解释成立。本次完整评分没有剔除任何无效patch，不能用全数据平均覆盖替代逐图达标要求。</p></section>
<section class="section" id="full-evaluation"><h2>完整训练与开发集评分</h2>
<p>3×3、7×7、15×15三个尺度各32个模板，共96个；非负秩4模板，训练8轮。
按1公里空间块划分新增原型层，同一全景四方向不跨组。最终测试集13,050个全景、52,200张方向图保持封存。</p>
<div class="table-wrap"><table><caption>相同固定判据下的完整评分</caption><thead><tr><th>划分</th><th>方向图像</th><th>平均patch覆盖</th><th>图像达标数（比例）</th><th>四方向全达标全景</th></tr></thead><tbody>'''
    for split, label in [('train', '训练'), ('dev', '开发')]:
        item = report['partitions'][split]
        content += (f'<tr><th scope="row">{label}</th><td>{item["images"]:,}</td>'
                    f'<td>{item["mean_image_patch_coverage"]:.2%}</td>'
                    f'<td>{item["images_meeting_target"]:,}（{item["image_pass_fraction"]:.3%}）</td>'
                    f'<td>{item["panoramas_all_four_pass"]:,}/{item["panoramas"]:,}</td></tr>')
    content += '</tbody></table></div>'
    content += fig('per_image_patch_coverage.png', '完整训练与开发集的逐图覆盖率分布；图像覆盖至少80%才达标。')
    content += f'''<p>训练与开发集平均覆盖接近，二者均明显低于目标。这一观察不能单独确定结构瓶颈，需进一步检查冻结模板能否通过更充分的响应求解提高覆盖。</p></section>
<section class="section" id="response-refit"><h2>冻结96个模板，只优化图像响应</h2>
<p>29座城市各取1个开发集全景，四方向共116张图。模板固定，从编码器响应开始，
每张图分别运行两条150步优化路线，不加响应稀疏惩罚；每条路线保留观测到的最佳硬覆盖。</p>
<div class="table-wrap"><table><caption>同一批116张图的配对比较；不是完整开发集的重跑</caption><thead><tr><th>方法</th><th>平均patch覆盖</th><th>低10%分位覆盖</th><th>达标图像</th><th>平均图像NMSE</th></tr></thead><tbody>'''
    for stage, label in [('encoder', '原编码器'), ('energy', '只降重建误差'), ('coverage', '直接优化逐图覆盖')]:
        item = refit['summaries'][stage]
        content += (f'<tr><th scope="row">{label}</th><td>{item["mean_image_patch_coverage"]:.1%}</td>'
                    f'<td>{item["coverage_quantiles"]["0.1"]:.1%}</td>'
                    f'<td>{item["images_meeting_target"]}/{item["images"]}</td>'
                    f'<td>{item["mean_image_nmse"]:.3f}</td></tr>')
    content += '</tbody></table></div>'
    content += fig('response_refit_diagnostic.png', '左：相同步数的平均覆盖变化；右：每张图观测到的最佳覆盖分布。虚线为逐图80%目标。')
    content += f'''<p><strong>直接优化覆盖带来{gain:.2f}个百分点的平均提升，但仍无图达标。</strong>
约50步后曲线趋于稳定；50到150步的平均覆盖只增加约0.006个百分点。
应优先检验模板数量、秩和空间约束，只延长当前响应优化可能收益有限。</p>
<p class="muted">有限步求解不是模板表达能力的理论上限，也不能单独确定哪项结构约束负责。
本次诊断在GPU 1运行约{refit['elapsed_seconds']:.0f}秒，原模板权重未改变，测试集未使用。</p></section>
<section class="section" id="examples"><h2>不同覆盖水平的原图实例</h2>
<p>按完整开发集覆盖分布的5%、25%、50%、75%、95%分位数选择。
绿色patch达到NMSE≤0.5，红色未达到，灰色为无有效目标激活。点击图片可放大；这些图展示重建覆盖，不是语义分割。</p><div class="two">'''
    for item in report['evidence']:
        caption = (f'{item["quantile"]:.0%}分位 · {item["city"]} · '
                   f'方向{item["heading"] * 90}° · 实际覆盖{item["patch_coverage"]:.1%}')
        content += fig(item['figure'], caption)
    content += '''</div></section><section class="section" id="next"><h2>后续待办</h2>
<ul><li>模板数量调参：相同划分下比较每尺度16 / 32 / 64个模板，检查覆盖、有效模板数、语义重复与跨种子稳定性。</li>
<li>模板结构对照：检验更高秩和空间约束变化是否改善难例；尚未启动。</li>
<li>核查模板组成和原图证据，独立判断语义一致性，不能用重建覆盖代替语义结论。</li></ul>
<p>冻结SAE曾见过部分开发和测试图像，因此本页空间留出仅针对新增原型层。</p>
<p><a href="../data/prototype-coverage.json">下载结果与数据身份 JSON</a> ·
<a href="../data/prototype-response-convergence.json">下载响应优化曲线 JSON</a></p></section>'''
    document = render_page('景观原型实例与评估', 'prototypes', content, snapshot)
    if gallery:
        document = document.replace('</head>', '<link rel="stylesheet" href="../assets/prototype-gallery.css?v=1"></head>')
        document = document.replace('</body>', '<script src="../assets/prototype-gallery.js?v=1" defer></script></body>')
    (target / 'index.html').write_text(document)

    gallery_home = (f'<p>{gallery["templates_with_examples"]}个有正响应的模板各展示8个不同点位，共{gallery["examples"]}组原图、贡献热图与裁剪；'
                    f'其余{gallery["templates"] - gallery["templates_with_examples"]}个槽位在开发集无正响应。</p>') if gallery else ''
    gallery_link = '<a href="prototypes/#prototype-gallery">逐原型查看街景实例 →</a> · ' if gallery else ''

    home = f'''<!-- prototype-results:start --><section class="section" id="prototype-results">
<div class="eyebrow">最新结果 · 景观原型评估</div><h2>96个局部模板：覆盖目标尚未达到</h2>
<p>完整开发集{dev['images']:,}张方向图的平均patch覆盖为<strong>{dev['mean_image_patch_coverage']:.1%}</strong>，
达到逐图80%要求的有{dev['images_meeting_target']:,}张（{dev['image_pass_fraction']:.3%}）。
冻结模板后，116张图平均覆盖从{baseline['mean_image_patch_coverage']:.1%}升至{final['mean_image_patch_coverage']:.1%}，仍为0/116达标。</p>
{gallery_home}<p>{gallery_link}<a href="prototypes/#full-evaluation">查看完整评分与优化曲线 →</a></p>
<p class="muted">这里按激活重建误差判定覆盖，语义质量待核查；模板数量调参仍为待办。</p></section><!-- prototype-results:end -->'''
    home_path = site / 'index.html'
    doc = home_path.read_text()
    if '<!-- prototype-results:start -->' in doc:
        doc = re.sub(r'<!-- prototype-results:start -->.*?<!-- prototype-results:end -->', lambda _: home, doc, flags=re.S)
    else:
        marker = '<div class="cards">'
        if marker not in doc:
            raise ValueError('Homepage entry point not found')
        doc = doc.replace(marker, home + '\n' + marker, 1)
    home_path.write_text(doc)
    combination = site / 'combinations/index.html'
    doc = combination.read_text()
    if 'id="prototype-results-link"' not in doc:
        start = doc.index('<p class="lead">')
        end = doc.index('</p>', start) + len('</p>')
        doc = doc[:end] + ('<div class="note" id="prototype-results-link">'
            '局部卷积模板的最新覆盖评估与冻结模板诊断已完成。'
            '<a href="../prototypes/">查看结果、失败区域与后续待办 →</a></div>') + doc[end:]
        combination.write_text(doc)


def update_existing_site(site):
    from build_main_site import nav, page
    from project_progress import PAGES, install as install_progress
    from element_hierarchy import install as install_elements
    snapshot = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M %Z')
    install(site, page, snapshot)
    for name in PAGES:
        path = site / name
        if not path.exists():
            continue
        active = 'overview' if name == 'index.html' else name.split('/')[0]
        prefix = '' if active == 'overview' else '../'
        doc = re.sub(r'<nav class="site-nav".*?</nav>', lambda _: nav(active, prefix), path.read_text(), count=1, flags=re.S)
        path.write_text(doc)
    element_summary = install_elements(site, page, snapshot)
    install_progress(site)
    manifest_path = site / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['prototype_coverage'] = {'page': 'prototypes/', 'summary': 'data/prototype-coverage.json',
                               'published_at': snapshot, 'target_achieved': False, 'semantic_review': 'pending'}
    if element_summary:
        manifest['element_hierarchy'] = {'page': 'elements/', 'identity': element_summary['identity'],
            'candidate_dimensions': element_summary['status_counts']['candidate_ready'],
            'semantic_validation': 'pending', 'graph_training': element_summary['graph_training']}
    gallery = site / 'data/prototype-gallery.json'
    if gallery.exists():
        value = json.loads(gallery.read_text())
        manifest['prototype_gallery'] = {k: value[k] for k in ['identity', 'templates', 'templates_with_examples', 'examples', 'semantic_review']}
        manifest['prototype_gallery']['page'] = 'prototypes/#prototype-gallery'
    manifest['file_count'] = sum(p.is_file() for p in site.rglob('*') if '.git' not in p.parts)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'page': str(site / 'prototypes/index.html'), 'published_at': snapshot}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, required=True)
    update_existing_site(parser.parse_args().site_dir.resolve())
