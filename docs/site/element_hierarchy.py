"""Publish the staged element hierarchy with traceable masked image evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/vitl30_relational_prototypes'
REPORTS = ROOT / 'experiments/vitl30_panorama_w4096_k32/reports'
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'
TEMPLATES = Path(__file__).with_name('templates')
SCREENING = ROOT / 'experiments/vitl30_panorama_w4096_k32/results/dimension_screening/screening.json'


def public_screened_row(original, screen):
    """Keep the old Qwen response traceable without publishing a weak label."""
    row = dict(original)
    row['raw_qwen_label_zh'] = original.get('label_zh', '')
    row['raw_qwen_family'] = original['family']
    row['screening'] = screen
    if not screen['label_eligible']:
        row['raw_qwen_interpretation'] = {key: original.get(key) for key in
            ['label_zh', 'fine_label_zh', 'semantic_type', 'cues_zh', 'caveats_zh', 'consistency',
             'confidence_self_reported']}
        row['label_zh'] = ''
        row['fine_label_zh'] = ''
        row['semantic_type'] = ''
        row['cues_zh'] = []
        row['caveats_zh'] = []
        row['consistency'] = ''
        row['confidence_self_reported'] = ''
        row['family'] = 'unresolved'
        row['family_zh'] = '未确定语义'
        row['semantic_role'] = 'unresolved'
        row['counts_as_element'] = False
        row['provisional_graph_candidate'] = False
        row['status'] = ('unobserved' if screen['status'] == 'no_positive_activation'
                         else 'insufficient_evidence')
        row['display_label_zh'] = ('无正激活证据' if row['status'] == 'unobserved'
                                   else '证据不足，暂不命名')
        row['quality_flags'] = [*row['quality_flags'], 'support_below_screening_minimum']
    else:
        row['display_label_zh'] = row.get('label_zh') or '未解释'
    return row


def build_category_graph(hierarchy):
    """Count Qwen labels within the provisional hierarchy without dropping dimensions."""
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in hierarchy['dimensions']:
        feature = row['feature']
        if feature in seen:
            raise ValueError(f'Duplicate SAE dimension f{feature}')
        seen.add(feature)
        groups[row['family']][row.get('display_label_zh') or row.get('label_zh') or '未解释'].append(feature)
    families = []
    for family in hierarchy['families']:
        labels = groups.pop(family['id'], {})
        children = [{'label': label, 'count': len(features), 'features': sorted(features),
                     'eligible': label not in {'无正激活证据', '证据不足，暂不命名'}}
                    for label, features in labels.items()]
        children.sort(key=lambda child: (-child['count'], child['label']))
        families.append({'id': family['id'], 'label': family['label_zh'],
                         'role': family['role'], 'count': sum(child['count'] for child in children),
                         'label_count': len(children), 'labels': children})
    if groups:
        raise ValueError(f'Unknown category families: {sorted(groups)}')
    families.sort(key=lambda family: (-family['count'], family['label']))
    total = sum(family['count'] for family in families)
    if total != len(hierarchy['dimensions']):
        raise ValueError('Category graph does not cover all SAE dimensions')
    return {'identity': hierarchy['identity'], 'total': total,
            'family_count': len(families),
            'label_count': sum(family['label_count'] for family in families),
            'families': families}


def install(site, render_page, snapshot):
    path = EXPERIMENT / 'results/hierarchy.json'
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    summary = obj['summary']
    screening = json.loads(SCREENING.read_text())
    if screening['protocol']['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Dimension screening uses another SAE checkpoint')
    screens = screening['dimensions']
    if len(screens) != 4096 or [row['feature'] for row in screens] != list(range(4096)):
        raise ValueError('Incomplete dimension screening')
    pilot = json.loads((EXPERIMENT / 'results/pilot_manifest_summary.json').read_text())
    if obj['protocol']['checkpoint_sha256'] != CHECKPOINT or pilot['hierarchy_identity'] != obj['identity']:
        raise ValueError('Element hierarchy or pilot uses another SAE/hierarchy')
    for source, record in obj['protocol']['sources'].items():
        source_path = REPORTS / source / 'dimension_interpretations.json'
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != record['records_sha256']:
            raise ValueError('VLM evidence changed; rebuild the hierarchy first')
    target = site / 'elements'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'inputs').mkdir(exist_ok=True)
    rows, jobs = [], []
    for original in obj['dimensions']:
        row = public_screened_row(original, screens[original['feature']])
        feature = row['feature']
        if row['status'] != 'unobserved':
            source = REPORTS / row['source'] / 'inputs' / f'f{feature:04d}_0.png'
            if not source.exists():
                raise ValueError(f'Missing actual masked VLM input for f{feature}')
            shared = site / 'interpretation/inputs' / f'f{feature:04d}_0.webp'
            if row['source'] == 'vlm_dimensions_top10_masked' and shared.exists():
                row['sheet'] = f'../interpretation/inputs/f{feature:04d}_0.webp'
            else:
                row['sheet'] = f'inputs/f{feature:04d}.webp'
                jobs.append((source, target / row['sheet'], 'lossless'))
        if row['family_disagreement']:
            source = REPORTS / 'vlm_dimensions/inputs' / f'f{feature:04d}_0.jpg'
            if source.exists():
                row['reference_sheet'] = f'inputs/f{feature:04d}_reference.webp'
                jobs.append((source, target / row['reference_sheet'], 'reference_preview'))
        rows.append(row)

    def copy(job):
        source, destination, mode = job
        if destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime:
            return
        with Image.open(source) as image:
            if mode == 'reference_preview':
                # The original unmasked sheet is 10 columns of 128px cells.
                # Retain its first row; full Top100 evidence already exists on
                # the dimension page, so don't duplicate all 50 thumbnails.
                if image.width != 1280 or image.height % 128:
                    raise ValueError('Unexpected reference contact sheet layout')
                image.crop((0, 0, 1280, 128)).convert('RGB').save(
                    destination, 'WEBP', quality=85, method=4)
            else:
                image.convert('RGB').save(destination, 'WEBP', lossless=True, method=4)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy, jobs))
    public_summary = {**summary, 'source_status_counts': summary['status_counts'],
                      'status_counts': dict(Counter(r['status'] for r in rows)),
                      'family_counts': dict(Counter(r['family'] for r in rows)),
                      'graph_candidate_family_counts': dict(Counter(
                          r['family'] for r in rows if r['provisional_graph_candidate'])),
                      'dimension_screening_identity': screening['identity'],
                      'prototype_route': 'local_multi_element_compositions'}
    public = {**obj, 'dimensions': rows, 'pilot': pilot,
              'summary': public_summary,
              'dimension_screening_identity': screening['identity'],
              'snapshot': snapshot}
    (target / 'hierarchy.json').write_text(json.dumps(public, ensure_ascii=False, separators=(',', ':')))
    try:
        from .label_activation_gallery import install as install_label_gallery
    except ImportError:
        from label_activation_gallery import install as install_label_gallery
    graph = build_category_graph({**obj, 'dimensions': rows})
    graph['dimension_screening_identity'] = screening['identity']
    gallery_stats = install_label_gallery(site, target, graph, REPORTS, obj['protocol']['sources'])
    (target / 'category-graph.json').write_text(json.dumps(
        graph, ensure_ascii=False, separators=(',', ':')))
    for obsolete in ['training-summary.json', 'prototype-examples.json',
                     'relational-gallery.css', 'relational-gallery.js']:
        (target / obsolete).unlink(missing_ok=True)
    if (target / 'examples').exists():
        shutil.rmtree(target / 'examples')
    if (target / 'open-panorama').exists():
        shutil.rmtree(target / 'open-panorama')
    if (target / 'local-compositions').exists():
        shutil.rmtree(target / 'local-compositions')
    if (site / 'combinations').exists():
        shutil.rmtree(site / 'combinations')
    for obsolete_figure in ['Fig_ViTL30_Landscape_Prototype_Overview.jpg',
                            'Fig_Dimension_and_Motif_Evidence.png']:
        (site / 'assets/figures' / obsolete_figure).unlink(missing_ok=True)
    for name in ['dimension_elements.csv', 'review_queue.csv']:
        # Normalize CSV line endings for the publishing repo; retain its BOM.
        (target / name).write_text((EXPERIMENT / 'results' / name).read_text())
    shutil.copyfile(SCREENING, target / 'dimension-screening.json')
    shutil.copyfile(SCREENING.with_name('screening.csv'), target / 'dimension-screening.csv')
    for suffix in ['css', 'js']:
        shutil.copyfile(TEMPLATES / f'element-hierarchy.{suffix}', target / f'elements.{suffix}')
    shutil.copyfile(TEMPLATES / 'element-training.css', target / 'element-training.css')
    counts = public_summary['status_counts']
    try:
        from .local_composition_gallery import install as install_local_gallery
    except ImportError:
        from local_composition_gallery import install as install_local_gallery
    relation_target = site / 'relations'
    relation_target.mkdir(exist_ok=True)
    local_gallery = install_local_gallery(relation_target)
    content = f'''<div class="eyebrow">Qwen labels · SAE activation evidence</div>
<h1>街景要素标签与激活</h1>
<p class="lead">浏览每个 Qwen 细标签对应的 SAE 维度，以及这些维度在四方向街景中的真实响应。</p>
<p><a href="../relations/#local-results">查看独立的要素关系页面 →</a></p>
<section class="section" id="qwen-taxonomy"><h2>Qwen 类别图谱</h2>
<p>左侧为暂定父类，右侧为 Qwen 对 SAE 维度高激活图像给出的细标签。支持度不足 20 个正激活全景的维度归入“证据不足，暂不命名”，旧 Qwen 猜测只在逐维证据中保留供核查。位置集中只提示复核，不自动判为位置编码。数字是 SAE 维度数，不是地物出现次数；类别尚未经人工语义验证。</p>
<div class="category-path" aria-label="类别层级">Qwen 逐维解释 <span aria-hidden="true">›</span> <span id="category-path-family">父类</span> <span aria-hidden="true">›</span> <span id="category-path-label">细标签</span></div>
<div class="category-graph"><div class="category-branch"><h3>父类 <span id="category-family-total"></span></h3><div id="category-families" class="category-families" aria-label="Qwen 类别父类"></div></div>
<div class="category-branch"><h3 id="category-label-heading">细标签</h3><div id="category-labels" class="category-labels" aria-label="Qwen 细标签"></div></div></div>
<div id="category-feature-detail" class="category-feature-detail" aria-live="polite"></div></section>
<section class="section" id="label-activations"><h2>标签的四方向激活实例</h2>
<p>{gallery_stats['labels']} 个标签共收录 {gallery_stats['examples']:,} 个全景实例；{gallery_stats['labels_with_five']} 个标签达到 5 例，{gallery_stats['labels'] - gallery_stats['labels_with_five']} 个不足 5 例（其中 {gallery_stats['labels_without_examples']} 个无正激活）。扩大样本补充 {gallery_stats['expanded_examples']} 例。</p>
<p>每个标签最多展示 5 个峰值响应较强的全景，优先覆盖不同 SAE 维度和点位。上行为四方向原图，下行为同一全景的激活高亮；颜色仅叠加在被选中维度的响应方向。热图按该方向峰值归一化，属于 patch 级响应，不是物体分割。高亮与标签不符时说明解释需要复核；部分稀缺标签不足 5 例或没有正激活。</p>
<div id="label-gallery" aria-live="polite"></div>
<dialog id="label-panorama-dialog" class="label-dialog" aria-labelledby="label-dialog-title"><div class="label-dialog-heading"><h2 id="label-dialog-title"></h2><form method="dialog"><button type="submit">关闭</button></form></div><div id="label-dialog-body" class="label-dialog-body"></div></dialog></section>
<section class="section"><h2>逐维候选与原图证据</h2>
<p>优先列出{summary['family_disagreements']:,}个遮罩/未遮罩解释父类不一致的维度。冲突标记仅表示需要检查，不能单独判断哪种解释正确。模型自述“高信心”不是准确率。</p>
<p><a href="dimension-screening.csv">下载逐维筛查 CSV</a> · <a href="dimension_elements.csv">下载原始要素表 CSV</a> · <a href="review_queue.csv">下载复核清单 CSV</a> · <a href="hierarchy.json">数据与来源 JSON</a></p>
<div class="controls"><input id="element-search" aria-label="搜索维度或语义" placeholder="编号、窗户、树冠…"><select id="element-family" aria-label="要素父类"><option value="">全部父类</option></select><select id="element-status" aria-label="证据状态"><option value="">全部状态</option></select><label><input type="checkbox" id="element-conflicts">仅看解释冲突</label></div>
<p id="element-count" role="status">载入候选表…</p>
<div class="element-layout"><div><div id="element-list" class="element-list" aria-label="维度列表"></div><button id="element-more" type="button">再显示100项</button></div><section id="element-detail" class="element-detail" aria-label="维度证据"></section></div></section>
<section class="section"><h2>范围与当前结论</h2><p>没有重新提取 DINO 特征或训练 SAE。验证全景优先用于激活实例；扩大样本用于补充稀缺标签，来源逐例标出。扩大样本的既有 Qwen 语义资料可能覆盖原型实验的训练、开发与测试图像，因此不构成独立语义验证。标签与图像的对应仍需人工复核。</p>
<p>{public_summary['status_counts'].get('context_or_material', 0):,}维背景/材料，{public_summary['status_counts'].get('needs_review', 0):,}维待核查，{public_summary['status_counts'].get('insufficient_evidence', 0):,}维证据不足，{public_summary['status_counts'].get('unobserved', 0):,}维仍无正激活证据。待核查项没有被删除，也没有被算作已验证的语义。</p></section>'''
    relation_content = f'''<div class="eyebrow">Recurring local scene compositions</div>
<h1>跨街景重复的要素关系</h1>
<p class="lead">先在每个方向的28×28 patch 激活中提取形状自适应的区域，再把同一点位四个方向放在一起，寻找不同要素及其相对关系的跨城重复实例。</p>
<p><a href="../elements/#qwen-taxonomy">查看组成要素的 Qwen 标签与逐维激活 →</a></p>
<div class="element-stages"><section><h2>01 · 四方向输入</h2><p>0°、90°、180°、270°同属一个全景，真实朝向和完整图像在实例中保留。</p></section>
<section><h2>02 · 局部 SAE 要素</h2><p>复用当前 ViT-L W4096/K32 逐 patch 激活，在方向图中产生非矩形区域遮罩。VLM 父类只用于探索性的要素归组。</p></section>
<section><h2>03 · 重复关系检索</h2><p>每个候选包含三个局部 SAE 要素，允许同一父类重复（例如建筑 + 乔木 + 乔木），检索同方向或跨方向的相对布局；数量由训练样本支持决定，不预设32或96种景观。</p></section>
<section><h2>04 · 留出样本核查</h2><p>开发集比较训练候选与同城要素及布局置换对照；测试集封存。语义仍需人工复核。</p></section></div>
<section class="section" id="local-method"><h2>当前实验如何找组合？</h2>
<p>以完整四方向全景为统计单位；每张方向图独立形成要素区域，再用方位、上下左右和分离的 mask 建立关系。关系搜索允许同一父类出现多个独立区域，同时过滤方向边界处的重复观测。探索性样本先看重复与置换对照，不将一个模型模板直接当作景观类别。</p></section>
{local_gallery}
<section class="section"><h2>用于解释的景观要素基础</h2>
<p>窗户、门和建筑墙面归入建筑，树冠和树干归入乔木，车窗和车轮归入车辆。父类只作为关系检索的辅助，仍需人工核查。要素标签及其激活图见 <a href="../elements/">标签与激活页</a>。</p></section>'''
    document = render_page('景观要素标签与激活', 'elements', content, snapshot,
        '<script src="elements.js" defer></script>')
    document = document.replace('</head>', '<link rel="stylesheet" href="elements.css">'
                                '<link rel="stylesheet" href="element-training.css"></head>')
    (target / 'index.html').write_text(document)
    relation_document = render_page('景观要素关系', 'relations', relation_content, snapshot)
    relation_document = relation_document.replace('</head>', '<link rel="stylesheet" href="../elements/elements.css"></head>')
    (relation_target / 'index.html').write_text(relation_document)
    # Keep the current route visible from every main page.
    try:
        from .build_main_site import nav
        from .project_progress import PAGES
    except ImportError:
        from build_main_site import nav
        from project_progress import PAGES
    for name in PAGES:
        document_path = site / name
        if not document_path.exists():
            continue
        active = 'overview' if name == 'index.html' else name.split('/')[0]
        prefix = '' if active == 'overview' else '../'
        doc = re.sub(r'<nav class="site-nav".*?</nav>', lambda _: nav(active, prefix),
                     document_path.read_text(), count=1, flags=re.S)
        if name == 'index.html':
            doc = re.sub(r'<section class="section"><h2>景观组合探索</h2>.*?</section>',
                         '<section class="section"><h2>跨街景的局部要素组合</h2>'
                         '<p>在完整四方向全景中提取自适应要素遮罩，检索跨街景重复的组合关系。'
                         '当前结果为探索性抽样，语义仍待核查。</p>'
                         '<a href="relations/#local-results">查看遮罩实例与置换对照 →</a></section>',
                         doc, flags=re.S)
            announcement = ('<section class="note" id="element-route">'
                '<strong>当前路线：跨街景重复的局部要素组合</strong>'
                '<p>从四方向图像提取局部区域，检索不同要素及相对关系的重复实例。</p>'
                f'<a href="{prefix}relations/">查看关系方法与实例 →</a> · '
                f'<a href="{prefix}elements/">查看标签激活 →</a></section>')
            if 'id="element-route"' in doc:
                doc = re.sub(r'<section class="note" id="element-route">.*?</section>',
                             lambda _: announcement, doc, count=1, flags=re.S)
            else:
                marker = '<main class="page">'
                if marker not in doc:
                    raise ValueError('Main page entry point not found')
                doc = doc.replace(marker, marker + announcement, 1)
        if name == 'index.html':
            doc = re.sub(r'<!-- prototype-results:start -->.*?<!-- prototype-results:end -->', '', doc, flags=re.S)
        if name == 'combinations/index.html':
            doc = re.sub(r'<div class="note" id="prototype-results-link">.*?</div>', '', doc, flags=re.S)
        document_path.write_text(doc)
    return {**public_summary, 'prototype_route': 'local_multi_element_compositions',
            'label_gallery': gallery_stats,
            'local_composition_status': 'complete' if (relation_target / 'local-compositions/results.json').exists() else 'pending'}


def update(site):
    from build_main_site import page
    from project_progress import install as install_progress
    manifest = json.loads((site / 'manifest.json').read_text())
    if manifest['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Publishing site uses another SAE checkpoint')
    snapshot = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M %Z')
    result = install(site, page, snapshot)
    install_progress(site)
    manifest['element_hierarchy'] = {'page': 'elements/', 'identity': result['identity'],
        'candidate_dimensions': result['status_counts']['candidate_ready'], 'semantic_validation': 'pending',
        'relation_page': 'relations/', 'label_gallery': result['label_gallery'],
        'prototype_route': result['prototype_route'],
        'local_composition_status': result['local_composition_status'],
        'published_at': snapshot}
    manifest.pop('prototype_coverage', None)
    manifest.pop('prototype_gallery', None)
    manifest.pop('panorama_display', None)
    manifest['file_count'] = sum(p.is_file() for p in site.rglob('*') if '.git' not in p.parts)
    manifest['bytes_before_manifest'] = sum(p.stat().st_size for p in site.rglob('*')
        if p.is_file() and '.git' not in p.parts and p.name != 'manifest.json')
    (site / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, required=True)
    update(parser.parse_args().site_dir.resolve())
