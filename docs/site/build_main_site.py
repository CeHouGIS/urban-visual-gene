#!/usr/bin/env python3
"""Build the public site from the single, explicitly identified main experiment.

This writes a separate static bundle. It never changes experiments, input images,
model outputs, or the git publishing checkout. Large Top100 crops are rendered
in-browser from their original image and stored bounding box.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image
from project_progress import install as install_progress
from element_hierarchy import install as install_element_hierarchy
from label_structure import install as install_label_structure
from feature_quality_review import install as install_quality_review

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/vitl30_panorama_w4096_k32"
TEMPLATES = Path(__file__).with_name("templates")
CHECKPOINT = "fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0"
NAV_ITEMS = [("overview", "主实验", ""), ("semantic", "语义地图", "semantic/"),
             ("instances", "图像实例", "instances/"), ("interpretation", "维度解释", "interpretation/"),
             ("elements", "要素标签", "elements/"), ("label-structure", "标签结构", "label-structure/"),
             ("quality", "维度复核", "quality/"),
             ("relations", "要素关系", "relations/")]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def write_compressed_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def nav(active, prefix=""):
    links = ''.join(f'<a href="{prefix}{url or "./"}"'
                    f'{" aria-current=page" if key == active else ""}>{label}</a>'
                    for key, label, url in NAV_ITEMS)
    return f'<nav class="site-nav" aria-label="主导航"><a class="brand" href="{prefix or "./"}">Urban Scene Primitives</a>{links}</nav>'


def page(title, active, content, snapshot, script=""):
    prefix = "" if active == "overview" else "../"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · Urban Scene Primitives</title>
<meta name="description" content="30 城市 DINOv3 ViT-L panorama-aware BatchTopK SAE，W4096 / K32 主实验结果。">
<link rel="stylesheet" href="{prefix}assets/site.css"></head><body>{nav(active, prefix)}
<main class="page">{content}</main><footer>30 城市 · DINOv3 ViT-L/16 · panorama-aware · W4096 / K32<br>
发布快照：{snapshot} · <a href="{prefix}manifest.json">数据与模型来源</a> · <a href="https://github.com/CeHouGIS/urban-visual-gene/tree/main/experiments/vitl30_panorama_w4096_k32">实验代码</a>
</footer>{script}</body></html>'''


def figure(filename, caption, prefix="assets/figures/"):
    return f'<figure><a href="{prefix}{filename}"><img loading="lazy" src="{prefix}{filename}" alt="{html.escape(caption)}"></a><figcaption>{caption}</figcaption></figure>'


def overview(out, snapshot, counts):
    content = '''<div class="eyebrow">30-city panorama-aware BatchTopK SAE</div>
<h1>Urban Scene Primitives</h1><p class="lead">从街景中学习稀疏视觉单元，观察它们响应什么，以及不同单元如何共同出现。</p>
<div class="stats"><div class="stat"><b>30</b><span>训练与验证城市</span></div><div class="stat"><b>89,031</b><span>完整四方向全景</span></div><div class="stat"><b>4,096</b><span>SAE 字典维度 · K=32</span></div><div class="stat"><b>28 × 28</b><span>每方向 patch 网格</span></div></div>
<p>四方向图像分别经过冻结的 DINOv3 ViT-L/16（448 × 448 → 28 × 28 × 1024），再进行环形全景邻域 context mixing；逐 patch 输入 BatchTopK SAE，完成 1024 → 4096 → 1024 的稀疏分解。</p>
<div class="cards"><a class="entry" href="semantic/"><span class="eyebrow">01 / Feature semantics</span><h2>维度语义地图 ↗</h2><p>2,863 个维度，各维 Top100 激活区域的 SigLIP embedding 汇总；悬停预览，点击核对原图、激活及四方向全景。</p><img loading="lazy" src="assets/figures/Fig_ViTL30_Dimension_Semantic_UMAP.png" alt="维度语义 UMAP"></a>
<a class="entry" href="instances/"><span class="eyebrow">02 / Image evidence</span><h2>Top10 图像实例 ↗</h2><p>29,934 个实例单独投影。按维度、64 组和激活排名筛选，查看同一维度响应的不同景观。</p><img loading="lazy" src="assets/figures/Fig_ViTL30_Top10_Image_Instance_UMAP.png" alt="图像实例 UMAP"></a>
<a class="entry" href="interpretation/"><span class="eyebrow">03 / Interpretation</span><h2>逐维 Qwen 解释 ↗</h2><p>每维 Top10 高激活区域裁剪并遮挡背景，查看模型看到的实际拼图、候选含义及局限。</p><p class="muted">@VLM_STATUS@</p></a></div>
<div class="note">图像语义分析使用可回溯原图的 29 城市、3,712 个验证全景（14,848 张方向图）。Moscow 尚未核实原图映射，未进入这部分解释。它与 30 城市模型训练范围不同。</div>
<section class="section" id="training"><h2>训练与激活</h2><p>85,191 个训练全景，3,840 个验证全景，每城固定留出 128 点。训练完成 16 轮，最优第 10 轮；sampled validation squared-L2 为 0.287344。K=32 是批次平均的激活预算，每个 patch 的非零数可以不同。</p>
<div class="two">@LOSS@@ACTIVATION@</div><p><a href="data/training/history.csv">训练记录 CSV</a> · <a href="data/training_summary.json">训练设置与数据身份</a></p></section>
<section class="section"><h2>城市与视觉组成</h2><p>4,096 维 decoder 按余弦距离与 average linkage 归纳为 64 组，用于组成和城市比较；分组颜色与语义地图一致。</p>
<div class="two">@CITIES@@COMPOSITION@</div><details><summary>展开城市相似性、共现与空间结果</summary><div class="two">@SIMILARITY@@COOCCURRENCE@@SHARED@@SPATIAL@</div></details></section>
<section class="section"><h2>跨街景的局部要素组合</h2><p>以完整四方向全景为统计单位，在各方向的 SAE 激活中提取自适应要素遮罩，检索重复的多要素关系。当前为探索性抽样，语义仍需核查。</p><a href="relations/#local-results">查看组合遮罩实例与置换对照 →</a></section>
<details class="section"><summary>方法与解释范围</summary><p>语义地图使用图像裁剪的 SigLIP embedding，城市、空间坐标、decoder 分组和 VLM 文字均不参与 UMAP 拟合。维度地图中至少 20 个有效实例才进入投影，故 4,096 个模型维度对应 2,863 个散点；图像实例地图则保留 3,178 维的最多 Top10 实例。64 组使用各自唯一颜色，部分组暂无足够实例进入维度投影。</p><p>二维邻近关系供浏览，不证明类别边界或统计显著性。Qwen 名称为待核查的解释；维度可能对应物体、部件、纹理或混合概念。激活图表示 patch 级响应，不是像素分割。</p><p><a href="data/group_interpretations.csv">64 组解释 CSV</a> · <a href="data/feature_semantic_umap_groups.csv">维度坐标与分组 CSV</a> · <a href="data/group_palette.json">64 组色表与许可</a></p></details>'''
    figures = {
        "LOSS": ("Fig_ViTL30_Training_Loss.png", "训练 loss 与抽样验证误差"),
        "ACTIVATION": ("Fig_ViTL30_Activation_Examples.png", "原图、激活热图与高响应区域"),
        "CITIES": ("Fig_ViTL30_Study_Cities.png", "主实验研究城市"),
        "COMPOSITION": ("Fig_ViTL30_City_Composition.png", "64 组视觉组成的城市比较"),
        "SIMILARITY": ("Fig_ViTL30_City_Similarity.png", "城市视觉组成相似性"),
        "COOCCURRENCE": ("Fig_ViTL30_Cooccurrence.png", "视觉组共同出现的描述性统计"),
        "SHARED": ("Fig_ViTL30_Shared_Distinctive.png", "共享与具有城市差异的视觉组"),
        "SPATIAL": ("Fig_ViTL30_Spatial_Structure.png", "视觉组成的空间结构"),
    }
    for key, (name, caption) in figures.items():
        content = content.replace('@'+key+'@', figure(name, caption))
    interpreted, unobserved = counts.get('interpreted', 0), counts.get('unobserved_in_sample', 0)
    insufficient = counts.get('insufficient_evidence', 0)
    remaining = 4096 - interpreted - unobserved - insufficient
    vlm_status = (f'待核查候选 {interpreted:,} 维；证据不足、暂停命名 {insufficient:,} 维；'
                  f'无正激活 {unobserved:,} 维。')
    if remaining:
        vlm_status += f'另有 {remaining:,} 维待补充。'
    content = content.replace('@VLM_STATUS@', vlm_status)
    (out/'index.html').write_text(page("主实验", "overview", content, snapshot))


def identity_guard(semantic_root, reports):
    for p in [semantic_root/'run.json', reports/'semantic_umap/summary.json',
              reports/'instance_umap/summary.json']:
        assert read_json(p)['checkpoint_sha256'] == CHECKPOINT, f'Checkpoint mismatch: {p}'


def copy_image(task):
    source, target, kind = task
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'copy':
        shutil.copyfile(source, target)
    else:
        with Image.open(source) as im:
            if kind == 'preview':
                im = im.convert('RGB')
                im.thumbnail((240, 200))
                im.save(target, 'WEBP', quality=85, method=3)
            else:
                # Preserve the exact pixels that Qwen received, including grey mask.
                im.save(target, 'WEBP', lossless=True, quality=100, method=3)


def semantic_pages(out, source, reports):
    target = out/'semantic'
    target.mkdir(parents=True)
    original = (source/'index.html').read_text()
    atlas, _ = json.JSONDecoder().raw_decode(original.split('const atlas=', 1)[1])
    points = {p['feature']: p for p in atlas['points']}
    full = read_json(source/'semantic_atlas.json')
    for point in full['points']:
        fid = point['feature']
        heats = re.findall(r'data-heat="([^"]+)"', (source/f'features/{fid}.html').read_text())
        assert len(heats) == len(point['examples']), fid
        rows = []
        for example, heat in zip(point['examples'], heats):
            image_id = int(re.search(r'/i(\d+)_', example['crop'])[1])
            rows.append({**{key: example[key] for key in ['rank', 'city', 'pano', 'heading', 'activation']},
                         'image': image_id, 'box': example['bbox_native_view'], 'heat': heat})
        write_compressed_json(target/f'features/{fid}.json.gz', {'feature': fid, 'group': points[fid]['group'],
                                                              'coherence': point['coherence'], 'examples': rows})
    original = original.replace("'features/'+id+'.html'", "'feature.html?v=panorama2&f='+id")
    original = original.replace('im.src=crop;', "setPreview(im,crop,'sources/');")
    original = original.replace('<script>const atlas=', '<script src="../assets/preview.js"></script><script>const atlas=')
    original = original.replace('</head>', '<link rel="stylesheet" href="../assets/site.css"></head>') if '</head>' in original else original.replace('<style>', '<link rel="stylesheet" href="../assets/site.css"><style>', 1)
    # This source document omits explicit head/body tags; insert at the actual header.
    original = original.replace('<header>', nav('semantic', '../')+'<header>', 1)
    original = original.replace('</script></html>', "addEventListener('hashchange',()=>{const id=Number(location.hash.replace('#f',''));if(byId.has(id)&&id!==selected)select(id)});</script></html>")
    (target/'index.html').write_text(original)
    for filename in ['feature.html', 'feature.js']:
        shutil.copyfile(TEMPLATES/filename, target/filename)

    instance = (reports/'instance_umap/index.html').read_text()
    instance = instance.replace('</head>', '<link rel="stylesheet" href="../assets/site.css"></head>')
    instance = instance.replace('<body>', '<body>'+nav('instances', '../'), 1)
    instance = re.sub(r'"source":"[^"]+"', '"source":"../semantic"', instance)
    instance = instance.replace('`${source}/features/${f}.html`', '`${source}/feature.html?f=${f}`')
    instance = instance.replace('crop.src=`${source}/${p[6]}`;', 'setPreview(crop,p[6],`${source}/sources/`);')
    instance = instance.replace('img.src=`${source}/${q[6]}`;', 'setPreview(img,q[6],`${source}/sources/`);')
    instance = instance.replace("tip.querySelector('img').src=`${source}/${p[6]}`;", "setPreview(tip.querySelector('img'),p[6],`${source}/sources/`);")
    instance = instance.replace('<script>const atlas=', '<script src="../assets/preview.js"></script><script>const atlas=')
    instance = instance.replace('select(initial);', "select(initial);if(!match){feature=null;document.querySelector('#feature').value='';draw()}")
    (out/'instances').mkdir()
    (out/'instances/index.html').write_text(instance)
    return [], len(full['points']), sum(p['count'] for p in full['points'])


def interpretation_pages(out, reports, snapshot):
    report = reports/'vlm_dimensions_top10_masked'
    expanded_report = reports/'vlm_dimensions_top10_expanded'
    # The combined JSON is an atomic, bounded snapshot; derive counts from it.
    records = read_json(report/'dimension_interpretations.json')['dimensions']
    expanded_records = read_json(expanded_report/'dimension_interpretations.json')['dimensions']
    assert len(records) == 4096 and {r['feature'] for r in records} == set(range(4096))
    assert len(expanded_records) == len(records) and all(r['feature'] == e['feature'] for r, e in zip(records, expanded_records))
    for feature, (record, extra) in enumerate(zip(records, expanded_records)):
        if record['status'] == 'unobserved_in_sample' and extra['status'] == 'interpreted':
            records[feature] = extra
            records[feature]['interpretation_source'] = 'expanded_posthoc'
        else:
            record['interpretation_source'] = 'fixed_validation'
    screening = read_json(EXPERIMENT/'results/dimension_screening/screening.json')
    assert screening['protocol']['checkpoint_sha256'] == CHECKPOINT
    screens = screening['dimensions']
    assert len(screens) == len(records) and all(r['feature'] == s['feature'] for r, s in zip(records, screens))
    for record, screen in zip(records, screens):
        record['source_status'] = record['status']
        record['raw_qwen_label_zh'] = record.get('label_zh', '')
        record['screening'] = screen
        if not screen['label_eligible']:
            record['raw_qwen_interpretation'] = {key: record.get(key) for key in
                ['label_zh', 'semantic_type', 'consistency', 'confidence_self_reported',
                 'cues_zh', 'alternatives_zh', 'caveats_zh']}
            record['label_zh'] = ''
            record['semantic_type'] = ''
            record['consistency'] = ''
            record['confidence_self_reported'] = ''
            record['cues_zh'] = []
            record['alternatives_zh'] = []
            record['caveats_zh'] = []
            record['status'] = ('unobserved_in_sample' if screen['status'] == 'no_positive_activation'
                                else 'insufficient_evidence')
    counts = dict(Counter(d['status'] for d in records))
    target = out/'interpretation'
    target.mkdir(exist_ok=True)
    tasks = []
    for d in records:
        d.pop('original_evidence', None)
        d.pop('record', None)
        sheet = (expanded_report if d['interpretation_source'] == 'expanded_posthoc' else report)/f'inputs/f{d["feature"]:04d}_0.png'
        if sheet.exists() and d['interpretation_source'] == 'expanded_posthoc':
            d['sheet'] = f'../elements/inputs/f{d["feature"]:04d}.webp'
        elif sheet.exists():
            d['sheet'] = f'inputs/f{d["feature"]:04d}_0.webp'
            tasks.append((sheet, target/d['sheet'], 'lossless'))
    write_json(target/'dimensions.json', {'snapshot_label': snapshot, 'counts': counts,
                                        'dimension_screening_identity': screening['identity'],
                                        'interpretation_identity': read_json(report/'identity.json'),
                                        'expanded_interpretation_identity': read_json(expanded_report/'identity.json'),
                                        'dimensions': records})
    pilot = EXPERIMENT/'results/dimension_screening/review_pilot.json'
    if pilot.exists():
        shutil.copyfile(pilot, target/'review-pilot.json')
    with (target/'dimensions.csv').open('w', newline='') as stream:
        fields = ['feature', 'status', 'crop_count', 'label_zh', 'raw_qwen_label_zh',
                  'semantic_type', 'consistency', 'confidence_self_reported']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    content = '''<div class="eyebrow">Masked Top10 · Qwen3-VL-2B</div><h1>逐维语义解释</h1>
<p>每维选前 10 个不同全景的最强方向，按该方向峰值约 50% 保留高激活 patch，裁到联合包围框，框内其余像素置灰（RGB 127）。固定验证样本无响应的维度另从扩大样本检索，来源逐维标出。Qwen 看的是 masked 原图拼图。</p>
<p class="note">支持度不足 20 个正激活全景的维度暂停命名，旧 Qwen 回答只保留供核对。其他名称与模型自述信心仍是待核查候选；patch mask 不是像素级分割。Top40 四组复核仅试跑 2 维：f0 的组间解释冲突，f165 的四组均指向行人；这批试跑尚未替换原始标签，也不能作为独立语义验证。</p><p id="snapshot"></p>
<p><a href="dimensions.csv">解释 CSV</a> · <a href="dimensions.json">全部记录 JSON</a> · <a href="review-pilot.json">Top40 试跑审计</a></p>
<div class="controls"><input id="search" placeholder="搜索编号、候选名称或类型" aria-label="搜索维度"><select id="type" aria-label="语义类型"><option value="">全部类型</option></select><select id="status" aria-label="解释状态"><option value="">全部状态</option><option value="interpreted">待核查候选</option><option value="insufficient_evidence">证据不足</option><option value="pending">待解释</option><option value="unobserved_in_sample">无正激活</option></select></div>
<p class="muted" id="match-count"></p><div class="vlm-layout"><div class="vlm-list" id="listing" aria-label="维度列表"></div><section class="vlm-detail" id="detail"><h2>选择一个维度</h2></section></div>'''
    (target/'index.html').write_text(page('维度解释', 'interpretation', content, snapshot, '<script src="interpretation.js"></script>'))
    shutil.copyfile(TEMPLATES/'interpretation.js', target/'interpretation.js')
    return tasks, counts


def combination_pages(out, reports, snapshot):
    target = out/'combinations'
    target.mkdir()
    nmf = read_json(reports/'landscape_prototypes/summary.json')
    motifs = read_json(reports/'unit_motifs/summary.json')
    # Preserve the completed analyses of this checkpoint; place detailed evidence
    # below the compact summary instead of introducing another experiment menu.
    nmf_html = (reports/'landscape_prototypes/index.html').read_text()
    cards = re.findall(r'<article\b.*?</article>', nmf_html, re.S)
    assert len(cards) == 32, len(cards)
    cards = [re.sub(r'href="[^"]*/features/(\d+)\.html"',
                    r'href="../semantic/feature.html?f=\1"', c) for c in cards]
    motif_html = (reports/'unit_motifs/index.html').read_text()
    motif_cards = [c for c in re.findall(r'<article\b.*?</article>', motif_html, re.S) if 'panoramas/' in c]
    assert motif_cards
    motif_cards = [re.sub(r'href="#f(\d+)"', r'href="../semantic/feature.html?f=\1"', c).replace('panoramas/', 'motifs/') for c in motif_cards]
    content = '''<div class="eyebrow">Composition · exploratory analysis</div><h1>景观组合探索</h1>
<p class="lead">在当前 4,096 维 SAE 激活上，寻找反复共同出现的组成与全景实例。</p>
<div class="note">这一页的两类结果均来自当前 ViT-L W4096/K32 模型。它们是候选组合，尚不能直接命名为独立的景观类型。实例按 0° → 90° → 180° → 270° 展示同一全景的四张原图，保留原图比例；每张透视图单独标记方向，未进行无缝投影拼接。</div>
<h2>全景组成模板</h2><p>对 3,712 个验证全景的维度组成做非负矩阵分解（NMF）。主展示为 32 个模板，并比较 16 / 32 / 64 个模板的跨城市重建表现；32 表示模型容量，不表示已确定有 32 类景观。</p>
@OVERVIEW@<details><summary>展开全部 32 个模板及四方向全景</summary>@NMF@</details>
<h2>跨城市共同出现的维度</h2><p>在空间稀疏化后的全景中，使用部分城市发现组合，再在留出城市验证共同出现与峰值关系。这里保留全部结果表，并展示 30 组代表实例；显著性只对应报告中明确检验的统计量。</p>
<p><a href="motifs.csv">全部候选组合 CSV</a> · <a href="dimension_audit.csv">4,096 维证据检查 CSV</a> · <a href="summary.json">统计设置与结果</a></p>
@EVIDENCE@<details><summary>展开 30 组共同出现的全景实例</summary>@MOTIFS@</details>
<p class="muted">共现、语义一致性和重复空间结构是不同证据。这里的标记只显示最强 patch 的位置，不能当作物体轮廓。完整判据与限制见 <a href="EXPERIMENT_DESIGN.md">实验设计</a>。</p>'''
    content = content.replace('@NMF@', ''.join(cards)).replace('@MOTIFS@', ''.join(motif_cards))
    content = content.replace('@OVERVIEW@', figure('Fig_ViTL30_Landscape_Prototype_Overview.jpg', '32 个 NMF 候选组成模板的代表全景', '../assets/figures/'))
    content = content.replace('@EVIDENCE@', figure('Fig_Dimension_and_Motif_Evidence.png', '逐维证据与组合验证的统计摘要', '../assets/figures/'))
    (target/'index.html').write_text(page('组合探索', 'combinations', content, snapshot))
    for filename in ['motifs.csv', 'dimension_audit.csv', 'EXPERIMENT_DESIGN.md']:
        shutil.copyfile(reports/'unit_motifs'/filename, target/filename)
    write_json(target/'summary.json', {'nmf': nmf, 'motifs': motifs})


def build(out, workers):
    assert not out.exists(), f'Use a new output directory: {out}'
    source = EXPERIMENT/'outputs/analysis/semantic_umap_top100'
    reports = EXPERIMENT/'reports'
    identity_guard(source, reports)
    snapshot = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M %Z')
    out.mkdir(parents=True)
    (out/'assets').mkdir()
    shutil.copyfile(TEMPLATES/'site.css', out/'assets/site.css')
    shutil.copyfile(TEMPLATES/'preview.js', out/'assets/preview.js')
    print('Building current main experiment pages', flush=True)
    tasks, dimensions, examples = semantic_pages(out, source, reports)
    masked, counts = interpretation_pages(out, reports, snapshot)
    tasks += masked
    image_rows = [json.loads(line) for line in (source/'images.jsonl').read_text().splitlines()]
    assert len(image_rows) == 14848
    for i in range(0, len(image_rows), 4):
        group = image_rows[i:i+4]
        assert len({(r['city'], r['pano']) for r in group}) == 1
        assert [r['heading'] for r in group] == [0, 90, 180, 270]
    tasks += [(source/f'sources/{i}.jpg', out/f'semantic/sources/{i}.jpg', 'copy') for i in range(len(image_rows))]
    overview(out, snapshot, counts)
    # Publish only figures actually used by the current pages.
    figure_names = set()
    for document in out.rglob('*.html'):
        figure_names.update(re.findall(r'assets/figures/([^"<>]+)', document.read_text()))
    (out/'assets/figures').mkdir()
    for filename in figure_names:
        shutil.copyfile(EXPERIMENT/'results/figures'/filename, out/'assets/figures'/filename)
    (out/'data/training').mkdir(parents=True)
    shutil.copyfile(EXPERIMENT/'results/training/history.csv', out/'data/training/history.csv')
    training = read_json(EXPERIMENT/'results/training/training_report.json')
    write_json(out/'data/training_summary.json', {key: training[key] for key in ['status', 'epoch', 'epochs', 'best_val_squared_l2', 'identity']})
    for name, dest in [('semantic_umap/feature_semantic_umap_groups.csv', 'feature_semantic_umap_groups.csv'),
                       ('semantic_umap/group_palette.json', 'group_palette.json'),
                       ('vlm_groups/group_interpretations.csv', 'group_interpretations.csv')]:
        shutil.copyfile(reports/name, out/'data'/dest)
    print(f'Packing {len(tasks):,} referenced images; {len(masked):,} exact Qwen input sheets', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(copy_image, tasks), 1):
            if i % 2000 == 0:
                print(f'Images {i:,}/{len(tasks):,}', flush=True)
    for src, dst, kind in masked[:2]:
        with Image.open(src) as a, Image.open(dst) as b:
            assert a.convert('RGB').tobytes() == b.convert('RGB').tobytes(), 'Lossless Qwen sheet check failed'
    # Share the primary masked sheets after the image queue has materialized.
    element_summary = install_element_hierarchy(out, page, snapshot)
    label_structure_counts = install_label_structure(out, page, snapshot)
    quality_counts = install_quality_review(out, page, snapshot)
    install_progress(out)
    (out/'.nojekyll').touch()
    # A compatibility entry contains only a redirect to the current homepage.
    (out/'latest.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=./"><title>Urban Scene Primitives</title><a href="./">主实验结果</a></html>')
    (out/'404.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Urban Scene Primitives</title>
<link rel="stylesheet" href="/urban-visual-gene/assets/site.css"><main class="page"><h1>页面已移除</h1>
<p>网站已整理为当前主实验。请从<a href="/urban-visual-gene/">主实验首页</a>继续查看结果。</p></main></html>''')
    workflows = out/'.github/workflows'
    workflows.mkdir(parents=True)
    (workflows/'pages.yml').write_text('''name: Deploy main experiment to GitHub Pages
on:
  push:
    branches: [gh-pages]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: '.'
      - id: deployment
        uses: actions/deploy-pages@v4
''')
    files = [p for p in out.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    assert size < 990_000_000, f'Website exceeds the intended Pages size budget: {size}'
    write_json(out/'manifest.json', {'experiment': EXPERIMENT.name, 'checkpoint_sha256': CHECKPOINT,
        'snapshot_label': snapshot, 'model': {'backbone': 'DINOv3 ViT-L/16', 'image_size': 448, 'grid': [28,28], 'input_dimension': 1024, 'dictionary_width': 4096, 'batchtopk_k': 32, 'panorama_context_mixing': True},
        'training': {'cities': 30, 'panoramas': 89031, 'train': 85191, 'validation': 3840},
        'image_semantics': {'cities': 29, 'panoramas': 3712, 'images': len(image_rows), 'dimensions': dimensions, 'projected_dimensions': 2863, 'top100_instances': examples, 'top10_umap_points': 29934},
        'vlm_status_counts': counts, 'label_structure': {'page': 'label-structure/',
            'counts': label_structure_counts, 'validation': 'provisional_text_only'},
        'quality_review': {'page': 'quality/', 'counts': quality_counts,
            'status': 'candidate_review_only'}, 'file_count': len(files)+1, 'bytes_before_manifest': size,
        'element_hierarchy': {'page': 'elements/', 'identity': element_summary['identity'],
                              'candidate_dimensions': element_summary['status_counts']['candidate_ready'],
                              'semantic_validation': 'pending',
                              'relation_page': 'relations/',
                              'label_gallery': element_summary['label_gallery'],
                              'prototype_route': element_summary['prototype_route'],
                              'local_composition_status': element_summary['local_composition_status']} if element_summary else None,
        'packaging': 'Original directional JPEGs copied unchanged; Top100 crops and previews reconstructed on demand using saved boxes; feature JSON compressed with gzip; Qwen masked input sheets encoded as pixel-exact lossless WebP.',
        'source_result_manifest_sha256': hashlib.sha256((EXPERIMENT/'results/manifest.json').read_bytes()).hexdigest()})
    print(json.dumps({'output': str(out), 'files': len(files)+1, 'MB': round(size/1e6,1), 'vlm': counts}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the static bundle')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    build(args.output.resolve(), max(1, min(args.workers, 8)))
