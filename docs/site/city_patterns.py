"""Publish verified 30-city spatial outputs without copying activation caches."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

FOLLOWUP = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/followup')
LABELS = {'barrier': '围挡／护栏', 'water': '水体', 'vehicle': '车辆',
          'construction': '施工', 'transport_structure': '交通构筑物',
          'low_vegetation': '低矮植被', 'terrain': '地形', 'utility': '市政设施'}


def read(path):
    return json.loads(path.read_text())


def checked_inputs(followup):
    source = read(followup / 'complete.json')
    identity = source['identity']['extract_run_identity']
    if source['cities'] != 30:
        raise ValueError('Expected a complete 30-city sample')
    stages = {}
    for stage in ('candidate_combinations', 'road_sides', 'prototype_city_distribution', 'chapter5_figures'):
        folder = followup / 'stages' / stage
        marker = read(folder / 'complete.json')
        if marker.get('status') != 'complete' or marker.get('run_identity') != identity:
            raise ValueError(f'Incomplete or mismatched publication source: {stage}')
        stages[stage] = folder
    report = read(stages['chapter5_figures'] / 'compile_report.json')
    if report.get('passed') is not True:
        raise ValueError('Thesis compile gate failed')
    return identity, source, stages


def label(candidate):
    return ' + '.join(LABELS.get(word, word) for word in candidate.split('+'))


def install(site, page, snapshot, followup=FOLLOWUP):
    identity, source, stages = checked_inputs(followup)
    target = site / 'city-patterns'
    target.mkdir(parents=True, exist_ok=True)
    data = target / 'data'
    data.mkdir(exist_ok=True)
    (site / 'assets/figures').mkdir(parents=True, exist_ok=True)
    figures = stages['chapter5_figures'] / 'figures'
    for image in figures.iterdir():
        if image.suffix in {'.png', '.pdf'}:
            shutil.copy2(image, site / 'assets/figures' / image.name)
    # Bounded public tables and protocols; no raw features or full-corpus masks.
    for stage, folder in stages.items():
        for name in ('protocol.json', 'summary.json', 'figure_inventory.json', 'compile_report.json'):
            if (folder / name).is_file():
                shutil.copy2(folder / name, data / f'{stage}_{name}')
    for stage, name in (('candidate_combinations', 'candidates.csv'),
                        ('prototype_city_distribution', 'city_prototype_distribution.csv')):
        shutil.copy2(stages[stage] / name, data / name)
    with (data / 'candidates.csv').open(newline='') as stream:
        candidates = [r for r in csv.DictReader(stream) if r['status'] == 'co_presence_candidate']
    audit = read(stages['road_sides'] / 'summary.json')
    distribution = read(stages['prototype_city_distribution'] / 'summary.json')
    examples = read(stages['road_sides'] / 'examples.json')
    public_examples = []
    for i, item in enumerate(examples):
        value = {k: v for k, v in item.items() if k not in {'original', 'overlay'}}
        for kind in ('original', 'overlay'):
            image = Path(item[kind])
            # Refuse an unexpected source path, including symlink escapes.
            image.resolve().relative_to((stages['road_sides'] / 'evidence').resolve())
            rel = Path('examples') / f'{i:03d}-{kind}.webp'
            (target / rel).parent.mkdir(exist_ok=True)
            shutil.copy2(image, target / rel)
            value[kind] = rel.as_posix()
        public_examples.append(value)
    (target / 'examples.json').write_text(json.dumps(public_examples, ensure_ascii=False, indent=2))
    manifest = {'run_identity': identity, 'cities': 30, 'panoramas': source['processed'],
                'cells': source['valid_cells'], 'candidates': len(candidates), 'audit': audit,
                'validated_prototypes': distribution['validated_prototypes'],
                'roadside_prevalence_available': False, 'scientific_review': 'pending'}
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def figure(stem, caption):
        caption = html.escape(caption)
        return (f'<figure><a href="../assets/figures/{stem}.png"><img loading="lazy" '
                f'src="../assets/figures/{stem}.png" alt="{caption}"></a><figcaption>{caption} '
                f'<a href="../assets/figures/{stem}.pdf">PDF</a></figcaption></figure>')

    sections = []
    for row in candidates:
        candidate = row['candidate']
        cards = []
        pool = [r for r in public_examples if r['candidate'] == candidate]
        for item in pool:
            city = item['city'].split('/')[-1]
            hypothesis = {'same_side': '同侧假设', 'opposite_side': '两侧假设',
                          'undetermined': '几何不明确'}[item['tentative_side']]
            cards.append(f'<details class="evidence"><summary>{html.escape(city)} · '
                f'{hypothesis} · 最终侧别待核查</summary><p>panoid: {html.escape(item["panoid"])} '
                f'· 格网 ({item["grid_x"]}, {item["grid_y"]})</p>'
                f'<a href="{item["original"]}"><img loading="lazy" src="{item["original"]}" '
                'alt="完整四方向原图，0／90／180／270 度"></a>'
                f'<a href="{item["overlay"]}"><img loading="lazy" src="{item["overlay"]}" '
                'alt="两个 SAE 要素响应和独立道路遮罩"></a></details>')
        sections.append(f'<section class="section" id="{html.escape(candidate)}"><h2>{label(candidate)}</h2>'
            f'<p><code>{candidate}</code> · 确认集 {int(row["confirmation_panoramas"]):,} 个全景 / '
            f'{int(row["confirmation_cells"]):,} 个格网 / {row["confirmation_cities"]} 城。'
            f'两种零假设 lift：{float(row["lift_city"]):.2f}、{float(row["lift_spatial"]):.2f}；'
            f'BH q={float(row["q_both"]):.4f}。这检验共同出现，尚未检验具体布局。</p>'
            + figure('Fig_CityPatterns_Map_' + candidate.replace('+', '_'),
                     '全部 30 城的观测格网：格内阳性全景比例；空白区域未采样。')
            + f'<h3>四方向证据 · {len(pool)} 个实例</h3><p>展开查看原图与高亮；可点击图片放大。</p>'
            + ''.join(cards) + '</section>')
    content = f'''<div class="eyebrow">30-city spatial sample · frozen SAE</div>
<h1>城市景观组成</h1><p class="lead">在扩大样本中检索重复要素组合，再回到完整四方向图像核查。</p>
<div class="stats"><div class="stat"><b>{source['processed']:,}</b><span>有效全景</span></div>
<div class="stat"><b>{source['valid_cells']:,}</b><span>观测 1 km 格网</span></div>
<div class="stat"><b>30</b><span>城市 · 每格最多 10 点</span></div>
<div class="stat"><b>{len(candidates)}</b><span>暂定共现候选</span></div></div>
<div class="note">当前结果是暂定 Qwen 要素父类的共同出现。已验证景观原型：0。
道路同侧／两侧需要原始相机标定和人工审阅；没有测量的发生率记为缺失，不记为 0。
原 SAE 的 89,031 个训练与验证全景保持原有范围；本页复用冻结模型。</div>
<section class="section"><h2>筛选与统计单位</h2><p>每个要素须占全景全部 4,096 维正激活质量至少 2%。
22 城发现、8 城确认；999 次保留边际频率的零假设抽样，分别控制城市和城市内 5 km 空间块。
使用两者较大的 p 值做 BH-FDR，要求 q≤0.05、两种 lift≥1.05，且每个划分至少覆盖 30 个全景、8 个格网及 3 城。
城市发生率先计算格内阳性全景比例，再对观测格网等权平均。置信区间按 5 km 块重抽样 999 次，未包含标签错误及候选选择的不确定性。</p></section>
<section class="section" id="distributions"><h2>跨城市分布</h2>
{figure('Fig_CityPatterns_Candidate_Cities', '7 个候选的格网平衡城市发生率。')}
{figure('Fig_CityPatterns_Candidate_Uncertainty', '城市发生率及 95% 空间块 bootstrap 区间；条件性描述统计。')}</section>
<section class="section"><h2>如何读取高亮</h2><p>琥珀色与青色对应候选的两个 SAE 要素；紫色为独立 SegFormer 预测的道路表面。
要素遮罩使用该全景各要素最大响应的 35% 阈值，重叠处按绘制顺序覆盖。这是 patch 响应归因，并非对象实例分割。</p>
<p>{audit['examples']} 个审阅实例覆盖 {audit['cities']} 城、{audit['cells']} 格。
最终同侧 {audit['same_side']}、两侧 {audit['opposite_side']}、无法确认 {audit['undetermined']}；
几何候选只用于审阅，不能从这些检索实例推算全样本侧别分布。</p></section>
<nav aria-label="候选跳转">{' · '.join(f'<a href="#{r["candidate"]}">{label(r["candidate"])}</a>' for r in candidates)}</nav>
{''.join(sections)}
<section class="section"><h2>数据与论文图表</h2><p><a href="data/candidates.csv">全部候选及检验 CSV</a> ·
<a href="data/city_prototype_distribution.csv">城市发生率 CSV</a> · <a href="examples.json">图像审阅记录</a> ·
<a href="data/candidate_combinations_protocol.json">共现协议</a> · <a href="data/road_sides_protocol.json">道路审阅协议</a> ·
<a href="data/prototype_city_distribution_protocol.json">分布协议</a> ·
<a href="data/chapter5_figures_figure_inventory.json">论文图表来源清单</a> ·
<a href="data/chapter5_figures_compile_report.json">论文编译报告</a></p>
<p>本次不重新训练 SAE。训练诊断、维度语义 UMAP 及历史 VLM 图像证据保留各自原始样本身份。
本次城市组成、共现、空间结构和格网 UMAP 使用扩大的空间样本。</p></section>'''
    styles = '<style>.evidence{margin:12px 0;padding:12px;border:1px solid #d6ddd9;border-radius:10px}.evidence img{width:100%;height:auto;display:block;margin:10px 0}.evidence p{overflow-wrap:anywhere}.evidence summary{cursor:pointer}</style>'
    (target / 'index.html').write_text(page('城市景观组成', 'city-patterns', styles + content, snapshot))
    # Retire the published pilot relation gallery while keeping links usable.
    old = site / 'relations'
    if old.exists():
        shutil.rmtree(old)
    old.mkdir()
    (old / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=../city-patterns/"><title>城市景观组成</title>'
        '<a href="../city-patterns/">查看当前 30 城组成结果</a></html>')
    # Refresh navigation on existing static pages without rebuilding image galleries.
    from build_main_site import nav
    import re
    for path in site.glob('*/index.html'):
        if path.parent.name in {'semantic-map', 'instances-map'}:
            continue
        text = path.read_text()
        if '<nav class="site-nav"' in text:
            text = re.sub(r'<nav class="site-nav".*?</nav>',
                          lambda _: nav(path.parent.name, '../'), text, count=1, flags=re.S)
        text = text.replace('../relations/#local-results', '../city-patterns/')
        text = text.replace('查看独立的要素空间关系', '查看城市组成与四方向审阅实例')
        path.write_text(text)
    home = site / 'index.html'
    text = home.read_text()
    text = re.sub(r'<nav class="site-nav".*?</nav>', lambda _: nav('overview'), text, count=1, flags=re.S)
    text = re.sub(r'<section class="note" id="element-route">.*?</section>',
        '<section class="note" id="element-route"><strong>当前路线：30 城景观组成与空间审阅</strong>'
        '<p>扩大样本中的共现筛选、城市分布和四方向证据已完成计算；语义与道路侧别仍待核查。</p>'
        '<a href="city-patterns/">查看当前结果 →</a> · <a href="elements/">查看标签激活 →</a></section>',
        text, flags=re.S)
    text = re.sub(r'<section class="section"><h2>跨街景的局部要素组合</h2>.*?</section>',
        f'<section class="section"><h2>30 城景观组成</h2><p>{source["processed"]:,} 个全景、'
        f'{source["valid_cells"]:,} 个格网；7 个暂定共现候选，城市分布及四方向审阅实例已更新。</p>'
        '<a href="city-patterns/">查看当前结果 →</a></section>', text, flags=re.S)
    text = text.replace('64 组视觉组成的城市比较', '64 组激活质量：格内平均后，各格等权的城市比较')
    text = text.replace('<b>89,031</b><span>完整四方向全景</span>',
                        '<b>89,031</b><span>SAE 训练与验证全景</span>')
    text = text.replace('视觉组共同出现的描述性统计', '格网级共现 PPMI：城市与格网等权；非显著性检验')
    text = text.replace('视觉组成的空间结构', '观测 1 km 格网的组成相似性；至少 20 对相邻格方可绘入')
    text = text.replace('原图、激活热图与高响应区域', '新四方向证据：琥珀／青色为两要素，紫色为独立道路预测；侧别待核查')
    if 'id="city-sample-scope"' not in text:
        text = text.replace('<h2>城市与视觉组成</h2>', '<h2>城市与视觉组成</h2>'
            '<p id="city-sample-scope">本节更新为 681,645 个全景、72,625 个观测格网、30 城；'
            '使用正激活质量，格内平均后格网等权。训练与语义图像证据各自保留原有样本身份。</p>')
    home.write_text(text)
    root_manifest = site / 'manifest.json'
    if root_manifest.exists():
        value = read(root_manifest)
        value['city_patterns'] = {'page': 'city-patterns/', **manifest}
        if value.get('element_hierarchy'):
            value['element_hierarchy'].update(relation_page='city-patterns/',
                prototype_route='spatial_family_co_presence', local_composition_status='retired_pilot')
        files = [p for p in site.rglob('*') if p.is_file() and '.git' not in p.parts]
        value['file_count'] = len(files)
        value['bytes_before_manifest'] = sum(p.stat().st_size for p in files if p != root_manifest)
        value['snapshot_label'] = snapshot
        root_manifest.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site-dir', type=Path, required=True)
    p.add_argument('--followup', type=Path, default=FOLLOWUP)
    args = p.parse_args()
    from build_main_site import page
    print(json.dumps(install(args.site_dir, page, datetime.now(timezone.utc).isoformat(), args.followup), ensure_ascii=False))
