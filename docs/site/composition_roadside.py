"""Publish a bounded image audit of roadside composition hypotheses."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def checked_audit(followup):
    compositions = followup / 'stages/composition_patterns'
    folder = followup / 'stages/composition_roadside_audit'
    marker = read(folder / 'complete.json')
    summary = read(folder / 'summary.json')
    protocol = read(folder / 'protocol.json')
    base = read(compositions / 'summary.json')
    if (marker['status'] != 'complete' or marker['run_identity'] != base['run_identity']
            or summary['run_identity'] != base['run_identity']
            or summary['protocol_identity'] != marker['identity']
            or protocol['identity'] != marker['identity']
            or protocol['composition_protocol_sha256'] != hashlib.sha256(
                (compositions / 'protocol.json').read_bytes()).hexdigest()):
        raise ValueError('Roadside audit and landscape composition identities differ')
    if (summary['patterns_total'] != base['recurrent_patterns']
            or summary['patterns_side_eligible'] + summary['patterns_not_side_eligible'] != base['recurrent_patterns']
            or summary['sampled_pattern_instances'] != marker['examples']
            or sum(summary['tentative'].values()) != marker['examples']
            or summary['human_validated'] or summary['calibration_verified']
            or summary['corpus_roadside_prevalence_estimable']
            or summary['final'] != {'same_side': 0, 'opposite_sides': 0,
                                    'undetermined': marker['examples']}):
        raise ValueError('Roadside audit cannot claim validated or corpus-wide prevalence')
    return folder, summary


def install(site, page, snapshot, followup):
    folder, summary = checked_audit(followup)
    target = site / 'road-sides'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'protocol.json', 'pattern_side_audit.csv'):
        shutil.copy2(folder / name, target / name)
    with (folder / 'pattern_side_audit.csv').open(newline='') as stream:
        patterns = list(csv.DictReader(stream))
    raw = read(folder / 'examples.json')
    if len(patterns) != summary['patterns_total'] or len(raw) != summary['sampled_pattern_instances']:
        raise ValueError('Roadside pattern or example table is incomplete')
    known = {r['pattern_id'] for r in patterns}
    if len(known) != len(patterns) or any(r['pattern_id'] not in known for r in raw):
        raise ValueError('Roadside example references an unknown pattern')
    for row in patterns:
        if int(row['sampled_panoramas']) != sum(int(row[k]) for k in (
                'same_side_hypotheses', 'opposite_sides_hypotheses', 'undetermined_hypotheses')):
            raise ValueError('Roadside per-pattern counts disagree')
    if sum(int(r['sampled_panoramas']) for r in patterns) != len(raw):
        raise ValueError('Roadside sample counts disagree with evidence')
    # All tentative cases are exposed for review; unresolved examples are a
    # small deterministic context sample, not chosen to improve apparent rates.
    tentative = [r for r in raw if r['tentative_side'] != 'undetermined']
    unresolved = [r for r in raw if r['tentative_side'] == 'undetermined'][:8]
    chosen = tentative + unresolved
    gallery = []
    for index, row in enumerate(chosen):
        public = {k: row[k] for k in ('pattern_id', 'city', 'panoid',
            'lateral_labels', 'tentative_side', 'final_side')}
        for kind in ('original', 'overlay', 'road_overlay'):
            source = Path(row[kind])
            source.resolve().relative_to((folder / 'evidence').resolve())
            dest = Path('evidence') / f'{index:03d}-{kind}.webp'
            (target / dest).parent.mkdir(exist_ok=True)
            shutil.copy2(source, target / dest)
            public[kind] = dest.as_posix()
        gallery.append(public)
    (target / 'gallery.json').write_text(json.dumps(gallery, ensure_ascii=False, indent=2))
    h = html.escape
    rows = []
    for item in patterns:
        if item['eligible_for_side_audit'] != 'True':
            continue
        side = int(item['same_side_hypotheses'])
        cross = int(item['opposite_sides_hypotheses'])
        unknown = int(item['undetermined_hypotheses'])
        rows.append(f'<tr data-label="{h(item["label"])}" data-class="{("both" if side and cross else "same" if side else "cross" if cross else "unknown")}">'
            f'<td>{h(item["label"])}</td><td>{int(item["sampled_panoramas"])}</td>'
            f'<td>{side}</td><td>{cross}</td><td>{unknown}</td></tr>')
    rows.sort(key=lambda row: ('data-class="unknown"' in row, row))
    evidence = []
    labels = {'same_side': '同侧假设', 'opposite_sides': '两侧假设',
              'undetermined': '无法判定'}
    for item in gallery:
        title = ' ＋ '.join(item['lateral_labels'])
        evidence.append(f'<details class="road-evidence"><summary>{h(labels[item["tentative_side"]])} · '
            f'{h(title)} · {h(item["city"])}</summary><p>panoid: {h(item["panoid"])}；'
            '该标签仅是未经相机标定与独立语义核查的几何假设。</p>'
            f'<img loading="lazy" src="{item["original"]}" alt="四方向原图">'
            f'<img loading="lazy" src="{item["overlay"]}" alt="两个或三个景观单元的SAE响应高亮">'
            f'<img loading="lazy" src="{item["road_overlay"]}" alt="独立道路分割，高亮为紫色"></details>')
    content = f'''<link rel="stylesheet" href="road-sides.css">
<div class="eyebrow">30-city sample · roadside image audit</div><h1>道路同侧与两侧</h1>
<p class="lead">按四方向激活与独立道路分割，审阅当前景观组合的道路侧别。</p>
<div class="stats"><div class="stat"><b>{summary['patterns_side_eligible']}</b><span>可定义侧别的组合</span></div>
<div class="stat"><b>{summary['tentative']['same_side']}</b><span>同侧几何假设</span></div>
<div class="stat"><b>{summary['tentative']['opposite_sides']}</b><span>两侧几何假设</span></div>
<div class="stat"><b>{summary['tentative']['undetermined']}</b><span>抽样实例无法判定</span></div></div>
<p class="note">每个可审阅组合最多抽取 {summary['per_pattern_requested']} 个不同城市的全景，
共 {summary['sampled_pattern_instances']} 个组合实例、{summary['unique_panoramas']} 个不同全景。
其他 {summary['patterns_not_side_eligible']} 个组合不足两个可定义道路侧别的要素。
这里的数字是抽样实例数，不是组合的总发生次数，也不能外推为 30 城全样本道路侧别比例。</p>
<p class="note">所有实例的最终核查状态仍为“未确认”：相机朝向和视场角未完成标定，
SAE 响应不是对象实例分割，部分 Qwen 标签存在语义错误。
“同侧／两侧假设”仅供挑选人工复核样本。</p>
<p><a href="../city-patterns/">返回景观组合图谱</a> ·
<a href="pattern_side_audit.csv">下载逐组合统计</a> · <a href="protocol.json">方法协议</a></p>
<section class="section"><h2>逐组合审阅统计</h2>
<div class="road-controls"><label>搜索组合 <input id="road-search" type="search" placeholder="建筑、乔木、围栏…"></label>
<label>几何假设 <select id="road-filter"><option value="">全部</option><option value="same">含同侧</option>
<option value="cross">含两侧</option><option value="unknown">全部无法判定</option></select></label></div>
<p id="road-count" aria-live="polite"></p><div class="road-table-wrap"><table id="road-table"><thead><tr>
<th>景观组合</th><th>抽样全景</th><th>同侧假设</th><th>两侧假设</th><th>无法判定</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>道路本身、人行横道、桥梁／铁路／隧道和跨路架空电线不视为位于某一侧的单元；
含道路的三元组合，仅比较另外两个具备侧别意义的单元。完整 185 组（包括 67 组不适用）见 CSV。</p></section>
<section class="section"><h2>四方向图像核查</h2><p>下列完整原图、单元响应与紫色道路分割来自同一全景。
全部自动同侧／两侧候选及前 8 个无法判定的样本供复核；它们不是挑选后的成功率样本。</p>
{''.join(evidence)}</section>
<script src="road-sides.js" defer></script>'''
    (target / 'index.html').write_text(page('道路侧别审阅', 'city-patterns', content, snapshot))
    shutil.copy2(Path(__file__).with_name('templates') / 'composition-roadside.css', target / 'road-sides.css')
    shutil.copy2(Path(__file__).with_name('templates') / 'composition-roadside.js', target / 'road-sides.js')
    return {'summary': summary, 'gallery_examples': len(gallery), 'page': 'road-sides/'}
