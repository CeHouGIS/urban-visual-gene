"""Install the exploratory fine-grained composition atlas beside its protocol."""
from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import zlib

FOLLOWUP = Path('/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/vitl30_city_patterns_1km10/followup')
TEMPLATES = Path(__file__).with_name('templates')


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def checked_compositions(followup):
    folder = followup / 'stages/composition_patterns'
    source = read(followup / 'complete.json')
    marker = read(folder / 'complete.json')
    summary = read(folder / 'summary.json')
    identity = source['identity']['extract_run_identity']
    if marker.get('status') != 'complete' or marker.get('run_identity') != identity or summary.get('run_identity') != identity:
        raise ValueError('Mismatched or incomplete composition revision')
    if summary['panoramas'] != source['processed'] or summary['cells'] != source['valid_cells']:
        raise ValueError('Incomplete composition corpus')
    if summary['statistical_confirmation'] or summary['layout_verified'] or summary['labels_human_validated']:
        raise ValueError('Exploratory revision cannot be published as validated')
    if not summary.get('landscape_filter_applied'):
        raise ValueError('Landscape scope filter must be applied before publication')
    return folder, summary


def checked_roadside_rows(followup, patterns):
    if __package__:
        from .composition_roadside import checked_audit
    else:
        from composition_roadside import checked_audit
    folder, summary = checked_audit(followup)
    rows = csv_rows(folder / 'pattern_side_audit.csv')
    pattern_ids = {p['pattern_id'] for p in patterns}
    ids = [r['pattern_id'] for r in rows]
    if len(ids) != len(pattern_ids) or set(ids) != pattern_ids:
        raise ValueError('Roadside audit does not match the current composition catalogue')
    counts = {'same_side': 0, 'opposite_sides': 0, 'undetermined': 0}
    eligible = 0
    result = {}
    for row in rows:
        sampled = int(row['sampled_panoramas'])
        tentative = {key: int(row[field]) for key, field in (
            ('same_side', 'same_side_hypotheses'),
            ('opposite_sides', 'opposite_sides_hypotheses'),
            ('undetermined', 'undetermined_hypotheses'))}
        is_eligible = row['eligible_for_side_audit'] == 'True'
        if row['eligible_for_side_audit'] not in ('True', 'False'):
            raise ValueError('Invalid roadside eligibility')
        if sampled != sum(tentative.values()) or (not is_eligible and sampled):
            raise ValueError('Roadside sample counts disagree')
        if (int(row['final_same_side']) or int(row['final_opposite_sides'])
                or int(row['final_undetermined']) != sampled):
            raise ValueError('Roadside audit contains unverified final labels')
        eligible += is_eligible
        for key in counts:
            counts[key] += tentative[key]
        result[row['pattern_id']] = {'eligible': is_eligible, 'sampled': sampled,
                                     'tentative': tentative}
    if (eligible != summary['patterns_side_eligible']
            or counts != summary['tentative']
            or sum(r['sampled'] for r in result.values()) != summary['sampled_pattern_instances']):
        raise ValueError('Roadside audit totals disagree with its summary')
    return summary, result


def install(site, page, snapshot, followup=FOLLOWUP):
    folder, summary = checked_compositions(followup)
    target = site / 'city-patterns'
    data = target / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'protocol.json', 'calibration.json', 'sensitivity.json',
                 'threshold_diagnostics.csv', 'city_distribution.csv', 'all_discovered_patterns.csv',
                 'coverage_by_city.csv', 'units.json'):
        shutil.copy2(folder / name, data / ('composition_' + name))
    units = read(folder / 'units.json')
    patterns = read(folder / 'patterns.json')
    excluded = [u for u in units if not u['fallback'] and not u.get('landscape_eligible', False)]
    (data / 'composition_excluded_units.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2))
    if any(not units[i].get('landscape_eligible', False) for p in patterns for i in p['units']):
        raise ValueError('Excluded object leaked into landscape patterns')
    rates = csv_rows(folder / 'city_distribution.csv')
    cities = sorted({r['city'] for r in rates})
    if len(patterns) != summary['recurrent_patterns'] or len(cities) != 30:
        raise ValueError('Missing pattern/city records')
    rates_by_pattern = {}
    for row in rates:
        rates_by_pattern.setdefault(row['pattern_id'], {})[row['city']] = {
            'rate': float(row['cell_weighted_rate']), 'positive': int(row['positive_panoramas']),
            'panoramas': int(row['sampled_panoramas']), 'cells': int(row['positive_cells']),
            'observed_cells': int(row['observed_cells'])}
    for pattern in patterns:
        pattern['rates'] = [rates_by_pattern[pattern['pattern_id']][c] for c in cities]
        pattern['family_ids'] = [units[i]['family'] for i in pattern['units']]
    roadside_summary = None
    if (followup / 'stages/composition_roadside_audit/complete.json').exists():
        roadside_summary, roadside_rows = checked_roadside_rows(followup, patterns)
        for pattern in patterns:
            pattern['roadside_audit'] = roadside_rows[pattern['pattern_id']]
    payload = {'summary': summary, 'patterns': patterns, 'cities': cities,
               'coverage': csv_rows(folder / 'coverage_by_city.csv'),
               'sensitivity': read(folder / 'sensitivity.json')}
    cells = csv_rows(folder / 'observed_cells.csv')
    if len(cells) != summary['cells'] or any(not 1 <= int(r['sampled_panoramas']) <= 10 for r in cells):
        raise ValueError('Invalid observed cell sample counts')
    cells_payload = {'cities': cities, 'cells': [[cities.index(r['city']), int(r['grid_x']), int(r['grid_y']),
                                               int(r['sampled_panoramas'])] for r in cells]}
    (data / 'composition_cells.json').write_text(json.dumps(cells_payload, separators=(',', ':')))
    compressed = {}
    samples = bytes(int(r['sampled_panoramas']) for r in cells)
    pattern_totals = {r['pattern_id']: r['positive_panoramas'] for r in patterns}
    for row in read(folder / 'cell_patterns.json'):
        counts = base64.b64decode(row['positive_counts_u8'])
        if len(counts) != summary['cells'] or any(a > b for a, b in zip(counts, samples)):
            raise ValueError('Cell count vector has invalid counts or length')
        if sum(counts) != pattern_totals[row['pattern_id']]:
            raise ValueError('Map counts disagree with pattern support')
        compressed[row['pattern_id']] = base64.b64encode(zlib.compress(counts, 9)).decode()
    if set(compressed) != set(pattern_totals):
        raise ValueError('Missing or duplicate map patterns')
    (data / 'composition_cell_counts.json').write_text(json.dumps(compressed, separators=(',', ':')))
    examples = []
    review_file = folder / 'evidence_review.json'
    if not review_file.exists():
        review_file = Path(__file__).resolve().parents[2] / 'experiments/vitl30_city_patterns/results/composition_evidence_review.json'
    review = read(review_file) if review_file.exists() else {}
    review_by_example = {(r['pattern_id'], r['city'], r['panoid']): r for r in review.get('examples', [])}
    if (folder / 'examples.json').exists():
        marker = read(folder / 'evidence_complete.json')
        protocol_hash = hashlib.sha256((folder / 'protocol.json').read_bytes()).hexdigest()
        if (marker['status'] != 'complete' or marker['run_identity'] != summary['run_identity']
                or marker.get('protocol_sha256') != protocol_hash):
            raise ValueError('Incomplete composition image evidence')
        # Remove images from the superseded exploratory composition revision.
        shutil.rmtree(target / 'composition-examples', ignore_errors=True)
        for i, row in enumerate(read(folder / 'examples.json')):
            if row['pattern_id'] not in pattern_totals:
                raise ValueError('Image evidence belongs to a superseded pattern')
            public = {k: v for k, v in row.items() if k not in {'original', 'overlay'}}
            audit = review_by_example.get((row['pattern_id'], row['city'], row['panoid']))
            if audit:
                public['spot_check_status'] = audit['status']
                public['review_note'] = audit['note']
            for kind in ('original', 'overlay'):
                source = Path(row[kind])
                source.resolve().relative_to((folder / 'evidence').resolve())
                relative = Path('composition-examples') / f'{i:03d}-{kind}.webp'
                (target / relative).parent.mkdir(exist_ok=True)
                shutil.copy2(source, target / relative)
                public[kind] = relative.as_posix()
            examples.append(public)
    for pattern in patterns:
        pattern['example_count'] = sum(e['pattern_id'] == pattern['pattern_id'] for e in examples)
        pattern['audit_flags'] = [e['review_note'] for e in examples if e['pattern_id'] == pattern['pattern_id']
                                  and e.get('spot_check_status') == 'suspected_semantic_mismatch']
    (data / 'compositions.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    (target / 'composition-examples.json').write_text(json.dumps(examples, ensure_ascii=False))
    (target / 'composition-evidence-review.json').write_text(json.dumps(review, ensure_ascii=False))
    for name in ('composition-atlas.css', 'composition-atlas.js'):
        shutil.copy2(TEMPLATES / name, target / name)
    s = summary
    roadside = None
    if roadside_summary:
        if __package__:
            from .composition_roadside import install as install_roadside
        else:
            from composition_roadside import install as install_roadside
        roadside = install_roadside(site, page, snapshot, followup)
    roadside_section = (f'''<section class="section" id="roadside-summary"><h2>道路同侧与两侧：抽样核查</h2>
<div class="stats"><div class="stat"><b>{roadside_summary['patterns_side_eligible']}</b><span>可定义侧别的组合／{roadside_summary['patterns_total']} 组</span></div>
<div class="stat"><b>{roadside_summary['tentative']['same_side']}</b><span>同侧几何假设 · 抽样实例</span></div>
<div class="stat"><b>{roadside_summary['tentative']['opposite_sides']}</b><span>两侧几何假设 · 抽样实例</span></div>
<div class="stat"><b>{roadside_summary['tentative']['undetermined']}</b><span>抽样实例无法判定</span></div></div>
<p>从 {roadside_summary['patterns_side_eligible']} 组可审阅组合各抽取最多 {roadside_summary['per_pattern_requested']} 个不同城市的实例，共 {roadside_summary['sampled_pattern_instances']} 个组合实例、
{roadside_summary['unique_panoramas']} 个不同全景；另有 {roadside_summary['patterns_not_side_eligible']} 组不适用。
以下是假设性几何判断，全部实例均未完成人工确认，不能外推为 30 城侧别比例。</p>
<p><a href="../road-sides/">查看四方向原图、道路分割与逐组合审阅 →</a></p></section>'''
                        if roadside_summary else '')
    content = f'''<link rel="stylesheet" href="composition-atlas.css">
<div class="eyebrow">30-city spatial sample · exploratory composition atlas</div>
<h1>城市景观组成</h1><p class="lead">从细分要素中挖掘反复出现的 2—4 类组合，查看它们在 30 城的响应分布。</p>
<div class="stats"><div class="stat"><b>{s['canonical_units']}</b><span>归并部件后的语义响应单元</span></div>
<div class="stat"><b>{s['recurrent_patterns']}</b><span>重复高响应组合 · 数量由支持度决定</span></div>
<div class="stat"><b>{s['equal_city_cell_coverage']:.1%}</b><span>全景匹配覆盖 · 城市与格网等权</span></div>
<div class="stat"><b>{s['panoramas']:,}</b><span>四方向全景 · {s['cells']:,} 个观测格网</span></div></div>
<p>本次景观构成保留建筑、植被、道路空间、水体地形及固定设施。已排除人、车辆、集装箱、施工机械及列明的临时对象，
共排除 {s['excluded_units']} 个响应单元；上级综合响应也排除这些维度。
<a href="data/composition_excluded_units.json">查看排除清单</a>。</p>
<p class="note">这些是 SAE 的相对高响应组合。Qwen 标签待核查；匹配比例不等于真实对象发生率，
全景匹配覆盖也不是 80% patch 解释率。对象数量、相邻关系与道路侧别仍需图像和几何验证。
本轮沿用已检查过的城市划分，属于探索性修订。</p>
<p class="note">图像抽查已发现部分“水体”响应落在天空、“桥梁”响应落在标志支架等语义疑点。
相关组合和实例已单独标记，请先核查高亮，再解释其城市分布。
<a href="composition-evidence-review.json">查看抽查记录</a>（局部抽查，不估计整体准确率）。</p>
{roadside_section}
<section class="section"><h2>算法怎样调整</h2>
<p>原来的 7 组来自“显著超额共现”筛选。新图谱改为先描述反复出现的构成：保留细标签与树种、设施类型，
将墙、窗、屋顶等同对象部件归并，每个对象父类每张全景只贡献一个响应单元。</p>
<p>各类阈值由 22 个发现城市的正响应第 60 百分位校准；
城市、格网和格内全景逐级等权。父类至少占全部激活质量的 0.5%，细单元至少 0.1%，以排除弱背景响应；
这些下限仍需人工校准，不能视为物体存在判据。阈值原样应用于全部城市，未按各城分别调阈值。
搜索 2—4 类组合，要求发现城市与复查城市的等权匹配比例均≥0.5%，且各覆盖≥30 全景、8 格网、3 城。
若增加要素仍保留原组合≥90%的发现支持，则合并近重复子组合。</p>
<p>这里不以显著性决定是否展示。空间块零假设的 lift 单独列出，未进行本轮显著性确认；
50／60／70 百分位阈值的敏感性结果见下方。</p></section>
<section class="section" id="compositions"><h2>组合目录</h2>
<div class="atlas-controls"><label>搜索要素 <input id="pattern-search" type="search" placeholder="例如 棕榈、建筑、路灯"></label>
<label>组合阶数 <select id="pattern-order"><option value="">全部</option><option value="2">2 类</option><option value="3">3 类</option><option value="4">4 类</option></select></label>
<label>排序 <select id="pattern-sort"><option value="support">重复程度</option><option value="complexity">复杂组合优先</option></select></label>
<label>图像证据 <select id="pattern-evidence"><option value="">全部组合</option><option value="yes">有四方向实例</option></select></label></div>
<p id="pattern-count" aria-live="polite">正在读取组合…</p><div id="pattern-cards" class="pattern-cards"></div>
<div class="atlas-controls"><button id="prev-patterns">上一页</button><span id="pattern-page"></span><button id="next-patterns">下一页</button></div></section>
<section class="section" id="distribution"><h2 id="selected-title">城市分布</h2>
<p id="selected-detail"></p><p id="selected-roadside" class="note"></p><div id="city-rates" class="city-rates"></div>
<h3>全部 30 城的观测格网</h3><p>每个小格的颜色表示格内匹配全景比例；灰色为有样本但未匹配，空白为未采样。
统一色标：0%—100%。鼠标停在格网上可查看实际匹配数与样本数；坐标为当地 UTM 公里。</p>
<div class="atlas-legend"><span>0%</span><i></i><span>100%</span></div><p id="map-status" aria-live="polite"></p>
<div id="city-maps" class="city-maps"></div><h3>四方向激活证据</h3>
<p>彩色部分为所列响应单元的 patch 归因；同一位置响应重叠时，显示归一化强度最大的单元。仍需人工核查语义。</p>
<div id="composition-evidence"></div></section>
<section class="section"><h2>阈值敏感性与覆盖</h2><div class="table-scroll"><table><thead><tr><th>正响应百分位</th><th>固定组合数</th><th>复查城市仍重复</th><th>格网平衡全景覆盖</th><th>原始全景覆盖</th></tr></thead><tbody>
{''.join(f'<tr><td>{r["quantile"]:.0%}</td><td>{r["frozen_patterns"]}</td><td>{r["patterns_repeated_evaluation"]}</td><td>{r["equal_city_cell_coverage"]:.1%}</td><td>{r["panorama_coverage"]:.1%}</td></tr>' for r in payload['sensitivity'])}
</tbody></table></div><p>敏感性分析冻结主设置发现的组合，不在各个阈值重新挑选组合。
原 7 组的格网平衡匹配覆盖为 {s['old_seven_equal_city_cell_coverage']:.1%}；新旧覆盖的判据不同，不能据此推断识别精度提高。</p></section>
<section class="section"><h2>下载与复核</h2><p>
<a href="data/composition_protocol.json">算法协议</a> · <a href="data/composition_calibration.json">冻结阈值</a> ·
<a href="data/composition_units.json">响应单元与 SAE 维度</a> ·
<a href="data/composition_all_discovered_patterns.csv">全部发现组合</a> · <a href="data/composition_city_distribution.csv">30 城匹配比例</a> ·
<a href="data/composition_threshold_diagnostics.csv">旧新阈值诊断</a> ·
<a href="data/composition_coverage_by_city.csv">逐城全景覆盖</a></p>
<details><summary>历史统计协议（不属于当前景观目录）</summary><p>原 7 组含现已排除的活动对象，仅供统计追溯，不纳入本次景观构成。
原检验回答特定 2% 规则下是否超出随机共现；本轮图谱回答哪些相对高响应构成反复出现。
两种结果的统计问题不同。<a href="data/candidates.csv">原检验表</a> ·
<a href="data/candidate_combinations_protocol.json">原协议</a> · <a href="examples.json">原四方向审阅记录</a>。
论文仍保留原 7 组的统计结论，本轮探索结果尚未写成新的确认性结论。</p></details></section>
<script src="composition-atlas.js" defer></script>'''
    (target / 'index.html').write_text(page('城市景观组成', 'city-patterns', content, snapshot))
    home = site / 'index.html'
    if home.exists():
        text = home.read_text()
        text = re.sub(r'<section class="note" id="element-route">.*?</section>',
            '<section class="note" id="element-route"><strong>当前路线：30 城景观组成与空间审阅</strong>'
            f'<p>{s["canonical_units"]} 个归并响应单元，{s["recurrent_patterns"]} 个重复高响应组合；'
            '新增细分组合目录、30 城格网分布和四方向实例。语义与道路布局待核查。</p>'
            '<a href="city-patterns/">查看当前结果 →</a> · <a href="elements/">查看标签激活 →</a></section>', text, flags=re.S)
        text = re.sub(r'<section class="section"><h2>30 城景观组成</h2>.*?</section>',
            '<section class="section"><h2>30 城景观组成</h2>'
            f'<p>{s["panoramas"]:,} 全景、{s["cells"]:,} 观测格网；{s["recurrent_patterns"]} 个探索性高响应组合，'
            '可按细标签、2—4 类组成和图像证据筛选，并查看各城分布。</p>'
            '<a href="city-patterns/">查看当前结果 →</a></section>', text, flags=re.S)
        home.write_text(text)
    manifest = {**summary, 'page': 'city-patterns/', 'method': 'fine_grained_relative_response_compositions',
                'previous_excess_candidates': 7, 'examples': len(examples), 'scientific_review': 'pending'}
    if roadside:
        manifest['roadside_audit'] = {'page': roadside['page'],
            'patterns_side_eligible': roadside['summary']['patterns_side_eligible'],
            'sampled_pattern_instances': roadside['summary']['sampled_pattern_instances'],
            'final_validation': 'pending'}
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    root_manifest = site / 'manifest.json'
    if root_manifest.exists():
        root = read(root_manifest)
        root['city_patterns'] = manifest
        root['snapshot_label'] = snapshot
        files = [p for p in site.rglob('*') if p.is_file() and '.git' not in p.parts]
        root['file_count'] = len(files)
        root['bytes_before_manifest'] = sum(p.stat().st_size for p in files if p != root_manifest)
        root_manifest.write_text(json.dumps(root, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--followup', type=Path, default=FOLLOWUP)
    args = parser.parse_args()
    from build_main_site import page
    print(json.dumps(install(args.site_dir, page, datetime.now(timezone.utc).isoformat(), args.followup), ensure_ascii=False))
