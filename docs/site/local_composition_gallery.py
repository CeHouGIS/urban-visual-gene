"""Publish only the current masked multi-element composition discovery pilot."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/vitl30_local_compositions/outputs'
RESULT = ROOT / 'experiments/vitl30_local_compositions/results'
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'


def install(target):
    source = OUT / 'results.json'
    if not source.exists():
        source = RESULT / 'results.json'
        if not source.exists():
            return '<section class="section" id="local-results"><h2>局部要素组合</h2><p>探索性实验正在计算，实例尚未发布。</p></section>'
        provenance = json.loads((RESULT / 'provenance.json').read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest() != provenance['results_sha256']:
            raise ValueError('Published local composition snapshot changed')
        gallery = RESULT / 'gallery'
    else:
        provenance = None
        gallery = OUT / 'gallery'
    data = json.loads(source.read_text())
    split_path = ROOT / 'experiments/vitl30_prototype_coverage/outputs/protocol.json'
    split_identity = (json.loads(split_path.read_text())['identity'] if split_path.exists()
                      else provenance['split_identity'] if provenance else None)
    hierarchy = ROOT / 'experiments/vitl30_relational_prototypes/results/hierarchy.json'
    families = {row['id']: row['label_zh']
                for row in json.loads(hierarchy.read_text()).get('families', [])}
    if (data['status'] != 'complete' or data['checkpoint_sha256'] != CHECKPOINT
            or data['split_identity'] != split_identity
            or data['hierarchy_sha256'] != hashlib.sha256(hierarchy.read_bytes()).hexdigest()
            or (provenance and (provenance['checkpoint_sha256'] != CHECKPOINT
                                or provenance['hierarchy_sha256'] != data['hierarchy_sha256']))
            or data['sampling']['cities'] != 29 or not data['patterns']
            or data['scope'] != 'exploratory balanced city sample; sealed test not accessed'):
        raise ValueError('Local composition results have mismatched provenance or are incomplete')
    target = Path(target)
    destination = target / 'local-compositions'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    # Angular variants can describe the same two elements and relative height.
    # Deduplicate the gallery only; the 298 tests and BH q-values stay intact.
    chosen, seen = [], set()
    for item in data['patterns']:
        relation = item['relation']
        signature = (tuple(sorted(item['families'])),
                     'cross' if relation.startswith('cross_') else 'same',
                     relation.rsplit('_', 1)[-1])
        if signature not in seen:
            seen.add(signature)
            chosen.append(item)
    public = {**data, 'patterns': [],
              'findings': {**data['findings'], 'source_displayed': data['findings']['displayed'],
                           'displayed': len(chosen)},
              'presentation': {'angular_variants_collapsed_for_gallery_only': True,
                               'significance_tests_recomputed': False}}
    cards = []
    for index, item in enumerate(chosen, 1):
        figures = []
        examples = []
        for example in item['examples']:
            image = example['image']
            if Path(image).name != image or not image.endswith('.webp'):
                raise ValueError('Invalid local composition image path')
            source_image = gallery / image
            if provenance and hashlib.sha256(source_image.read_bytes()).hexdigest() != provenance['gallery_sha256'][image]:
                raise ValueError(f'Published local composition image changed: {image}')
            shutil.copy2(source_image, destination / image)
            city = html.escape(example['city'])
            figures.append(f'<figure><a href="local-compositions/{image}" target="_blank" rel="noopener">'
                           f'<img loading="lazy" src="local-compositions/{image}" alt="{city} 的四方向要素遮罩"></a>'
                           f'<figcaption>{city}<br>panoid: {html.escape(example["panoid"])}</figcaption></figure>')
            examples.append(example)
        public['patterns'].append({**item, 'examples': examples})
        left, right = (html.escape(families.get(key, key)) for key in item['families'])
        relation = html.escape(item['relation'])
        if relation.startswith('cross_'):
            _, angle, vertical = relation.split('_')
            position = {'U': '后者在画面中更高', 'D': '后者在画面中更低', 'M': '画面高度相近'}[vertical]
            layout = f'跨方向，朝向相差{angle}°，{position}'
        else:
            vertical, horizontal = relation[5:]
            parts = []
            if vertical != 'M':
                parts.append('上方' if vertical == 'U' else '下方')
            if horizontal != 'M':
                parts.append('左侧' if horizontal == 'L' else '右侧')
            layout = '同方向，后者位于前者' + '及'.join(parts)
        cards.append(f'<details class="local-composition" id="local-p{index}"'
                     f'{" open" if index == 1 else ""}><summary>候选 {index:02d} · '
                     f'<span class="first-family">{left}</span> + <span class="second-family">{right}</span> '
                     f'· {html.escape(layout)} · 开发集 {item["dev_panoramas"]} 全景 / {item["dev_cities"]} 城</summary>'
                     f'<p class="note">原始关系代码：<code>{relation}</code>；两种要素分别以红、蓝遮罩。</p>'
                     f'<p>训练样本：{item["train_panoramas"]} 全景、{item["train_blocks"]} 空间块、'
                     f'{item["train_cities"]} 城；开发样本：{item["dev_panoramas"]} 全景、'
                     f'{item["dev_blocks"]} 空间块、{item["dev_cities"]} 城。'
                     f'同城方向内要素置换期望 {item["null_means"]["cooccurrence"]:.1f}，'
                     f'布局置换期望 {item["null_means"]["layout"]:.1f}；'
                     f'BH q 值分别 {item["q"]["cooccurrence"]:.3f} / '
                     f'{item["q"]["layout"]:.3f}。</p>'
                     f'<div class="local-examples">{"".join(figures)}</div></details>')
    (destination / 'results.json').write_text(json.dumps(public, ensure_ascii=False, separators=(',', ':')))
    count = data['sampling']
    findings = data['findings']
    return f'''<section class="section" id="local-results"><h2>局部要素组合 · 探索性结果</h2>
<p>从 {count['cities']} 城抽样 {count['train_panoramas']:,} 个训练全景和 {count['dev_panoramas']:,} 个开发全景，共 {count['sampled_images']:,} 张方向图像。每张图的 SAE patch 激活被归入暂定要素父类，再形成非矩形区域 mask。每个候选关系包含两个不同父类，可位于同方向或同一全景的不同方向；最终组合数量没有预设。</p>
<p>训练样本提出 {findings['all_candidates']} 个重复关系候选；原始图库选出 {findings['displayed']} 个方向关系，本页把同一要素对与上下关系的 90°／180°／270°变体仅在展示层合并，呈现 {len(chosen)} 个不同组合布局。统计检验仍针对原始 {findings['all_candidates']} 个候选，未合并 p 或 q 值。两个零假设均通过 BH-FDR q&lt;0.05 的候选有 {findings['q_below_005_both_nulls']} 个；这是样本内探索性检验，不代表 {findings['q_below_005_both_nulls']} 种独立景观。每张图为同一点位的 0°、90°、180°、270° 四方向；<span class="first-family">红色</span>和<span class="second-family">蓝色</span>分别遮罩两种要素。</p>
<p class="note">遮罩依据 SAE 激活和事后 VLM 父类推断，不是独立分割标注；建筑墙面和窗户归为同一父类，不能靠它们自身形成多要素组合。实例还需人工审阅对象完整性、关系稳定性以及跨城泛化。测试集保持封存，80% 语义覆盖目标尚未验证。</p>
<p><a href="local-compositions/results.json">查看参数、置换结果与全部展示实例 JSON</a></p>
<style>.local-composition{{margin:16px 0;padding:16px;border:1px solid #d4ddd8;border-radius:10px}}
.local-composition summary{{cursor:pointer;font-weight:700}}.local-examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,540px),1fr));gap:16px}}
.local-examples figure{{margin:0}}.local-examples img{{width:100%;height:auto;border-radius:6px}}
.local-examples figcaption{{font-size:12px;color:#53665d;overflow-wrap:anywhere}}
.first-family{{color:#c43830;font-weight:700}}.second-family{{color:#1682c5;font-weight:700}}</style>
{''.join(cards)}</section>'''
