"""Publish the current panorama-level multi-element relation results.

The repository contains an older, direction-level two-family pilot for
backwards-compatible unit tests.  When the panorama motif result exists it is
the only source used for the public site; the old snapshot is never mixed into
the published page.
"""
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

FAMILY_LABELS = {
    'building': '建筑及附属部件',
    'tree': '乔木及其部件',
    'low_vegetation': '草灌及低矮植被',
    'road': '道路及车行路面',
    'pedestrian': '步行空间及附属部件',
    'barrier': '围合与防护设施',
    'utility': '市政杆线设施',
    'vehicle': '车辆及其部件',
    'sign': '标识与交通控制设施',
    'water': '水体',
    'terrain': '自然地形',
    'person': '行人及人体部件',
    'furniture': '街道家具与小设施',
    'construction': '施工设施与设备',
}


def _install_legacy(target):
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


def _label(family):
    return FAMILY_LABELS.get(family, family)


def _layout_text(layout):
    """Make the canonical three-node layout readable on the public page."""
    angle = {'near': '近邻', 'adjacent': '相邻', 'wide': '较宽间隔',
             'very_wide': '很宽间隔', 'opposite': '相对方向'}
    vertical = {'up': '后者更高', 'down': '后者更低', 'level': '高度相近'}
    parts = []
    for first, second in layout:
        parts.append(f'{angle.get(first, first)} / {vertical.get(second, second)}')
    return '；'.join(parts)


def _install_panorama(target, source):
    """Publish the new three-node, complete-panorama relation catalogue."""
    data = json.loads(source.read_text())
    sampling = data.get('sampling', {})
    method = data.get('method', {})
    counts = data.get('counts', {})
    if (data.get('status') != 'exploratory'
            or data.get('scope') != 'complete four-heading panorama; sealed test partition not accessed'
            or data.get('checkpoint_sha256') != CHECKPOINT
            or len(data.get('patterns', [])) == 0
            or sampling.get('cities') != 29
            or method.get('motif_size') != 3
            or method.get('view_order') != [0, 90, 180, 270]):
        raise ValueError('Panorama motif results have mismatched provenance or are incomplete')

    target = Path(target)
    destination = target / 'local-compositions'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    gallery = source.parent / 'gallery'
    patterns = []
    cards = []
    copied = set()
    for index, item in enumerate(data['patterns'], 1):
        examples = []
        figures = []
        for example in item.get('examples', []):
            image = example.get('image', '')
            if Path(image).name != image or not image.endswith('.webp'):
                raise ValueError('Invalid panorama motif image path')
            source_image = gallery / image
            if not source_image.exists():
                raise ValueError(f'Missing panorama motif image: {image}')
            if image not in copied:
                shutil.copy2(source_image, destination / image)
                copied.add(image)
            city = html.escape(str(example.get('city', '')))
            panoid = html.escape(str(example.get('panoid', '')))
            figures.append(
                f'<figure><a href="local-compositions/{image}" target="_blank" rel="noopener">'
                f'<img loading="lazy" src="local-compositions/{image}" '
                f'alt="{city} 的四方向关系遮罩实例"></a>'
                f'<figcaption>{city}<br>panoid: {panoid}</figcaption></figure>')
            examples.append(example)
        public_item = {**item, 'examples': examples}
        patterns.append(public_item)
        family_text = ' + '.join(_label(f) for f in item['families'])
        family_code = ' + '.join(item['families'])
        layout = _layout_text(item['layout'])
        q = item.get('q', {})
        cards.append(
            f'<details class="local-composition" id="local-m{index}"'
            f'{" open" if index == 1 else ""}><summary>关系 {index:02d} · '
            f'<span class="motif-families">{html.escape(family_text)}</span>'
            f' <small>({html.escape(family_code)})</small></summary>'
            f'<p><b>全景关系：</b>{html.escape(layout)}。'
            f'训练集 {item["train_panoramas"]} 个全景 / {item["train_cities"]} 城；'
            f'开发集 {item["dev_panoramas"]} 个全景 / {item["dev_cities"]} 城。'
            f'共现置换 q={q.get("cooccurrence", float("nan")):.3f}，'
            f'布局置换 q={q.get("layout", float("nan")):.3f}。</p>'
            f'<div class="local-examples">{"".join(figures)}</div></details>')

    public = {**data, 'patterns': patterns,
              'presentation': {
                  'source': 'outputs/panorama_motifs/results.json',
                  'gallery_patterns': len(patterns),
                  'gallery_images': len(copied),
                  'legacy_direction_level_results_published': False,
              }}
    (destination / 'results.json').write_text(
        json.dumps(public, ensure_ascii=False, separators=(',', ':')))

    cities = int(sampling.get('cities', 0))
    panoramas = int(sampling.get('panoramas', 0))
    train = int(sampling.get('train_panoramas', 0))
    dev = int(sampling.get('dev_panoramas', 0))
    directional = int(sampling.get('directional_images', 0))
    observed = int(counts.get('all_observed_keys', 0))
    candidates = int(counts.get('train_candidates', 0))
    significant = int(counts.get('both_nulls_q_below_005', 0))
    fov = method.get('fov_deg', 90)
    return f'''<section class="section" id="local-results"><h2>全景级多要素关系 · 探索性结果</h2>
<p>当前网页只发布全新的全景级关系结果：以 {cities} 城的完整四方向全景为统计单位，使用 {panoramas:,} 个全景（{train:,} train、{dev:,} dev）和 {directional:,} 张方向图像。每个候选由三个局部 SAE 要素组成，允许同一父类重复，例如“建筑 + 乔木 + 乔木”；旧的单方向二元组合快照已从网页和发布目录删除。</p>
<p>每个方向保留 28×28 patch 激活区域，并用 {fov}° 视场假设映射到同一条环形水平坐标，再跨 0°、90°、180°、270° 查找关系。下方每张图都是同一点位的四方向条带，颜色遮罩表示候选 SAE family 区域。条带用于阅读和对照，不是经过球面校正的全景投影。</p>
<div class="motif-stats"><span><b>{observed:,}</b> 个观测关系 key</span><span><b>{candidates:,}</b> 个训练候选</span><span><b>{len(patterns)}</b> 个展示布局</span><span><b>{significant:,}</b> 个候选的两个置换 q&lt;0.05</span></div>
<p class="note">置换检验和 BH-FDR 只提供探索性优先级，不能把 {significant:,} 个候选直接称为已确认原型。SAE family mask 是特征归因区域，不是物体分割；同一物体可能被拆成多个区域，需人工复核。测试集未读取，80% 覆盖目标也未在此结果中验证。</p>
<p><a href="local-compositions/results.json">查看完整候选、协议、p/q 值与实例 JSON</a></p>
<style>.motif-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}.motif-stats span{{padding:12px;border:1px solid #d4ddd8;border-radius:8px;background:#f5f8f6}}.motif-stats b{{display:block;font-size:1.25rem}}.local-composition{{margin:16px 0;padding:16px;border:1px solid #d4ddd8;border-radius:10px}}.local-composition summary{{cursor:pointer;font-weight:700}}.local-composition small{{font-weight:400;color:#53665d}}.local-examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,540px),1fr));gap:16px}}.local-examples figure{{margin:0}}.local-examples img{{width:100%;height:auto;border-radius:6px}}.local-examples figcaption{{font-size:12px;color:#53665d;overflow-wrap:anywhere}}.motif-families{{color:#1f5b4b}}</style>
{''.join(cards)}</section>'''


def _install_structured(target, source):
    data = json.loads(source.read_text())
    sampling, method = data.get('sampling', {}), data.get('method', {})
    rules = Path(__file__).with_name('label_structure_rules.json')
    if (data.get('status') != 'exploratory' or data.get('checkpoint_sha256') != CHECKPOINT
            or data.get('rules_sha256') != hashlib.sha256(rules.read_bytes()).hexdigest()
            or sampling.get('cities') != 29 or sampling.get('train') != 1392
            or sampling.get('dev') != 464 or method.get('unit') != 'complete four-heading panorama'
            or method.get('motif_size') != 3 or not data.get('patterns')):
        raise ValueError('Structured motif results have mismatched provenance or are incomplete')
    destination = Path(target) / 'local-compositions'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    cards = []
    for index, item in enumerate(data['patterns'], 1):
        figures = []
        for example in item['examples']:
            name = example['image']
            if Path(name).name != name or not name.endswith('.webp'):
                raise ValueError('Invalid structured motif image path')
            image = source.parent / 'gallery' / name
            if not image.exists():
                raise ValueError(f'Missing structured motif image: {name}')
            shutil.copy2(image, destination / name)
            roles = ' · '.join(
                f'{node["observed_label"]} → {node["entity_label"]}'
                for node in example['nodes'])
            city = html.escape(str(example['city']))
            figures.append(
                f'<figure><a href="local-compositions/{name}" target="_blank" rel="noopener">'
                f'<img loading="lazy" src="local-compositions/{name}" alt="{city} 的四方向关系遮罩实例"></a>'
                f'<figcaption>{city} · {html.escape(roles)}<br>'
                f'panoid: {html.escape(str(example["panoid"]))}</figcaption></figure>')
        fine = item.get('fine_compositions', [])[:5]
        fine_text = ''.join('<li>' + html.escape(' + '.join(role[1] for role in row['roles']))
                            + f'：训练 {row["train_panoramas"]} / 开发 {row["dev_panoramas"]} 全景</li>'
                            for row in fine)
        family_text = ' + '.join(_label(f) for f in item['families'])
        support = item['support']
        cards.append(
            f'<details class="local-composition" id="local-m{index}"'
            f'{" open" if index == 1 else ""}><summary>关系 {index:02d} · '
            f'{html.escape(family_text)}</summary>'
            f'<p>四方向相对布局：{html.escape(_layout_text(item["layout"]))}。'
            f'训练 {support["train"]["panoramas"]} 全景 / {support["train"]["cities"]} 城；'
            f'开发 {support["dev"]["panoramas"]} 全景 / {support["dev"]["cities"]} 城。'
            f'共现置换 q={item["q"]["cooccurrence"]:.3f}，'
            f'布局置换 q={item["q"]["layout"]:.3f}。</p>'
            f'{"<h3>重复的细标签组合</h3><ul>" + fine_text + "</ul>" if fine_text else ""}'
            f'<div class="local-examples">{"".join(figures)}</div></details>')
    (destination / 'results.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
    counts = data['counts']
    return f'''<section class="section" id="local-results"><h2>标签结构约束的全景组成 · 探索性结果</h2>
<p>复用 29 城、{sampling['panoramas']:,} 个四方向全景和原有 SAE 稀疏激活。从每个候选区域贡献最大的合格 SAE 维度读取 Qwen 细标签；只将标签结构中明确连接的标签用于本轮关系检索。<code>is_a</code> 保留类型差异，<code>part_of</code> 部件归回所属对象。建筑的窗户与墙面不会算成两个建筑要素；相隔足够远的两个树区域仍只是待复核的两棵树候选。</p>
<p>训练集提出 {counts['train_candidates']:,} 个三要素布局 key，开发集进行同城方向内的标签置换与全景内的布局置换；其中 {counts['both_nulls_q_below_005']:,} 个 key 的两种探索性 BH q 值均低于 0.05。本页展示 {len(data['patterns'])} 个组合及四方向遮罩实例。细标签组合的频数是描述统计，q 值仅适用于对应的宽类布局 key。</p>
<p class="note">Qwen 标签和 SAE 遮罩尚未经过对象实例验证。候选跨城重复不意味着对象数量、语义或景观原型已经确认；测试集激活保持封存。实例中的箭头表示标签归并，不代表视觉识别出的对象边界。</p>
<p><a href="../label-structure/">核查标签上下级与并列结构</a> · <a href="local-compositions/results.json">下载全部候选和实验协议 JSON</a></p>
<style>.local-composition{{margin:16px 0;padding:16px;border:1px solid #d4ddd8;border-radius:8px}}.local-composition summary{{cursor:pointer;font-weight:700}}.local-examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,540px),1fr));gap:16px}}.local-examples figure{{margin:0}}.local-examples img{{width:100%;height:auto;border-radius:6px}}.local-examples figcaption{{font-size:12px;color:#53665d;overflow-wrap:anywhere}}</style>
{''.join(cards)}</section>'''


def install(target):
    """Install the newest panorama result, falling back only for old tests."""
    structured = OUT / 'structured_motifs' / 'results.json'
    if structured.exists():
        return _install_structured(target, structured)
    panorama = OUT / 'panorama_motifs' / 'results.json'
    if panorama.exists():
        return _install_panorama(target, panorama)
    return _install_legacy(target)
