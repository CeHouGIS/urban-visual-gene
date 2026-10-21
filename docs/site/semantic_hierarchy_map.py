"""Publish a two-level map of provisional Qwen labels and SAE activation evidence."""
from __future__ import annotations

import json
import posixpath
from pathlib import Path
import shutil


TEMPLATES = Path(__file__).with_name('templates')
ENGLISH_FAMILIES = {
    'unresolved': 'Unresolved or low evidence',
    'building': 'Buildings and components',
    'ground_unspecified': 'Ground, unspecified',
    'tree': 'Trees and components',
    'road': 'Road and carriageway',
    'sky': 'Sky and clouds',
    'vegetation_unspecified': 'Vegetation, unspecified',
    'vehicle': 'Vehicles and components',
    'material_unspecified': 'Material, object unclear',
    'low_vegetation': 'Low vegetation',
    'scene_context': 'Scene context',
    'sign': 'Signs and traffic control',
    'barrier': 'Barriers and enclosures',
    'transport_structure': 'Transport structures',
    'utility': 'Utility infrastructure',
    'person': 'People and body parts',
    'terrain': 'Natural terrain',
    'pedestrian': 'Pedestrian space',
    'construction': 'Construction equipment',
    'furniture': 'Street furniture',
    'water': 'Water',
    'spatial_encoding': 'Spatial encoding candidates',
}


def _example_url(view):
    return '../' + posixpath.normpath('elements/' + view)


def _family_examples(family, limit=4):
    labels = [label for label in family['labels'] if label['eligible']]
    candidates = [(label, example) for label in labels for example in label['examples']]
    selected, seen_panoramas, seen_labels = [], set(), set()
    for distinct_label in (True, False):
        for label, example in candidates:
            pano = (example['city'], example['pano'])
            if pano in seen_panoramas or (distinct_label and label['label'] in seen_labels):
                continue
            selected.append({
                'label': label['label'], 'feature': example['feature'],
                'city': example['city'], 'heading': example['heading'],
                'image': _example_url(example['views'][example['heading'] // 90]),
                'heat': example['heat'],
                'family': family['id'],
            })
            seen_panoramas.add(pano)
            seen_labels.add(label['label'])
            if len(selected) == limit:
                return selected
    return selected


def _weak_examples(rows, evidence, limit=4):
    """Select real weak positive responses; zero-response dimensions cannot supply images."""
    ranked = sorted((row for row in rows if row['status'] == 'insufficient_evidence'),
                    key=lambda row: (-row['screening']['positive_panoramas'], row['feature']))
    selected, features, cities, panoramas = [], set(), set(), set()
    for new_city in (True, False):
        for row in ranked:
            if row['feature'] in features:
                continue
            for crop in evidence[row['feature']]['crops']:
                if crop.get('activation', 0) <= 0 or 'image_id' not in crop:
                    continue
                pano = (crop['city'], crop['pano'])
                if pano in panoramas or (new_city and crop['city'] in cities):
                    continue
                start = crop['image_id'] - crop['image_id'] % 4
                selected.append({
                    'label': '证据不足', 'feature': row['feature'],
                    'city': crop['city'], 'heading': crop['heading'],
                    'image': f'../semantic/sources/{start + crop["heading"] // 90}.jpg',
                    'heat': crop['heatmap_png'], 'family': 'unresolved',
                })
                features.add(row['feature'])
                cities.add(crop['city'])
                panoramas.add(pano)
                if len(selected) == limit:
                    return selected
                break
    return selected


def build_map(graph, rows, evidence):
    if graph['total'] != len(rows) or graph['family_count'] != len(graph['families']):
        raise ValueError('The two-level map requires the complete SAE hierarchy')
    weak = [row for row in rows if row['status'] == 'insufficient_evidence']
    zero = [row for row in rows if row['status'] == 'unobserved']
    families = []
    for family in graph['families']:
        if family['id'] == 'unresolved':
            member_ids = {feature for label in family['labels'] for feature in label['features']}
            weak_ids = {row['feature'] for row in weak}
            zero_ids = {row['feature'] for row in zero}
            other_ids = member_ids - weak_ids - zero_ids
            if len(other_ids) + len(weak_ids) + len(zero_ids) != family['count']:
                raise ValueError('Evidence counts exceed unresolved family size')
            children = [
                {'label': '待复核的未归类响应', 'count': len(other_ids),
                 'features': sorted(other_ids)},
                {'label': '证据不足／无正激活', 'count': len(weak) + len(zero),
                 'features': sorted(weak_ids | zero_ids)},
            ]
            examples = _weak_examples(rows, evidence)
        else:
            children = [{'label': label['label'], 'count': label['count'],
                         'features': label['features']}
                        for label in family['labels']]
            examples = _family_examples(family)
        if sum(child['count'] for child in children) != family['count']:
            raise ValueError(f'Child counts do not cover {family["id"]}')
        if len(examples) != 4:
            raise ValueError(f'Need four real activation images for {family["id"]}; got {len(examples)}')
        families.append({
            'id': family['id'], 'label': family['label'],
            'label_en': ENGLISH_FAMILIES.get(family['id'], family['label']),
            'role': family['role'],
            'count': family['count'], 'children': children, 'examples': examples,
        })
    leaves = [feature for family in families for child in family['children']
              for feature in child['features']]
    if sorted(leaves) != list(range(graph['total'])):
        raise ValueError('The hierarchy does not have exactly one leaf per SAE dimension')
    return {
        'identity': graph['identity'], 'dimension_screening_identity': graph['dimension_screening_identity'],
        'total': graph['total'], 'family_count': len(families),
        'weak_count': len(weak), 'zero_count': len(zero), 'leaf_count': len(leaves),
        'families': families,
        'interpretation': 'Provisional Qwen taxonomy; branch positions do not encode cluster distance.',
    }


def install(site, render_page, snapshot, graph, rows, reports):
    evidence = json.loads((reports / 'vlm_dimensions_top10_masked/evidence.json').read_text())
    result = build_map(graph, rows, evidence['dimensions'])
    target = site / 'semantic-map'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'map.json').write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
    try:
        from .render_semantic_map import render as render_static_map
    except ImportError:
        from render_semantic_map import render as render_static_map
    render_static_map(site, target / 'semantic-hierarchy.png')
    for extension in ('css', 'js'):
        shutil.copyfile(TEMPLATES / f'semantic-hierarchy-map.{extension}', target / f'map.{extension}')
    content = f'''<div class="eyebrow">4096 SAE dimensions · Qwen semantic taxonomy</div>
<h1>逐维解释层级图</h1>
<p class="lead">{result['family_count']} 个暂定父类、4096 个维度叶节点和对应的 SAE 激活证据。</p>
<p><a href="semantic-hierarchy.png">下载完整总览图 PNG</a> · <a href="../elements/">查看逐维解释与四方向全景</a> · <a href="../label-structure/">查看标签间关系</a></p>
<div class="map-summary" id="map-summary" role="status">正在载入语义层级…</div>
<div class="map-head" aria-hidden="true"><span>父类</span><span>子类 · 编号</span><span>激活实例 · 每类 4 张</span></div>
<div id="semantic-hierarchy-map" class="semantic-hierarchy-map" aria-label="Qwen 两层语义归类图"></div>
<p class="map-caveat">此图整理 Qwen 的暂定名称，并非按距离训练的聚类树；编号只用于查找细标签。图片显示真实正激活区域的 patch 级高亮，不是物体分割。空间编码仅是基于固定竖向激活带的审阅候选。证据不足和无正激活合为一个归档子类；无正激活维度无法提供激活图片，该父类的四张图来自证据不足但仍有少量正激活的维度。</p>'''
    document = render_page('逐维解释层级图', 'elements', content, snapshot,
                           '<script defer src="map.js"></script>')
    document = document.replace('</head>', '<link rel="stylesheet" href="map.css"></head>')
    (target / 'index.html').write_text(document)
    return {'page': 'semantic-map/', 'families': result['family_count'],
            'dimensions': result['total'], 'weak': result['weak_count'], 'zero': result['zero_count']}
