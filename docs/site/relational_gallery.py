"""Publish development examples from the trained relation dictionary."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/vitl30_relational_prototypes'
TEMPLATES = Path(__file__).with_name('templates')


def install(site):
    source = EXPERIMENT / 'outputs/instance_gallery'
    index = json.loads((source / 'index.json').read_text())
    pilot = json.loads((EXPERIMENT / 'results/pilot_manifest_summary.json').read_text())
    report = json.loads((EXPERIMENT / 'results/relational_pilot_summary.json').read_text())
    if (index['pilot_identity'] != pilot['identity'] or index['partition'] != 'dev'
            or index['test_used'] or len(index['slots']) != 32
            or index['development_regions'] != report['eligible_dev_regions']):
        raise ValueError('Relation gallery is not the audited development set')
    target = site / 'elements'
    image_dir = target / 'examples'
    image_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for slot in index['slots']:
        if slot['development_regions'] != report['slot_usage'][slot['slot']]['development_regions']:
            raise ValueError('Slot count changed')
        if len(slot['examples']) != index['examples_per_slot']:
            raise ValueError('Incomplete slot examples')
        layout = slot.get('layout', {})
        if (layout.get('grid') != [4, 4] or len(layout.get('dominant_families', [])) != 16
                or len(layout.get('relative_strengths', [])) != 16):
            raise ValueError('Incomplete learned slot layout')
        cities = set()
        for item in slot['examples']:
            if item['slot'] != slot['slot'] or item['city'] in cities or item['region_id'] in seen:
                raise ValueError('Duplicate or mismatched example')
            cities.add(item['city'])
            seen.add(item['region_id'])
            shutil.copyfile(source / item['image'], image_dir / item['image'])
    (target / 'prototype-examples.json').write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')))
    for suffix in ('css', 'js'):
        shutil.copyfile(TEMPLATES / f'relational-gallery.{suffix}', target / f'relational-gallery.{suffix}')
    return '''<section class="section" id="relation-examples"><h2>关系槽位的布局与真实街景实例</h2>
<p><strong>这里的“布局”是模型为一个槽位学到的15类景观要素在4×4空间格中的强度模板。</strong>它不是整张图片的类别，也不是物体分割。每个格可以同时包含多类响应；网页用强度最高的父类给该格着色，透明度表示模板强度。</p>
<p>32个槽位各取4个开发集实例，来自不同城市。图中前四格按0°、90°、180°、270°展示全景方向；黄色框是选中的14×14、21×21或28×28 patch区域。第五格把该槽位的学习布局直接mask到选中方向上，框外压暗；最右是区域裁剪。</p>
<p>同一槽位的4张实例使用同一布局mask，用于检查它是否在不同街景中表达稳定结构。实例由关系模型重新推理挑选；名称和语义一致性仍需核查。关系模型开发MSE为0.2166，高于完整布局对照的0.2121。</p>
<div class="relation-tools"><label for="relation-slot">候选槽位</label><select id="relation-slot"></select><span id="relation-slot-count" role="status"></span></div>
<div id="relation-detail" class="relation-detail" aria-live="polite"></div>
<p><a href="prototype-examples.json">下载实例索引与来源身份</a></p></section>'''
