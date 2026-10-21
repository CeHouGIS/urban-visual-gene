"""Publish dev exemplars belonging to the current patch-coverage checkpoint."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

from PIL import Image


def load_gallery(experiment, protocol, model_hash):
    folder = experiment / 'outputs/gallery'
    if not (folder / 'complete.json').exists():
        return None
    complete = json.loads((folder / 'complete.json').read_text())
    bundle = json.loads((folder / 'prototypes.json').read_text())
    identity = bundle['identity']
    if (identity != complete['identity'] or
            identity['experiment_identity'] != protocol['identity'] or
            identity['model_sha256'] != model_hash or
            identity['sae_sha256'] != protocol['checkpoint_sha256'] or
            identity['split'] != 'dev' or identity['test_used']):
        raise ValueError('Prototype gallery belongs to another model or partition')
    records = bundle['prototypes']
    expected = len(protocol['model']['scales']) * protocol['model']['per_scale']
    if [p['prototype'] for p in records] != list(range(expected)):
        raise ValueError('Gallery must retain every prototype, including inactive ones')
    for record in records:
        if record['identity'] != identity or record['model_sha256'] != model_hash:
            raise ValueError('Stale per-template gallery record')
        if record['scale'] != protocol['model']['scales'][record['prototype'] // protocol['model']['per_scale']]:
            raise ValueError('Prototype scale mismatch')
        used = set()
        for item in record['instances']:
            if (item['coefficient_fp32'] <= 0 or item['removing_response_delta_sse'] <= 0 or
                    item['pano_row'] in used):
                raise ValueError('Gallery exemplar lacks positive contribution or duplicates a panorama')
            used.add(item['pano_row'])
    if (sum(len(p['instances']) for p in records) != complete['examples'] or
            sum(bool(p['instances']) for p in records) != complete['templates_with_examples']):
        raise ValueError('Gallery completion counts disagree')
    return folder, complete, bundle


def install(site, experiment, protocol, model_hash):
    loaded = load_gallery(experiment, protocol, model_hash)
    if loaded is None:
        return '', None
    folder, complete, bundle = loaded
    assets = site / 'assets/prototype-gallery'
    public = {'identity': bundle['identity'], 'templates': complete['templates'],
              'templates_with_examples': complete['templates_with_examples'],
              'examples': complete['examples'], 'semantic_review': 'pending', 'prototypes': []}
    for record in bundle['prototypes']:
        name = f'p{record["prototype"]:03d}'
        source, target = folder / name, assets / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'template.png', target / 'template.png')
        item = {k: record[k] for k in ['prototype', 'scale', 'status', 'top_dimensions', 'top20_weight_fraction']}
        item['template'] = f'../assets/prototype-gallery/{name}/template.png'
        item['instances'] = []
        for row in record['instances']:
            example = {k: row[k] for k in ['rank', 'panoid', 'city', 'heading', 'y', 'x',
                'coefficient_fp32', 'box_patch', 'removing_response_delta_sse', 'native_size', 'native_box']}
            for kind in ['original', 'overlay', 'crop', 'masked']:
                filename = Path(row[kind])
                if filename.name != row[kind]:
                    raise ValueError('Image filename must stay within its template folder')
                output_name = filename.stem + '.jpg'
                with Image.open(source / filename) as image:
                    image = image.convert('RGB')
                    image.thumbnail((1120, 800), Image.Resampling.LANCZOS)
                    image.save(target / output_name, quality=88, optimize=True)
                example[kind] = f'../assets/prototype-gallery/{name}/{output_name}'
            item['instances'].append(example)
        public['prototypes'].append(item)
    data = site / 'data/prototype-gallery.json'
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
    templates = Path(__file__).with_name('templates')
    for filename in ['prototype-gallery.css', 'prototype-gallery.js']:
        shutil.copyfile(templates / filename, site / 'assets' / filename)
    missing = complete['templates'] - complete['templates_with_examples']
    content = f'''<section class="section" id="prototype-gallery" aria-labelledby="gallery-title">
<div class="eyebrow">Prototype atlas · real street-view evidence</div>
<h2 id="gallery-title">逐原型查看真实街景实例</h2>
<p>当前模型的{complete['templates']}个模板中，{complete['templates_with_examples']}个有可展示实例，共{complete['examples']}组。
每个模板最多8个不同全景点位，优先跨城市选取；候选来自开发集全部{bundle['identity']['images']:,}张方向图。{f'另外{missing}个模板暂无通过核验的实例，目录中保留并标明原因。' if missing else ''}</p>
<p class="muted">按模板峰值响应检索，再核验：去掉该位置的模板响应后，重建平方误差必须增大。
热图显示这个局部模板的贡献强度；裁剪范围对应模板在图中的空间支持，不是物体分割。
这些是待核查的候选景观组合，尚未赋予已验证的语义名称。</p>
<div id="prototype-atlas" aria-busy="true"><p role="status">正在载入原型实例…</p></div>
<noscript>请启用JavaScript，或<a href="../data/prototype-gallery.json">下载完整原型实例索引</a>。</noscript>
<p class="muted">实例检索没有使用封存测试集。模板身份：<code>{html.escape(model_hash[:16])}</code>。
<a href="../data/prototype-gallery.json">下载全部实例与组成维度</a></p>
</section>'''
    return content, public
