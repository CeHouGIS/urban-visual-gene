"""Publish only the current complete-panorama exploratory evidence, if complete."""
from __future__ import annotations
import html
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/vitl30_open_panorama_prototypes/outputs'
CHECKPOINT = 'fccf7c29fea802341802e365f07fa3bdb4077203ba784b03f046ac89823113e0'

def install(target,status):
    evidence = OUT/'gallery/index.json'
    if status.get('status')!='complete' or not evidence.exists():
        return ''
    payload = json.loads(evidence.read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    if (payload['checkpoint_sha256']!=CHECKPOINT or payload['test_used'] or
            payload['identity']!=protocol['identity'] or not payload['full_panorama'] or
            payload['headings']!=[0,90,180,270] or payload['native_grid']!=[28,112] or
            payload['prototype_checkpoint_sha256']!=hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest()):
        raise ValueError('Open panorama evidence does not match the current corpus/model')
    target = Path(target)
    destination = target/'open-panorama'
    destination.mkdir(exist_ok=True)
    public = {**payload,'prototypes':[]}
    cards, expected_images = [], set()
    for prototype in payload['prototypes']:
        record = {**prototype,'examples':[]}
        images = []
        for example in prototype['examples']:
            if example['partition']!='dev':
                raise ValueError('Only development examples may be published')
            filename = example['image']
            if Path(filename).name!=filename:
                raise ValueError('Invalid gallery image filename')
            expected_images.add(filename)
            shutil.copy2(OUT/'gallery'/filename,destination/filename)
            record['examples'].append({k:v for k,v in example.items() if k!='source_images'})
            city = html.escape(example['city'])
            images.append(f'<figure><a href="open-panorama/{filename}" target="_blank" rel="noopener">'
                f'<img loading="lazy" src="open-panorama/{filename}" alt="完整四方向原图、贡献热图和mask：{city}"></a>'
                f'<figcaption>{city} · NMSE {example["nmse"]:.3f} · 原型mask面积 {example["mask_fraction"]:.1%}'
                f'<br>panoid: {html.escape(example["panoid"])}</figcaption></figure>')
        group_labels = []
        for group in prototype['top_groups'][:5]:
            names = ' / '.join(str(pair[0]) for pair in group['families'][:2])
            group_labels.append(f'G{group["decoder_group"]:02d} {html.escape(names)} ({group["weight_share"]:.1%})')
        public['prototypes'].append(record)
        count = len(record['examples'])
        opened = ' open' if not cards else ''
        cards.append(f'<details class="open-prototype" id="open-p{prototype["id"]}"{opened}>'
            f'<summary>候选 P{prototype["id"]:04d} · {prototype["train_support"]:,} 训练全景 / '
            f'{prototype["train_cities"]} 城 · {prototype["dev_support"]:,} 开发全景 · {count} 个实例</summary>'
            '<p>主要decoder组（组内含多种语义，尚未核查）：'+ '；'.join(group_labels)+'</p>'
            '<div class="open-examples">'+''.join(images)+'</div></details>')
    for old_image in destination.glob('*.webp'):
        if old_image.name not in expected_images:
            old_image.unlink()
    (destination/'index.json').write_text(json.dumps(public,ensure_ascii=False,separators=(',',':')))
    training = payload['training_summary']
    dev = training['dev']
    events = training['events']
    births = sum(e['event']=='birth' for e in events)
    merges = sum(e['event']=='merge' for e in events)
    retired = sum(len(e['ids']) for e in events if e['event']=='retire')
    count = len(payload['prototypes'])
    return f'''<section class="section" id="open-results"><h2>首轮完整全景候选结果</h2>
<p>{count} 个候选模板；从4个初始化候选开始，本轮新增{births}次、合并{merges}次、淘汰{retired}个。
最终原型数量仍需不同种子和阈值的稳定性验证，不能据此认定景观有{count}种。</p>
<p>全部{dev['panoramas']:,}个开发全景：组级归一化重建NMSE {dev['nmse_unit_normalized']:.4f}；
组级池化网格覆盖 {dev['group_pooled_patch_coverage_all_denominator']:.1%}；
四方向各≥80%网格达标的全景比例 {dev['four_directions_each_at_least_80pct']:.1%}。</p>
<p class="note">计算输入由4096维聚合为64个decoder组，并将每方向28×28池化到14×14。
这里的覆盖是该近似空间的重建覆盖，不代表原4096维覆盖或语义解释覆盖。语义验证、跨种子、完整打乱对照待完成，测试集继续封存。</p>
<p>每个实例上排为原图，中排为该模板的贡献份额热图，下排为贡献份额≥0.35的mask。
均按0°、90°、180°、270°展示，不是无缝球面全景。mask是模型归因，不能当作物体分割。
完整全景模板的mask可能覆盖大部分图像；这表示其贡献分布广，不等于找到了精细物体边界。点击候选标题展开实例，点击图片放大。</p>
<p><a href="open-panorama/index.json">候选、生命周期、指标与实例数据</a></p>
<style>.open-prototype{{margin:18px 0;padding:16px;border:1px solid #d4ddd8;border-radius:10px}}
.open-prototype summary{{cursor:pointer;font-weight:700}}.open-examples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:18px}}
.open-examples figure{{margin:0}}.open-examples img{{width:100%;height:auto;display:block;border-radius:6px}}
.open-examples figcaption{{font-size:12px;color:#53665d;overflow-wrap:anywhere}}</style>
{''.join(cards)}</section>'''
