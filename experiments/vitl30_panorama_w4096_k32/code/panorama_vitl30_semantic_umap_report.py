"""Scientific figures and local image evidence browser for semantic UMAP."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from sae_experiments.panorama_vitl30 import OUTPUT
from sae_experiments.panorama_vitl30_analysis import GROUPS, checked_groups
from formal.analysis.semantic_feature_umap import digest, write_json


# Colorcet 3.2.1, glasbey_dark, first 64 colors in the original order.
# Attribution: Peter Kovesi / Colorcet contributors, CC BY 4.0.
# https://colorcet.holoviz.org/user_guide/Categorical.html
GROUP_COLORS = (
    '#d60000', '#8c3bff', '#018700', '#00acc6', '#e6a500', '#ff7ed1', '#6b004f', '#573b00',
    '#005659', '#15e18c', '#0000dd', '#a17569', '#bcb6ff', '#bf03b8', '#645472', '#790000',
    '#0774d8', '#729a7c', '#ff7752', '#004b00', '#8e7b01', '#f2007b', '#8eba00', '#a57bb8',
    '#5901a3', '#e2afaf', '#a03a52', '#a1c8c8', '#9e4b00', '#546744', '#bac389', '#5e7b87',
    '#60383b', '#8287ff', '#380000', '#e252ff', '#2f5282', '#7ecaff', '#c4668e', '#008069',
    '#919eb6', '#cc7407', '#7e2a8e', '#00bda3', '#2db152', '#4d33ff', '#00e400', '#ff00cd',
    '#c85748', '#e49cff', '#1ca1ff', '#6e70aa', '#c89a69', '#77563b', '#03dae6', '#c1a3c3',
    '#ff6989', '#ba00fd', '#915280', '#9e0174', '#93a14f', '#364424', '#af6dff', '#596d00',
)
PALETTE_INFO = {
    'name': 'Glasbey dark: 64 unique categorical colors',
    'source': 'Colorcet 3.2.1 glasbey_dark, first 64 entries in original order',
    'url': 'https://colorcet.holoviz.org/user_guide/Categorical.html',
    'attribution': 'Peter Kovesi / Colorcet contributors',
    'license': 'CC BY 4.0',
    'license_url': 'https://creativecommons.org/licenses/by/4.0/',
}

# Fixed display annotations, reviewed against the saved crops and activations.
# These choices do not enter the embeddings, UMAP fit, or decoder clustering.
ILLUSTRATED_DIMENSIONS = (2333, 2536, 3029, 698, 16, 2740)
ILLUSTRATION_SELECTION = {
    'method': 'manual replacement of two display dimensions after visual review',
    'retained_dimensions': [2536, 3029, 698, 16],
    'replacements': [
        {'previous': 2763, 'feature': 2333,
         'visual_description': 'pedestrian lower bodies, especially trousers and legs'},
        {'previous': 14, 'feature': 2740,
         'visual_description': 'windows with curtains'},
    ],
    'review': 'top eight crops, ranks 25/50/75/100, and top two original-image activation overlays',
    'examples': 'unchanged activation ranks: 1-8 for crops, 1-2 for original images and overlays',
    'interpretation': 'provisional visual descriptions; not semantic accuracy estimates',
}


def group_palette():
    """Use a fixed, unique color for every group, including unprojected groups."""
    rgb = np.array([list(bytes.fromhex(c[1:])) for c in GROUP_COLORS], dtype=np.uint8)
    if rgb.shape != (GROUPS, 3) or len(np.unique(rgb, axis=0)) != GROUPS:
        raise ValueError('Every decoder group must have a unique RGB color')
    return rgb


DETAIL_JS = r"""
document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{
 if(!d.open||d.dataset.loaded)return;d.dataset.loaded='1';
 d.querySelectorAll('img[data-src]').forEach(im=>im.src=im.dataset.src);
 const canvas=d.querySelector('canvas');if(!canvas)return;
 const source=new Image(), heat=new Image();
 Promise.all([new Promise(r=>{source.onload=r;source.src=d.dataset.image}),new Promise(r=>{heat.onload=r;heat.src='data:image/png;base64,'+d.dataset.heat})]).then(()=>{
  const w=Math.min(640,source.width),h=Math.round(w*source.height/source.width);canvas.width=w;canvas.height=h;
  const ctx=canvas.getContext('2d');ctx.filter='grayscale(1) brightness(.53)';ctx.drawImage(source,0,0,w,h);ctx.filter='none';
  const hc=document.createElement('canvas');hc.width=w;hc.height=h;const hx=hc.getContext('2d');hx.drawImage(heat,0,0,w,h);const pixels=hx.getImageData(0,0,w,h),values=pixels.data;
  for(let i=0;i<values.length;i+=4){const v=values[i]/255,c=lut[Math.round(v*255)];for(let k=0;k<3;k++)values[i+k]=c[k];values[i+3]=Math.round(Math.pow(v,.4)*.96*255);}
  hx.putImageData(pixels,0,0);ctx.drawImage(hc,0,0);const b=JSON.parse(d.dataset.box),sx=w/source.width,sy=h/source.height;
  ctx.strokeStyle='#000';ctx.lineWidth=4;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);
  ctx.strokeStyle='#5af5ff';ctx.lineWidth=2;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);
  d.dataset.rendered='1';
 });
}));
"""

INDEX_HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ViT-L SAE 图像语义 UMAP</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f6fa;color:#162036;font:14px system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}header{padding:17px 25px;background:white;border-bottom:1px solid #d9dfeb}h1{font-size:23px;margin:0 0 8px}p{line-height:1.6;margin:5px 0;color:#56617a}main{display:grid;grid-template-columns:57% 43%;flex:1;min-height:0}.plot{padding:15px;min-width:0;display:flex;flex-direction:column;gap:6px}#plot{width:100%;flex:1;min-height:160px;background:white;border:1px solid #d9dfeb;border-radius:10px;touch-action:none}iframe{height:100%;width:100%;border:0;border-left:1px solid #d9dfeb}button,input{padding:7px;margin:2px;border:1px solid #bac5d8;border-radius:5px;background:white}button{cursor:pointer}#legend{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:3px}#legend button{font-size:11px;padding:4px 2px;margin:0;white-space:nowrap}#legend button[aria-pressed=true]{outline:2px solid #162036}#legend button:disabled{color:#9299a3;background:#f5f6fa;cursor:default}.swatch{display:inline-block;width:9px;height:9px;margin-right:4px;border-radius:50%}#tip{position:fixed;display:none;background:white;border:1px solid #bac5d8;padding:10px;border-radius:8px;pointer-events:none;box-shadow:0 4px 20px #14223d30;z-index:10}#tip img{width:85px;height:85px;object-fit:contain;margin:3px}#status,#group-status{color:#56617a;font-size:12px;min-height:17px}@media(max-width:850px){#legend{grid-template-columns:repeat(4,minmax(0,1fr))}body{height:auto}main{display:block;min-height:auto}.plot{height:95vh;min-height:640px}iframe{height:80vh}}
</style><header><h1>SAE 维度的图像语义 UMAP · 64 组上色</h1><p id="summary"></p><p>每个点代表一个 SAE 维度，颜色对应已有的 64 组 decoder 聚类，每组固定使用一种独立颜色。坐标仍来自 Top 激活图像裁剪的 SigLIP 语义。点击图例高亮分组；悬停预览，点击查看原图、激活图和四方向全景。</p></header>
<main><section class="plot"><canvas id="plot"></canvas><div><button id="reset">重置视图</button><button id="all-groups">显示全部分组</button><input id="feature" type="number" min="0" max="4095" placeholder="维度编号"><button id="jump">查看</button></div><div id="group-status"></div><div id="legend" aria-label="64组聚类图例"></div><div id="status"></div></section><iframe id="detail" title="逐维度图像证据"></iframe></main><div id="tip"></div>
<script>const atlas=__DATA__;
const canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d'),points=atlas.points.filter(p=>p.x!==null),byId=new Map(atlas.points.map(p=>[p.feature,p]));
document.querySelector('#summary').textContent=`当前 ViT-L 模型 · W4096 / K32 · ${points.length} 个维度进入投影 · ${atlas.panoramas} 个验证全景 · ${atlas.cities} 个映射可靠城市 · SigLIP 图像编码`;
const xs=points.map(p=>p.x),ys=points.map(p=>p.y),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys),tip=document.querySelector('#tip');let width=1,height=1,scale=1,px=0,py=0,drag=null,selected=null,activeGroup=null;
function groupName(g){return 'G'+String(g).padStart(2,'0')}
function color(p){return 'rgb('+atlas.colors[p.group].join(',')+')'}
const legend=document.querySelector('#legend');
atlas.group_counts.forEach((n,g)=>{const button=document.createElement('button');button.dataset.group=g;button.disabled=!n;button.setAttribute('aria-pressed','false');button.title=`${groupName(g)}：${n} 个已投影维度 / ${atlas.group_totals[g]} 个全部维度`;button.innerHTML=`<span class="swatch" style="background:rgb(${atlas.colors[g].join(',')})"></span>${groupName(g)} (${n}/${atlas.group_totals[g]})`;button.onclick=()=>setGroup(activeGroup===g?null:g);legend.append(button)});
function setGroup(g){activeGroup=g;tip.style.display='none';for(const b of legend.children)b.setAttribute('aria-pressed',Number(b.dataset.group)===g?'true':'false');document.querySelector('#group-status').textContent=g===null?`${atlas.group_counts.filter(n=>n>0).length}/64 组有投影；括号为已投影数 / 总维度数。计数为 0 的分组暂无足够图像实例。`:`${groupName(g)} · ${atlas.group_counts[g]} 个已投影维度 / ${atlas.group_totals[g]} 个全部维度 · 其余分组淡化`;draw()}
function pos(p){const unit=Math.min((width-60)/(hiX-loX||1),(height-60)/(hiY-loY||1));return [(width/2+(p.x-(loX+hiX)/2)*unit)*scale+px,(height/2-(p.y-(loY+hiY)/2)*unit)*scale+py]}
function draw(){ctx.clearRect(0,0,width,height);const ordered=activeGroup===null?points:[...points.filter(p=>p.group!==activeGroup),...points.filter(p=>p.group===activeGroup)];for(const p of ordered){const [x,y]=pos(p),dimmed=activeGroup!==null&&p.group!==activeGroup;ctx.beginPath();ctx.arc(x,y,p.feature===selected?6:3,0,Math.PI*2);ctx.fillStyle=dimmed?'#e7eaf0':color(p);ctx.fill();if(p.feature===selected&&!dimmed){ctx.strokeStyle='#111';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#111';ctx.fillText('f'+p.feature,x+9,y-5)}}}
function resize(){const b=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;width=b.width;height=b.height;canvas.width=width*d;canvas.height=height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function nearest(x,y){let best=null,dist=12;for(const p of points){if(activeGroup!==null&&p.group!==activeGroup)continue;const [a,b]=pos(p),v=Math.hypot(x-a,y-b);if(v<dist){best=p;dist=v}}return best}
function select(id){const p=byId.get(id);if(!p)return;selected=id;if(activeGroup!==null&&p.group!==activeGroup)setGroup(p.group);document.querySelector('#feature').value=id;document.querySelector('#detail').src='features/'+id+'.html';document.querySelector('#status').textContent=`f${id} · ${groupName(p.group)} · ${p.count} 个不同全景实例 · 一致性 ${p.coherence.toFixed(3)}${p.x===null?' · 实例不足，未进入投影':''}`;location.hash='f'+id;draw()}
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.offsetX,y:e.offsetY,px,py,moved:false};tip.style.display='none'});
canvas.addEventListener('pointermove',e=>{if(drag){const dx=e.offsetX-drag.x,dy=e.offsetY-drag.y;if(Math.hypot(dx,dy)>4)drag.moved=true;px=drag.px+dx;py=drag.py+dy;draw();return}const p=nearest(e.offsetX,e.offsetY);if(!p){tip.style.display='none';return}tip.replaceChildren();const title=document.createElement('div');title.textContent=`f${p.feature} · ${groupName(p.group)} · ${p.count} 例 · 一致性 ${p.coherence.toFixed(3)}`;tip.append(title);for(const crop of p.preview){const im=document.createElement('img');im.src=crop;tip.append(im)}tip.style.display='block';tip.style.left=Math.min(e.clientX+15,innerWidth-310)+'px';tip.style.top=Math.min(e.clientY+15,innerHeight-145)+'px'});
canvas.addEventListener('pointerleave',()=>tip.style.display='none');canvas.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const p=nearest(e.offsetX,e.offsetY);if(p)select(p.feature)}drag=null});canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();const next=Math.max(.4,Math.min(20,scale*Math.exp(-e.deltaY*.001))),q=next/scale;px=e.offsetX-(e.offsetX-px)*q;py=e.offsetY-(e.offsetY-py)*q;scale=next;draw()},{passive:false});
document.querySelector('#reset').onclick=()=>{scale=1;px=py=0;draw()};document.querySelector('#all-groups').onclick=()=>setGroup(null);document.querySelector('#jump').onclick=()=>select(Number(document.querySelector('#feature').value));document.querySelector('#feature').addEventListener('keydown',e=>{if(e.key==='Enter')select(Number(e.target.value))});window.addEventListener('resize',resize);resize();setGroup(null);const requested=Number(location.hash.replace('#f',''));select(byId.has(requested)?requested:points[0].feature);
</script></html>"""


def gallery(root, rows, crops, data, run, labels, palette):
    folder = root / 'features'
    sources = root / 'sources'
    folder.mkdir(exist_ok=True)
    sources.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        path = sources / f'{i}.jpg'
        if not path.exists():
            path.symlink_to(row['image'])
    lut = np.rint(colormaps['inferno'](np.linspace(0, 1, 256))[:, :3] * 255).astype(int).tolist()
    groups = defaultdict(list)
    for row in crops:
        groups[row['feature']].append(row)
    points = []
    for j in range(run['width']):
        examples = sorted(groups[j], key=lambda r: r['rank'])
        points.append({'feature': j, 'group': int(labels[j]), 'x': float(data['xy'][j, 0]) if data['eligible'][j] else None,
                       'y': float(data['xy'][j, 1]) if data['eligible'][j] else None,
                       'count': int(data['count'][j]), 'coherence': float(data['coherence'][j]),
                       'preview': [r['crop'] for r in examples[:3]]})
        body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
                '<style>body{font:14px system-ui,sans-serif;margin:18px;color:#162036}h1{font-size:22px}p{color:#56617a;line-height:1.6}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.card{border:1px solid #d9dfeb;padding:10px;border-radius:7px;min-width:0}.crop{width:100%;height:140px;object-fit:contain}summary{cursor:pointer;color:#2462a9;padding:7px 0}canvas,.original{width:100%;height:auto}.pano{display:flex;overflow-x:auto}.pano img{width:50%;object-fit:contain}small{display:block;overflow-wrap:anywhere;color:#56617a}a{color:#2462a9}</style>',
                f'<h1>维度 f{j} · G{labels[j]:02d}</h1><p>所属组来自固定的 64 组 decoder 聚类。<br>{len(examples)} 个不同全景实例 · embedding 一致性 {data["coherence"][j]:.3f}<br>按稀疏激活峰值排名；裁剪只描述检索证据。激活热图颜色按本方向峰值归一化，强度比较请查看原始 peak。</p><div class="cards">']
        for row in examples:
            source = f'../sources/{row["image_id"]}.jpg'
            pano_start = row['image_id'] // 4 * 4
            body.extend([f'<article class="card"><img class="crop" loading="lazy" src="../{html.escape(row["crop"], quote=True)}" alt="高激活区域">',
                         f'<b>#{row["rank"]} · {html.escape(row["city"].split("/")[-1])} · {row["heading"]}°</b>',
                         f'<small>peak={row["activation"]:.5f}<br>{html.escape(row["pano"])}</small>',
                         f'<details data-image="{source}" data-heat="{row["heatmap_png"]}" data-box="{json.dumps(row["bbox_native_view"])}"><summary>原图与激活图</summary><a href="{source}" target="_blank"><img class="original" data-src="{source}" alt="原始方向图"></a><canvas></canvas></details>',
                         '<details><summary>四方向全景</summary><div class="pano">' + ''.join(f'<img data-src="../sources/{pano_start+h}.jpg" alt="{h*90}度">' for h in range(4)) + '</div></details></article>'])
        body.extend(['</div><script>const lut=' + json.dumps(lut) + ';', DETAIL_JS, '</script></html>'])
        (folder / f'{j}.html').write_text('\n'.join(body))
    atlas = {'points': points, 'panoramas': run['panoramas'], 'cities': len(run['cities']),
             'colors': palette.tolist(), 'group_counts': np.bincount(labels[data['eligible']], minlength=GROUPS).tolist(),
             'group_totals': np.bincount(labels, minlength=GROUPS).tolist()}
    (root / 'index.html').write_text(INDEX_HTML.replace('__DATA__', json.dumps(atlas).replace('<', '\\u003c')))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=OUTPUT / 'analysis/semantic_umap_top100')
    parser.add_argument('--report', type=Path, default=Path('/root/code/urban-visual-gene/reports/vitl30_semantic_umap'))
    args = parser.parse_args()
    root = args.result
    run = json.loads((root / 'run.json').read_text())
    labels = checked_groups(root.parent.parent, run['checkpoint_sha256'])
    clustering = json.loads((root.parent / 'decoder_groups.json').read_text())
    palette = group_palette()
    projection_hash = digest(root / 'feature_semantic_umap.npz')
    reduction = json.loads((root / 'reduction_run.json').read_text())
    rows = [json.loads(x) for x in (root / 'images.jsonl').read_text().splitlines()]
    crops = [json.loads(x) for x in (root / 'crops.jsonl').read_text().splitlines()]
    with np.load(root / 'feature_semantic_umap.npz') as archive:
        data = {k: archive[k] for k in archive.files}
    prototypes = np.load(root / 'feature_semantic_embeddings.f32.npy')
    if not np.isfinite(prototypes).all() or not np.isfinite(data['xy'][data['eligible']]).all():
        raise ValueError('Invalid projection')
    # Check panorama uniqueness for every dimension, not just image uniqueness.
    seen = set()
    for row in crops:
        key = row['feature'], row['city'], row['pano']
        if key in seen:
            raise ValueError('A panorama contributes twice to one feature')
        seen.add(key)
        if not (root / row['crop']).is_file():
            raise FileNotFoundError(row['crop'])
    args.report.mkdir(parents=True, exist_ok=True)
    chosen = list(ILLUSTRATED_DIMENSIONS)
    if any(not data['eligible'][j] or data['count'][j] < run['top'] for j in chosen):
        raise ValueError('Every illustrated dimension must have a complete projected Top100')
    fig, ax = plt.subplots(figsize=(12, 11))
    eligible = data['eligible']
    counts = np.bincount(labels[eligible], minlength=GROUPS)
    totals = np.bincount(labels, minlength=GROUPS)
    ax.scatter(*data['xy'][eligible].T, c=palette[labels[eligible]] / 255,
               s=13, alpha=1, linewidths=0, rasterized=True)
    for feature in chosen:
        x, y = data['xy'][feature]
        ax.scatter(x, y, s=90, facecolors='none', edgecolors='#202a35', linewidths=1.1)
        ax.annotate(f'f{feature} / G{labels[feature]:02d}', (x, y), xytext=(7, 7), textcoords='offset points', fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 1.5})
    handles = [Line2D([], [], marker='o', linestyle='', markersize=5,
                      color=palette[g] / 255,
                      label=f'G{g:02d} ({counts[g]}/{totals[g]})') for g in range(GROUPS)]
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .055), ncol=8,
                        frameon=False, fontsize=8, columnspacing=1.4, handletextpad=.3,
                        title='Existing decoder groups G00–G63 (projected / total dimensions)', title_fontsize=9)
    for text, count in zip(legend.get_texts(), counts):
        if not count:
            text.set_color('#9299a3')
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2',
           title='Image semantics of SAE dimensions, colored by 64 decoder groups')
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.08, .016, f'{int(eligible.sum()):,} supported dimensions | {run["panoramas"]:,} held-out panoramas | '
             f'{len(run["cities"])} mapped cities | {np.count_nonzero(counts)}/64 groups represented\n'
             'Coordinates: Top100 panorama crop embeddings (SigLIP), cosine UMAP. '
             'Colors: decoder cosine / average linkage; labels do not enter UMAP.', fontsize=8.5, color='#526071')
    fig.subplots_adjust(left=.08, right=.98, top=.94, bottom=.28)
    name = 'Fig_ViTL30_Dimension_Semantic_UMAP'
    for suffix in ['png', 'pdf']:
        fig.savefig(args.report / f'{name}.{suffix}', dpi=200)
    plt.close(fig)
    groups = gallery(root, rows, crops, data, run, labels, palette)
    fig, axes = plt.subplots(len(chosen), 8, figsize=(16, 2*len(chosen)), squeeze=False)
    for y, feature in enumerate(chosen):
        for x, ax in enumerate(axes[y]):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            examples = sorted(groups[feature], key=lambda row: row['rank'])
            if x < len(examples):
                row = examples[x]
                with Image.open(root / row['crop']) as crop:
                    ax.imshow(crop.convert('RGB'))
                ax.set_title(f'#{row["rank"]} | {row["city"].split("/")[-1]}', fontsize=8)
            if x == 0:
                ax.set_ylabel(f'f{feature} / G{labels[feature]:02d}\nn={data["count"][feature]}\nagreement={data["coherence"][feature]:.2f}', fontsize=9,
                              color=palette[labels[feature]] / 255)
    fig.suptitle('Actual top-activation crops for six illustrated SAE dimensions', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    for suffix in ['png', 'pdf']:
        fig.savefig(args.report / f'Fig_ViTL30_Semantic_UMAP_Examples.{suffix}', dpi=170)
    plt.close(fig)
    support = np.load(root / 'positive_panorama_support.npy')
    summary = {**reduction, 'candidate_cities': len(run['cities']), 'candidate_panoramas': run['panoramas'],
               'directional_images': run['images'], 'crops': len(crops),
               'unique_encoded_crops': json.loads((root / 'embedding_run.json').read_text())['unique_crops'],
               'full_top100_dimensions': int((data['count'] >= run['top']).sum()),
               'zero_response_in_sample': int((support == 0).sum()), 'illustrated_dimensions': chosen,
               'illustration_selection': ILLUSTRATION_SELECTION,
               'checkpoint_sha256': run['checkpoint_sha256'], 'crop_manifest_sha256': digest(root / 'crops.jsonl'),
               'coloring': {'source': 'existing decoder groups', 'groups': GROUPS,
                            'method': clustering['method'], 'labels_sha256': clustering['labels_sha256'],
                            'palette': PALETTE_INFO,
                            'unique_colors': len(np.unique(palette, axis=0)),
                            'projected_group_counts': counts.tolist(), 'total_group_counts': totals.tolist(),
                            'represented_groups': int(np.count_nonzero(counts)),
                            'unrepresented_groups': np.flatnonzero(counts == 0).tolist()},
               'projection_npz_sha256': projection_hash,
               'all_dimensions_have_unique_panorama_examples': True,
               'finite_eligible_coordinates': True, 'interactive_gallery': str(root / 'index.html')}
    write_json(args.report / 'summary.json', summary)
    write_json(args.report / 'group_palette.json', {
        'colors_rgb': palette.tolist(), 'group_ids': list(range(GROUPS)),
        'colors_hex': list(GROUP_COLORS), 'palette': PALETTE_INFO,
        'labels_sha256': clustering['labels_sha256'], 'checkpoint_sha256': run['checkpoint_sha256']})
    with (args.report / 'feature_semantic_umap_groups.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'decoder_group', 'umap_1', 'umap_2', 'crop_count', 'coherence', 'status'])
        for j in range(run['width']):
            xy = data['xy'][j].tolist() if eligible[j] else ['', '']
            writer.writerow([j, int(labels[j]), *xy, int(data['count'][j]), float(data['coherence'][j]),
                             'eligible' if eligible[j] else 'excluded'])
    for source in ['feature_semantic_umap.csv', 'run.json', 'embedding_run.json', 'reduction_run.json']:
        shutil.copy2(root / source, args.report / source)
    for path in args.report.glob('Fig_ViTL30_*'):
        shutil.copy2(path, OUTPUT / 'figures' / path.name)
    body = ['# Current ViT-L SAE: dimension-level image semantic UMAP', '',
            f'使用当前 W4096、K32 模型的固定验证集：{len(run["cities"])} 个映射可靠城市，{run["panoramas"]:,} 个完整全景，{run["images"]:,} 张方向图。未重新提取 DINO 特征或训练 SAE。Moscow 因映射尚未核对而排除。', '',
            '逐维度按全景内最大正稀疏激活排名，同一全景只保留最佳方向。每个全景的全部 3,136 个 patch 构成独立 BatchTopK 预算，使用 FP32 推理。候选集为训练前固定的验证点位，不混入之前按 64 组检索的实例。', '',
            f'每维最多 Top100；峰值 50% 阈值连通区域外扩一个 patch，最小 3×3 patch，映射回原始图像裁剪。得到 {len(crops):,} 个逐维实例，其中 {summary["unique_encoded_crops"]:,} 个不同裁剪区域。相同像素区域只编码一次，汇总时仍保留各维度自己的等权实例。', '',
            '干净裁剪经 SigLIP base patch16 224 图像 encoder，逐例 L2 归一化，逐维等权平均后再次归一化。坐标不使用空间位置、城市标签、decoder 权重、64 组标签或 VLM 文字描述。embedding 一致性仍在逐维详情中保留，它不是语义准确率。', '',
            f'散点颜色改为当前 SAE 已有的 64 组 decoder 聚类标签 G00–G63（decoder 余弦距离、平均链接）；每个点仍是一个原始 SAE 维度，并非只有 64 个点。沿用标签及组编号，不重新聚类或拟合 UMAP。当前 {np.count_nonzero(counts)} 组有足够实例进入投影；G42、G44、G51 暂无投影成员，图例保留所有 64 种独立组色，暂无投影的组只将文字灰显。图例括号显示“已投影数 / 总维度数”，例如 G42 为 0/9；0 不表示组内没有 SAE 维度。组色仅辅助对照 decoder 分组与图像语义之间的关系，不预设二者完全一致。', '',
            f'使用余弦 UMAP，n_neighbors=30、min_dist=0.1、seed=42。至少 20 个有效实例才投影：{reduction["included"]:,} 个维度纳入，{reduction["excluded"]:,} 个维度证据不足或原型退化，{summary["full_top100_dimensions"]:,} 个维度有完整 Top100。', '',
            '图中六个标注用于实例展示：保留 f2536、f3029、f698、f16，将 f2763 替换为 f2333，将 f14 替换为 f2740。两个替换维度经过人工目视核对：检查前八名及第 25、50、75、100 名裁剪，并检查前两名原图与激活图。f2333 主要呈现行人下半身、裤装与腿部，f2740 主要呈现带窗帘的窗户；这是基于实例的暂定描述。人工选择仅决定展示哪些维度，不参与 embedding、UMAP 拟合或聚类。每个展示维度都有完整 Top100，实例图仍严格展示激活排名前八例，原图与激活图严格展示前两例，不人工换图。', '',
            'UMAP 只辅助观察局部关系；不据二维距离宣称类别边界、统计显著性或冗余。平均原型可能掩盖多义特征，应查看完整实例；Top100 仅针对本候选集。', '',
            f'[本地交互页面]({root / "index.html"})：点击图例高亮指定组，其余点淡化；悬停预览、点击维度，逐例查看干净裁剪、原图、激活和四方向全景。页面原图通过本机 NAS 路径链接，分享前需另行打包。', '',
            '色表使用 Colorcet 3.2.1 的 Glasbey dark 前 64 色，一组一色，固定顺序；静态散点与交互页面均使用不透明的相同 RGB 值。暂无投影的组也保留自己的颜色。来源：Peter Kovesi / Colorcet contributors，CC BY 4.0；[色表说明](https://colorcet.holoviz.org/user_guide/Categorical.html)。', '',
            '[全部维度的分组与坐标](feature_semantic_umap_groups.csv) · [可复现组色](group_palette.json)', '',
            f'![Dimension semantic UMAP]({name}.png)', '',
            '![Representative crops](Fig_ViTL30_Semantic_UMAP_Examples.png)', '']
    (args.report / 'REPORT.md').write_text('\n'.join(body))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
