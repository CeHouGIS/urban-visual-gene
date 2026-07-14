"""Build a self-contained HTML tunnel-labeling page: CLIP tunnel candidates as a
grid (images embedded as base64), click to toggle 隧道/非隧道, live label output
to copy back. Saves the index->path map so labels can be mapped back."""
import base64, io, json, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
DIAG=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2/genes/diag")
OUT=DIAG/"qc_label.html"; MAP=DIAG/"qc_label_candidates.parquet"
d=pd.read_parquet(DIAG/"qc_clip_scores.parquet")
cand=d[d.tunnel_prob>0.28].sort_values("tunnel_prob",ascending=False).head(320).reset_index(drop=True)
cand.to_parquet(MAP)
items=[]
for i,r in cand.iterrows():
    try:
        im=Image.open(r["path"]).convert("RGB"); im.thumbnail((104,104))
        b=io.BytesIO(); im.save(b,"JPEG",quality=58)
        src="data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
    except: src=""
    items.append({"i":int(i),"p":round(float(r["tunnel_prob"]),2),"d":int(r["tunnel_prob"]>0.5),"src":src})
J=json.dumps(items)
html=f"""<!doctype html><html lang=zh><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>隧道标注</title><style>
body{{margin:0;background:#0b0f18;color:#dbe4f0;font-family:"PingFang SC","Microsoft YaHei",Arial}}
#bar{{position:sticky;top:0;background:#0b0f18ee;backdrop-filter:blur(8px);padding:10px 14px;border-bottom:1px solid #1e2d47;z-index:5}}
#bar b{{color:#39d6ff}} .hint{{color:#8a9bb5;font-size:12.5px;margin:4px 0 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:6px;padding:12px}}
.c{{position:relative;border:3px solid #444;border-radius:8px;overflow:hidden;cursor:pointer;aspect-ratio:1}}
.c img{{width:100%;height:100%;object-fit:cover;display:block}}
.c.y{{border-color:#22e0a1}} .c.n{{border-color:#ff5d5d}}
.c .tag{{position:absolute;top:2px;left:2px;font-size:11px;font-weight:800;padding:1px 5px;border-radius:4px;background:#000a}}
.c.y .tag{{color:#22e0a1}} .c.n .tag{{color:#ff5d5d}}
.c .p{{position:absolute;bottom:2px;right:2px;font-size:9.5px;color:#cfe;background:#000a;padding:0 3px;border-radius:3px}}
textarea{{width:100%;height:60px;background:#0a1120;color:#cfe;border:1px solid #1e2d47;border-radius:8px;font-family:monospace;font-size:12px}}
button{{background:#39d6ff;color:#04121a;border:0;border-radius:7px;padding:7px 14px;font-weight:700;cursor:pointer;margin-right:8px}}
</style></head><body>
<div id=bar>
 <b>隧道标注</b> —— 点图片切换:绿框=隧道✓，红框=非隧道✗。默认按 CLIP 概率预设(>0.5 绿)，你只需<b>改错的</b>。
 <div class=hint>标完点「生成」，把下面文本框内容<b>复制回贴给我</b>。 <span id=stat></span></div>
 <div style="margin-top:8px"><button onclick=gen()>生成标签</button><button onclick=allY()>全设隧道</button><button onclick=allN()>全设非隧道</button><button onclick=inv()>反选</button></div>
 <textarea id=out placeholder="点「生成标签」后这里出现,复制它贴回给我"></textarea>
</div>
<div class=grid id=g></div>
<script>
const D={J}; const st={{}};
const g=document.getElementById('g');
D.forEach(o=>{{st[o.i]=o.d; const c=document.createElement('div'); c.className='c '+(o.d?'y':'n'); c.id='c'+o.i;
 c.innerHTML=`<img src="${{o.src}}"><span class=tag>${{o.d?'✓':'✗'}}</span><span class=p>${{o.p}}</span>`;
 c.onclick=()=>{{st[o.i]=st[o.i]?0:1; c.className='c '+(st[o.i]?'y':'n'); c.querySelector('.tag').textContent=st[o.i]?'✓':'✗'; upd();}};
 g.appendChild(c);}});
function upd(){{let y=0,n=0;for(const k in st){{st[k]?y++:n++;}}document.getElementById('stat').textContent=`当前 隧道 ${{y}} · 非隧道 ${{n}}`;}}
function setall(v){{D.forEach(o=>{{st[o.i]=v;const c=document.getElementById('c'+o.i);c.className='c '+(v?'y':'n');c.querySelector('.tag').textContent=v?'✓':'✗';}});upd();}}
function allY(){{setall(1);}} function allN(){{setall(0);}}
function inv(){{D.forEach(o=>{{st[o.i]=st[o.i]?0:1;const c=document.getElementById('c'+o.i);c.className='c '+(st[o.i]?'y':'n');c.querySelector('.tag').textContent=st[o.i]?'✓':'✗';}});upd();}}
function gen(){{const y=[],n=[];for(const k in st){{(st[k]?y:n).push(+k);}}
 document.getElementById('out').value='TUNNEL_LABELS tunnel='+y.join(',')+' | not='+n.join(',');}}
upd();
</script></body></html>"""
OUT.write_text(html,encoding="utf-8")
print(f"saved {OUT}  ({len(items)} candidates, {OUT.stat().st_size/1e6:.1f}MB)")
print("default tunnel(>0.5):",int((cand.tunnel_prob>0.5).sum()),"of",len(cand))
