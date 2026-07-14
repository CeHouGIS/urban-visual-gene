"""Visualize co-expression MODULES on real street scenes: for each module, find
the images where its member genes co-fire most, and jet-overlay the module's
combined activation. Proves the gene COMBINATION expresses a coherent concept."""
import os, json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.cm as cm
G2=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2"))
GEN=G2/"genes"; DG=GEN/"diag"
man=json.load(open(GEN/"web"/"manifest.json")); CATN={c["id"]:c["name"] for c in man["categories"]}
g2c=np.load(G2/"gene2cat.npy")
co=json.load(open(DG/"coexpr.json")); modules=co["modules"]
z=np.load(GEN/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"].astype(np.int32); val=z["val"].astype(np.float32); N,P,T=idx.shape
th=[str(GEN/"thumbs"/f"{i}.jpg") for i in range(N)]
def log(*a): import time;print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

NMOD=8; NEX=6; TH=150
def coact(members):
    mem=np.array(members)
    mask=np.isin(idx,mem)                       # (N,784,32)
    per_patch=(val*mask).sum(2)                 # (N,784) summed member activation
    score=per_patch.sum(1)                      # (N,) total module expression
    return per_patch, score
def overlay(tp,m28):
    m=m28.astype(np.float32); m=(m-m.min())/(m.max()-m.min()+1e-8)
    up=np.array(Image.fromarray((m*255).astype(np.uint8)).resize((TH,TH),Image.BILINEAR),np.float32)/255
    heat=(cm.jet(up)[...,:3]*255).astype(np.float32)
    img=np.array(Image.open(tp).convert("RGB").resize((TH,TH)),np.float32)
    a=(0.30+0.55*up)[...,None]; return (img*(1-a)+heat*a).astype(np.uint8)

sel=modules[:NMOD]
W=170+NEX*TH; H=NMOD*TH
canvas=Image.new("RGB",(W,H),(7,11,20)); dr=ImageDraw.Draw(canvas)
info=[]
for mi,md in enumerate(sel):
    per_patch,score=coact(md["genes"])
    top=np.argsort(-score)[:NEX]
    y=mi*TH
    dr.text((6,y+8),f"M{mi}",fill=(255,255,255))
    dr.text((6,y+30),f"{md['n']} genes",fill=(150,170,200))
    dr.text((6,y+48),f"coh {md['coh']}",fill=(150,170,200))
    for ci,i in enumerate(top):
        ov=overlay(th[int(i)],per_patch[int(i)].reshape(28,28))
        canvas.paste(Image.fromarray(ov),(170+ci*TH,y))
    catmix=[(CATN[c],n) for c,n in md["cats"][:4]]
    info.append((mi,md["n"],md["coh"],catmix,md["genes"][:8]))
    log(f"M{mi}: {md['n']} genes coh {md['coh']} {catmix}")
canvas.save(DG/"coexpr_modules.png")
log(f"[done] -> coexpr_modules.png")
json.dump(info,open(DG/"module_info.json","w"),ensure_ascii=False)
