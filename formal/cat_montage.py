"""Render the NEW dict's 10 KMeans categories as rows of representative gene
overlays (from the already-rendered genes/web/exem), for naming. CPU."""
import json, numpy as np
from pathlib import Path
from PIL import Image
G2=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2/genes")
man=json.load(open(G2/"web"/"manifest.json"))
genes=man["genes"]; NCAT=10; NTOP=8; TH=128
by={c:[] for c in range(NCAT)}
for gid,g in genes.items(): by[g["cat"]].append(g)
for c in by: by[c].sort(key=lambda g:-g["peak"])
rows=[]
for c in range(NCAT):
    tops=by[c][:NTOP]
    imgs=[]
    for g in tops:
        p=G2/"web"/g["ex"][0].replace("genes/","")     # ex path is 'genes/exem/..'
        try: im=Image.open(p).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(20,20,30))
        imgs.append((im,g["id"],g["peak"]))
    rows.append((c,len(by[c]),imgs))
from PIL import ImageDraw
W=NTOP*TH+70; H=NCAT*TH
canvas=Image.new("RGB",(W,H),(10,14,22)); dr=ImageDraw.Draw(canvas)
for ri,(c,n,imgs) in enumerate(rows):
    y=ri*TH
    dr.text((4,y+TH//2-8),f"cat{c}\n(n={n})",fill=(220,230,240))
    for ci,(im,gid,pk) in enumerate(imgs):
        x=70+ci*TH; canvas.paste(im,(x,y))
        dr.text((x+2,y+2),f"#{gid}",fill=(255,255,120))
canvas.save(G2/"diag"/"cat_montage.png")
print("saved cat_montage.png")
for c in range(NCAT):
    print(f"cat{c}: n={len(by[c])} top genes:", [g['id'] for g in by[c][:NTOP]])
