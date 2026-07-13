"""Interpret-page category montage from the branch categories (CPU, no GPU):
one row per category, top-8 representative gene overlays. -> gallery PNG."""
import os, json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
G2=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2"))
GEN=G2/"genes"; GAL=G2/"web"/"gallery"; GAL.mkdir(parents=True,exist_ok=True)
man=json.load(open(GEN/"web"/"manifest.json"))
genes=man["genes"]; cats=man["categories"]; NCAT=len(cats)
by={c["id"]:[] for c in cats}
for gid,g in genes.items(): by[g["cat"]].append(g)
for c in by: by[c].sort(key=lambda g:-g["peak"])
NTOP=8; TH=132; LAB=150
W=LAB+NTOP*TH; H=NCAT*TH
canvas=Image.new("RGB",(W,H),(7,11,20)); dr=ImageDraw.Draw(canvas)
for c in cats:
    i=c["id"]; y=i*TH
    dr.rectangle([0,y,10,y+TH],fill=tuple(int(c["color"].lstrip('#')[k:k+2],16) for k in (0,2,4)))
    dr.text((16,y+TH//2-10),f"cat {i}  (n={len(by[i])})",fill=(219,228,240))
    for ci,g in enumerate(by[i][:NTOP]):
        p=GEN/"web"/g["ex"][0].replace("genes/","")
        try: im=Image.open(p).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(20,20,30))
        x=LAB+ci*TH; canvas.paste(im,(x,y)); dr.text((x+3,y+3),f"#{g['id']}",fill=(255,255,130))
canvas.save(GAL/"global_category_montage.png")
print(f"saved global_category_montage.png ({NCAT} cats) -> {GAL}")
