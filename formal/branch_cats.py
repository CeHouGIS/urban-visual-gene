"""Derive gene->category from the phylogeny TREE super-branches (not KMeans).
Each top-level super-branch = one category. Saves gene2cat.npy + renders a
naming montage (one row per branch, top genes' overlays)."""
import json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
G2=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2/genes")
man=json.load(open(G2/"web"/"manifest.json"))
tree=man["tree"]; genes=man["genes"]
NBR=len(tree); print(f"{NBR} super-branches")
# gene -> branch index (branch order = tree order = dendrogram leaf order)
g2c=np.full(512,-1,int); branch_genes=[]
for i,br in enumerate(tree):
    gids=br["genes"] if "genes" in br else [g for c in br["children"] for g in c["genes"]]
    for g in gids: g2c[g]=i
    branch_genes.append(sorted(gids,key=lambda g:-genes[str(g)]["peak"]))
assert (g2c>=0).all(), f"{(g2c<0).sum()} genes unassigned"
np.save(G2.parent/"gene2cat.npy", g2c)   # formal_out_global2/gene2cat.npy
print("branch sizes:",[len(b) for b in branch_genes])
# montage: one row per branch, top-8 genes' rep overlays
NTOP=8; TH=128
W=NTOP*TH+80; H=NBR*TH
canvas=Image.new("RGB",(W,H),(10,14,22)); dr=ImageDraw.Draw(canvas)
for i,b in enumerate(branch_genes):
    y=i*TH; dr.text((4,y+TH//2-8),f"br{i}\n(n={len(b)})",fill=(220,230,240))
    for ci,g in enumerate(b[:NTOP]):
        p=G2/"web"/genes[str(g)]["ex"][0].replace("genes/","")
        try: im=Image.open(p).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(20,20,30))
        x=80+ci*TH; canvas.paste(im,(x,y)); dr.text((x+2,y+2),f"#{g}",fill=(255,255,120))
canvas.save(G2/"diag"/"branch_montage.png")
print("saved branch_montage.png")
for i,b in enumerate(branch_genes): print(f"br{i}: n={len(b)} top:",b[:8])
