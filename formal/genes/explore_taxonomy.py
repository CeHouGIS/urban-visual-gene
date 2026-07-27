"""Explore a 2-LEVEL category taxonomy from the ward tree: cut HIGH for macro
parents (merge redundant flat cats), then sub-cut each parent into children
(keep real distinctions like trunk vs canopy). Prints structure + montage."""
import os, sys, json, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import linkage, to_tree
G2=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2"))
GEN=G2/"genes"; man=json.load(open(GEN/"web"/"manifest.json"))
genes=man["genes"]; oldCN=[c["name"] for c in man["categories"]]; g2c_old=np.load(G2/"gene2cat.npy")
d=torch.load(G2/"sae_448_k512.pt",map_location="cpu"); A=d["state"]["dec.weight"].T.numpy()
An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
Z=linkage(An,method="ward",metric="euclidean"); root=to_tree(Z)
def split_to_cap(node,cap):
    cl=[node]
    while True:
        over=[c for c in cl if c.count>cap and not c.is_leaf()]
        if not over: break
        big=max(over,key=lambda c:c.count); cl.remove(big); cl+=[big.left,big.right]
    return cl
def leaves(n): return n.pre_order(lambda x:x.id)
def topgenes(gids,k=8): return sorted(gids,key=lambda g:-genes[str(g)]["peak"])[:k]

PCAP=int(os.environ.get("PCAP","150")); CCAP=int(os.environ.get("CCAP","45"))
parents=split_to_cap(root,PCAP)
# order by leaf position
from scipy.cluster.hierarchy import dendrogram
order=dendrogram(Z,no_plot=True)["leaves"]; pos={g:i for i,g in enumerate(order)}
parents.sort(key=lambda n:min(pos[g] for g in leaves(n)))
print(f"PCAP={PCAP} -> {len(parents)} parents; CCAP={CCAP}\n")
struct=[]
for pi,pn in enumerate(parents):
    pg=leaves(pn); kids=split_to_cap(pn,CCAP); kids.sort(key=lambda n:min(pos[g] for g in leaves(n)))
    # which old-14 cats fall in this parent (to see merges)
    from collections import Counter
    oc=Counter(oldCN[g2c_old[g]] for g in pg)
    print(f"P{pi} ({len(pg)} genes)  old-cats: {dict(oc.most_common(5))}")
    print(f"    top genes: {topgenes(pg)}")
    for ci,kn in enumerate(kids):
        kg=leaves(kn); print(f"    -c{ci} ({len(kg)}): {topgenes(kg,6)}")
    struct.append((pn,pg,kids))
print()

# montage: one row-block per parent; within it show top genes of each child
TH=120; NG=6
rows=[]
for pi,(pn,pg,kids) in enumerate(struct):
    for ci,kn in enumerate(kids):
        rows.append((pi,ci,topgenes(leaves(kn),NG)))
W=190+NG*TH; H=len(rows)*TH
cv=Image.new("RGB",(W,H),(7,11,20)); dr=ImageDraw.Draw(cv)
for ri,(pi,ci,gs) in enumerate(rows):
    y=ri*TH; dr.text((6,y+TH//2-8),f"P{pi}-c{ci}",fill=(219,228,240))
    for j,g in enumerate(gs):
        p=GEN/"web"/genes[str(int(g))]["ex"][0].replace("genes/","")
        try: im=Image.open(p).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(20,20,30))
        cv.paste(im,(190+j*TH,y)); dr.text((190+j*TH+2,y+2),f"#{int(g)}",fill=(255,255,130))
cv.save(GEN/"diag"/"taxonomy.png"); print("saved taxonomy.png,",len(rows),"child rows")
