"""Diagnose category REDUNDANCY: how similar are the 14 tree-branch categories,
and what DISTINGUISHES each from its neighbours. -> similarity matrix, a category
dendrogram, and a 'distinctive genes' montage (genes most specific to each cat)."""
import json, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
G2=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2")
GEN=G2/"genes"; DG=GEN/"diag"
man=json.load(open(GEN/"web"/"manifest.json")); CN=[c["name"] for c in man["categories"]]; NCAT=len(CN)
genes=man["genes"]; g2c=np.load(G2/"gene2cat.npy")
d=torch.load(G2/"sae_448_k512.pt",map_location="cpu"); A=d["state"]["dec.weight"].T.numpy()
A=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)

# category centroids + cosine similarity
cen=np.stack([A[g2c==c].mean(0) for c in range(NCAT)]); cen=cen/(np.linalg.norm(cen,axis=1,keepdims=True)+1e-9)
S=np.clip(cen@cen.T,-1,1)
print("=== category centroid cosine similarity (nearest neighbour each) ===")
for c in range(NCAT):
    o=[(S[c,k],k) for k in range(NCAT) if k!=c]; o.sort(reverse=True)
    nn=", ".join(f"{CN[k]}={s:.2f}" for s,k in o[:3])
    print(f"  {c:2d} {CN[c]:8s} ~ {nn}")

# dendrogram of the 14 categories -> which are near-duplicates
D=1-S; np.fill_diagonal(D,0); D=(D+D.T)/2
Z=linkage(squareform(D,checks=False),method="average")
print("\n=== category groups at cos>=0.6 (merge height<=0.4) ===")
from scipy.cluster.hierarchy import fcluster
gr=fcluster(Z,t=0.4,criterion="distance")
for g in sorted(set(gr)):
    mem=[CN[c] for c in range(NCAT) if gr[c]==g]
    if len(mem)>1: print(f"  REDUNDANT GROUP: {mem}")
    else: print(f"  distinct: {mem[0]}")

# distinctive genes per category: atom most aligned to (own centroid - mean of other centroids)
other=np.stack([np.delete(cen,c,0).mean(0) for c in range(NCAT)])
disc=cen-other; disc=disc/(np.linalg.norm(disc,axis=1,keepdims=True)+1e-9)   # discriminative direction
def distinctive(c,n=6):
    mem=np.where(g2c==c)[0]
    sc=A[mem]@disc[c]                       # how discriminative each member gene is
    return mem[np.argsort(-sc)][:n]

# montage: each cat row = its most DISTINCTIVE genes
NTOP=6; TH=132; LAB=150
canvas=Image.new("RGB",(LAB+NTOP*TH,NCAT*TH),(7,11,20)); dr=ImageDraw.Draw(canvas)
for c in range(NCAT):
    y=c*TH; dr.text((6,y+TH//2-8),f"c{c} n={int((g2c==c).sum())}",fill=(219,228,240))
    for ci,g in enumerate(distinctive(c,NTOP)):
        p=GEN/"web"/genes[str(int(g))]["ex"][0].replace("genes/","")
        try: im=Image.open(p).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(20,20,30))
        canvas.paste(im,(LAB+ci*TH,y)); dr.text((LAB+ci*TH+2,y+2),f"#{int(g)}",fill=(255,255,130))
canvas.save(DG/"cat_distinctive.png")
# category dendrogram figure
fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor("#070b14"); ax.set_facecolor("#070b14")
dendrogram(Z,labels=[f"c{c}" for c in range(NCAT)],ax=ax,color_threshold=0.4,above_threshold_color="#8a9bb5")
ax.set_title("category similarity (lower merge = more redundant)",color="#dbe4f0")
ax.tick_params(colors="#8a9bb5"); [s.set_visible(False) for s in ax.spines.values()]
fig.tight_layout(); fig.savefig(DG/"cat_dendro.png",dpi=110,facecolor="#070b14"); plt.close(fig)
print("\nsaved cat_distinctive.png + cat_dendro.png")
