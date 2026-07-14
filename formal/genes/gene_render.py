"""Render the 512-gene dashboard: per-gene top-20 jet-overlay exemplars +
512-gene hierarchical-clustering dendrogram + manifest. CPU, from sparse acts."""
import os, json, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.cm as cm, matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
OUT=Path(os.environ.get("GENESDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global/genes"))
EX=OUT/"web"/"exem"; EX.mkdir(parents=True,exist_ok=True)
K=512; G=28; NEX=20; NCAT=10
CATN=["住宅立面·地面","立面·车辆","植被·开阔地","路面·商业立面","密集植被·田野","路面·商铺","道路走廊·绿化","树冠","密集城市立面","墙面·植被"]
CATC=["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff","#f6c945","#9b8cff","#ff9e64","#73daca"]
def log(*a): import time; print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

z=np.load(OUT/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"].astype(np.int64); val=z["val"].astype(np.float32)   # (N,784,32)
N=idx.shape[0]; th=[str(OUT/"thumbs"/f"{i}.jpg") for i in range(N)]
atoms=np.load(OUT/"atoms.npy")                                    # (512,1024)
g2c=KMeans(NCAT,n_init=10,random_state=0).fit_predict(atoms)
log(f"{N} imgs, computing per-gene max ...")
# per-image per-gene max activation
maxmat=np.zeros((N,K),np.float32)
for i in range(N):
    np.maximum.at(maxmat[i], idx[i].ravel(), val[i].ravel())
maxmat[:,0]*=0 if False else 1
peak=maxmat.max(0); nimg=(maxmat>0.05*(peak+1e-9)).sum(0)

def gmap(i,k):
    m=np.where(idx[i]==k, val[i], 0).max(1)                       # (784,)
    return m.reshape(G,G)
def overlay(tp,gm):
    m=gm.astype(np.float32); m=(m-m.min())/(m.max()-m.min()+1e-8)
    up=np.array(Image.fromarray((m*255).astype(np.uint8)).resize((224,224),Image.BILINEAR),np.float32)/255
    heat=(cm.jet(up)[...,:3]*255).astype(np.float32)
    img=np.array(Image.open(tp).convert("RGB").resize((224,224)),np.float32)
    a=(0.3+0.55*up)[...,None]
    return (img*(1-a)+heat*a).astype(np.uint8)

log("rendering 512 x top-20 overlays ...")
genes=[]
order=np.argsort(-peak)                                           # most-used genes first
for rank,k in enumerate(order):
    k=int(k)
    if peak[k]<1e-6:
        genes.append({"id":k,"cat":int(g2c[k]),"peak":0.0,"nimg":0,"ex":[]}); continue
    top=np.argsort(-maxmat[:,k])[:NEX]; top=[int(i) for i in top if maxmat[i,k]>1e-6]
    gd=EX/f"g{k}"; gd.mkdir(exist_ok=True); ex=[]
    for r,i in enumerate(top):
        Image.fromarray(overlay(th[i],gmap(i,k))).resize((128,128)).save(gd/f"{r}.jpg",quality=72)
        ex.append(f"genes/exem/g{k}/{r}.jpg")
    genes.append({"id":k,"cat":int(g2c[k]),"peak":round(float(peak[k]),2),"nimg":int(nimg[k]),"ex":ex})
    if rank%50==0: log(f"  {rank}/{K} genes")

# ---- interactive hierarchy: multi-level cut of the cosine-distance dendrogram ----
log("building nested tree ...")
A=atoms/(np.linalg.norm(atoms,axis=1,keepdims=True)+1e-9)
D=1-np.clip(A@A.T,-1,1); np.fill_diagonal(D,0); D=(D+D.T)/2
Z=linkage(squareform(D,checks=False),method="average")
leaves=dendrogram(Z,no_plot=True)["leaves"]                # dendrogram leaf order (adjacent = similar)
L1=6; L2=22                                                 # super-branches, sub-branches
lab1=fcluster(Z,t=L1,criterion="maxclust")                 # (512,) 1..L1
lab2=fcluster(Z,t=L2,criterion="maxclust")                 # refinement of lab1 (same tree, deeper cut)
gdict={g["id"]:g for g in genes}
def has(gid): return bool(gdict[gid]["ex"])
def dom_cat(gids):                                          # dominant category among genes with exemplars
    cs=[gdict[g]["cat"] for g in gids if has(g)] or [gdict[g]["cat"] for g in gids]
    return max(set(cs),key=cs.count)
def rep(gids):                                             # representative overlay = strongest gene's top img
    cand=[g for g in gids if has(g)]
    if not cand: return None
    best=max(cand,key=lambda g:gdict[g]["peak"]); return gdict[best]["ex"][0]
def node(gids,name_seen):                                   # build a cluster node
    live=[g for g in gids if has(g)]
    if not live: return None
    c=dom_cat(gids); base=CATN[c]; name_seen[base]=name_seen.get(base,0)+1
    nm=base if name_seen[base]==1 else f"{base} {chr(64+name_seen[base])}"   # 植被, 植被 B, ...
    return {"cat":int(c),"color":CATC[c],"name":nm,"n":len(live),"rep":rep(gids)}
# order clusters by dendrogram leaf order
order_l1=list(dict.fromkeys(lab1[i] for i in leaves))
seen1={}
tree=[]
for c1 in order_l1:
    g1=[gi for gi in leaves if lab1[gi]==c1]               # gene ids in this super-branch, leaf-ordered
    nd1=node(g1,seen1)
    if not nd1: continue
    order_l2=list(dict.fromkeys(lab2[i] for i in g1))
    seen2={}; kids=[]
    for c2 in order_l2:
        g2=[gi for gi in g1 if lab2[gi]==c2]
        nd2=node(g2,seen2)
        if not nd2: continue
        nd2["genes"]=[gi for gi in g2 if has(gi)]          # leaf gene ids, leaf-ordered
        kids.append(nd2)
    nd1["children"]=kids; tree.append(nd1)

man={"categories":[{"id":c,"name":CATN[c],"color":CATC[c]} for c in range(NCAT)],
     "n_genes":K,"n_active":int((peak>1e-6).sum()),"n_imgs":N,
     "genes":{str(g["id"]):g for g in genes if g["ex"]},    # data store keyed by gene id
     "tree":tree}                                           # nested hierarchy referencing gene ids
json.dump(man,open(OUT/"web"/"manifest.json","w"),ensure_ascii=False)
sz=sum(f.stat().st_size for f in EX.rglob("*.jpg"))/1e6
log(f"[done] {len(man['genes'])} genes w/ exemplars, {len(tree)} super-branches, "
    f"{sum(len(t['children']) for t in tree)} sub-branches, {len(list(EX.rglob('*.jpg')))} overlays, {sz:.0f}MB")
