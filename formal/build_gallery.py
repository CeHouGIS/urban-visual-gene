"""K=512 4-city feature gallery + hierarchy + category montage (for naming).
CPU only: argmax gene maps + heading-0 thumbnails from formal_out."""
import os, json
import numpy as np, torch
from pathlib import Path
from PIL import Image
OUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_clean")
GAL=OUT/"web"/"gallery"; GAL.mkdir(parents=True,exist_ok=True)
K=512; G=28; NCAT=10
CITIES=["HongKong","Singapore","Amsterdam","CapeTown"]
PALETTE=["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff","#f6c945","#9b8cff","#ff9e64","#73daca"]

# atoms + categories (same KMeans as explorer)
d=torch.load(OUT/f"sae_448_k{K}.pt",map_location="cpu")
atoms=d["state"]["dec.weight"].T.numpy()
os.environ["OMP_NUM_THREADS"]="1"; from sklearn.cluster import KMeans
g2c=KMeans(NCAT,n_init=10,random_state=0).fit_predict(atoms)

# load heading-0 gene maps + thumbnails across cities
maps=[]; thumbs=[]
for city in CITIES:
    z=np.load(OUT/f"streets_{city}_k{K}.npz",allow_pickle=True)
    gm=z["gmaps"]                                        # clean dict: artifact removed at source
    for pi in range(gm.shape[0]):
        tp=OUT/"thumbs"/city/f"{pi}.jpg"
        if tp.exists(): maps.append(gm[pi,0]); thumbs.append(str(tp))
maps=np.stack(maps); N=len(maps)
print(f"{N} heading-0 images",flush=True)

# feature prevalence stats
prev=np.zeros((N,K),np.float32)                       # fraction of patches where k is argmax
for i in range(N):
    m=maps[i][maps[i]>=0];
    if len(m)==0: continue
    u,c=np.unique(m,return_counts=True); prev[i,u]=c/ (G*G)
peak=prev.max(0); nimg=(prev>0.02).sum(0); meanp=prev.mean(0)
alive=peak>0.02
selective=alive & (meanp<0.03) & (nimg>=6)
score=peak*(prev.std(0)+1e-6); score[~selective]=-1
topf=np.argsort(-score)[:12]
print("top features:",topf.tolist(),flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BG,FG,MUT="#070b14","#dbe4f0","#8a9bb5"
def hi(th,gm,k):                                       # highlight patches where argmax==k
    mask=(gm==k).astype(np.float32)
    up=np.array(Image.fromarray((mask*255).astype(np.uint8)).resize((224,224),Image.NEAREST),np.float32)/255
    img=np.array(Image.open(th).convert("RGB").resize((224,224)),np.float32)
    hot=np.array([255,60,60],np.float32)
    a=(up*0.55)[...,None]
    return (img*(1-a)+hot*a).astype(np.uint8)

def strip(k,path):
    order=np.argsort(-prev[:,k])[:4]
    fig,ax=plt.subplots(1,4,figsize=(10.4,2.6))
    for c,i in enumerate(order):
        ax[c].imshow(hi(thumbs[i],maps[i],k)); ax[c].axis("off")
    fig.subplots_adjust(0,0,1,1,0.02,0); fig.savefig(path,dpi=100,bbox_inches="tight",pad_inches=0.02); plt.close(fig)

meta={"K":K,"features":[]}
for k in topf:
    fn=f"feat_{int(k)}.png"; strip(int(k),GAL/fn)
    meta["features"].append({"id":int(k),"file":fn,"cat":int(g2c[k]),
                             "peak":round(float(peak[k]),2),"n_imgs":int(nimg[k])})
json.dump(meta,open(GAL/"gallery_meta.json","w"),ensure_ascii=False,indent=2)

# category montage: for each cat, 3 representative features (by peak among cat members)
fig,axes=plt.subplots(NCAT,3,figsize=(9.5,3.1*NCAT)); fig.patch.set_facecolor(BG); axes=np.atleast_2d(axes)
cat_reps={}
for c in range(NCAT):
    members=[k for k in range(K) if g2c[k]==c and peak[k]>0.02]
    members=sorted(members,key=lambda k:-peak[k])[:3]
    cat_reps[c]=members
    for j in range(3):
        ax=axes[c,j]; ax.set_facecolor(BG); ax.axis("off")
        if j<len(members):
            k=members[j]; i=int(np.argmax(prev[:,k]))
            ax.imshow(hi(thumbs[i],maps[i],k)); ax.set_title(f"#{k}",color=PALETTE[c],fontsize=9)
    axes[c,0].set_ylabel(f"cat {c}",color=FG,fontsize=11,rotation=0,labelpad=28,va="center"); axes[c,0].axis("on")
    axes[c,0].set_xticks([]); axes[c,0].set_yticks([])
    for s in axes[c,0].spines.values(): s.set_visible(False)
fig.suptitle("K=512 categories — representative features (name each)",color=FG,fontsize=12)
fig.tight_layout(rect=[0,0,1,0.99]); fig.savefig(GAL/"category_montage.png",dpi=105,facecolor=BG); plt.close(fig)
print("cat reps:",{c:cat_reps[c] for c in range(NCAT)},flush=True)

# dendrogram of atoms colored by category (subset for readability: topf + extras)
from scipy.cluster.hierarchy import linkage,dendrogram,set_link_color_palette
from scipy.spatial.distance import squareform
sel=np.argsort(-peak)[:44]                             # 44 most prevalent features
Xs=atoms[sel]; sim=np.clip(Xs@Xs.T,-1,1); dist=1-sim; np.fill_diagonal(dist,0); dist=(dist+dist.T)/2
Z=linkage(squareform(dist,checks=False),method="average")
set_link_color_palette(PALETTE)
fig,ax=plt.subplots(figsize=(9,12)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
dendrogram(Z,orientation="right",labels=[f"#{int(k)}" for k in sel],ax=ax,color_threshold=Z[-6,2]-1e-9,above_threshold_color=MUT,leaf_font_size=9)
ax.set_title("K=512 visual-gene phylogeny (44 most-prevalent, atom cosine dist)",color=FG,fontsize=11)
ax.tick_params(colors=FG); [s.set_visible(False) for s in ax.spines.values()]
for l in ax.get_ymajorticklabels(): l.set_color(FG)
fig.tight_layout(); fig.savefig(GAL/"dendrogram_512.png",dpi=115,facecolor=BG); plt.close(fig)
print("saved gallery + category_montage + dendrogram_512",flush=True)
