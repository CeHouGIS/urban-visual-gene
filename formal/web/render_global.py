"""Render 12-city GLOBAL dictionary results for the website: feature gallery,
category montage, and a 12-city coverage map. Samples fresh images across all
12 cities, encodes with the global K=512 dict (artifact projection ON). GPU."""
import os, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from formal.gpu_run import Extractor, stratified_panos, imgpath, SAE, HEADINGS, CITY_DIR

OUT=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global"))
PROJ=os.environ.get("GENE_PROJ",str(OUT.parent/"formal_out"/"artifact_dirs_mean.npy"))
WEB=OUT/"web"; GAL=WEB/"gallery"; GAL.mkdir(parents=True,exist_ok=True)
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo","MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
PER=40; K=512; G=28; NCAT=10
PALETTE=["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff","#f6c945","#9b8cff","#ff9e64","#73daca"]
def log(*a): print(*a,flush=True)

ext=Extractor(448, proj_path=PROJ)
d=torch.load(OUT/"sae_448_k512.pt",map_location="cpu")
dev="cuda" if torch.cuda.is_available() else "cpu"
sae=SAE(d["D"],K,d["topk"]).to(dev); sae.load_state_dict(d["state"]); sae.eval()

# --- sample fresh images across 12 cities, encode heading-0 argmax gene maps + thumbs ---
maps=[]; thumbs=[]; citytag=[]; centroids={}
tdir=GAL/"thumbs"; tdir.mkdir(exist_ok=True)
gi=0
for city in CITIES:
    cands=stratified_panos(city, PER, seed=1)
    on=[p for p in cands if os.path.exists(imgpath(city,p,0))][:PER]
    lat=[]; lon=[]
    for pid in on:
        try: im=Image.open(imgpath(city,pid,0)).convert("RGB").resize((448,448),Image.BILINEAR)
        except: continue
        feats,_=ext.batch([im]); zt=feats[0].to(dev)         # (G*G, 1024) — one image
        with torch.no_grad(): a=sae.encode(zt).cpu().numpy() # (G*G, K)
        maps.append(a.argmax(1).reshape(G,G).astype(np.int16))
        tp=tdir/f"{gi}.jpg"; im.resize((256,256)).save(tp,quality=72); thumbs.append(str(tp)); citytag.append(city); gi+=1
    log(f"{city}: {sum(1 for c in citytag if c==city)} imgs encoded")
maps=np.stack(maps); N=len(maps); log(f"total {N} imgs")

# --- feature selection + strips ---
prev=np.zeros((N,K),np.float32)
for i in range(N):
    m=maps[i][maps[i]>=0]
    if len(m): u,c=np.unique(m,return_counts=True); prev[i,u]=c/(G*G)
peak=prev.max(0); nimg=(prev>0.02).sum(0)
good=(peak>0.05)&(prev.mean(0)<0.04)&(nimg>=5)
score=peak*(prev.std(0)+1e-6); score[~good]=-1
topf=np.argsort(-score)[:12]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BG,FG,MUT="#070b14","#dbe4f0","#8a9bb5"
def hi(th,gm,k):
    mask=(gm==k).astype(np.float32)
    up=np.array(Image.fromarray((mask*255).astype(np.uint8)).resize((224,224),Image.NEAREST),np.float32)/255
    img=np.array(Image.open(th).convert("RGB").resize((224,224)),np.float32)
    a=(up*0.55)[...,None]; return (img*(1-a)+np.array([255,60,60],np.float32)*a).astype(np.uint8)
def strip(k,path):
    order=np.argsort(-prev[:,k])[:4]
    fig,ax=plt.subplots(1,4,figsize=(10.4,2.6))
    for c,i in enumerate(order): ax[c].imshow(hi(thumbs[i],maps[i],k)); ax[c].set_title(citytag[i],color="k",fontsize=8); ax[c].axis("off")
    fig.subplots_adjust(0,0,1,0.92,0.02,0); fig.savefig(path,dpi=100,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
meta={"features":[]}
if (OUT/"gene2cat.npy").exists():
    g2c=np.load(OUT/"gene2cat.npy"); log("loaded gene2cat.npy")
else:
    from sklearn.cluster import KMeans
    g2c=KMeans(NCAT,n_init=10,random_state=0).fit_predict(d["state"]["dec.weight"].T.numpy())
for k in topf:
    fn=f"gfeat_{int(k)}.png"; strip(int(k),GAL/fn)
    meta["features"].append({"id":int(k),"file":fn,"cat":int(g2c[k])})
json.dump(meta,open(GAL/"global_gallery.json","w"))

# category montage
fig,axes=plt.subplots(NCAT,3,figsize=(9.5,3.0*NCAT)); fig.patch.set_facecolor(BG); axes=np.atleast_2d(axes)
for c in range(NCAT):
    mem=[k for k in range(K) if g2c[k]==c and peak[k]>0.02]; mem=sorted(mem,key=lambda k:-peak[k])[:3]
    for j in range(3):
        ax=axes[c,j]; ax.set_facecolor(BG); ax.axis("off")
        if j<len(mem):
            k=mem[j]; i=int(np.argmax(prev[:,k])); ax.imshow(hi(thumbs[i],maps[i],k)); ax.set_title(f"#{k}",color=PALETTE[c],fontsize=9)
    axes[c,0].set_ylabel(f"cat {c}",color=FG,fontsize=11,rotation=0,labelpad=26,va="center"); axes[c,0].axis("on")
    axes[c,0].set_xticks([]); axes[c,0].set_yticks([])
    for s in axes[c,0].spines.values(): s.set_visible(False)
fig.suptitle("Global K=512 categories (12 cities) — representative features",color=FG,fontsize=12)
fig.tight_layout(rect=[0,0,1,0.99]); fig.savefig(GAL/"global_category_montage.png",dpi=105,facecolor=BG); plt.close(fig)
log("saved global gallery + montage")
