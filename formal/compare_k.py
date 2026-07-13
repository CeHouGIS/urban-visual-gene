"""Compare K=256/512/1024 dictionaries: metrics + granularity visualization.
Runs on CPU (login node) from formal_out artifacts."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
OUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out")
KS=[256,512,1024]; G=28

class SAE(nn.Module):
    def __init__(s,D,K,topk):
        super().__init__(); s.topk=topk
        s.b_pre=nn.Parameter(torch.zeros(D)); s.enc=nn.Linear(D,K); s.dec=nn.Linear(K,D,bias=False)
    def encode(s,z):
        pre=s.enc(z-s.b_pre); val,idx=pre.topk(s.topk,1)
        a=torch.zeros_like(pre); a.scatter_(1,idx,F.relu(val)); return a
    def forward(s,z): a=s.encode(z); return a,s.dec(a)+s.b_pre

def load(K):
    d=torch.load(OUT/f"sae_448_k{K}.pt",map_location="cpu")
    m=SAE(d["D"],K,d["topk"]); m.load_state_dict(d["state"]); m.eval(); return m

# ---- metrics on a dict-sample subset ----
Z=np.load(OUT/"dict_sample.f16.npy"); idx=np.random.default_rng(0).choice(len(Z),300000,replace=False)
Zt=torch.tensor(Z[idx].astype(np.float32)); print(f"metrics on {len(Zt):,} patches")
rows=[]
for K in KS:
    m=load(K)
    with torch.no_grad():
        A=[]; recon=0.0
        for st in range(0,len(Zt),20000):
            zb=Zt[st:st+20000]; a,zh=m(zb); A.append(a.cpu()); recon+=(1-F.cosine_similarity(zb,zh,1)).sum().item()
        A=torch.cat(A); recon/=len(Zt)
    used=(A>1e-6).any(0).sum().item(); dead=K-used
    usage=(A>1e-6).float().mean(0).numpy()          # per-atom activation frequency
    # granularity proxy: mean #unique dominant atoms per patch-block already fixed by topk;
    # use entropy of usage distribution (higher = more evenly used dictionary)
    p=usage/usage.sum(); ent=-(p[p>0]*np.log(p[p>0])).sum()
    rows.append((K,1-recon,dead,used,ent))
    print(f"K={K:4d}  recon_cos={1-recon:.4f}  dead={dead:3d}/{K}  active={used}  usage_entropy={ent:.2f}")

# ---- granularity visualization: same HK streets at 3 K ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
BG,FG,MUT="#070b14","#dbe4f0","#8a9bb5"
def colorize(gm,K):
    # deterministic color per gene id (hash to hsv)
    h=((gm.astype(np.int64)*2654435761)%360)/360.0
    import colorsys
    flat=h.ravel(); rgb=np.array([colorsys.hsv_to_rgb(x,0.65,0.95) for x in flat]).reshape(G,G,3)
    up=np.array(Image.fromarray((rgb*255).astype(np.uint8)).resize((224,224),Image.NEAREST))
    return up

st={K:np.load(OUT/f"streets_HongKong_k{K}.npz",allow_pickle=True) for K in KS}
tdir=OUT/"thumbs"/"HongKong"
pts=[5,40,120,200]   # sample points
fig,ax=plt.subplots(len(pts),4,figsize=(12,3*len(pts))); fig.patch.set_facecolor(BG)
for r,pi in enumerate(pts):
    try: th=np.array(Image.open(tdir/f"{pi}.jpg").resize((224,224)))
    except: th=np.zeros((224,224,3),np.uint8)
    ax[r,0].imshow(th); ax[r,0].set_title("original" if r==0 else "",color=FG,fontsize=11)
    for c,K in enumerate(KS):
        gm=st[K]["gmaps"][pi,0]                       # heading 0
        nu=len(np.unique(gm[gm>=0]))
        ax[r,c+1].imshow(colorize(gm,K))
        ax[r,c+1].set_title((f"K={K}" if r==0 else "")+f"  ({nu} genes)",color=FG,fontsize=10)
    for c in range(4): ax[r,c].axis("off")
fig.suptitle("Same HK street @448 — dominant-gene map at K=256 / 512 / 1024 (finer with larger K)",color=FG,fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(OUT/"compare_k_granularity.png",dpi=115,facecolor=BG); plt.close(fig)

# ---- metrics bar ----
fig,ax=plt.subplots(1,3,figsize=(13,3.6)); fig.patch.set_facecolor(BG)
Kx=[str(k) for k in KS]
for a,(vals,ttl) in zip(ax,[([r[1] for r in rows],"reconstruction cos ↑"),
                            ([r[2] for r in rows],"dead atoms ↓"),
                            ([r[4] for r in rows],"usage entropy (even use) ↑")]):
    a.bar(Kx,vals,color=["#39d6ff","#ffb454","#22e0a1"]); a.set_facecolor(BG); a.set_title(ttl,color=FG,fontsize=11)
    a.tick_params(colors=MUT); [s.set_color(MUT) for s in a.spines.values()]
    for i,v in enumerate(vals): a.text(i,v,f"{v:.3f}" if v<10 else f"{int(v)}",ha="center",va="bottom",color=FG,fontsize=9)
fig.suptitle("K sweep metrics (4-city shared dict, 3.2M patches @448, topk=32)",color=FG,fontsize=12)
fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig(OUT/"compare_k_metrics.png",dpi=115,facecolor=BG); plt.close(fig)
print("saved compare_k_granularity.png, compare_k_metrics.png")
