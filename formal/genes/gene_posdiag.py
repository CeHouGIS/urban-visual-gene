"""A: quantify how POSITIONAL each gene is (fires at fixed patch locations
regardless of content). Outputs a ranked list + a montage of mean activation maps."""
import os, json, numpy as np, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.cm as cm, matplotlib.pyplot as plt
OUT=Path(os.environ.get("GENESDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global/genes"))
DG=OUT/"diag"; DG.mkdir(exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
K=512; P=784; G=28
z=np.load(OUT/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"].astype(np.int64); val=z["val"].astype(np.float32)   # (N,784,32)
N=idx.shape[0]; log(f"{N} imgs")
man=json.load(open(OUT/"web"/"manifest.json")); CATN={c["id"]:c["name"] for c in man["categories"]}

# accumulate per-gene per-patch total activation (meanacc) and sum of squares (suma2)
meanacc=np.zeros((K,P)); suma2=np.zeros(K)
Pcol=np.repeat(np.arange(P),idx.shape[2])                        # patch id for each of the 32 slots
for c0 in range(0,N,500):
    ii=idx[c0:c0+500]; vv=val[c0:c0+500]; b=ii.shape[0]
    g=ii.reshape(-1); w=vv.reshape(-1); pc=np.tile(Pcol,b)
    meanacc+=np.bincount(g*P+pc, weights=w, minlength=K*P).reshape(K,P)
    suma2 +=np.bincount(g, weights=w*w, minlength=K)
log("accumulated; computing scores ...")

# posR2 = 1 - SS_res/SS_tot ; position-only predictor = per-patch mean map
summ =meanacc.sum(1)/N                                           # sum_p m[p]
summ2=(meanacc**2).sum(1)/N**2                                   # sum_p m[p]^2
SS_res=suma2 - N*summ2
SS_tot=suma2 - N*(summ**2)/P
posR2=np.where(SS_tot>1e-9, 1-SS_res/np.maximum(SS_tot,1e-9), 0.0)
posR2=np.clip(posR2,0,1)

M=meanacc.reshape(K,G,G); tot=M.sum((1,2))+1e-9
ring=np.zeros((G,G),bool); ring[0]=ring[-1]=True; ring[:,0]=ring[:,-1]=True
corner=np.zeros((G,G),bool)
for r in (slice(0,3),slice(G-3,G)):
    for c in (slice(0,3),slice(G-3,G)): corner[r,c]=True
border_frac=M[:,ring].sum(1)/tot                                 # area baseline .138
corner_frac=M[:,corner].sum(1)/tot                              # area baseline .046
rowE=M.sum(2); rowE=rowE/(rowE.sum(1,keepdims=True)+1e-9)
colE=M.sum(1); colE=colE/(colE.sum(1,keepdims=True)+1e-9)
row_conc=np.sort(rowE,1)[:,-3:].sum(1); col_conc=np.sort(colE,1)[:,-3:].sum(1)

active=suma2>np.percentile(suma2,5)                              # ignore near-dead genes
def gtype(k):
    if posR2[k]<0.35: return "semantic"
    if corner_frac[k]>0.18: return "corner"
    if border_frac[k]>0.30: return "edge"
    if row_conc[k]>0.55 and col_conc[k]<0.38: return "h-line"
    if col_conc[k]>0.55 and row_conc[k]<0.38: return "v-line"
    return "pos-other"
types=[gtype(k) for k in range(K)]

rows=[]
for k in range(K):
    rows.append(dict(id=k,cat=CATN[man["genes"].get(str(k),{}).get("cat",0)],
                     posR2=round(float(posR2[k]),3),border=round(float(border_frac[k]),3),
                     corner=round(float(corner_frac[k]),3),rowc=round(float(row_conc[k]),3),
                     colc=round(float(col_conc[k]),3),type=types[k]))
rows_pos=sorted([r for r in rows if r["type"]!="semantic" and active[r["id"]]],
                key=lambda r:-r["posR2"])
from collections import Counter
cnt=Counter(r["type"] for r in rows if active[r["id"]])
log("=== type counts (active genes) ==="); [log(f"  {t:9s} {n}") for t,n in cnt.most_common()]
for th in (0.3,0.4,0.5,0.6,0.7):
    log(f"  posR2 > {th}: {int((posR2[active]>th).sum())} genes")
log("=== top-25 positional genes ===")
for r in rows_pos[:25]:
    log(f"  #{r['id']:3d} posR2={r['posR2']:.2f} {r['type']:8s} border={r['border']:.2f} corner={r['corner']:.2f} rowc={r['rowc']:.2f} colc={r['colc']:.2f}  [{r['cat']}]")
json.dump({"n_positional":len(rows_pos),"counts":dict(cnt),"genes":rows},
          open(DG/"posdiag.json","w"),ensure_ascii=False)

# montage of mean activation maps for the top positional genes
top=rows_pos[:36]
ncol=6; nrow=(len(top)+ncol-1)//ncol
fig,axs=plt.subplots(nrow,ncol,figsize=(ncol*2.1,nrow*2.35)); fig.patch.set_facecolor("#0a1120")
for ax,r in zip(axs.ravel(),top):
    mp=M[r["id"]]; mp=(mp-mp.min())/(mp.max()-mp.min()+1e-9)
    ax.imshow(mp,cmap="jet"); ax.set_title(f"#{r['id']} {r['type']}\nR2={r['posR2']:.2f}",color="#dbe4f0",fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axs.ravel()[len(top):]: ax.axis("off")
fig.suptitle("Mean activation map (over 6000 imgs) — top positional genes",color="#fff",fontsize=11)
fig.tight_layout(); fig.savefig(DG/"posmaps.png",dpi=110,facecolor="#0a1120"); plt.close(fig)
log(f"[done] {len(rows_pos)} positional genes -> diag/posdiag.json ; montage -> diag/posmaps.png")
