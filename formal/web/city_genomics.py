"""Cross-city 'genomics': treat each city as a 512-gene frequency profile (fraction
of its patches whose dominant/argmax gene is k), then compute
  - city phylogeny: Jensen-Shannon distance between profiles -> dendrogram (PNG)
  - pangenome: per-gene cross-city prevalence -> core / accessory / unique spectrum
  - marker genes: per-city enrichment (lift vs global mean) -> signature genes
  - visual diversity: Shannon entropy / effective #genes / richness per city
CPU, reads genes/sparse_acts.npz (balanced 12 cities x1200 imgs). -> citygenome/.
Gene thumbnails are referenced from the already-deployed genes/exem/g{k}/0.jpg."""
import os, json, numpy as np
from pathlib import Path
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
DICT=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global3"))
GEN=DICT/"genes"; OUT=DICT/"web"/"citygenome"; OUT.mkdir(parents=True,exist_ok=True)
def log(*a): import time;print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
K=512
ZH={"HongKong":"香港","Singapore":"新加坡","Amsterdam":"阿姆斯特丹","CapeTown":"开普敦","Paris":"巴黎",
    "SaoPaulo":"圣保罗","MexicoCity":"墨西哥城","Sydney":"悉尼","Jakarta":"雅加达","Dhaka":"达卡","NewDelhi":"新德里","Manila":"马尼拉"}

# ---- per-city gene frequency profile (argmax gene per patch) ----
z=np.load(GEN/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"]; city=np.array([str(c) for c in z["city"]])
top=idx[:,:,0].astype(np.int64)                        # (N,784) dominant gene per patch (topk col0 = max)
CITIES=sorted(set(city), key=lambda c:list(z["city"]).index(np.str_(c)))
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo","MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
P=np.zeros((len(CITIES),K))
for ci,c in enumerate(CITIES):
    g=top[city==c].ravel(); cnt=np.bincount(g,minlength=K).astype(float); P[ci]=cnt/cnt.sum()
Npatch=top.size; log(f"{top.shape[0]} imgs, {Npatch:,} patches, profile {P.shape}")

# ---- taxonomy names for genes ----
g2c=np.load(DICT/"gene2cat.npy"); g2ch=np.load(DICT/"gene2child.npy")
tax=json.load(open(DICT/"taxonomy.json")); catn={p["id"]:p["name"] for p in tax["parents"]}
catc={p["id"]:p["color"] for p in tax["parents"]}; childn={c["id"]:c["name"] for c in tax["children"]}
man=json.load(open(GEN/"web"/"manifest.json")); has_ex=set(int(g) for g in man["genes"])   # genes with exemplars
def gene_info(k):
    k=int(k); return {"gene":k,"name":childn.get(int(g2ch[k]),catn[int(g2c[k])]),
                      "cat":catn[int(g2c[k])],"color":catc[int(g2c[k])],
                      "thumb":(f"genes/exem/g{k}/0.jpg" if k in has_ex else "")}

# ---- city phylogeny: Jensen-Shannon distance ----
def kl(a,b): a=a+1e-12; b=b+1e-12; return float(np.sum(a*np.log2(a/b)))
def jsd(a,b): m=0.5*(a+b); return 0.5*kl(a,m)+0.5*kl(b,m)          # 0..1 (bits)
n=len(CITIES); D=np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n): D[i,j]=D[j,i]=np.sqrt(max(jsd(P[i],P[j]),0))   # JS distance metric
Z=linkage(squareform(D,checks=False),method="average")
leaves=dendrogram(Z,no_plot=True)["leaves"]
# dendrogram PNG (english labels; dark)
BG,FG,AC="#0b1322","#dbe4f0","#39d6ff"
fig,ax=plt.subplots(figsize=(9.6,3.4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
dendrogram(Z,labels=CITIES,ax=ax,color_threshold=0.5*D.max(),above_threshold_color="#5a6b86")
ax.set_ylabel("Jensen–Shannon distance",color=FG,fontsize=10)
ax.tick_params(colors=FG); [t.set_color(FG) for t in ax.get_xticklabels()]
for s in ax.spines.values(): s.set_color("#22304d")
plt.setp(ax.get_xticklabels(),rotation=35,ha="right",fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"city_dendro.png",dpi=120,facecolor=BG); plt.close(fig)
log("dendrogram saved")
# similarity matrix in leaf order (1 - Dnorm)
Dn=D/ (D.max()+1e-9); sim=(1-Dn)
sim_ord=sim[np.ix_(leaves,leaves)]
labels_ord=[CITIES[i] for i in leaves]

# ---- diversity: entropy / effective #genes / richness ----
div=[]
for ci,c in enumerate(CITIES):
    p=P[ci]; nz=p[p>0]; H=float(-(nz*np.log2(nz)).sum())
    div.append({"city":c,"zh":ZH[c],"H":round(H,2),"eff":int(round(2**H)),
                "richness":int((p>=1e-4).sum())})

# ---- pangenome: cross-city prevalence ----
THRESH=5e-4                                            # gene "present" in a city if >=0.05% of its patches
present=P>=THRESH; prevalence=present.sum(0)           # per gene, in how many cities (0..12)
used=prevalence>=1
spectrum=[int((prevalence==k).sum()) for k in range(1,n+1)]   # #genes present in exactly k cities
n_core=int((prevalence==n).sum()); n_unique=int(((prevalence>=1)&(prevalence<=2)).sum())
n_acc=int(((prevalence>2)&(prevalence<n)).sum())
# core examples: present in all, strongest by min-across-cities share
core_ids=[k for k in range(K) if prevalence[k]==n]
core_ids=sorted(core_ids,key=lambda k:-P[:,k].min())[:8]
core=[{**gene_info(k),"minshare":round(float(P[:,k].min()),4),"prev":int(prevalence[k])} for k in core_ids]
# unique/rare examples: present in 1-2 cities, strongest enrichment
meanp=P.mean(0)+1e-12
rare_ids=[k for k in range(K) if 1<=prevalence[k]<=2 and k in has_ex]
def top_city_for(k): ci=int(np.argmax(P[:,k])); return ci
rare_ids=sorted(rare_ids,key=lambda k:-(P[:,k].max()))[:8]
unique=[{**gene_info(k),"city":CITIES[top_city_for(k)],"zh":ZH[CITIES[top_city_for(k)]],
         "prev":int(prevalence[k]),"share":round(float(P[:,k].max()),4),
         "lift":round(float(P[:,k].max()/meanp[k]),1)} for k in rare_ids]

# ---- marker genes per city: enrichment lift vs global mean ----
lift=P/meanp
markers=[]
for ci,c in enumerate(CITIES):
    cand=[k for k in range(K) if P[ci,k]>=8e-4 and k in has_ex]     # enough support + has thumb
    cand.sort(key=lambda k:-lift[ci,k])
    gs=[{**gene_info(k),"lift":round(float(lift[ci,k]),1),"share":round(float(P[ci,k]),4)} for k in cand[:5]]
    markers.append({"city":c,"zh":ZH[c],"genes":gs})

data={"stats":{"n_imgs":int(top.shape[0]),"n_patches":int(Npatch),"n_cities":n,
               "n_used_genes":int(used.sum()),"threshold":THRESH},
      "leaf_order":labels_ord,"sim":[[round(float(x),3) for x in row] for row in sim_ord],
      "sim_zh":[ZH[c] for c in labels_ord],
      "diversity":sorted(div,key=lambda d:-d["H"]),
      "pangenome":{"threshold":THRESH,"spectrum":spectrum,"n_core":n_core,"n_accessory":n_acc,
                   "n_unique":n_unique,"n_used":int(used.sum()),"core":core,"unique":unique},
      "markers":markers,"categories":tax["parents"]}
json.dump(data,open(OUT/"data.json","w"),ensure_ascii=False)
log(f"[done] core={n_core} accessory={n_acc} unique(<=2)={n_unique} used={int(used.sum())} -> citygenome/data.json")
log("diversity(H bits): "+", ".join(f"{d['city']}={d['H']}" for d in data["diversity"]))
log("phylo order: "+" > ".join(labels_ord))
