"""Gene CO-EXPRESSION analysis: which visual genes fire TOGETHER in the same
patch (combinatorial expression, like gene modules). Complements the hierarchy
(similar genes) with co-firing (complementary genes that combine). CPU.
  -> coexpr.json (pairs+modules) + coexpr_network.png"""
import os, json, numpy as np
from pathlib import Path
from scipy import sparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G2=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2"))
GEN=G2/"genes"; DG=GEN/"diag"; DG.mkdir(exist_ok=True)
def log(*a): import time; print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
K=512
man=json.load(open(GEN/"web"/"manifest.json"))
CATN={c["id"]:c["name"] for c in man["categories"]}; CATC={c["id"]:c["color"] for c in man["categories"]}
g2c=np.load(G2/"gene2cat.npy")
z=np.load(GEN/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"].astype(np.int32); val=z["val"].astype(np.float32)   # (N,784,32)
N,P,T=idx.shape; Npatch=N*P; log(f"{N} imgs, {Npatch:,} patches")

# --- active mask: a gene is "expressed" at a patch if it contributes non-trivially ---
THRESH=0.15
pmax=val.max(2,keepdims=True)
active=(val>=THRESH)&(val>=0.25*pmax)
log(f"THRESH={THRESH}: avg active genes/patch = {active.sum()/Npatch:.2f}")

# --- co-occurrence M (KxK), marginal f (K), chunked ---
M=np.zeros((K,K)); CH=300
for c0 in range(0,N,CH):
    ii=idx[c0:c0+CH]; aa=active[c0:c0+CH]; np_=ii.shape[0]*P
    gi=ii.reshape(-1,T); am=aa.reshape(-1,T)
    rows=np.repeat(np.arange(gi.shape[0]),T)[am.ravel()]
    cols=gi[am]
    A=sparse.csr_matrix((np.ones(len(cols),np.float32),(rows,cols)),shape=(gi.shape[0],K))
    M+=np.asarray((A.T@A).todense())
f=np.diag(M).copy()                                   # patches where gene present
log(f"marginals: median {int(np.median(f))}, max {int(f.max())}")

# --- lift / PMI with support filter ---
fi=f[:,None]; fj=f[None,:]
lift=np.where((fi>0)&(fj>0), M*Npatch/(fi*fj+1e-9), 0.0)
np.fill_diagonal(lift,0); np.fill_diagonal(M,0)
MINSUP=max(200,int(0.0005*Npatch))                    # co-occur in >= this many patches
strong=(M>=MINSUP)&(lift>=2.0)
log(f"MINSUP={MINSUP}: {int(strong.sum()//2)} strong co-expressing pairs")

# --- top pairs (by lift*log-support), prefer cross-category = true combinations ---
pairs=[]
ii,jj=np.where(np.triu(strong,1))
for i,j in zip(ii,jj):
    pairs.append((int(i),int(j),float(lift[i,j]),int(M[i,j]),int(g2c[i]),int(g2c[j])))
pairs.sort(key=lambda p:-(p[2]*np.log(p[3])))
log("=== top 25 co-expressing gene pairs (lift, support) ===")
for i,j,lf,su,ci,cj in pairs[:25]:
    x="  ✱cross" if ci!=cj else ""
    log(f"  #{i:3d}+#{j:3d}  lift={lf:5.1f} n={su:6d}  [{CATN[ci]} + {CATN[cj]}]{x}")

# --- modules: community detection on the STRONG (lift>=EDGE_LIFT) subgraph ---
EDGE_LIFT=float(os.environ.get("EDGE_LIFT","8"))
RES=float(os.environ.get("RES","1.4"))
sp=[(i,j,lf,su,ci,cj) for i,j,lf,su,ci,cj in pairs if lf>=EDGE_LIFT]
log(f"module graph: {len(sp)} edges at lift>={EDGE_LIFT}")
import networkx as nx
Gr=nx.Graph()
for i,j,lf,su,ci,cj in sp: Gr.add_edge(i,j,weight=np.log(lf))
try:
    comms=nx.community.louvain_communities(Gr,weight="weight",resolution=RES,seed=1)
    log(f"louvain (res={RES}) communities: {len(comms)}")
except Exception as e:
    log(f"louvain unavailable ({e}); greedy_modularity")
    comms=list(nx.community.greedy_modularity_communities(Gr,weight="weight"))

modules=[]
for m in comms:
    m=[int(g) for g in m if f[g]>=MINSUP]
    if len(m)<3: continue
    sub=lift[np.ix_(m,m)]; coh=sub[np.triu_indices(len(m),1)].mean()
    cats=[int(g2c[g]) for g in m]; from collections import Counter
    catmix=Counter(cats)
    modules.append({"genes":sorted(m,key=lambda g:-f[g]),"n":len(m),"coh":round(float(coh),2),
                    "cats":sorted(catmix.items(),key=lambda x:-x[1]),
                    "cross":len(set(cats))})
modules.sort(key=lambda x:-(x["coh"]*np.sqrt(x["n"])))
log(f"=== {len(modules)} co-expression modules (lift>={EDGE_LIFT}) ===")
for mi,md in enumerate(modules[:20]):
    cs="+".join(f"{CATN[c]}×{n}" for c,n in md["cats"][:5])
    log(f"  M{mi}: {md['n']} genes, coh(lift)={md['coh']}, {md['cross']} cats [{cs}]")
    log(f"       genes {md['genes'][:14]}")

json.dump({"pairs":[{"i":i,"j":j,"lift":round(lf,2),"n":su,"ci":ci,"cj":cj} for i,j,lf,su,ci,cj in pairs[:400]],
           "modules":modules,"minsup":MINSUP},open(DG/"coexpr.json","w"),ensure_ascii=False)

# --- network figure: top-frequent genes, strong edges, colored by category ---
try:
    import networkx as nx
    top=np.argsort(-f)[:200]; ts=set(top.tolist())
    Gp=nx.Graph()
    for g in top: Gp.add_node(int(g))
    for i,j,lf,su,ci,cj in sp:
        if i in ts and j in ts: Gp.add_edge(i,j,weight=np.log(lf))
    Gp.remove_nodes_from([n for n in list(Gp.nodes) if Gp.degree(n)==0])
    pos=nx.spring_layout(Gp,weight="weight",k=0.35,seed=1,iterations=120)
    fig,ax=plt.subplots(figsize=(14,11)); fig.patch.set_facecolor("#070b14"); ax.set_facecolor("#070b14")
    ews=[Gp[u][v]["weight"] for u,v in Gp.edges]
    nx.draw_networkx_edges(Gp,pos,alpha=0.10,width=[0.3+0.5*w for w in ews],edge_color="#5a6b8a",ax=ax)
    nx.draw_networkx_nodes(Gp,pos,node_size=[30+0.02*f[n] for n in Gp.nodes],
        node_color=[CATC[int(g2c[n])] for n in Gp.nodes],linewidths=0,ax=ax)
    ax.set_title("Visual-gene co-expression network (nodes=genes by category, edges=co-firing)",color="#dbe4f0")
    ax.axis("off"); fig.tight_layout(); fig.savefig(DG/"coexpr_network.png",dpi=110,facecolor="#070b14"); plt.close(fig)
    log("saved coexpr_network.png")
except Exception as e: log(f"network fig skipped: {e}")
log("[done] -> diag/coexpr.json + coexpr_network.png")
