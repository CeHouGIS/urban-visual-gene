"""Build web assets for the co-expression page: gene co-firing network (with a
precomputed layout), Louvain modules with on-scene combination exemplars, and
top cross-category combination pairs. CPU. -> formal_out_global2/genes/web/coexpr/"""
import os, json, numpy as np
from pathlib import Path
from scipy import sparse
import networkx as nx
from PIL import Image
import matplotlib.cm as cm
G2=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2")
GEN=G2/"genes"; OUT=GEN/"web"/"coexpr"; OUT.mkdir(parents=True,exist_ok=True)
def log(*a): import time;print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
K=512
man=json.load(open(GEN/"web"/"manifest.json")); CATN={c["id"]:c["name"] for c in man["categories"]}
CATC={c["id"]:c["color"] for c in man["categories"]}; genesM=man["genes"]
g2c=np.load(G2/"gene2cat.npy")
z=np.load(GEN/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"].astype(np.int32); val=z["val"].astype(np.float32); N,P,T=idx.shape; Npatch=N*P
th=[str(GEN/"thumbs"/f"{i}.jpg") for i in range(N)]
log(f"{N} imgs {Npatch:,} patches")

# ---- co-occurrence + lift ----
pmax=val.max(2,keepdims=True); active=(val>=0.15)&(val>=0.25*pmax)
M=np.zeros((K,K)); CH=300
for c0 in range(0,N,CH):
    ii=idx[c0:c0+CH]; aa=active[c0:c0+CH]
    gi=ii.reshape(-1,T); am=aa.reshape(-1,T)
    rows=np.repeat(np.arange(gi.shape[0]),T)[am.ravel()]; cols=gi[am]
    A=sparse.csr_matrix((np.ones(len(cols),np.float32),(rows,cols)),shape=(gi.shape[0],K))
    M+=np.asarray((A.T@A).todense())
f=np.diag(M).copy(); fi=f[:,None]; fj=f[None,:]
lift=np.where((fi>0)&(fj>0),M*Npatch/(fi*fj+1e-9),0.0); np.fill_diagonal(lift,0); np.fill_diagonal(M,0)
MINSUP=max(200,int(0.0005*Npatch))
pairs=[]
ii,jj=np.where(np.triu((M>=MINSUP)&(lift>=2.0),1))
for i,j in zip(ii,jj): pairs.append((int(i),int(j),float(lift[i,j]),int(M[i,j])))
pairs.sort(key=lambda p:-(p[2]*np.log(p[3])))
log(f"{len(pairs)} strong pairs")

# ---- modules (Louvain on lift>=8) ----
EDGE=8.0
Gr=nx.Graph()
for i,j,lf,su in pairs:
    if lf>=EDGE: Gr.add_edge(i,j,weight=np.log(lf))
comms=nx.community.louvain_communities(Gr,weight="weight",resolution=1.4,seed=1)
mods=[]
for m in comms:
    m=[int(g) for g in m if f[g]>=MINSUP]
    if len(m)<3: continue
    sub=lift[np.ix_(m,m)]; coh=float(sub[np.triu_indices(len(m),1)].mean())
    from collections import Counter
    cats=Counter(int(g2c[g]) for g in m)
    mods.append({"genes":sorted(m,key=lambda g:-f[g]),"coh":round(coh,2),
                 "cats":sorted(cats.items(),key=lambda x:-x[1])})
mods.sort(key=lambda x:-(x["coh"]*np.sqrt(len(x["genes"]))))
NAMES=["车辆·建成","树冠·密林","立面·门窗","绿化·树丛","街道走廊","店面·招牌","架空线·电杆",
       "墙面·立面","路面·立面","绿篱·沿墙","立面·结构","道路·沿街","路口·路面","路面·地物"]
MPAL=["#ff5d5d","#22e0a1","#f6c945","#7ee787","#4dd0e1","#ff9e64","#b388ff","#73daca",
      "#ff7ac6","#5ad1ff","#9b8cff","#8a9bb5","#ffd24d","#c2b280"]
gene2mod={}
for mi,md in enumerate(mods):
    md["id"]=mi; md["name"]=NAMES[mi] if mi<len(NAMES) else CATN[md["cats"][0][0]]
    md["color"]=MPAL[mi%len(MPAL)]
    for g in md["genes"]: gene2mod[g]=mi
log(f"{len(mods)} modules")

# ---- module on-scene combination exemplars ----
def coact(members):
    mask=np.isin(idx,np.array(members)); pp=(val*mask).sum(2); return pp, pp.sum(1)
def overlay(tp,m28,SZ):
    m=m28.astype(np.float32); m=(m-m.min())/(m.max()-m.min()+1e-8)
    up=np.array(Image.fromarray((m*255).astype(np.uint8)).resize((SZ,SZ),Image.BILINEAR),np.float32)/255
    heat=(cm.jet(up)[...,:3]*255).astype(np.float32)
    img=np.array(Image.open(tp).convert("RGB").resize((SZ,SZ)),np.float32)
    a=(0.30+0.55*up)[...,None]; return (img*(1-a)+heat*a).astype(np.uint8)
NEX=8; SZ=200
for md in mods:
    pp,score=coact(md["genes"]); top=np.argsort(-score)[:NEX]
    d=OUT/f"mod{md['id']}"; d.mkdir(exist_ok=True); ex=[]
    for r,i in enumerate(top):
        Image.fromarray(overlay(th[int(i)],pp[int(i)].reshape(28,28),SZ)).save(d/f"{r}.jpg",quality=76)
        ex.append(f"coexpr/mod{md['id']}/{r}.jpg")
    md["ex"]=ex
log("module exemplars rendered")

# ---- network layout (lift>=8 graph, spring) ----
pos=nx.spring_layout(Gr,weight="weight",k=0.5,seed=1,iterations=200)
xs=np.array([pos[n][0] for n in pos]); ys=np.array([pos[n][1] for n in pos])
def norm(a): a=(a-a.min())/(a.max()-a.min()+1e-9); return 40+a*920
X={n:float(v) for n,v in zip(pos, norm(xs))}; Y={n:float(v) for n,v in zip(pos, norm(ys))}
nodes=[{"id":int(n),"x":round(X[n],1),"y":round(Y[n],1),"m":gene2mod.get(int(n),-1),
        "cat":int(g2c[n]),"f":int(f[n])} for n in Gr.nodes]
edges=[[int(i),int(j),round(float(np.log(lf)),2)] for i,j,lf,su in pairs if lf>=EDGE
       and i in pos and j in pos]

# ---- gene metadata (thumb = its top overlay) ----
gmeta={}
for n in set(list(Gr.nodes)+[g for md in mods for g in md["genes"]]):
    g=genesM.get(str(n))
    gmeta[int(n)]={"cat":int(g2c[n]),"color":CATC[int(g2c[n])],"f":int(f[n]),
                   "th":(g["ex"][0] if g else "")}

# ---- top cross-category combination pairs ----
xpairs=[{"i":i,"j":j,"lift":round(lf,1),"n":su,"ci":int(g2c[i]),"cj":int(g2c[j])}
        for i,j,lf,su in pairs if g2c[i]!=g2c[j]][:40]

json.dump({"categories":man["categories"],
           "modules":[{"id":m["id"],"name":m["name"],"color":m["color"],"coh":m["coh"],
                       "genes":m["genes"],"cats":m["cats"],"ex":m["ex"]} for m in mods],
           "nodes":nodes,"edges":edges,"genes":gmeta,"pairs":xpairs,
           "stats":{"avg_active":round(float(active.sum()/Npatch),1),"n_pairs":len(pairs),
                    "n_modules":len(mods),"max_lift":round(float(max(p[2] for p in pairs)),0)}},
          open(OUT/"data.json","w"),ensure_ascii=False)
log(f"[done] {len(nodes)} nodes, {len(edges)} edges, {len(mods)} modules -> coexpr/data.json")
