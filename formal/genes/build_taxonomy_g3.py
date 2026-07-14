"""global3-NATIVE taxonomy (no global2). 5 parents: 植被/建筑立面/路面·道路/车辆/
天空·远景. First 4 from global3's own ward clusters (cap=30, 28 fine clusters
mapped by hand); 天空 identified data-driven (genes whose mean activation
concentrates in the top image rows = sky). Writes gene2cat/gene2child/taxonomy."""
import os, json, numpy as np, torch
from pathlib import Path
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
G3=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global3")
GEN=G3/"genes"; G=28
d=torch.load(G3/"sae_448_k512.pt",map_location="cpu"); A=d["state"]["dec.weight"].T.numpy()
An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
Z=linkage(An,method="ward",metric="euclidean"); root=to_tree(Z)
def split_to_cap(n,cap):
    cl=[n]
    while True:
        over=[c for c in cl if c.count>cap and not c.is_leaf()]
        if not over: break
        b=max(over,key=lambda c:c.count); cl.remove(b); cl+=[b.left,b.right]
    return cl
order=dendrogram(Z,no_plot=True)["leaves"]; pos={g:i for i,g in enumerate(order)}
P=split_to_cap(root,30); P.sort(key=lambda n:min(pos[g] for g in n.pre_order(lambda x:x.id)))
Pcl=[n.pre_order(lambda x:x.id) for n in P]
print(f"{len(Pcl)} fine clusters")

# --- data-driven SKY: mean activation concentrated in top rows ---
z=np.load(GEN/"sparse_acts.npz",allow_pickle=True); idx=z["idx"].astype(np.int64); val=z["val"].astype(np.float32)
N,Pn,T=idx.shape; meanacc=np.zeros((512,Pn)); Pcol=np.repeat(np.arange(Pn),T)
for c0 in range(0,N,500):
    ii=idx[c0:c0+500]; vv=val[c0:c0+500]; b=ii.shape[0]
    meanacc+=np.bincount((ii.reshape(-1)*Pn+np.tile(Pcol,b)),weights=vv.reshape(-1),minlength=512*Pn).reshape(512,Pn)
M=meanacc.reshape(512,G,G); tot=M.sum((1,2))+1e-9
topfrac=M[:,:G//4,:].sum((1,2))/tot          # fraction of activation in top 25% rows
sky=set(int(g) for g in np.argsort(-topfrac)[:10] if topfrac[g]>0.5)
print(f"SKY genes ({len(sky)}): {sorted(sky)}  topfrac range {topfrac[list(sky)].min():.2f}-{topfrac[list(sky)].max():.2f}")

# --- parents + cluster->child mapping (5 parents; hand-mapped from the cap=30 montage) ---
PARENTS=[("植被","#22e0a1"),("建筑立面","#f6c945"),("路面·道路","#8a9bb5"),("车辆","#ff9e64"),("天空·远景","#39d6ff")]
# P-index -> (parent_id, child_name)
CMAP={
 23:(0,"树丛·灌木"),24:(0,"沿墙绿篱"),25:(0,"树冠"),
 2:(1,"窗·竖构件"),11:(1,"窗·竖构件"),4:(1,"住宅·楼房"),19:(1,"住宅·楼房"),21:(1,"住宅·楼房"),
 15:(1,"沿街立面"),16:(1,"沿街立面"),22:(1,"沿街立面"),7:(1,"建筑·结构"),17:(1,"建筑·结构"),
 18:(1,"建筑·结构"),20:(1,"建筑·结构"),26:(1,"矮墙·巷弄"),27:(1,"矮墙·巷弄"),
 0:(2,"车行道"),5:(2,"车行道"),6:(2,"车行道"),9:(2,"车行道"),10:(2,"车行道"),12:(2,"车行道"),
 1:(2,"路口·人行道"),13:(2,"路口·人行道"),3:(2,"路肩·开阔地"),
 8:(3,"车辆"),14:(3,"车辆·店面"),
}
assert set(CMAP)==set(range(len(Pcl))), f"unmapped clusters: {set(range(len(Pcl)))-set(CMAP)}"
# child registry (name -> global id), parent 4 gets a sky child
childreg={}; g2p=np.full(512,-1,int); g2ch=np.full(512,-1,int)
def child_id(pid,name):
    k=(pid,name)
    if k not in childreg: childreg[k]=len(childreg)
    return childreg[k]
for pi,gids in enumerate(Pcl):
    pid,cn=CMAP[pi]; cid=child_id(pid,cn)
    for g in gids: g2p[g]=pid; g2ch[g]=cid
# sky override
scid=child_id(4,"天空·远景")
for g in sky: g2p[g]=4; g2ch[g]=scid
assert (g2p>=0).all()
np.save(G3/"gene2cat.npy",g2p); np.save(G3/"gene2child.npy",g2ch)
# taxonomy.json
children=[{"id":cid,"parent":pid,"name":nm,"n":int((g2ch==cid).sum())} for (pid,nm),cid in childreg.items()]
parents=[{"id":i,"name":n,"color":c,"n":int((g2p==i).sum()),
          "children":[{"id":ch["id"],"name":ch["name"],"n":ch["n"]} for ch in children if ch["parent"]==i]}
         for i,(n,c) in enumerate(PARENTS)]
json.dump({"parents":parents,"children":children},open(G3/"taxonomy.json","w"),ensure_ascii=False)
# categories_g3.json (5 parents) for the builders
json.dump([{"id":i,"name":n,"color":c} for i,(n,c) in enumerate(PARENTS)],
          open(Path(str(G3.parent))/"categories_g3.json","w"),ensure_ascii=False)
print("parents:",{p["name"]:p["n"] for p in parents})
print("children:",[(c["name"],c["n"]) for c in children])
