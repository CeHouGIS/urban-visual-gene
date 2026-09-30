"""Fix the 2-level SEMANTIC taxonomy: 22 fine ward-clusters -> 6 parents / 16
children (manual semantic grouping). Writes gene2cat.npy(=parent), gene2child.npy,
taxonomy.json, and categories_new.json(=6 parents)."""
import os, json, numpy as np, torch
from pathlib import Path
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
G2=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2"))
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
d=torch.load(G2/"sae_448_k512.pt",map_location="cpu"); A=d["state"]["dec.weight"].T.numpy()
An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
Z=linkage(An,method="ward",metric="euclidean"); root=to_tree(Z)
def split_to_cap(node,cap):
    cl=[node]
    while True:
        over=[c for c in cl if c.count>cap and not c.is_leaf()]
        if not over: break
        big=max(over,key=lambda c:c.count); cl.remove(big); cl+=[big.left,big.right]
    return cl
order=dendrogram(Z,no_plot=True)["leaves"]; pos={g:i for i,g in enumerate(order)}
P=split_to_cap(root,45); P.sort(key=lambda n:min(pos[g] for g in n.pre_order(lambda x:x.id)))
Pcl=[n.pre_order(lambda x:x.id) for n in P]     # 22 fine clusters, gene ids
assert len(Pcl)==22, len(Pcl)

# parents (id,name,color) ; children (name, [fine-cluster indices])
PARENTS=[("植被","#22e0a1"),("建筑立面","#f6c945"),("路面·道路","#8a9bb5"),
         ("车辆","#ff9e64"),("天空·远景","#39d6ff"),("围墙·栅栏","#b388ff")]
CHILDREN=[  # (parent_id, child_name, [P indices])
 (0,"树冠",[0]),(0,"树丛·灌木",[1]),(0,"沿墙绿篱",[3]),
 (1,"住宅·楼房",[7,19]),(1,"商业·店面",[6,11]),(1,"窗·竖构件",[20]),(1,"矮墙·巷弄",[2]),(1,"建筑·结构",[21]),
 (2,"车行道",[9,10]),(2,"路口·人行道",[13,14,16]),(2,"道路灭点",[18]),(2,"路肩·开阔地",[12,15]),
 (3,"车辆",[8]),(3,"车辆·店面",[17]),
 (4,"天空·远景",[5]),
 (5,"围栏·门",[4]),
]
g2p=np.full(512,-1,int); g2ch=np.full(512,-1,int)
children_out=[]
for ch_id,(pid,cname,pis) in enumerate(CHILDREN):
    gids=sorted([g for pi in pis for g in Pcl[pi]])
    for g in gids: g2p[g]=pid; g2ch[g]=ch_id
    children_out.append({"id":ch_id,"parent":pid,"name":cname,"n":len(gids),"genes":gids})
assert (g2p>=0).all(), f"{(g2p<0).sum()} genes unassigned"
np.save(G2/"gene2cat.npy", g2p)          # site category = parent
np.save(G2/"gene2child.npy", g2ch)
# taxonomy.json (nested)
tax={"parents":[{"id":i,"name":n,"color":c,"n":int((g2p==i).sum()),
      "children":[{"id":ch["id"],"name":ch["name"],"n":ch["n"]} for ch in children_out if ch["parent"]==i]}
      for i,(n,c) in enumerate(PARENTS)],
     "children":children_out}
json.dump(tax,open(G2/"taxonomy.json","w"),ensure_ascii=False)
json.dump([{"id":i,"name":n,"color":c} for i,(n,c) in enumerate(PARENTS)],
          open(REPO/"formal"/"categories_new.json","w"),ensure_ascii=False)
print("parents:",[(p["name"],p["n"]) for p in tax["parents"]])
print("children:",[(c["name"],c["n"]) for c in children_out])
print("total genes:",int((g2p>=0).sum()))
