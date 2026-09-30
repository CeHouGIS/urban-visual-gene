"""Rebuild ONLY the nested `tree` in genes/web/manifest.json with a more BALANCED
hierarchy: ward linkage + 'split the largest cluster' cutting. Touches no images.
  --dry : just print size distributions for a few settings, don't write."""
import os, sys, json, numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
OUT=Path(os.environ.get("GENESDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global/genes"))
atoms=np.load(OUT/"atoms.npy")                      # (512,1024) decoder directions
A=atoms/(np.linalg.norm(atoms,axis=1,keepdims=True)+1e-9)
man=json.load(open(OUT/"web"/"manifest.json"))
CATN={c["id"]:c["name"] for c in man["categories"]}
CATC={c["id"]:c["color"] for c in man["categories"]}
G=man["genes"]                                      # {"86":{id,cat,peak,ex,...}}
cat=lambda gid: G[str(gid)]["cat"]

def build(method):
    Z=linkage(A, method=method, metric="euclidean")  # ward valid on unit vecs (euclid~cos)
    order=dendrogram(Z, no_plot=True)["leaves"]
    pos={g:i for i,g in enumerate(order)}
    root=to_tree(Z)
    return Z, root, pos

def split_to_k(node, k):
    """split into k clusters by repeatedly breaking the LARGEST splittable cluster"""
    cl=[node]
    while len(cl) < k:
        sp=[c for c in cl if not c.is_leaf()]
        if not sp: break
        big=max(sp, key=lambda c:c.count)
        cl.remove(big); cl += [big.left, big.right]
    return cl

def split_to_cap(node, cap):
    """split until every cluster has <= cap leaves (only breaks over-cap clusters)"""
    cl=[node]
    while True:
        over=[c for c in cl if c.count>cap and not c.is_leaf()]
        if not over: break
        big=max(over, key=lambda c:c.count)
        cl.remove(big); cl += [big.left, big.right]
    return cl

def leaves(node): return node.pre_order(lambda x: x.id)   # gene ids under node

def dom(gids):
    cs=[cat(g) for g in gids]; return max(set(cs), key=cs.count)

def name_dedup(base, seen):
    seen[base]=seen.get(base,0)+1
    return base if seen[base]==1 else f"{base} {chr(64+seen[base])}"

def make_tree(root, pos, SUPER_CAP, SUB_CAP):
    supers=split_to_cap(root, SUPER_CAP)
    supers.sort(key=lambda n: min(pos[g] for g in leaves(n)))
    seen1={}; tree=[]
    for sn in supers:
        sg=sorted(leaves(sn), key=lambda g:pos[g])
        subs=split_to_cap(sn, SUB_CAP); subs.sort(key=lambda n: min(pos[g] for g in leaves(n)))
        seen2={}; kids=[]
        for bn in subs:
            bg=sorted(leaves(bn), key=lambda g:pos[g])
            dc=dom(bg)
            kids.append({"cat":int(dc),"color":CATC[dc],"name":name_dedup(CATN[dc],seen2),
                         "n":len(bg),"rep":G[str(max(bg,key=lambda g:G[str(g)]['peak']))]["ex"][0],
                         "genes":bg})
        dc=dom(sg)
        tree.append({"cat":int(dc),"color":CATC[dc],"name":name_dedup(CATN[dc],seen1),
                     "n":len(sg),"rep":G[str(max(sg,key=lambda g:G[str(g)]['peak']))]["ex"][0],
                     "children":kids})
    return tree

if "--dry" in sys.argv:
    Z,root,pos=build("ward")
    for SC,BC in ((120,30),(110,28),(100,26),(90,24)):
        t=make_tree(root,pos,SC,BC)
        supsz=sorted([s['n'] for s in t],reverse=True)
        subsz=sorted([c['n'] for s in t for c in s['children']],reverse=True)
        print(f"ward SUPER_CAP={SC} SUB_CAP={BC}: {len(t)} supers {supsz}")
        print(f"     {sum(len(s['children']) for s in t)} subs; sub min/med/max = "
              f"{min(subsz)}/{int(np.median(subsz))}/{max(subsz)}  sizes={subsz}\n")
    sys.exit(0)

# ---- write ----
SC=int(sys.argv[sys.argv.index("--super")+1]) if "--super" in sys.argv else 110
BC=int(sys.argv[sys.argv.index("--sub")+1]) if "--sub" in sys.argv else 28
METHOD=sys.argv[sys.argv.index("--method")+1] if "--method" in sys.argv else "ward"
Z,root,pos=build(METHOD)
man["tree"]=make_tree(root,pos,SC,BC)
json.dump(man, open(OUT/"web"/"manifest.json","w"), ensure_ascii=False)
supsz=[s["n"] for s in man["tree"]]; subsz=[c["n"] for s in man["tree"] for c in s["children"]]
print(f"[written] method={METHOD} SUPER_CAP={SC} SUB_CAP={BC}")
print(f"  supers ({len(supsz)}): {sorted(supsz,reverse=True)}")
print(f"  subs   ({len(subsz)}): min {min(subsz)} / med {int(np.median(subsz))} / max {max(subsz)}")
