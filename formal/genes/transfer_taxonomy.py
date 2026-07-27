"""Transfer the human-built 6-parent/16-child taxonomy from global2 to a retrained
dict (global3) by nearest-neighbour atom matching — each new gene inherits the
parent/child of its most cosine-similar global2 gene. Avoids re-labeling."""
import os, json, numpy as np, torch
from pathlib import Path
REPO=Path("/global/scratch/users/cehou/urban-visual-gene/formal")
SRC=Path(os.environ.get("SRC_DICT",str(REPO/"formal_out_global2")))
DST=Path(os.environ.get("DST_DICT",str(REPO/"formal_out_global3")))
def atoms(d):
    a=torch.load(d/"sae_448_k512.pt",map_location="cpu")["state"]["dec.weight"].T.numpy()
    return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
A2,A3=atoms(SRC),atoms(DST)
g2cat=np.load(SRC/"gene2cat.npy"); g2child=np.load(SRC/"gene2child.npy")
sim=np.clip(A3@A2.T,-1,1); nn=sim.argmax(1); mx=sim.max(1)
g3cat=g2cat[nn].astype(int); g3child=g2child[nn].astype(int)
np.save(DST/"gene2cat.npy",g3cat); np.save(DST/"gene2child.npy",g3child)
# rebuild taxonomy.json (same parent/child NAMES from source, new counts)
srctax=json.load(open(SRC/"taxonomy.json"))
childname={c["id"]:c["name"] for c in srctax["children"]}
childparent={c["id"]:c["parent"] for c in srctax["children"]}
parents=[]
for p in srctax["parents"]:
    kids=[{"id":c["id"],"name":c["name"],"n":int((g3child==c["id"]).sum())} for c in p["children"]]
    parents.append({"id":p["id"],"name":p["name"],"color":p["color"],"n":int((g3cat==p["id"]).sum()),"children":kids})
json.dump({"parents":parents,"children":srctax["children"]},open(DST/"taxonomy.json","w"),ensure_ascii=False)
print(f"transferred taxonomy global2 -> {DST.name}")
print(f"  match cosine: min {mx.min():.2f}  median {np.median(mx):.2f}  <0.7: {(mx<0.7).sum()} genes")
print("  parent sizes:", {p["name"]:p["n"] for p in parents})
