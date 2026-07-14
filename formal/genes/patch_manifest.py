"""Patch a dict's genes manifest with the (transferred) taxonomy: set per-gene
parent cat + child, embed the 6-parent taxonomy, and tag positional genes from
posdiag.json. Run AFTER gene_render + gene_posdiag, BEFORE gene_tree."""
import os, json, numpy as np
from pathlib import Path
REPO=Path("/global/scratch/users/cehou/urban-visual-gene/formal")
DICT=Path(os.environ.get("DICTDIR",str(REPO/"formal_out_global3")))
cj=json.load(open(REPO/"categories_new.json"))                 # 6 parents (names/colors)
tax=json.load(open(DICT/"taxonomy.json"))
childname={c["id"]:c["name"] for c in tax["children"]}
g2c=np.load(DICT/"gene2cat.npy"); g2ch=np.load(DICT/"gene2child.npy")
man=json.load(open(DICT/"genes"/"web"/"manifest.json"))
man["categories"]=[{"id":c["id"],"name":c["name"],"color":c["color"]} for c in cj]
man["taxonomy"]=tax["parents"]
for gid,g in man["genes"].items():
    k=int(gid); g["cat"]=int(g2c[k]); g["child"]=childname[int(g2ch[k])]
# positional tags from posdiag (if present)
pd=DICT/"genes"/"diag"/"posdiag.json"
if pd.exists():
    d=json.load(open(pd)); pos={r["id"]:r for r in d["genes"] if r["type"]!="semantic" and str(r["id"]) in man["genes"]}
    man["positional"]=sorted(pos.keys())
    for gid,g in man["genes"].items():
        if int(gid) in pos: g["pt"]=pos[int(gid)]["type"]; g["pr2"]=pos[int(gid)]["posR2"]
    print(f"tagged {len(man['positional'])} positional genes")
json.dump(man,open(DICT/"genes"/"web"/"manifest.json","w"),ensure_ascii=False)
print(f"patched manifest: {len(man['categories'])} parent categories + per-gene child + taxonomy")
