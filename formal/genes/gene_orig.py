"""Save the paired ORIGINAL crop for every gene exemplar (g{k}/{r}_o.jpg),
using the exact same top-20 selection as gene_render.py so r matches 1:1."""
import os, numpy as np, time
from pathlib import Path
from PIL import Image
OUT=Path(os.environ.get("GENESDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global/genes"))
EX=OUT/"web"/"exem"; K=512; NEX=20
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
z=np.load(OUT/"sparse_acts.npz",allow_pickle=True)
idx=z["idx"].astype(np.int64); val=z["val"].astype(np.float32)
N=idx.shape[0]; th=[str(OUT/"thumbs"/f"{i}.jpg") for i in range(N)]
maxmat=np.zeros((N,K),np.float32)
for i in range(N): np.maximum.at(maxmat[i], idx[i].ravel(), val[i].ravel())
log("saving paired originals ...")
cache={}
n=0
for k in range(K):
    if maxmat[:,k].max()<1e-6: continue
    top=[int(i) for i in np.argsort(-maxmat[:,k])[:NEX] if maxmat[i,k]>1e-6]
    gd=EX/f"g{k}"
    if not gd.exists(): continue
    for r,i in enumerate(top):
        if i not in cache:
            cache[i]=Image.open(th[i]).convert("RGB").resize((128,128))
        cache[i].save(gd/f"{r}_o.jpg",quality=72); n+=1
    if k%50==0: log(f"  {k}/{K}")
log(f"[done] {n} original crops saved (_o.jpg)")
