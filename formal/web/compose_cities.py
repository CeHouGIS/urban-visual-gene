"""All-12-city composition & classification data for the redesigned site.
Per city: road-weighted spatial sample of panos, encode ALL 4 headings
[0,90,180,270] of each pano with the global K=512 dict and pool their patches ->
per-pano 360° CATEGORY composition (配比) + lat/lon + a 2x2 montage thumbnail of
the four views. So a point's composition reflects the full surroundings, not just
the forward image. GPU.
Output: web2/panos.parquet (city,pano_id,lat,lon,cat0..catN) + web2/thumbs/<city>/.
"""
import os, json, sqlite3, collections, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from formal.gpu_run import Extractor, SAE, imgpath, CITY_DIR, meta_db, HEADINGS

OUT=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global"))
PROJ=os.environ.get("GENE_PROJ",str(OUT.parent/"formal_out"/"artifact_dirs_mean.npy"))
W2=OUT/"web2"; TH=W2/"thumbs"; TH.mkdir(parents=True,exist_ok=True)
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo","MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
PER=250; K=512; G=28; NCAT=10; GRID=3
def log(*a): print(*a,flush=True)

ext=Extractor(448, proj_path=PROJ)
d=torch.load(OUT/"sae_448_k512.pt",map_location="cpu")
dev="cuda" if torch.cuda.is_available() else "cpu"
sae=SAE(d["D"],K,d["topk"]).to(dev); sae.load_state_dict(d["state"]); sae.eval()
os.environ["OMP_NUM_THREADS"]="8"
if (OUT/"gene2cat.npy").exists():
    g2c=np.load(OUT/"gene2cat.npy"); log("loaded gene2cat.npy")
else:
    from sklearn.cluster import KMeans
    g2c=KMeans(NCAT,n_init=10,random_state=0).fit_predict(d["state"]["dec.weight"].T.numpy())
    np.save(W2/"gene2cat.npy", g2c)
NCAT=int(np.asarray(g2c).max())+1; log(f"NCAT={NCAT}")

def sample_latlon(city, quota, seed=1):
    """road-weighted spatial sample -> [(pid,lat,lon)] on-disk-verified."""
    try:
        con=sqlite3.connect(f"file:{meta_db(city)}?mode=ro&immutable=1",uri=True)
        rows=[r for r in con.execute("SELECT panoid,lat,lon FROM gsv WHERE download=1") if r[1] and r[2]]; con.close()
    except Exception: rows=[]
    if len(rows)<quota:   # Vienna-style fallback: walk disk (no latlon) -> skip city here
        return []
    cell=lambda la,lo:(round(la,GRID),round(lo,GRID))
    roadw=collections.Counter(); import glob
    country=CITY_DIR[city].split("/")[0]
    for db in glob.glob(f"/global/scratch/users/cehou/data/SVIs/GSV/metadata/{country}/{city}/sampling_points_db/*.db"):
        try:
            c=sqlite3.connect(f"file:{db}?mode=ro&immutable=1",uri=True)
            for la,lo in c.execute("SELECT lat,lon FROM points"):
                if la and lo: roadw[cell(la,lo)]+=1
            c.close()
        except Exception: pass
    pc=[cell(r[1],r[2]) for r in rows]; ppc=collections.Counter(pc)
    w=np.array([(roadw.get(c,0)+1.0)/ppc[c] for c in pc],float); w/=w.sum()
    idx=np.random.default_rng(seed).choice(len(rows),size=min(len(rows),6*quota),replace=False,p=w)
    out=[]
    for i in idx:
        pid,la,lo=rows[i]
        if os.path.exists(imgpath(city,pid,0)): out.append((pid,float(la),float(lo)))
        if len(out)>=quota: break
    return out

# 2x2 montage layout: heading -> quadrant (front=TL, right=TR, back=BR, left=BL)
POS={0:(0,0),90:(128,0),180:(128,128),270:(0,128)}
recs=[]
for ci,city in enumerate(CITIES):
    samp=sample_latlon(city, PER)
    tdir=TH/city; tdir.mkdir(parents=True,exist_ok=True)
    got=0
    for pid,la,lo in samp:
        pils=[]; hs=[]                                          # all available headings of this pano
        for h in HEADINGS:
            p=imgpath(city,pid,h)
            if not os.path.exists(p): continue
            try: pils.append(Image.open(p).convert("RGB").resize((448,448),Image.BILINEAR)); hs.append(h)
            except: pass
        if not pils: continue
        feats,_=ext.batch(pils)                                 # (n,784,1024), order matches pils
        comp=np.zeros(NCAT,np.float32)                          # pool patches over all 4 views
        for j in range(len(pils)):
            zt=feats[j].to(dev)
            with torch.no_grad(): gm=sae.encode(zt).argmax(1).cpu().numpy()   # (G*G,) gene per patch
            comp+=np.bincount(g2c[gm],minlength=NCAT).astype(np.float32)
        s=comp.sum()
        if s<=0: continue
        comp/=s
        mont=Image.new("RGB",(256,256),(11,19,34))             # 2x2 montage of the four views
        for pj,h in zip(pils,hs): mont.paste(pj.resize((128,128)),POS[h])
        mont.save(tdir/f"{got}.jpg",quality=72)
        recs.append({"city":city,"pano_id":pid,"lat":la,"lon":lo,"ti":got,
                     **{f"cat{c}":float(comp[c]) for c in range(NCAT)}})
        got+=1
    log(f"{city}: {got} panos composed (4-heading pooled)")
import pandas as pd
pd.DataFrame(recs).to_parquet(W2/"panos.parquet")
log(f"[done] {len(recs)} panos -> web2/panos.parquet")
