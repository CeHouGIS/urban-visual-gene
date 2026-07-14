"""4-city interactive street explorer assets from formal K=512 gene maps.
CPU only: reloads 4-heading images from disk, renders category overlays.
Outputs explorer_out/{manifest.json, street_data/<city>/...}."""
import os, json, shutil
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
OUT=REPO/"formal"/os.environ.get("EXPDIR","formal_out_expglobal"); EXP=OUT/"web"/"explorer";
DICTDIR=REPO/"formal"/os.environ.get("DICTDIR_NAME","formal_out_global")
import json as _json
CATJSON=os.environ.get("CATJSON","")
if EXP.exists(): shutil.rmtree(EXP)
DATA=EXP/"street_data"; DATA.mkdir(parents=True)
IMROOT=Path("/global/scratch/users/cehou/data/SVIs/GSV/images")
CITY_DIR={"HongKong":"China/HongKong","Singapore":"Singapore/Singapore",
          "Amsterdam":"Netherlands/Amsterdam","CapeTown":"SouthAfrica/CapeTown"}
K=512; G=28; NCAT=10; RES=224; HEADINGS=[0,90,180,270]
ARTIFACT=[57,306]                                       # DINOv3 left/right border-artifact genes
ROADS_PER_CITY=30; PTS=4
PALETTE=["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff","#f6c945","#9b8cff","#ff9e64","#73daca"]
def imgpath(city,pid,h): return IMROOT/CITY_DIR[city]/pid[0].lower()/pid[1].lower()/pid[2].lower()/f"{pid}_{h}.jpg"
def hx(c): c=c.lstrip('#'); return np.array([int(c[i:i+2],16) for i in (0,2,4)],np.float32)

# gene -> category (KMeans on K=512 atoms)
d=torch.load(DICTDIR/f"sae_448_k{K}.pt",map_location="cpu")  # categories from the chosen dict
atoms=d["state"]["dec.weight"].T.numpy()               # (K, D)
os.environ["OMP_NUM_THREADS"]="1"
if (DICTDIR/"gene2cat.npy").exists():
    g2c=np.load(DICTDIR/"gene2cat.npy"); print("loaded gene2cat.npy",flush=True)
else:
    from sklearn.cluster import KMeans
    g2c=KMeans(NCAT,n_init=10,random_state=0).fit_predict(atoms)
if CATJSON and Path(CATJSON).exists():
    _cj=_json.load(open(CATJSON)); CATS=[{"id":c["id"],"name":c["name"],"color":c["color"]} for c in _cj]
    PALETTE=[c["color"] for c in _cj]
else:
    CATS=[{"id":c,"name":f"类别{c}","color":PALETTE[c]} for c in range(NCAT)]
NCAT=len(CATS)
palrgb=np.stack([hx(PALETTE[c]) for c in range(NCAT)])
print("categories ready",flush=True)

def overlay(orig,gm):
    cat=np.where(gm>=0,g2c[np.clip(gm,0,K-1)],-1)
    col=np.zeros((G,G,3),np.float32)
    for c in range(NCAT): col[cat==c]=palrgb[c]
    up=np.array(Image.fromarray(col.astype(np.uint8)).resize((RES,RES),Image.NEAREST),np.float32)
    a=np.where((cat>=0)[...,None],0.5,0.0)
    a=np.array(Image.fromarray((np.repeat(a,3,2)*255).astype(np.uint8)).resize((RES,RES),Image.NEAREST),np.float32)/255
    return (orig.astype(np.float32)*(1-a)+up*a).astype(np.uint8)

import pandas as pd
cities_json=[]
for city in CITY_DIR:
    z=np.load(OUT/f"streets_{city}_k{K}.npz",allow_pickle=True)
    gm=z["gmaps"]                                        # clean dict: artifact removed at source (projection)
    road=z["road"]; pid=z["pano_id"]; lat=z["lat"]; lon=z["lon"]
    df=pd.DataFrame({"road":road,"pano_id":pid,"lat":lat,"lon":lon,"pi":np.arange(len(road))})
    uroads=list(dict.fromkeys(road))[:ROADS_PER_CITY]
    cdir=DATA/city; cdir.mkdir(parents=True)
    roads_json=[]
    for ri,rid in enumerate(uroads):
        sub=df[df["road"]==rid]
        if len(sub)>PTS: sub=sub.iloc[np.linspace(0,len(sub)-1,PTS).round().astype(int)]
        rdir=cdir/f"r{ri}"; rdir.mkdir()
        comp=np.zeros(NCAT); pts=[]
        for pj,(_,row) in enumerate(sub.iterrows()):
            pi=int(row["pi"]); imgs=[]
            for hj,h in enumerate(HEADINGS):
                try: orig=np.asarray(Image.open(imgpath(city,row["pano_id"],h)).convert("RGB").resize((RES,RES)))
                except: continue
                ov=overlay(orig,gm[pi,hj])
                of=f"{city}/r{ri}/p{pj}_{h}.jpg"; vf=f"{city}/r{ri}/p{pj}_{h}v.jpg"
                Image.fromarray(orig).save(DATA/of,quality=70); Image.fromarray(ov).save(DATA/vf,quality=70)
                imgs.append({"h":int(h),"o":"explorer/street_data/"+of,"v":"explorer/street_data/"+vf})
                cat=np.where(gm[pi,hj]>=0,g2c[np.clip(gm[pi,hj],0,K-1)],-1)
                for c in range(NCAT): comp[c]+=int((cat==c).sum())
            pts.append({"lat":float(row["lat"]),"lon":float(row["lon"]),"imgs":imgs})
        comp=comp/max(comp.sum(),1)
        roads_json.append({"id":f"r{ri}","label":str(rid)[:30],
            "composition":[{"cat":int(c),"frac":round(float(comp[c]),3)} for c in np.argsort(-comp) if comp[c]>0.01],
            "points":pts})
    lons=[p["lon"] for r in roads_json for p in r["points"]]; lats=[p["lat"] for r in roads_json for p in r["points"]]
    cities_json.append({"name":city,"roads":roads_json,
        "bounds":{"minlon":min(lons),"maxlon":max(lons),"minlat":min(lats),"maxlat":max(lats)}})
    print(f"{city}: {len(roads_json)} roads done",flush=True)

json.dump({"categories":CATS,"cities":cities_json},open(EXP/"manifest.json","w"),ensure_ascii=False)
sz=sum(f.stat().st_size for f in DATA.rglob("*.jpg"))/1e6
print(f"[done] {len(list(DATA.rglob('*.jpg')))} images, {sz:.1f} MB -> {EXP}",flush=True)
