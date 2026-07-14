"""Analyze all-12-city composition (配比) + classify scene morphotypes (分类).
CPU. Reads web2/panos.parquet -> cross-city composition figure, morphotype
figure, and manifest.json for the interactive site."""
import os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
W2=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global"))/"web2"
KTYPE=7
CATJSON=os.environ.get("CATJSON","")
if CATJSON and Path(CATJSON).exists():
    _cj=json.load(open(CATJSON)); CATN=[c["name"] for c in _cj]; CATC=[c["color"] for c in _cj]
else:
    CATN=["住宅立面·地面","立面·车辆","植被·开阔地","路面·商业立面","密集植被·田野",
          "路面·商铺","道路走廊·绿化","树冠","密集城市立面","墙面·植被"]
    CATC=["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff","#f6c945","#9b8cff","#ff9e64","#73daca"]
NCAT=len(CATN)
TYPC=["#ff5d5d","#4dd0e1","#ffd24d","#7ee787","#b388ff","#ff9e64","#5ad1ff"]
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo","MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
ZH={"HongKong":"香港","Singapore":"新加坡","Amsterdam":"阿姆斯特丹","CapeTown":"开普敦","Paris":"巴黎",
    "SaoPaulo":"圣保罗","MexicoCity":"墨西哥城","Sydney":"悉尼","Jakarta":"雅加达","Dhaka":"达卡","NewDelhi":"新德里","Manila":"马尼拉"}

df=pd.read_parquet(W2/"panos.parquet")
C=df[[f"cat{c}" for c in range(NCAT)]].values.astype(float)
print(f"{len(df)} panos, {df['city'].nunique()} cities")

# ---- 分类: cluster per-pano composition (CLR->PCA->KMeans) ----
def clr(x): x=x+1e-3; x=x/x.sum(1,keepdims=True); return np.log(x)-np.log(x).mean(1,keepdims=True)
Xp=PCA(n_components=min(9,NCAT-1)).fit_transform(clr(C))
lab=KMeans(KTYPE,n_init=10,random_state=0).fit_predict(Xp); df["type"]=lab
# name each type by its DISTINCTIVE category (highest lift vs global mean) + top raw
gmean=C.mean(0)+1e-6; typename={}; typeprof={}
for t in range(KTYPE):
    mc=C[lab==t].mean(0); lift=mc/gmean
    dist=max(range(NCAT),key=lambda i:lift[i] if mc[i]>0.04 else 0)     # most over-represented
    top=int(np.argmax(mc))                                              # most abundant
    a=CATN[dist].split("·")[0]; b=CATN[top].split("·")[0]
    typename[t]=a if a==b else f"{a}·{b}"
    typeprof[t]=[{"cat":int(i),"frac":round(float(mc[i]),3)} for i in np.argsort(-mc) if mc[i]>0.03]
print("type sizes:",np.bincount(lab).tolist())
print("type names:",typename)

# ---- manifest (charts rendered as HTML in the page; no CJK-broken matplotlib) ----
ct=pd.crosstab(df["city"],df["type"],normalize="index").reindex(CITIES).fillna(0)
cities_json=[]
for city in CITIES:
    sub=df[df["city"]==city]
    if len(sub)==0: continue
    panos=[{"lat":float(r.lat),"lon":float(r.lon),"t":int(r.type),"ti":int(r.ti),
            "c":[round(float(getattr(r,f"cat{k}")),2) for k in range(NCAT)]} for r in sub.itertuples()]
    cities_json.append({"name":city,"zh":ZH[city],"n":len(panos),
        "bounds":{"minlon":float(sub.lon.min()),"maxlon":float(sub.lon.max()),
                  "minlat":float(sub.lat.min()),"maxlat":float(sub.lat.max())},
        "comp":[round(float(sub[f"cat{k}"].mean()),3) for k in range(NCAT)],
        "typedist":[round(float(ct.loc[city,t]) if t in ct.columns else 0.0,3) for t in range(KTYPE)],
        "panos":panos})
man={"categories":[{"id":c,"name":CATN[c],"color":CATC[c]} for c in range(NCAT)],
     "types":[{"id":t,"name":typename[t],"color":TYPC[t],"profile":typeprof[t],"n":int((lab==t).sum())} for t in range(KTYPE)],
     "cities":cities_json}
json.dump(man,open(W2/"manifest.json","w"),ensure_ascii=False)
print("type names:",{t:man['types'][t]['name'] for t in range(KTYPE)})
print("saved manifest.json ("+str(len(cities_json))+" cities)")
