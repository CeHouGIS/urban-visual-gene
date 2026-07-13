"""4-city street morphotypes from formal K=512 gene maps (CPU, no GPU).
Renders a per-city map + representative streets + cross-city composition."""
import numpy as np, pandas as pd, os
from pathlib import Path
from PIL import Image
OUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_clean")
WEB=OUT/"web"; WEB.mkdir(exist_ok=True)
K=512; G=28; NBAND=4; KCLUST=6
CITIES=["HongKong","Singapore","Amsterdam","CapeTown"]
band_of=np.repeat(np.arange(NBAND),int(np.ceil(G/NBAND)))[:G]

# ---- load all cities, build per-street band-composition descriptors ----
rows=[]   # (city, road, lat, lon, pi_global)
descs=[]; thumbs=[]
gi=0
for city in CITIES:
    z=np.load(OUT/f"streets_{city}_k{K}.npz",allow_pickle=True)
    gm=z["gmaps"]                                        # clean dict: artifact removed at source
    road=z["road"]; lat=z["lat"]; lon=z["lon"]
    df=pd.DataFrame({"road":road,"lat":lat,"lon":lon}); df["pi"]=np.arange(len(df))
    # per-point band histogram
    def pdesc(m):
        d=np.zeros((NBAND,K))
        for h in range(m.shape[0]):
            for r in range(G):
                for c in range(G):
                    g=m[h,r,c]
                    if g>=0: d[band_of[r],g]+=1
        return (d/np.clip(d.sum(1,keepdims=True),1,None)).reshape(-1)
    pt=np.stack([pdesc(gm[i]) for i in range(len(df))])
    for rid,sub in df.groupby("road"):
        idx=sub["pi"].values
        rows.append((city,str(rid),float(sub["lat"].mean()),float(sub["lon"].mean())))
        descs.append(pt[idx].mean(0))
        # representative thumbnail = middle point of the road
        thumbs.append((city,int(sub["pi"].values[len(sub)//2])))
    print(f"{city}: {df['road'].nunique()} roads")
R=pd.DataFrame(rows,columns=["city","road","lat","lon"]); D=np.stack(descs)
print(f"total {len(R)} streets across {R['city'].nunique()} cities, desc dim {D.shape[1]}")

# ---- CLR -> PCA -> KMeans (joint across cities) ----
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
os.environ["OMP_NUM_THREADS"]="1"
def clr(x): x=x+1e-4; x=x/x.sum(1,keepdims=True); return np.log(x)-np.log(x).mean(1,keepdims=True)
Xp=PCA(n_components=20).fit_transform(clr(D))
lab=KMeans(KCLUST,n_init=10,random_state=0).fit_predict(Xp)
R["type"]=lab
print("morphotype sizes:",np.bincount(lab).tolist())
print("city × type:\n",pd.crosstab(R["city"],R["type"]))

# ---- render ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BG,FG,MUT="#070b14","#dbe4f0","#8a9bb5"; pal=["#39d6ff","#ffb454","#22e0a1","#ff7ac6","#7c5cff","#5ad1ff"]
fig=plt.figure(figsize=(17,10)); fig.patch.set_facecolor(BG)
# --- top row: 4 city maps ---
for ci,city in enumerate(CITIES):
    ax=fig.add_axes([0.035+0.245*ci,0.70,0.205,0.24]); ax.set_facecolor(BG)
    sub=R[R["city"]==city]
    for t in range(KCLUST):
        m=sub["type"]==t; ax.scatter(sub["lon"][m],sub["lat"][m],s=16,c=pal[t],edgecolors="none")
    ax.set_title(city,color=FG,fontsize=11); ax.tick_params(colors=MUT,labelsize=6)
    [s.set_color(MUT) for s in ax.spines.values()]
# --- bottom-left: cross-city composition ---
axc=fig.add_axes([0.05,0.08,0.24,0.50]); axc.set_facecolor(BG)
ct=pd.crosstab(R["city"],R["type"],normalize="index").reindex(CITIES)
bottom=np.zeros(len(CITIES))
for t in range(KCLUST):
    v=ct[t].values if t in ct.columns else np.zeros(len(CITIES))
    axc.bar(range(len(CITIES)),v,bottom=bottom,color=pal[t],label=f"type {t}"); bottom+=v
axc.set_xticks(range(len(CITIES))); axc.set_xticklabels(CITIES,rotation=20,ha="right",fontsize=8)
axc.set_title("morphotype composition per city",color=FG,fontsize=11)
axc.tick_params(colors=MUT); [s.set_color(MUT) for s in axc.spines.values()]
axc.legend(facecolor=BG,edgecolor=MUT,labelcolor=FG,fontsize=8,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.13))
# --- bottom-right: representative streets per type (6 rows x 4 thumbs) ---
fig.text(0.34,0.62,"representative streets per morphotype",color=FG,fontsize=11)
rh=0.093
for t in range(KCLUST):
    members=[i for i in range(len(R)) if lab[i]==t]
    cen=Xp[members].mean(0); order=sorted(members,key=lambda i:((Xp[i]-cen)**2).sum())[:4]
    ytop=0.585-t*rh
    fig.text(0.34,ytop-0.01,f"● type {t}",color=pal[t],fontsize=10.5,fontweight="bold")
    fig.text(0.34,ytop-0.035,f"{len(members)} st.",color=MUT,fontsize=8)
    for j,i in enumerate(order):
        cty,pi=thumbs[i]
        try: im=np.array(Image.open(OUT/"thumbs"/cty/f"{pi}.jpg").resize((150,150)))
        except: im=np.zeros((150,150,3),np.uint8)
        a=fig.add_axes([0.40+j*0.145,ytop-0.075,0.085,0.078]); a.imshow(im); a.axis("off")
        for s in a.spines.values(): s.set_visible(True); s.set_color(pal[t]); s.set_linewidth(2)
fig.suptitle(f"4-city street morphotypes · formal K={K} dict @448 (240 streets, joint clustering)",color=FG,fontsize=14)
fig.savefig(WEB/"morphotypes_4city.png",dpi=115,facecolor=BG); plt.close(fig)
R.to_parquet(WEB/"morphotypes.parquet")
print("saved web/morphotypes_4city.png")
