"""Run the trained QC CNN over each city's FULL downloaded panos (heading-0).
Shardable for a CPU SLURM array (--shard S --nshards N, round-robin over
(city,chunk) work units) + a --combine pass that merges parts -> qc_full/<city>.
parquet (pano_id, p_good/dark/blur/tunnel, pred, is_bad). stratified_panos
auto-excludes is_bad.

  # array task:   python -m formal.qc_net_cities --shard $ID --nshards 40 --chunk 100000
  # combine:      python -m formal.qc_net_cities --combine --thr 0.6
"""
import os, argparse, sqlite3, time, shutil, numpy as np, torch, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from formal.gpu_run import meta_db, imgpath, CITY_DIR
from formal.qc_net import Net, RES, CLASSES, MODEL
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
QCF=REPO/"formal"/"qc_full"; QCF.mkdir(exist_ok=True)
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo",
        "MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
COLS=["p_good","p_dark","p_blur","p_tunnel"]
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

class DS(Dataset):
    def __init__(s,city,ids): s.city=city; s.ids=ids
    def __len__(s): return len(s.ids)
    def __getitem__(s,i):
        try:
            im=Image.open(imgpath(s.city,s.ids[i],0)).convert("RGB").resize((RES,RES))
            return i, torch.from_numpy(np.asarray(im,np.float32)/255).permute(2,0,1)
        except Exception: return i, None
def collate(b):
    idx=[x[0] for x in b if x[1] is not None]; px=[x[1] for x in b if x[1] is not None]
    return (idx, torch.stack(px)) if px else ([],None)

def city_ids(city):
    con=sqlite3.connect(f"file:{meta_db(city)}?mode=ro&immutable=1",uri=True)
    ids=[r[0] for r in con.execute("SELECT panoid FROM gsv WHERE download=1")]; con.close()
    return ids

def build_units(chunk):
    units=[]
    for city in CITIES:
        con=sqlite3.connect(f"file:{meta_db(city)}?mode=ro&immutable=1",uri=True)
        n=con.execute("SELECT COUNT(*) FROM gsv WHERE download=1").fetchone()[0]; con.close()
        for ck in range((n+chunk-1)//chunk): units.append((city,ck))
    return units

def combine(thr):
    for city in CITIES:
        cdir=QCF/f"_tmp_{city}"; parts=sorted(cdir.glob("part_*.parquet")) if cdir.exists() else []
        if not parts: log(f"{city}: no parts — skip"); continue
        df=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True).drop_duplicates("pano_id")
        V=df[COLS].values; pred=V.argmax(1); mx=V.max(1)
        df["pred"]=[CLASSES[p] for p in pred]; df["is_bad"]=(pred!=0)&(mx>thr)
        df.to_parquet(QCF/f"{city}.parquet")
        cnt={CLASSES[c]:int((pred==c).sum()) for c in range(4)}
        log(f"{city} -> {len(df):,} | {cnt} | BAD {int(df['is_bad'].sum()):,} ({100*df['is_bad'].mean():.1f}%)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshards",type=int,default=1)
    ap.add_argument("--chunk",type=int,default=100000); ap.add_argument("--batch",type=int,default=512)
    ap.add_argument("--workers",type=int,default=8); ap.add_argument("--thr",type=float,default=0.6)
    ap.add_argument("--combine",action="store_true")
    a=ap.parse_args()
    if a.combine: combine(a.thr); log("[combine done]"); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=Net().to(dev); net.load_state_dict(torch.load(MODEL,map_location=dev)); net.eval()
    units=build_units(a.chunk)
    mine=[units[u] for u in range(len(units)) if u%a.nshards==a.shard]
    log(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(units)} work units, dev={dev}")
    idc={}; t0=time.time(); done=0
    for city,ck in mine:
        if city not in idc: idc[city]=city_ids(city)
        cdir=QCF/f"_tmp_{city}"; cdir.mkdir(exist_ok=True)
        pf=cdir/f"part_{ck:04d}.parquet"
        if pf.exists(): continue
        cids=idc[city][ck*a.chunk:(ck+1)*a.chunk]
        dl=DataLoader(DS(city,cids),batch_size=a.batch,num_workers=a.workers,collate_fn=collate,
                      prefetch_factor=4,persistent_workers=False)
        rows=[]
        for idx,px in dl:
            if px is None: continue
            with torch.no_grad(): P=net(px.to(dev)).softmax(1).cpu().numpy()
            for r,i in enumerate(idx): rows.append((cids[i],*[float(v) for v in P[r]]))
            done+=len(idx)
        pd.DataFrame(rows,columns=["pano_id"]+COLS).to_parquet(pf)
        log(f"  {city} chunk {ck} -> {len(rows)} panos ({done/(time.time()-t0):.0f} img/s cum)")
    log(f"shard {a.shard} done: {done:,} imgs")

if __name__=="__main__": main()
