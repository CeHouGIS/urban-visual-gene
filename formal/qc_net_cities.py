"""Run the trained QC CNN over each city's FULL downloaded panos (heading-0),
GPU-batched, chunked+resumable, -> qc_full/<city>.parquet (pano_id, p_good/dark/
blur/tunnel, pred, is_bad). stratified_panos auto-excludes is_bad at sample time.

  python -m formal.qc_net_cities --thr 0.6 --batch 512 --workers 8
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--thr",type=float,default=0.6); ap.add_argument("--batch",type=int,default=512)
    ap.add_argument("--workers",type=int,default=8); ap.add_argument("--chunk",type=int,default=200000)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=Net().to(dev); net.load_state_dict(torch.load(MODEL,map_location=dev)); net.eval()
    log(f"QC net loaded dev={dev}, thr={a.thr}")
    COLS=["p_good","p_dark","p_blur","p_tunnel"]
    for city in CITIES:
        out=QCF/f"{city}.parquet"
        if out.exists(): log(f"{city}: done — skip"); continue
        con=sqlite3.connect(f"file:{meta_db(city)}?mode=ro&immutable=1",uri=True)
        ids=[r[0] for r in con.execute("SELECT panoid FROM gsv WHERE download=1")]; con.close()
        log(f"{city}: {len(ids):,} downloaded panos")
        cdir=QCF/f"_tmp_{city}"; cdir.mkdir(exist_ok=True)
        nch=(len(ids)+a.chunk-1)//a.chunk; t0=time.time(); done=0
        for ck in range(nch):
            pf=cdir/f"part_{ck:04d}.parquet"
            if pf.exists(): done+=a.chunk; continue
            cids=ids[ck*a.chunk:(ck+1)*a.chunk]
            dl=DataLoader(DS(city,cids),batch_size=a.batch,num_workers=a.workers,
                          collate_fn=collate,prefetch_factor=4,persistent_workers=False)
            rows=[]
            for idx,px in dl:
                if px is None: continue
                with torch.no_grad(): P=net(px.to(dev)).softmax(1).cpu().numpy()
                for r,i in enumerate(idx): rows.append((cids[i],*[float(v) for v in P[r]]))
                done+=len(idx)
                if done % (a.batch*80) < a.batch:
                    log(f"  {city} {done:,}/{len(ids):,} ({done/(time.time()-t0):.0f} img/s)")
            pd.DataFrame(rows,columns=["pano_id"]+COLS).to_parquet(pf)
        df=pd.concat([pd.read_parquet(p) for p in sorted(cdir.glob("part_*.parquet"))],ignore_index=True)
        V=df[COLS].values; pred=V.argmax(1); mx=V.max(1)
        df["pred"]=[CLASSES[p] for p in pred]; df["is_bad"]=(pred!=0)&(mx>a.thr)
        df.to_parquet(out); shutil.rmtree(cdir)
        n=len(df); bad=int(df["is_bad"].sum())
        cnt={CLASSES[c]:int((pred==c).sum()) for c in range(4)}
        log(f"{city} DONE {n:,} panos | {cnt} | BAD {bad:,} ({100*bad/max(n,1):.1f}%)")
    log("[all done] blocklists -> qc_full/<city>.parquet")

if __name__=="__main__": main()
