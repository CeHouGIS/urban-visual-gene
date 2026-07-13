"""Run the quality filter over each city's actual sampling candidates (the panos
the pipeline draws) and write a per-city QC blocklist -> formal/qc/<city>.parquet.
stratified_panos() then auto-excludes is_bad panos on every future run."""
import os, numpy as np, pandas as pd, time
from multiprocessing import Pool
from pathlib import Path
from formal.gpu_run import stratified_panos, imgpath
from formal.quality_filter import analyze, flag_row
REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
QC=REPO/"formal"/"qc"; QC.mkdir(exist_ok=True)
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo",
        "MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
PER=3000; CAP=6000; SEEDS=[0,1,7]     # union of the seeds the pipeline uses
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

tot=0
for city in CITIES:
    seen=set(); items=[]
    for sd in SEEDS:
        for pid in stratified_panos(city, PER, seed=sd):
            if pid in seen: continue
            seen.add(pid); p=imgpath(city,pid,0)
            if os.path.exists(p): items.append((pid,str(p)))
        if len(items)>=CAP: break
    items=items[:CAP]
    with Pool(16) as pool: res=pool.map(analyze,[p for _,p in items])
    rows=[]
    for (pid,_),r in zip(items,res):
        b,l,t=flag_row(*r[1:])
        rows.append((str(pid),round(r[1],1),round(r[4],0),round(r[5],3),round(r[7],1),b,l,t,(b or l or t)))
    df=pd.DataFrame(rows,columns=["pano_id","bright","lapvar","blur_tilefrac","skytop","black","blur","tunnel","is_bad"])
    df.to_parquet(QC/f"{city}.parquet"); tot+=int(df["is_bad"].sum())
    log(f"{city:10s} QC {len(df):5d} | black {int(df.black.sum()):3d} blur {int(df.blur.sum()):3d} "
        f"tunnel {int(df.tunnel.sum()):3d} -> BAD {int(df.is_bad.sum()):3d} ({100*df.is_bad.mean():.1f}%)")
log(f"[done] blocklists -> formal/qc/<city>.parquet ; total flagged {tot:,}")
