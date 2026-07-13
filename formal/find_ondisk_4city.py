"""Precompute, per city, the matched panos that exist on disk with all 4
headings — fast (one listdir per prefix dir). Writes formal/ondisk/<city>.parquet
with [pano_id, matched_road_id, chainage_m, lat, lon] for on-disk-4-heading panos.
The formal GPU run's S2 reads these so street selection is robust (no inline guessing).
"""
import os, pandas as pd
from collections import defaultdict
from pathlib import Path

REPO=Path("/global/scratch/users/cehou/urban-visual-gene")
IMROOT=Path("/global/scratch/users/cehou/data/SVIs/GSV/images")
OD=REPO/"formal"/"ondisk"; OD.mkdir(parents=True,exist_ok=True)
CITY_DIR={"HongKong":"China/HongKong","Singapore":"Singapore/Singapore",
          "Amsterdam":"Netherlands/Amsterdam","CapeTown":"SouthAfrica/CapeTown"}

def scan(city):
    out=OD/f"{city}.parquet"
    if out.exists(): print(f"[{city}] exists, skip",flush=True); return
    d=REPO/"outputs"/CITY_DIR[city]
    rmp=pd.read_parquet(d/"road_matched_panos.parquet")
    keep=[c for c in ["pano_id","matched_road_id","chainage_m","lat","lon"] if c in rmp.columns]
    rmp=rmp[keep].copy()
    if "lat" not in rmp.columns:
        pf=pd.read_parquet(d/"pano_features.parquet",columns=["pano_id","lat","lon"])
        rmp=rmp.merge(pf,on="pano_id",how="left")
    root=IMROOT/CITY_DIR[city]
    dirs=defaultdict(list)
    for pid in rmp["pano_id"]:
        if len(pid)>=3: dirs[(pid[0],pid[1],pid[2])].append(pid)
    print(f"[{city}] {len(rmp)} panos, {len(dirs)} prefix dirs",flush=True)
    present=set(); done=0
    for (a,b,c),pids in dirs.items():
        try: files=set(os.listdir(root/a/b/c))
        except FileNotFoundError: files=set()
        for pid in pids:
            if all(f"{pid}_{h}.jpg" in files for h in (0,90,180,270)): present.add(pid)
        done+=1
        if done%15000==0: print(f"[{city}] {done}/{len(dirs)} dirs, {len(present)} on-disk",flush=True)
    on=rmp[rmp["pano_id"].isin(present)].copy()
    on.to_parquet(out)
    ppr=on.groupby("matched_road_id").size()
    print(f"[{city}] on-disk-4h: {len(on)} panos, roads>=3: {int((ppr>=3).sum())} -> {out.name}",flush=True)

if __name__=="__main__":
    import sys
    cities=sys.argv[1:] or list(CITY_DIR)
    for c in cities: scan(c)
    print("ALL DONE",flush=True)
