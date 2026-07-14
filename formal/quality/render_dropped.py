"""Montage of the images the QC filter DROPPED, grouped by reason (black/blur/
tunnel), so they can be eyeballed. Reads formal/qc/<city>.parquet."""
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from formal.gpu_run import imgpath, CITY_DIR
QC=Path("/global/scratch/users/cehou/urban-visual-gene/formal/qc")
OUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global2/genes/diag/qc_dropped.png")
REASONS=[("black","纯黑 / 空白"),("blur","大片模糊"),("tunnel","隧道(启发式)")]
CAP=80; TH=128; COLS=10; rng=np.random.default_rng(0)
by={r:[] for r,_ in REASONS}; totals={}
for pf in sorted(QC.glob("*.parquet")):
    city=pf.stem
    if city not in CITY_DIR: continue
    df=pd.read_parquet(pf)
    for r,_ in REASONS:
        for pid in df.loc[df[r],"pano_id"]: by[r].append((city,str(pid)))
for r,_ in REASONS:
    totals[r]=len(by[r])
    if len(by[r])>CAP: by[r]=[by[r][i] for i in rng.choice(len(by[r]),CAP,replace=False)]

# layout
blocks=[]; H=0
for r,name in REASONS:
    n=len(by[r]); rows=(n+COLS-1)//COLS
    blocks.append((r,name,rows)); H+=24+rows*TH
W=COLS*TH
cv=Image.new("RGB",(W,H),(9,12,20)); dr=ImageDraw.Draw(cv); y=0
for r,name,rows in blocks:
    dr.rectangle([0,y,W,y+22],fill=(24,34,54))
    dr.text((8,y+5),f"{name}  —  共 {totals[r]} 张（示 {len(by[r])}）",fill=(220,230,245)); y+=24
    for i,(city,pid) in enumerate(by[r]):
        x=(i%COLS)*TH; yy=y+(i//COLS)*TH
        try: im=Image.open(imgpath(city,pid,0)).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(40,0,0))
        cv.paste(im,(x,yy)); dr.text((x+2,yy+2),city[:4],fill=(255,235,120))
    y+=rows*TH
cv.save(OUT)
print(f"saved {OUT}  totals: "+", ".join(f"{r}={totals[r]}" for r,_ in REASONS))
