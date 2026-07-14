"""Fast low-quality image filter: flag PURE-BLACK, LARGE-BLUR and TUNNEL images.
CPU, multiprocessed, NO training / NO CNN.
  black : brightness + dark-pixel fraction
  blur  : tiled Laplacian focus (content present but edges smeared; skips sky/walls)
  tunnel: dim + NO bright sky at top (enclosed) + smooth walls  (heuristic)

  python -m formal.quality_filter --glob '/path/**/*.jpg' --out qc.parquet --workers 16
  python -m formal.quality_filter --metrics qc.parquet --calibrate
  python -m formal.quality_filter --metrics qc.parquet --apply
"""
import os, sys, argparse, glob as _glob, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
RES=256; GRID=4; TILE=RES//GRID
COLS=["path","bright","dark","std","lapvar","blur_tilefrac","warm","skytop","glow"]
# calibrated defaults (see --calibrate)
THRESH=dict(black_mean=14.0,dark_frac=0.97,std_min=6.0,
            blur_tilefrac=0.4,blur_lapvar=48.0,
            tunnel_bright=100.0,tunnel_sky=95.0,tunnel_lapvar=280.0,tunnel_warm=4.0,tunnel_glow=42.0)

def analyze(path):
    try:
        rgb=np.asarray(Image.open(path).convert("RGB").resize((RES,RES)),np.float32)
    except Exception:
        return (path,-1,-1,-1,-1,-1,0,-1,0)
    a=rgb@np.array([0.299,0.587,0.114],np.float32)          # luminance (RES,RES)
    bright=float(a.mean()); dark=float((a<20).mean()); std=float(a.std())
    lap=ndimage.laplace(a); lapvar=float(lap.var())
    at=a.reshape(GRID,TILE,GRID,TILE).transpose(0,2,1,3).reshape(GRID*GRID,-1)
    lt=lap.reshape(GRID,TILE,GRID,TILE).transpose(0,2,1,3).reshape(GRID*GRID,-1)
    astd=at.std(1); tvar=lt.var(1)
    blur_tilefrac=float(((tvar<12)&(astd>8)).mean())         # content but smeared
    warm=float(rgb[...,0].mean()-rgb[...,2].mean())          # R-B (sodium tunnels)
    skytop=float(a[:RES//4].mean())                          # top 25%: sky=bright, tunnel ceiling=dark
    thr=np.percentile(a,95); glow=float(a[a>=thr].mean()-bright)   # bright spot in dark scene
    return (path,bright,dark,std,lapvar,blur_tilefrac,warm,skytop,glow)

def flag_row(bright,dark,std,lapvar,blur_tilefrac,warm,skytop,glow):
    """-> (is_black, is_blur, is_tunnel)."""
    T=THRESH
    black=(bright<T['black_mean'])or(dark>T['dark_frac'])or(std<T['std_min'])or(bright<0)
    blur =(blur_tilefrac>T['blur_tilefrac'])or(0<=lapvar<T['blur_lapvar'])
    tunnel=(not black)and(bright<T['tunnel_bright'])and(skytop<T['tunnel_sky'])and(lapvar<T['tunnel_lapvar'])\
           and((warm>T['tunnel_warm'])or(glow>T['tunnel_glow']))   # smooth walls + directional/warm light
    return bool(black),bool(blur),bool(tunnel)

def montage(paths,title,out,n=40):
    paths=paths[:n]; cols=8; rows=(len(paths)+cols-1)//cols; TH=140
    cv=Image.new("RGB",(cols*TH,rows*TH+22),(12,12,18))
    from PIL import ImageDraw; ImageDraw.Draw(cv).text((6,4),title,fill=(230,230,240))
    for i,p in enumerate(paths):
        try: im=Image.open(p).convert("RGB").resize((TH,TH))
        except: im=Image.new("RGB",(TH,TH),(30,0,0))
        cv.paste(im,((i%cols)*TH,22+(i//cols)*TH))
    cv.save(out); print("  montage ->",out)

def flags_df(df):
    import numpy as np
    b,l,t=[],[],[]
    for r in df[COLS[1:]].itertuples(index=False):
        fb,fl,ft=flag_row(*r); b.append(fb); l.append(fl); t.append(ft)
    return np.array(b),np.array(l),np.array(t)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob"); ap.add_argument("--list"); ap.add_argument("--out",default="qc_metrics.parquet")
    ap.add_argument("--metrics"); ap.add_argument("--workers",type=int,default=max(1,os.cpu_count()-1))
    ap.add_argument("--calibrate",action="store_true"); ap.add_argument("--apply",action="store_true")
    ap.add_argument("--no-tunnel",action="store_true"); ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    import pandas as pd
    if a.metrics: df=pd.read_parquet(a.metrics)
    else:
        if a.glob: paths=sorted(_glob.glob(a.glob,recursive=True))
        elif a.list: paths=[l.strip() for l in open(a.list) if l.strip()]
        else: sys.exit("need --glob or --list (or --metrics)")
        if a.limit: paths=paths[:a.limit]
        print(f"analyzing {len(paths):,} images with {a.workers} workers ...",flush=True)
        os.environ["OMP_NUM_THREADS"]="1"; from multiprocessing import Pool; import time; t0=time.time()
        rows=[]
        with Pool(a.workers) as pool:
            for i,r in enumerate(pool.imap_unordered(analyze,paths,chunksize=64)):
                rows.append(r)
                if i%10000==0 and i: print(f"  {i:,}/{len(paths):,} ({i/(time.time()-t0):.0f} img/s)",flush=True)
        df=pd.DataFrame(rows,columns=COLS); df.to_parquet(a.out)
        print(f"[metrics] {len(df):,} imgs -> {a.out}  ({len(df)/(time.time()-t0):.0f} img/s)")

    ok=df[df.bright>=0]
    print(f"\n{len(df):,} imgs ({len(df)-len(ok)} unreadable)")
    for c in ["bright","std","lapvar","blur_tilefrac","skytop","warm"]:
        q=np.percentile(ok[c],[1,5,50,95,99]); print(f"  {c:14s} p1/5/50/95/99: "+" ".join(f"{v:.1f}" for v in q))
    isb,isl,ist=flags_df(df)
    if a.no_tunnel: ist[:]=False
    drop=isb|isl|ist
    print(f"\n  BLACK {int(isb.sum()):,} | BLUR {int(isl.sum()):,} | TUNNEL {int(ist.sum()):,}"
          f"  -> DROP {int(drop.sum()):,} ({100*drop.mean():.1f}%)  KEEP {int((~drop).sum()):,}")
    if a.calibrate:
        d=Path(a.metrics or a.out).parent
        montage(ok.sort_values("bright").path.tolist(),"darkest",d/"qc_darkest.png")
        montage(ok.sort_values("blur_tilefrac",ascending=False).path.tolist(),"large-blur",d/"qc_bigblur.png")
        tun=ok.assign(ts=ok.skytop+ok.bright+ok.lapvar/8).sort_values("ts")  # low = tunnel-like
        montage(tun.path.tolist(),"most tunnel-like (dim + no sky + smooth)",d/"qc_tunnel.png")
    if a.apply:
        base=Path(a.metrics or a.out).with_suffix("")
        df[~drop].path.to_csv(f"{base}.keep.txt",index=False,header=False)
        why=np.where(isb,"black","")+np.where(isl,"|blur","")+np.where(ist,"|tunnel","")
        df.assign(why=why)[drop][["path","why","bright","blur_tilefrac","skytop","lapvar"]].to_csv(f"{base}.drop.csv",index=False)
        print(f"\n[apply] keep -> {base}.keep.txt  drop -> {base}.drop.csv")

if __name__=="__main__": main()
