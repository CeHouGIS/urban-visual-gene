"""FULL-sample batched encoder: encode ALL downloaded panos of a city (heading-0)
with the global K=512 dict -> per-pano 10-category composition + lat/lon.
Multi-worker image loading + batched GPU forward + chunked resumable parquet.

  python -m formal.encode_city --city Jakarta --batch 64 --chunk 50000
"""
import os, argparse, sqlite3, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from formal.gpu_run import CITY_DIR, meta_db, imgpath, SAE

MID="facebook/dinov3-vitl16-pretrain-lvd1689m"
TOK=(Path.home()/".cache/huggingface/token").read_text().strip()
GOUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global")
ENC=GOUT/"encoded"; RES=448; G=28; K=512; NCAT=10
def log(*a): import time; print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

_proc=None
def get_proc():
    global _proc
    if _proc is None:
        from transformers import AutoImageProcessor
        _proc=AutoImageProcessor.from_pretrained(MID, token=TOK)
    return _proc

class PanoDS(Dataset):
    def __init__(self, city, ids): self.city=city; self.ids=ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        pid=self.ids[i]
        try:
            im=Image.open(imgpath(self.city,pid,0)).convert("RGB")
            px=get_proc()(images=im,return_tensors="pt",size={"height":RES,"width":RES})["pixel_values"][0]
            return i, px
        except Exception:
            return i, None

def collate(batch):
    idx=[b[0] for b in batch if b[1] is not None]
    px=[b[1] for b in batch if b[1] is not None]
    if not px: return [], None
    return idx, torch.stack(px)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--city",required=True)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--chunk",type=int,default=50000)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--max-chunks",type=int,default=0,help="0=all; >0 limits (smoke test)")
    args=ap.parse_args()
    outdir=ENC/args.city; outdir.mkdir(parents=True,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"

    # model + projection + SAE + gene->category
    from transformers import AutoModel
    model=AutoModel.from_pretrained(MID,token=TOK).eval().to(dev)
    if dev=="cuda": model=model.to(torch.bfloat16)
    with torch.no_grad():
        d0=get_proc()(images=Image.new("RGB",(RES,RES)),return_tensors="pt",size={"height":RES,"width":RES})
        px0=d0["pixel_values"].to(dev);  px0=px0.to(torch.bfloat16) if dev=="cuda" else px0
        prefix=model(px0).last_hidden_state.shape[1]-G*G
    A=torch.tensor(np.load(GOUT.parent/"formal_out"/"artifact_dirs_mean.npy"),dtype=torch.float32)
    Q,_=torch.linalg.qr(A.T); Pd=Q.T.to(dev)                       # (2,1024) projection basis
    dd=torch.load(GOUT/"sae_448_k512.pt",map_location="cpu")
    sae=SAE(dd["D"],K,dd["topk"]).to(dev); sae.load_state_dict(dd["state"]); sae.eval()
    os.environ["OMP_NUM_THREADS"]="4"; from sklearn.cluster import KMeans
    g2c=KMeans(NCAT,n_init=10,random_state=0).fit_predict(dd["state"]["dec.weight"].T.numpy())
    g2c=torch.tensor(g2c,dtype=torch.long,device=dev)
    log(f"{args.city} model ready, prefix={prefix}, dev={dev}")

    # all downloaded panos + latlon
    con=sqlite3.connect(f"file:{meta_db(args.city)}?mode=ro&immutable=1",uri=True)
    rows=[r for r in con.execute("SELECT panoid,lat,lon FROM gsv WHERE download=1") if r[1] and r[2]]; con.close()
    ids=[r[0] for r in rows]; lat={r[0]:r[1] for r in rows}; lon={r[0]:r[2] for r in rows}
    log(f"{args.city}: {len(ids):,} downloaded panos")

    import pandas as pd
    nchunks=(len(ids)+args.chunk-1)//args.chunk
    for ck in range(nchunks):
        pf=outdir/f"part_{ck:04d}.parquet"
        if pf.exists(): log(f"  chunk {ck}/{nchunks} cached"); continue
        cids=ids[ck*args.chunk:(ck+1)*args.chunk]
        dl=DataLoader(PanoDS(args.city,cids),batch_size=args.batch,num_workers=args.workers,
                      collate_fn=collate,prefetch_factor=4,persistent_workers=False)
        recs=[]; done=0
        for idx,px in dl:
            if px is None: continue
            px=px.to(dev)
            if dev=="cuda": px=px.to(torch.bfloat16)
            with torch.no_grad():
                h=model(px).last_hidden_state[:,prefix:,:].float()      # (B,784,1024)
                h=h-(h@Pd.t())@Pd                                       # project out artifact dirs
                h=F.normalize(h,dim=2)
                a=sae.encode(h.reshape(-1,1024)).argmax(1)             # (B*784,) gene
                cat=g2c[a].reshape(h.shape[0],G*G)                     # (B,784) category
                comp=torch.zeros(h.shape[0],NCAT,device=dev)
                comp.scatter_add_(1,cat,torch.ones_like(cat,dtype=torch.float32))
                comp=(comp/comp.sum(1,keepdim=True)).cpu().numpy()
            for r,i in enumerate(idx):
                pid=cids[i]; recs.append((pid,lat[pid],lon[pid],*[round(float(comp[r,c]),3) for c in range(NCAT)]))
            done+=len(idx)
            if done % (args.batch*40) < args.batch: log(f"  {args.city} chunk {ck}: {done:,}/{len(cids):,}")
        cols=["pano_id","lat","lon"]+[f"cat{c}" for c in range(NCAT)]
        pd.DataFrame(recs,columns=cols).to_parquet(pf)
        log(f"  {args.city} chunk {ck}/{nchunks} -> {pf.name} ({len(recs):,} panos)")
    log(f"{args.city} DONE")

if __name__=="__main__": main()
