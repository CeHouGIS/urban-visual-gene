"""Encode a cross-city pool for the 512-gene dashboard: per-image sparse Top-K
activations (idx+val over 784 patches) + 224px thumbnails, so any gene's
activation map can be reconstructed for jet overlays. GPU.
Each sampled pano contributes ALL 4 headings [0,90,180,270] as separate images —
so a gene's representative regions can be drawn from any viewing direction, not
just the forward view. Each row (=one heading image) has its own thumbnail; the
original<->activation toggle reads that thumbnail, so correspondence is exact."""
import os, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from formal.gpu_run import Extractor, SAE, stratified_panos, imgpath, CITY_DIR, HEADINGS
GOUT=Path(os.environ.get("DICTDIR","/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global"))
PROJ=os.environ.get("GENE_PROJ",str(GOUT.parent/"formal_out"/"artifact_dirs_mean.npy"))
OUT=GOUT/"genes"; TH=OUT/"thumbs"; TH.mkdir(parents=True,exist_ok=True)
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo","MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
PER=300; K=512; G=28; TOPK=32     # panos/city; x4 headings => ~1200 imgs/city
def log(*a): import time; print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

ext=Extractor(448, proj_path=PROJ)
dd=torch.load(GOUT/"sae_448_k512.pt",map_location="cpu")
dev="cuda" if torch.cuda.is_available() else "cpu"
sae=SAE(dd["D"],K,dd["topk"]).to(dev); sae.load_state_dict(dd["state"]); sae.eval()

idxs=[]; vals=[]; meta=[]; heads=[]; gi=0
for city in CITIES:
    cands=stratified_panos(city, PER, seed=7)
    on=[p for p in cands if os.path.exists(imgpath(city,p,0))][:PER]   # heading-0 = anchor for selection
    for pid in on:
        pils=[]; hs=[]                                          # all available headings of this pano
        for h in HEADINGS:
            p=imgpath(city,pid,h)
            if not os.path.exists(p): continue
            try: pils.append(Image.open(p).convert("RGB").resize((448,448),Image.BILINEAR)); hs.append(h)
            except: pass
        if not pils: continue
        feats,_=ext.batch(pils)                                 # (n,784,1024), order matches pils
        for j in range(len(pils)):
            zt=feats[j].to(dev)                                 # (784,1024)
            with torch.no_grad():
                a=sae.encode(zt)                                # (784,512) top-k sparse
                v,ix=a.topk(TOPK,dim=1)                         # (784,32)
            idxs.append(ix.cpu().numpy().astype(np.int16)); vals.append(v.cpu().numpy().astype(np.float16))
            pils[j].resize((224,224)).save(TH/f"{gi}.jpg",quality=80)
            meta.append((city,pid)); heads.append(hs[j]); gi+=1
    log(f"{city}: {sum(1 for m in meta if m[0]==city)} images encoded ({sum(1 for m in meta if m[0]==city)//4}~ panos x4 headings)")
idxs=np.stack(idxs); vals=np.stack(vals)                       # (N,784,32)
np.savez_compressed(OUT/"sparse_acts.npz", idx=idxs, val=vals,
                    city=np.array([m[0] for m in meta]), pano=np.array([m[1] for m in meta]),
                    heading=np.array(heads,np.int16))
np.save(OUT/"atoms.npy", dd["state"]["dec.weight"].T.numpy())  # (512,1024) for dendrogram
log(f"[done] {len(meta)} imgs, sparse {idxs.shape} -> genes/sparse_acts.npz")
