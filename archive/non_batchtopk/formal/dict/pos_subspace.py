"""Estimate the POSITIONAL SUBSPACE of DINOv3 patch features: the directions along
which the per-grid-position MEAN feature deviates from the global mean. Projecting
these out (before L2-normalize) removes line/corner/edge/center positional genes at
the root. Outputs artifact_dirs_pos.npy (r,1024) for gpu_run --project-dirs."""
import os, numpy as np, torch, time
from pathlib import Path
from PIL import Image
from formal.gpu_run import stratified_panos, imgpath, MID, TOK
GOUT=Path("/global/scratch/users/cehou/urban-visual-gene/formal/formal_out_global")
RES=448; G=28
CITIES=["HongKong","Singapore","Amsterdam","CapeTown","Paris","SaoPaulo",
        "MexicoCity","Sydney","Jakarta","Dhaka","NewDelhi","Manila"]
PER=250   # panos/city, heading 0 (positional structure is shared across headings)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

from transformers import AutoImageProcessor, AutoModel
dev="cuda" if torch.cuda.is_available() else "cpu"
proc=AutoImageProcessor.from_pretrained(MID, token=TOK)
model=AutoModel.from_pretrained(MID, token=TOK).to(dev).eval()
if dev=="cuda": model=model.to(torch.bfloat16)
with torch.no_grad():
    d0=proc(images=Image.new("RGB",(RES,RES)),return_tensors="pt",size={"height":RES,"width":RES})
    px0=d0["pixel_values"].to(dev); px0=px0.to(torch.bfloat16) if dev=="cuda" else px0
    prefix=model(px0).last_hidden_state.shape[1]-G*G
log(f"model ready dev={dev} prefix={prefix}")

Spos=np.zeros((G*G,1024),np.float64); cnt=0
def flush(pils):
    global Spos,cnt
    d=proc(images=pils,return_tensors="pt",size={"height":RES,"width":RES})
    px=d["pixel_values"].to(dev); px=px.to(torch.bfloat16) if dev=="cuda" else px
    with torch.no_grad():
        h=model(px).last_hidden_state[:,prefix:,:].float()     # (B,784,1024) RAW pre-normalize
    Spos+=h.sum(0).double().cpu().numpy(); cnt+=h.shape[0]

buf=[]
for city in CITIES:
    cand=stratified_panos(city, PER, seed=1)
    on=[p for p in cand if os.path.exists(imgpath(city,p,0))][:PER]
    for pid in on:
        try: buf.append(Image.open(imgpath(city,pid,0)).convert("RGB").resize((RES,RES),Image.BILINEAR))
        except: continue
        if len(buf)>=32: flush(buf); buf=[]
    log(f"{city}: cumulative {cnt} imgs")
if buf: flush(buf)
log(f"accumulated {cnt} imgs")

Fpos=Spos/cnt                                                   # (784,1024) per-position mean
gmean=Fpos.mean(0,keepdims=True)
C=Fpos-gmean                                                    # position-dependent deviation
U,S,Vt=np.linalg.svd(C, full_matrices=False)                    # C = U diag(S) Vt
var=S**2; cum=np.cumsum(var)/var.sum()
log("singular values (top 20): "+", ".join(f"{s:.2f}" for s in S[:20]))
log("cum variance   (top 20): "+", ".join(f"{c:.3f}" for c in cum[:20]))
np.save(GOUT/"pos_dirs_full.npy", Vt[:20].astype(np.float32))   # keep spectrum for re-slicing
np.save(GOUT/"pos_singvals.npy", S.astype(np.float32))
renv=os.environ.get("POS_R")
r=int(renv) if renv else max(8,min(16,int(np.searchsorted(cum,0.90)+1)))
A=Vt[:r].astype(np.float32)                                     # (r,1024) positional directions
np.save(GOUT/"artifact_dirs_pos.npy", A)
log(f"[done] r={r} (cumvar={cum[r-1]:.3f}) -> artifact_dirs_pos.npy shape {A.shape}")
